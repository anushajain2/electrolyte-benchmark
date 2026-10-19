"""Score saved predictions against frozen manifests.

Labels are joined by measurement_id. A prediction file with duplicate,
missing or unexpected IDs for any seed is rejected, so a model can never be
scored against the wrong row order.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import Config, RunManifest
from .curate import load_processed
from .metrics import group_table, paired_bootstrap, row_errors, summarize
from .splits import load_folds, load_manifest
from .train import log_reserved_exposure, safe

MAIN_MODELS = ["median", "knn", "ridge", "extra_trees", "hist_gb"]
DIAGNOSTIC_MODELS = ["ridge_raw"]
LOCAL_MODELS = ["arrhenius_local"]


class PredictionIDError(ValueError):
    pass


def check_prediction_ids(pred: pd.DataFrame, expected: np.ndarray, where: str) -> None:
    for seed, p in pred.groupby("seed"):
        ids = p.measurement_id
        dup = ids[ids.duplicated()].unique()
        missing = np.setdiff1d(expected, ids.to_numpy())
        unexpected = np.setdiff1d(ids.to_numpy(), expected)
        if len(dup) or len(missing) or len(unexpected):
            raise PredictionIDError(
                f"{where} seed {seed}: {len(dup)} duplicate, {len(missing)} missing, {len(unexpected)} unexpected IDs"
                + (f" (e.g. dup={dup[:3].tolist()}, missing={missing[:3].tolist()}, unexpected={unexpected[:3].tolist()})")
            )


def _group_keys(task_kind: str, score_group: str, fold_unit: str, meta: pd.DataFrame) -> pd.Series:
    if score_group == "heldout_component":
        return pd.Series(fold_unit, index=meta.index)
    return meta[score_group]


def score_fold(pred: pd.DataFrame, test: pd.DataFrame, truth: pd.Series, meta: pd.DataFrame, fold: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-row seed-averaged errors and per-seed group tables for one (fold, model)."""
    expected = test.measurement_id.to_numpy()
    check_prediction_ids(pred, expected, f"{fold.task_id}/{fold.track_id}/{fold.fold_id}/{pred.model.iat[0]}")
    per_seed = []
    gts = []
    for seed, p in pred.groupby("seed", sort=True):
        p = p.set_index("measurement_id").loc[expected].reset_index()
        e = row_errors(p, truth)
        e["seed"] = seed
        e["prediction_mS_cm"] = p.prediction_mS_cm.to_numpy()
        per_seed.append(e)
        groups = _group_keys(fold.kind, fold.score_group, fold.heldout_unit, meta.loc[expected])
        gt = group_table(e, groups)
        gt["seed"] = seed
        gts.append(gt)
    es = pd.concat(per_seed)
    rows = es.groupby("measurement_id", sort=False).agg(
        y_mS_cm=("y_mS_cm", "first"),
        abs_log_err=("abs_log_err", "mean"),
        signed_log_err=("signed_log_err", "mean"),
        abs_err_mS_cm=("abs_err_mS_cm", "mean"),
        valid_frac=("valid", "mean"),
        pred_geomean_mS_cm=("prediction_mS_cm", lambda v: float(10 ** np.mean(np.log10(v[v > 0]))) if (v > 0).any() else np.nan),
        pred_sd_log10=("prediction_mS_cm", lambda v: float(np.std(np.log10(v[v > 0]))) if (v > 0).sum() > 1 else 0.0),
    ).reindex(expected).reset_index()
    rows["valid"] = rows.valid_frac == 1.0
    rows["subset"] = test.set_index("measurement_id").subset.reindex(expected).to_numpy()
    rows["group"] = _group_keys(fold.kind, fold.score_group, fold.heldout_unit, meta.loc[expected]).to_numpy()
    return rows, pd.concat(gts).reset_index()


def _subsets(kind: str) -> dict[str, callable]:
    s = {"all": lambda r: np.ones(len(r), bool)}
    if kind == "source_transfer":
        s["exact_formulation_match"] = lambda r: (r.subset == "exact_formulation_match").to_numpy()
    if kind == "external":
        s["common_support"] = lambda r: (r.subset == "common_support").to_numpy()
        s["outside_support"] = lambda r: (r.subset == "outside_support").to_numpy()
    if kind == "component_transfer":
        s["supported_folds"] = lambda r: (r.interpretation_label == "supported_comparison").to_numpy()
        s["combined_folds"] = lambda r: (r.interpretation_label == "combined_shift").to_numpy()
        s["familiar_source_rows"] = lambda r: (r.subset == "familiar_source").to_numpy()
    return s


def _paired_and_jackknife(Gm: pd.DataFrame, ref: str, task: str, track: str, comparison: str, loo_rows: list) -> list[dict]:
    """Per-group paired differences against the reference model, and the
    group-mean difference recomputed with each group left out in turn."""
    rows = []
    if ref not in Gm.columns:
        return rows
    G = Gm.dropna()
    for m in G.columns:
        if m == ref:
            continue
        d = G[m] - G[ref]
        for g, v in d.items():
            rows.append({"task_id": task, "track_id": track, "comparison": comparison, "model": m, "group_id": g, "mae_model": G.at[g, m], "mae_reference": G.at[g, ref], "diff_vs_reference": v})
        full = float(d.mean())
        for g in G.index:
            rest = d.drop(index=g)
            loo_rows.append({"task_id": task, "track_id": track, "comparison": comparison, "model": m, "left_out_group": g, "diff_vs_reference_full": full, "diff_vs_reference_without_group": float(rest.mean()) if len(rest) else np.nan, "sign_flips": bool(len(rest) and np.sign(rest.mean()) != np.sign(full))})
    return rows


def evaluate(cfg: Config, include_reserved: bool = False) -> dict[str, pd.DataFrame]:
    man = RunManifest("evaluate", cfg)
    meas, _ = load_processed(cfg)
    meta = meas.set_index("measurement_id")
    truth = meta.conductivity_mS_cm
    folds = load_folds(cfg)
    if not include_reserved:
        folds = folds[~folds.reserved]
    pred_root = cfg.p("results") / "predictions"
    all_rows, all_gt = [], []
    for fold in folds.itertuples(index=False):
        manifest = load_manifest(cfg, fold.task_id, fold.track_id)
        test = manifest[(manifest.fold_id == fold.fold_id) & (manifest.role == "test")]
        d = pred_root / fold.task_id / fold.track_id / safe(fold.fold_id)
        for path in sorted(d.glob("*.parquet")):
            pred = pd.read_parquet(path)
            rows, gt = score_fold(pred, test, truth, meta, pd.Series(fold._asdict()))
            for df in (rows, gt):
                df.insert(0, "model", path.stem)
                df.insert(0, "fold_id", fold.fold_id)
                df.insert(0, "track_id", fold.track_id)
                df.insert(0, "task_id", fold.task_id)
            rows["interpretation_label"] = fold.interpretation_label
            gt["interpretation_label"] = fold.interpretation_label
            all_rows.append(rows)
            all_gt.append(gt)
        if fold.reserved:
            log_reserved_exposure(cfg, man.run_id, "evaluate", f"{fold.task_id}/{fold.track_id}: reserved labels scored")
    rows = pd.concat(all_rows, ignore_index=True)
    elig = pd.read_csv(cfg.p("splits") / "eligibility.csv")
    level = elig.groupby(["task_id", "track_id"]).reporting_level.first()
    kinds = folds.groupby(["task_id", "track_id"]).kind.first()

    summary_rows, group_rows, boot_rows, rank_rows, pg_rows, loo_rows = [], [], [], [], [], []
    nb, bs = cfg["evaluation"]["n_bootstrap"], cfg["evaluation"]["bootstrap_seed"]
    ref = cfg["models"]["reference_model"]
    for (task, track), r_tt in rows.groupby(["task_id", "track_id"], sort=True):
        kind = kinds[(task, track)]
        for subset, sel in _subsets(kind).items():
            r = r_tt[sel(r_tt)]
            if not len(r):
                continue
            # group-level, seed-averaged
            G = {}
            for model, rm in r.groupby("model"):
                gt = group_table(rm.rename(columns={}), rm.group)
                G[model] = gt.mae_log10
                s = summarize(rm, gt)
                summary_rows.append({"task_id": task, "track_id": track, "subset": subset, "model": model, "reporting_level": level.get((task, track), ""), **s})
                if subset == "all":
                    lab = rm.groupby("group").interpretation_label.first()
                    group_rows.append(gt.assign(task_id=task, track_id=track, model=model, interpretation_label=lab).reset_index())
            # paired bootstrap on rows valid for every compared model
            for family_set, name in [(MAIN_MODELS, "main"), (MAIN_MODELS + LOCAL_MODELS, "with_local")]:
                present = [m for m in family_set if m in G]
                if name == "with_local" and not any(m in G for m in LOCAL_MODELS):
                    continue
                rr = r[r.model.isin(present)]
                ok = rr.groupby("measurement_id").valid.all()
                common = ok[ok].index
                rr = rr[rr.measurement_id.isin(common)]
                Gm = rr.groupby(["group", "model"]).abs_log_err.mean().unstack()[present]
                b = paired_bootstrap(Gm, nb, bs, reference=ref)
                b.insert(0, "comparison", name)
                b.insert(0, "subset", subset)
                b.insert(0, "track_id", track)
                b.insert(0, "task_id", task)
                b["n_common_rows"] = len(common)
                b["note"] = "few groups: read group-level results individually" if len(Gm.dropna()) < cfg["reporting"]["headline_min_units"] else ""
                boot_rows.append(b)
                if subset == "all" and name in ("main", "with_local"):
                    pg_rows.extend(_paired_and_jackknife(Gm, ref, task, track, name, loo_rows))
                if subset == "all" and name == "main":
                    ranks = Gm.rank(axis=1, method="min")
                    for g, rk in ranks.iterrows():
                        rank_rows.append({"task_id": task, "track_id": track, "group": g, **{f"rank_{m}": int(v) for m, v in rk.items() if pd.notna(v)}, "best_model": rk.idxmin()})
    # confirmation runs never overwrite the reproducible (non-reserved) tables
    out = cfg.p("results") / ("tables_confirm" if include_reserved else "tables")
    out.mkdir(parents=True, exist_ok=True)
    tables = {
        "summary": pd.DataFrame(summary_rows),
        "group_metrics": pd.concat(group_rows, ignore_index=True).rename(columns={"group": "group_id"}),
        "bootstrap": pd.concat(boot_rows, ignore_index=True),
        "group_rankings": pd.DataFrame(rank_rows),
        "seed_group_metrics": pd.concat(all_gt, ignore_index=True),
        "paired_group_differences": pd.DataFrame(pg_rows),
        "leave_one_group_out": pd.DataFrame(loo_rows),
    }
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
        man.add_output(out / f"{name}.csv")
    rows.to_parquet(out / "row_errors.parquet", index=False)
    man.add_output(out / "row_errors.parquet")
    man.notes = {"include_reserved": include_reserved, "n_scored_rows": int(len(rows))}
    man.write(cfg.p("results") / "runs")
    (out / "README.json").write_text(
        json.dumps(
            {
                "primary_metric": "group_mean_mae_log10: MAE of log10(conductivity / mS cm^-1) averaged within each score group, then equally across groups",
                "seeds": "stochastic models: metrics are computed per seed; row_errors averages errors over seeds",
                "bootstrap": "paired over groups, same resampled groups for every model, restricted to rows valid for all compared models",
                "invalid": "rows with status != ok or nonpositive predictions count as invalid and are excluded from log metrics; see invalid_rate",
            },
            indent=1,
        )
    )
    return tables
