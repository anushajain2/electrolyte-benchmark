"""Separate "better labels" from "an easier test set" (v0.5).

A label-reliability rule (the digitization floor, R14) can change a score in
two ways: by changing which cases are evaluated, and by changing what the model
is trained on. Both comparisons below keep the reference version's fold
membership fixed.

  evaluation_filtering  predictions held fixed; scores on all test cases and on
                        the retained subset, with coverage and what disappears
  training_cleaning     the same outer folds refit with and without the floored
                        training rows; both scored on identical, audited test
                        cases (those the rule would retain)

Floor definitions come from investigation.label_effects in the config:
evidence tiers (verified only / verified + inferred) x thresholds.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from .config import Config
from .models import make_model
from .splits import load_folds, load_manifest


def floor_mask(cfg: Config, meas: pd.DataFrame, statuses: list[str], threshold: float) -> pd.Series:
    """True for measurements the digitization floor would remove."""
    le = cfg["investigation"]["label_effects"]
    pres = pd.read_csv(cfg.root / le["presentation_map"])
    src = set(pres[(pres.presentation == "linear_axis_figure") & pres.status.isin(statuses)].source_doi)
    k = meas.conductivity_mS_cm
    return meas.source_doi.isin(src) & (k > 0) & (k < threshold)


def _definitions(cfg: Config):
    le = cfg["investigation"]["label_effects"]
    for tier, statuses in le["tiers"].items():
        for thr in le["thresholds"]:
            yield f"{tier}@{thr}", statuses, thr


def evaluation_filtering(cfg: Config, meas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    track = cfg["investigation"]["primary_track"]
    rows = pd.read_parquet(cfg.p("results") / "tables" / "row_errors.parquet")
    rows = rows[(rows.track_id == track) & rows.valid]
    midx = meas.set_index("measurement_id")
    scores, lost = [], []
    for name, statuses, thr in _definitions(cfg):
        drop = set(midx.index[floor_mask(cfg, midx.reset_index(), statuses, thr).to_numpy()])
        for (task, model), r in rows.groupby(["task_id", "model"]):
            keep = ~r.measurement_id.isin(drop)
            full = r.groupby("group").abs_log_err.mean()
            kept = r[keep].groupby("group").abs_log_err.mean()
            scores.append(
                {
                    "definition": name,
                    "task_id": task,
                    "model": model,
                    "cases_all": len(r),
                    "cases_retained": int(keep.sum()),
                    "coverage": float(keep.mean()),
                    "groups_all": int(full.size),
                    "groups_retained": int(kept.size),
                    "group_mean_mae_all": float(full.mean()),
                    "group_mean_mae_retained": float(kept.mean()) if kept.size else np.nan,
                    "change": float(kept.mean() - full.mean()) if kept.size else np.nan,
                }
            )
        # what disappears (model-independent): per task
        for task, r in rows[rows.model == rows.model.iloc[0]].groupby("task_id"):
            gone = midx.loc[r.measurement_id[r.measurement_id.isin(drop)]]
            if not len(gone):
                continue
            stay = midx.loc[r.measurement_id[~r.measurement_id.isin(drop)]]
            tb = pd.cut(gone.temperature_K, [0, 243.2, 263.2, 283.2, 303.2, 400], labels=["<=243 K", "243-263 K", "263-283 K", "283-303 K", ">303 K"])
            lost.append(
                {
                    "definition": name,
                    "task_id": task,
                    "cases_removed": len(gone),
                    "formulations_removed_entirely": len(set(gone.formulation_id) - set(stay.formulation_id)),
                    "sources_affected": "; ".join(sorted(gone.source_doi.unique())),
                    "source_groups_removed_entirely": len(set(gone.source_group_id) - set(stay.source_group_id)),
                    **{f"removed_{k}": int(v) for k, v in tb.value_counts().sort_index().items()},
                }
            )
    return pd.DataFrame(scores), pd.DataFrame(lost)


def _fit_errors(model, params, X, midx, tr, te):
    with threadpool_limits(1):
        m = make_model(model, 0, params)
        m.fit(X.loc[tr], midx.loc[tr, "conductivity_mS_cm"].to_numpy(), midx.loc[tr])
        p = m.predict(X.loc[te], midx.loc[te]).prediction_mS_cm.to_numpy()
        return np.abs(np.log10(p) - np.log10(midx.loc[te, "conductivity_mS_cm"].to_numpy()))


def training_cleaning(cfg: Config, meas: pd.DataFrame, X: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    le = cfg["investigation"]["label_effects"]
    track = cfg["investigation"]["primary_track"]
    midx = meas.set_index("measurement_id")
    folds = load_folds(cfg)
    folds = folds[(folds.track_id == track) & ~folds.reserved]
    runs = pd.read_csv(cfg.p("results") / "train_runs.csv")
    thr = le["training_threshold"]
    plans, info = [], []
    for name, statuses in le["tiers"].items():
        drop = set(midx.index[floor_mask(cfg, midx.reset_index(), statuses, thr).to_numpy()])
        for f in folds.itertuples():
            man = load_manifest(cfg, f.task_id, track)
            o = man[man.fold_id == f.fold_id]
            tr = o.measurement_id[o.role == "train"].to_numpy()
            te = o.measurement_id[o.role == "test"].to_numpy()
            te_audited = te[~np.isin(te, list(drop))]  # identical test cases for both arms
            tr_clean = tr[~np.isin(tr, list(drop))]
            if not len(te_audited) or len(tr_clean) == len(tr):
                continue
            for model in le["models"]:
                r = runs[(runs.task_id == f.task_id) & (runs.track_id == track) & (runs.fold_id == f.fold_id) & (runs.model == model)]
                params = json.loads(r.chosen_config.iat[0]) if len(r) else cfg["investigation"]["fixed_configs"].get(model, {})
                for arm, trn in (("trained_on_all", tr), ("trained_on_cleaned", tr_clean)):
                    plans.append((name, f.task_id, f.fold_id, model, arm, params, trn, te_audited))
            info.append({"tier": name, "task_id": f.task_id, "fold_id": f.fold_id, "train_rows": len(tr), "train_rows_removed": len(tr) - len(tr_clean), "test_cases_audited": len(te_audited), "test_cases_all": len(te)})
    if not plans:
        return pd.DataFrame(columns=["tier", "task_id", "model", "trained_on_all", "trained_on_cleaned", "training_effect"])
    errs = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_fit_errors)(m, p, X, midx, trn, te) for _, _, _, m, _, p, trn, te in plans)
    out = []
    for (name, task, fold, model, arm, _, _, te), e in zip(plans, errs):
        grp = midx.loc[te, "source_group_id"].to_numpy() if task != "component_transfer" else np.full(len(te), fold)
        g = pd.Series(e).groupby(grp).mean()
        out.append({"tier": name, "task_id": task, "fold_id": fold, "model": model, "arm": arm, "cases": len(te), **{f"mae_group::{k}": v for k, v in g.items()}})
    res = pd.DataFrame(out)
    # task-level: average groups across folds (groups are disjoint across folds except interpolation)
    long = res.melt(id_vars=["tier", "task_id", "fold_id", "model", "arm", "cases"], var_name="group", value_name="mae").dropna(subset=["mae"])
    long["group"] = long.group.str.replace("mae_group::", "", regex=False)
    task = long.groupby(["tier", "task_id", "model", "arm", "group"]).mae.mean().groupby(["tier", "task_id", "model", "arm"]).mean().unstack("arm").reset_index()
    task["training_effect"] = task["trained_on_cleaned"] - task["trained_on_all"]
    counts = pd.DataFrame(info).groupby(["tier", "task_id"]).agg(folds=("fold_id", "size"), train_rows_removed=("train_rows_removed", "sum"), test_cases_audited=("test_cases_audited", "sum"), test_cases_all=("test_cases_all", "sum")).reset_index()
    return task.merge(counts, on=["tier", "task_id"], how="left")
