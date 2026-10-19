"""Evaluate your own model on the frozen benchmark.

    from elbench.config import load_config
    from elbench.submission import iter_folds, score_predictions

    cfg = load_config("configs/benchmark_v0.1.yaml")
    preds = []
    for fold in iter_folds(cfg, "source_transfer", "molal_w"):
        model.fit(fold.X_train, fold.y_train)            # labels for training rows only
        preds.append(fold.prediction_frame(model.predict(fold.X_test)))
    result = score_predictions(cfg, "source_transfer", "molal_w", pd.concat(preds))

Prediction frames need columns fold_id, measurement_id, prediction_mS_cm and
optionally status (default "ok"). Rows are joined to labels by ID; duplicate,
missing or unexpected IDs are rejected. Development folds are exposed through
``fold.dev_folds`` so tuning never needs the outer test labels.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .curate import load_processed
from .evaluate import MAIN_MODELS, check_prediction_ids, _group_keys
from .features import track_features
from .metrics import group_table, paired_bootstrap, row_errors, summarize
from .splits import load_folds, load_manifest


@dataclass
class FoldData:
    task_id: str
    track_id: str
    fold_id: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    rows_train: pd.DataFrame
    rows_test: pd.DataFrame  # canonical inputs and provenance, without the target
    interpretation_label: str
    dev_folds: list["FoldData"] = field(default_factory=list)
    y_dev: pd.Series | None = None  # only set on development folds

    def prediction_frame(self, pred_mS_cm, status=None) -> pd.DataFrame:
        pred = np.asarray(pred_mS_cm, dtype=float)
        return pd.DataFrame(
            {
                "fold_id": self.fold_id,
                "measurement_id": self.X_test.index.to_numpy(),
                "prediction_mS_cm": pred,
                "status": status if status is not None else np.where(np.isfinite(pred), "ok", "failed"),
            }
        )


TARGET_COLUMNS = ["conductivity_mS_cm", "conductivity_original", "log10_conductivity", "conductivity_uncertainty_mS_cm"]


def iter_folds(cfg: Config, task_id: str, track_id: str, include_dev: bool = True, include_reserved: bool = False) -> Iterator[FoldData]:
    folds = load_folds(cfg)
    folds = folds[(folds.task_id == task_id) & (folds.track_id == track_id)]
    if folds.reserved.any() and not include_reserved:
        raise PermissionError(f"{task_id} is a reserved evaluation; pass include_reserved=True and record the exposure")
    meas, comps = load_processed(cfg)
    desc = pd.read_parquet(cfg.p("cache") / "descriptors.parquet")
    X = track_features(cfg, meas, comps, desc, track_id)
    rows = meas.set_index("measurement_id").loc[X.index]
    y = rows.conductivity_mS_cm
    man = load_manifest(cfg, task_id, track_id)

    def make(fid, test_role, label):
        m = man[man.fold_id == fid]
        tr = m.measurement_id[m.role == "train"].to_numpy()
        te = m.measurement_id[m.role == test_role].to_numpy()
        fd = FoldData(task_id, track_id, fid, X.loc[tr], y.loc[tr], X.loc[te], rows.loc[tr], rows.loc[te].drop(columns=TARGET_COLUMNS), label)
        if test_role == "dev":
            fd.y_dev = y.loc[te]
        return fd

    for f in folds.itertuples():
        fd = make(f.fold_id, "test", f.interpretation_label)
        if include_dev and isinstance(f.dev_folds, str) and f.dev_folds:
            fd.dev_folds = [make(d, "dev", f.interpretation_label) for d in f.dev_folds.split("|")]
        yield fd


def score_predictions(cfg: Config, task_id: str, track_id: str, predictions: pd.DataFrame, compare_to_baselines: bool = True, baselines: tuple[str, ...] = tuple(MAIN_MODELS)) -> dict[str, pd.DataFrame]:
    """Score a submission with the benchmark's metrics and paired bootstrap."""
    meas, _ = load_processed(cfg)
    meta = meas.set_index("measurement_id")
    truth = meta.conductivity_mS_cm
    folds = load_folds(cfg)
    folds = folds[(folds.task_id == task_id) & (folds.track_id == track_id)]
    man = load_manifest(cfg, task_id, track_id)
    predictions = predictions.copy()
    if "status" not in predictions:
        predictions["status"] = "ok"
    predictions["seed"] = predictions.get("seed", 0)
    unknown = set(predictions.fold_id) - set(folds.fold_id)
    if unknown:
        raise ValueError(f"unknown fold ids {sorted(unknown)[:3]}")
    errs = []
    for f in folds.itertuples():
        test = man[(man.fold_id == f.fold_id) & (man.role == "test")].measurement_id.to_numpy()
        p = predictions[predictions.fold_id == f.fold_id]
        check_prediction_ids(p, test, f"submission/{f.fold_id}")
        e = row_errors(p.set_index("measurement_id").loc[test].reset_index(), truth)
        e["fold_id"] = f.fold_id
        e["group"] = _group_keys(f.kind, f.score_group, f.heldout_unit, meta.loc[test]).to_numpy()
        errs.append(e)
    e = pd.concat(errs, ignore_index=True)
    gt = group_table(e, e.group)
    out = {"summary": pd.DataFrame([summarize(e, gt)]), "groups": gt.reset_index()}
    if compare_to_baselines:
        out["comparison"], out["comparison_cases"] = compare_on_common_cases(cfg, task_id, track_id, e, baselines)
    return out


def compare_on_common_cases(cfg: Config, task_id: str, track_id: str, sub: pd.DataFrame, baselines) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired comparison on one common set of cases.

    A case is (fold_id, measurement_id). Only cases with a valid prediction from
    the submission AND every compared baseline are used, so every model is
    scored on exactly the same cases; baselines with any invalid prediction are
    left out (incomplete). Returns the bootstrap table and a case-count table."""
    rows = pd.read_parquet(cfg.p("results") / "tables" / "row_errors.parquet")
    rows = rows[(rows.task_id == task_id) & (rows.track_id == track_id) & rows.model.isin(baselines)]
    complete = [m for m, g in rows.groupby("model") if bool(g.valid.all())]
    incomplete = sorted(set(rows.model) - set(complete))
    rows = rows[rows.model.isin(complete)]
    keys = ["fold_id", "measurement_id"]
    cases = set(map(tuple, sub.loc[sub.valid, keys].to_numpy()))
    for _, g in rows.groupby("model"):
        cases &= set(map(tuple, g.loc[g.valid, keys].to_numpy()))
    idx = pd.MultiIndex.from_tuples(sorted(cases), names=keys)
    frames = {m: g.set_index(keys).reindex(idx) for m, g in rows.groupby("model")}
    frames["submission"] = sub.set_index(keys).reindex(idx)
    G = pd.DataFrame({m: f.groupby("group").abs_log_err.mean() for m, f in frames.items()})
    counts = pd.DataFrame([{"model": m, "cases_used": int(f.abs_log_err.notna().sum())} for m, f in frames.items()])
    counts["common_cases"] = len(idx)
    counts["excluded_incomplete_baselines"] = ",".join(incomplete)
    return paired_bootstrap(G, cfg["evaluation"]["n_bootstrap"], cfg["evaluation"]["bootstrap_seed"], reference="submission"), counts
