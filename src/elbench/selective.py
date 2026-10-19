"""Selective prediction for condition extrapolation (v0.5, exploratory).

Question: can unreliable predictions be identified before their labels are
known? For the temperature-extrapolation tasks, each base model is paired with
inexpensive confidence signals:

  condition_distance  kelvin beyond the coldest (or warmest) training
                      temperature of the same formulation (larger = less confident)
  feature_distance    distance to the nearest training row in standardized
                      feature space (training-only scaler)
  ensemble_spread     standard deviation of per-tree log predictions (extra trees only)
  random              random rejection, averaged over repeats (reference)
  oracle              sorted by the actual error (diagnostic bound, not a method)

Outputs per task, base model and signal: risk-coverage on the test cases at
fixed coverage levels, and a development-fixed threshold (chosen on the task's
nested development fold for the target coverage) applied to the test cases,
with realized coverage, error and rejection rates by source group and
temperature band. Every test label here was inspected in earlier versions, so
all results are exploratory.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .config import Config
from .models import make_model
from .splits import load_manifest

SIGNALS_BY_MODEL = {
    "extra_trees": ["condition_distance", "feature_distance", "ensemble_spread"],
    "ridge": ["condition_distance", "feature_distance"],
    "arrhenius_local": ["condition_distance"],
}


def _fit(model, params, X, midx, tr, te):
    m = make_model(model, 0, params)
    m.fit(X.loc[tr], midx.loc[tr, "conductivity_mS_cm"].to_numpy(), midx.loc[tr])
    p = m.predict(X.loc[te], midx.loc[te])
    err = np.abs(np.log10(p.prediction_mS_cm.to_numpy()) - np.log10(midx.loc[te, "conductivity_mS_cm"].to_numpy()))
    return m, err


def _uncertainty(signal, m, X, midx, tr, te, colder_is_test: bool) -> np.ndarray:
    """Larger value = less confident."""
    if signal == "condition_distance":
        t_tr = midx.loc[tr].groupby("formulation_id").temperature_K
        edge = (t_tr.min() if colder_is_test else t_tr.max()).reindex(midx.loc[te, "formulation_id"]).to_numpy()
        T = midx.loc[te, "temperature_K"].to_numpy()
        return np.nan_to_num(edge - T if colder_is_test else T - edge, nan=np.inf)
    if signal == "feature_distance":
        sc = StandardScaler().fit(X.loc[tr].to_numpy())
        nn = NearestNeighbors(n_neighbors=1).fit(sc.transform(X.loc[tr].to_numpy()))
        return nn.kneighbors(sc.transform(X.loc[te].to_numpy()))[0][:, 0]
    if signal == "ensemble_spread":
        est = m.model
        Z = X.loc[te].to_numpy()
        return np.stack([t.predict(Z) for t in est.estimators_]).std(axis=0)
    raise ValueError(signal)


def _group_mean(err, groups) -> float:
    return float(pd.Series(err).groupby(groups).mean().mean()) if len(err) else np.nan


def run_selective(cfg: Config, meas: pd.DataFrame, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sel = cfg["investigation"]["selective"]
    track = cfg["investigation"]["primary_track"]
    midx = meas.set_index("measurement_id")
    runs = pd.read_csv(cfg.p("results") / "train_runs.csv")
    rng = np.random.default_rng(cfg["seed"])
    curves, fixed, rejected = [], [], []
    for task in sel["tasks"]:
        colder = cfg["tasks"][task]["kind"] == "temperature_transfer_low"
        man = load_manifest(cfg, task, track)
        outer = f"{task}/all"
        dev = f"{outer}::dev0"
        o, d = man[man.fold_id == outer], man[man.fold_id == dev]
        tr, te = o.measurement_id[o.role == "train"].to_numpy(), o.measurement_id[o.role == "test"].to_numpy()
        dtr, dte = d.measurement_id[d.role == "train"].to_numpy(), d.measurement_id[d.role == "dev"].to_numpy()
        groups = midx.loc[te, "source_group_id"].to_numpy()
        dgroups = midx.loc[dte, "source_group_id"].to_numpy()
        for model, signals in SIGNALS_BY_MODEL.items():
            r = runs[(runs.task_id == task) & (runs.track_id == track) & (runs.fold_id == outer) & (runs.model == model)]
            params = json.loads(r.chosen_config.iat[0]) if len(r) else {}
            with threadpool_limits(1):
                m, err = _fit(model, params, X, midx, tr, te)
                dm, derr = _fit(model, params, X, midx, dtr, dte)
                unc = {s: _uncertainty(s, m, X, midx, tr, te, colder) for s in signals}
                dunc = {s: _uncertainty(s, dm, X, midx, dtr, dte, colder) for s in signals}
            unc["oracle"] = err
            n = len(te)
            for s, u in unc.items():
                order = np.argsort(u, kind="stable")
                for c in sel["coverage_levels"]:
                    k = int(round(c * n))
                    idx = order[:k]
                    curves.append({"task_id": task, "model": model, "signal": s, "target_coverage": c, "cases": k, "group_mean_mae_log10": _group_mean(err[idx], groups[idx]), "sample_mae_log10": float(err[idx].mean()), "diagnostic_only": s == "oracle"})
            for c in sel["coverage_levels"]:
                k = int(round(c * n))
                vals = [(_group_mean(err[p[:k]], groups[p[:k]]), float(err[p[:k]].mean())) for p in (rng.permutation(n) for _ in range(sel["random_repeats"]))]
                curves.append({"task_id": task, "model": model, "signal": "random", "target_coverage": c, "cases": k, "group_mean_mae_log10": float(np.mean([v[0] for v in vals])), "sample_mae_log10": float(np.mean([v[1] for v in vals])), "diagnostic_only": False})
            # development-fixed thresholds
            tgt = sel["development_target_coverage"]
            T = midx.loc[te, "temperature_K"].to_numpy()
            band = pd.cut(T, [0, 233.2, 243.2, 253.2, 323.0, 400], labels=["<=233 K", "233-243 K", "243-253 K", "253-323 K", ">323 K"]).astype(str)
            for s in signals:
                thr = float(np.quantile(dunc[s][np.isfinite(dunc[s])], tgt)) if np.isfinite(dunc[s]).any() else np.inf
                keep = unc[s] <= thr
                fixed.append(
                    {
                        "task_id": task,
                        "model": model,
                        "signal": s,
                        "dev_target_coverage": tgt,
                        "dev_threshold": thr,
                        "dev_realized_coverage": float((dunc[s] <= thr).mean()),
                        "dev_mae_retained": _group_mean(derr[dunc[s] <= thr], dgroups[dunc[s] <= thr]),
                        "test_realized_coverage": float(keep.mean()),
                        "test_group_mean_mae_all": _group_mean(err, groups),
                        "test_group_mean_mae_retained": _group_mean(err[keep], groups[keep]),
                        "test_groups_fully_rejected": int(len(set(groups)) - len(set(groups[keep]))),
                    }
                )
                for kind, key in (("source_group", groups), ("temperature_band", band)):
                    t = pd.DataFrame({"key": key, "rejected": ~keep}).groupby("key").rejected.agg(["size", "mean"]).reset_index()
                    for row in t.itertuples():
                        rejected.append({"task_id": task, "model": model, "signal": s, "by": kind, "value": row.key, "cases": int(row.size), "rejection_rate": float(row.mean)})
    return pd.DataFrame(curves), pd.DataFrame(fixed), pd.DataFrame(rejected)
