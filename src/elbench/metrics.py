"""Metrics: log-scale MAE averaged within score groups, then across groups.

Primary metric: group-mean MAE of log10(conductivity / mS cm^-1). Rows whose
prediction is not scorable on the log scale (status != ok or prediction <= 0)
are never dropped silently: they are counted as invalid and reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FACTOR_OF_TWO_LOG10 = np.log10(2.0)  # ~0.301


def row_errors(pred: pd.DataFrame, truth_mS_cm: pd.Series) -> pd.DataFrame:
    """``pred`` has measurement_id, prediction_mS_cm, status; ``truth`` is indexed by measurement_id."""
    y = truth_mS_cm.reindex(pred.measurement_id).to_numpy()
    p = pred.prediction_mS_cm.to_numpy()
    valid = (pred.status.to_numpy() == "ok") & np.isfinite(p) & (p > 0)
    out = pred[["measurement_id"]].copy()
    out["y_mS_cm"] = y
    out["valid"] = valid
    out["abs_log_err"] = np.where(valid, np.abs(np.log10(np.where(valid, p, 1.0)) - np.log10(y)), np.nan)
    out["abs_err_mS_cm"] = np.where(np.isfinite(p), np.abs(p - y), np.nan)
    out["signed_log_err"] = np.where(valid, np.log10(np.where(valid, p, 1.0)) - np.log10(y), np.nan)
    return out


def group_table(err: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    e = err.assign(group=groups.to_numpy())
    g = e.groupby("group", sort=True)
    return pd.DataFrame(
        {
            "n": g.size(),
            "n_invalid": g.valid.apply(lambda v: int((~v).sum())),
            "mae_log10": g.abs_log_err.mean(),
            "median_ae_log10": g.abs_log_err.median(),
            "bias_log10": g.signed_log_err.mean(),
            "mae_mS_cm": g.abs_err_mS_cm.mean(),
            "frac_within_factor2": g.abs_log_err.apply(lambda v: float((v.dropna() <= FACTOR_OF_TWO_LOG10).mean()) if v.notna().any() else np.nan),
        }
    )


def summarize(err: pd.DataFrame, gt: pd.DataFrame) -> dict:
    gm = gt.mae_log10.dropna()
    q75 = gm.quantile(0.75) if len(gm) else np.nan
    return {
        "n_rows": int(len(err)),
        "n_groups": int(len(gt)),
        "invalid_rate": float((~err.valid).mean()) if len(err) else np.nan,
        "group_mean_mae_log10": float(gm.mean()) if len(gm) else np.nan,
        "sample_mae_log10": float(err.abs_log_err.mean()),
        "group_median_mae_log10": float(gm.median()) if len(gm) else np.nan,
        "worst_quartile_mae_log10": float(gm[gm >= q75].mean()) if len(gm) else np.nan,
        "group_mean_mae_mS_cm": float(gt.mae_mS_cm.mean()) if len(gt) else np.nan,
        "sample_mae_mS_cm": float(err.abs_err_mS_cm.mean()),
        "frac_within_factor2": float((err.abs_log_err.dropna() <= FACTOR_OF_TWO_LOG10).mean()) if err.abs_log_err.notna().any() else np.nan,
    }


def group_mean_mae(err: pd.DataFrame, groups) -> float:
    """Fast path used during tuning."""
    v = err.abs_log_err.to_numpy()
    if not np.isfinite(v).any():
        return float("nan")  # e.g. every raw-scale prediction nonpositive
    if groups is None:
        return float(np.nanmean(v))
    return float(pd.Series(v).groupby(np.asarray(groups)).mean().mean())


def paired_bootstrap(G: pd.DataFrame, n_boot: int, seed: int, reference: str | None = None) -> pd.DataFrame:
    """Paired bootstrap that resamples score groups (intervals are conditional on the grouping).

    ``G`` is groups x models of per-group MAE. The same resampled group
    indices are used for every model, so differences are paired.
    """
    G = G.dropna(axis=0, how="any")
    n = len(G)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n)) if n else np.zeros((n_boot, 0), dtype=int)
    vals = G.to_numpy()
    boots = vals[idx].mean(axis=1) if n else np.full((n_boot, G.shape[1]), np.nan)  # (n_boot, n_models)
    rows = []
    ref_j = list(G.columns).index(reference) if reference in G.columns else None
    for j, m in enumerate(G.columns):
        r = {
            "model": m,
            "n_groups": n,
            "group_mean_mae_log10": float(vals[:, j].mean()) if n else np.nan,
            "ci_low": float(np.percentile(boots[:, j], 2.5)) if n else np.nan,
            "ci_high": float(np.percentile(boots[:, j], 97.5)) if n else np.nan,
        }
        if ref_j is not None and n:
            d = boots[:, j] - boots[:, ref_j]
            r.update(
                {
                    "diff_vs_reference": float(vals[:, j].mean() - vals[:, ref_j].mean()),
                    "diff_ci_low": float(np.percentile(d, 2.5)),
                    "diff_ci_high": float(np.percentile(d, 97.5)),
                    "p_model_worse": float((d > 0).mean()),
                }
            )
        rows.append(r)
    return pd.DataFrame(rows)
