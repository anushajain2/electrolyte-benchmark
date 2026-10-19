"""Fixed, documented feature set.

Features are a pure function of each measurement's inputs (components,
temperature, concentration). Nothing is fitted here and no target or source
information enters, so the table can be computed once for all rows; every
fitted transformation (scaling, model fitting) happens inside the model
adapters on training rows only.

Columns (32):
  solv_mean_<d>, solv_var_<d>   composition-weighted mean and variance of 10 solvent descriptors
  n_solvents                    number of solvent components
  anion_<d>                     6 anion descriptors (salt represented as ions, not as a category)
  T_K, inv_T_1000, conc, sqrt_conc, log_conc
Weights are mole fractions on mole-fraction tracks and volume fractions on the
volume track (where conversion to mole fractions is unsupported).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .chemistry import ANION_DESCRIPTORS, SOLVENT_DESCRIPTORS

CONDITION_COLUMNS = ["T_K", "inv_T_1000", "conc", "sqrt_conc", "log_conc"]
FEATURE_COLUMNS = (
    [f"solv_mean_{d}" for d in SOLVENT_DESCRIPTORS]
    + [f"solv_var_{d}" for d in SOLVENT_DESCRIPTORS]
    + ["n_solvents"]
    + [f"anion_{d}" for d in ANION_DESCRIPTORS]
    + CONDITION_COLUMNS
)


def formulation_features(comps: pd.DataFrame, desc: pd.DataFrame, weight: str) -> pd.DataFrame:
    """Composition features per formulation (no conditions)."""
    sd = desc[desc.role == "solvent"].set_index("canonical_id")[[f"solv_{d}" for d in SOLVENT_DESCRIPTORS]]
    ad = desc[desc.role == "salt"].set_index("canonical_id")[[f"anion_{d}" for d in ANION_DESCRIPTORS]]
    solv = comps[comps.role == "solvent"].copy()
    solv["w"] = solv.mole_fraction if weight == "mole_fraction" else solv.amount_value
    if solv.w.isna().any():
        bad = solv[solv.w.isna()].formulation_id.unique()[:5]
        raise ValueError(f"composition weight {weight} unavailable for formulations {list(bad)}")
    solv = solv.join(sd, on="component_id")
    vals = solv[sd.columns].to_numpy()
    w = solv.w.to_numpy()[:, None]
    solv_w = pd.DataFrame(vals * w, columns=sd.columns, index=solv.index)
    solv_w["formulation_id"] = solv.formulation_id.to_numpy()
    mean = solv_w.groupby("formulation_id").sum()
    wsum = solv.groupby("formulation_id").w.sum()
    mean = mean.div(wsum, axis=0)
    sq = pd.DataFrame(((vals - mean.loc[solv.formulation_id].to_numpy()) ** 2) * w, columns=sd.columns, index=solv.index)
    sq["formulation_id"] = solv.formulation_id.to_numpy()
    var = sq.groupby("formulation_id").sum().div(wsum, axis=0)
    # Components with equal descriptor values (e.g. TPSA of EC and DMC) leave
    # float noise (~1e-29) instead of an exact zero; a training-only scaler
    # would then blow that noise up for unseen mixtures.
    var = var.mask(var.abs() < 1e-9, 0.0)
    out = pd.concat(
        [
            mean.rename(columns=lambda c: c.replace("solv_", "solv_mean_")),
            var.rename(columns=lambda c: c.replace("solv_", "solv_var_")),
            solv.groupby("formulation_id").size().rename("n_solvents").astype(float),
        ],
        axis=1,
    )
    salt = comps[comps.role == "salt"].set_index("formulation_id").component_id
    out = out.join(ad.reindex(salt.reindex(out.index)).set_axis(out.index))
    return out


def build_features(meas: pd.DataFrame, comps: pd.DataFrame, desc: pd.DataFrame, weight: str) -> pd.DataFrame:
    """Feature matrix indexed by measurement_id, columns FEATURE_COLUMNS."""
    ff = formulation_features(comps[comps.formulation_id.isin(meas.formulation_id)], desc, weight)
    X = ff.reindex(meas.formulation_id).set_axis(meas.measurement_id)
    T = meas.temperature_K.to_numpy()
    c = meas.concentration.to_numpy()
    X["T_K"] = T
    X["inv_T_1000"] = 1000.0 / T
    X["conc"] = c
    X["sqrt_conc"] = np.sqrt(c)
    X["log_conc"] = np.log10(c)
    X = X[FEATURE_COLUMNS].astype(float)
    if X.isna().any().any():
        cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"missing features {cols}")
    return X


def track_features(cfg, meas: pd.DataFrame, comps: pd.DataFrame, desc: pd.DataFrame, track: str) -> pd.DataFrame:
    sub = meas[(meas.inclusion_status == "included") & (meas.track_id == track)]
    return build_features(sub, comps, desc, cfg["tracks"][track]["composition_weight"])
