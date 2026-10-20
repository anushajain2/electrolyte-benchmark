"""Temperature-representation ablation (v0.5.1).

Why do temperature-aware baselines beat tree ensembles below the coldest
training temperature? Two explanations compete:

  representation  the models lack a usable temperature term (1/T is the
                  Arrhenius-natural coordinate)
  model family    trees predict a constant beyond the training range of any
                  feature, whatever the representation

Each model is refit on the same outer folds with temperature columns removed:
all features; T only (no 1/T); 1/T only (no T); no temperature at all. If ridge
depends on 1/T while extra trees are unaffected by any choice, the reversal is a
model-family effect that the representation cannot fix for trees.
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

VARIANTS = {
    "all features": [],
    "T only (no 1/T)": ["inv_T_1000"],
    "1/T only (no T)": ["T_K"],
    "no temperature": ["T_K", "inv_T_1000"],
}


def _run(model, params, X, midx, tr, te, drop):
    with threadpool_limits(1):
        cols = [c for c in X.columns if c not in drop]
        m = make_model(model, 0, params)
        m.fit(X.loc[tr, cols], midx.loc[tr, "conductivity_mS_cm"].to_numpy(), midx.loc[tr])
        p = m.predict(X.loc[te, cols], midx.loc[te]).prediction_mS_cm.to_numpy()
        e = np.abs(np.log10(p) - np.log10(midx.loc[te, "conductivity_mS_cm"].to_numpy()))
        b = np.log10(p) - np.log10(midx.loc[te, "conductivity_mS_cm"].to_numpy())
        return e, b


def temperature_ablation(cfg: Config, meas: pd.DataFrame, X: pd.DataFrame, n_jobs: int = -1) -> pd.DataFrame:
    ab = cfg["investigation"]["temperature_ablation"]
    track = cfg["investigation"]["primary_track"]
    midx = meas.set_index("measurement_id")
    runs = pd.read_csv(cfg.p("results") / "train_runs.csv")
    folds = load_folds(cfg)
    plans = []
    for task in ab["tasks"]:
        man = load_manifest(cfg, task, track)
        for f in folds[(folds.task_id == task) & (folds.track_id == track)].itertuples():
            o = man[man.fold_id == f.fold_id]
            tr, te = o.measurement_id[o.role == "train"].to_numpy(), o.measurement_id[o.role == "test"].to_numpy()
            for model in ab["models"]:
                r = runs[(runs.task_id == task) & (runs.track_id == track) & (runs.fold_id == f.fold_id) & (runs.model == model)]
                params = json.loads(r.chosen_config.iat[0]) if len(r) else {}
                for name, drop in VARIANTS.items():
                    plans.append((task, f.fold_id, model, name, params, tr, te, drop))
    out = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_run)(m, p, X, midx, tr, te, d) for _, _, m, _, p, tr, te, d in plans)
    rows = []
    for (task, fold, model, name, _, _, te, _), (e, b) in zip(plans, out):
        g = midx.loc[te, "source_group_id"].to_numpy()
        rows.append(pd.DataFrame({"task_id": task, "fold_id": fold, "model": model, "variant": name, "group": g, "err": e, "bias": b}))
    long = pd.concat(rows, ignore_index=True)
    per_group = long.groupby(["task_id", "model", "variant", "group"]).agg(mae=("err", "mean"), bias=("bias", "mean")).reset_index()
    res = per_group.groupby(["task_id", "model", "variant"]).agg(group_mean_mae_log10=("mae", "mean"), mean_bias_log10=("bias", "mean"), n_groups=("group", "nunique")).reset_index()
    order = {k: i for i, k in enumerate(VARIANTS)}
    return res.sort_values(["task_id", "model", "variant"], key=lambda s: s.map(order) if s.name == "variant" else s).reset_index(drop=True)
