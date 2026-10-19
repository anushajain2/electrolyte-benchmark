"""Section 6-7 investigation: is apparent chemical generalization entangled with source?

Analyses (all on the primary track):
  A  ranking_shift       model rankings per task vs grouped interpolation
  B  transfer_gaps       same rows scored under interpolation vs a shift task
  C  source_familiarity  controlled, equal-size training sets with / without
                         other data from the target source (primary contrast)
  D  source_diversity    matched budget, varying number of sources (secondary, exploratory)
  E  failure_cases       worst held-out formulations traced to raw records

Only C with ``--confirm`` touches reserved labels, and that use is logged.
"""
from __future__ import annotations

import json
import zlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import kendalltau
from threadpoolctl import threadpool_limits

from .config import Config, RunManifest
from .curate import load_processed
from .evaluate import MAIN_MODELS
from .features import track_features
from .metrics import paired_bootstrap
from .models import make_model
from .splits import benchmark_frame, common_support, load_folds, load_manifest, solvent_matrix, support_spec
from .train import log_reserved_exposure


# --------------------------------------------------------------------------
# A, B: from evaluation tables
# --------------------------------------------------------------------------
def ranking_shift(summary: pd.DataFrame, track: str) -> pd.DataFrame:
    s = summary[(summary.track_id == track) & (summary.subset == "all") & summary.model.isin(MAIN_MODELS)]
    if "complete" in s:
        s = s[s.complete.astype(bool)]  # incomplete models are never ranked
    piv = s.pivot(index="task_id", columns="model", values="group_mean_mae_log10")
    ranks = piv.rank(axis=1, method="min")
    base = ranks.loc["interp_formulation"] if "interp_formulation" in ranks.index else None
    rows = []
    for task, rk in ranks.iterrows():
        tau = kendalltau(base, rk).statistic if base is not None else np.nan
        rows.append({"task_id": task, "best_model": rk.idxmin(), "kendall_tau_vs_interpolation": tau, **{f"mae_{m}": piv.at[task, m] for m in piv.columns}, **{f"rank_{m}": int(rk[m]) for m in rk.index}})
    return pd.DataFrame(rows)


def transfer_gaps(rows: pd.DataFrame, track: str, model: str) -> pd.DataFrame:
    """For rows scored both in interpolation and in a shift task, compare errors.

    The difference is measured on identical rows, so it isolates what the
    shift task removed from training (other formulations of the same source,
    the component, the other side of a boundary). It is not a causal effect."""
    r = rows[(rows.track_id == track) & (rows.model == model)]
    base = r[r.task_id == "interp_formulation"].set_index("measurement_id").abs_log_err
    out = []
    for (task, group), g in r[r.task_id != "interp_formulation"].groupby(["task_id", "group"]):
        g = g[g.measurement_id.isin(base.index) & g.valid]
        if not len(g):
            continue
        b = base.reindex(g.measurement_id).to_numpy()
        out.append(
            {
                "task_id": task,
                "group": group,
                "interpretation_label": g.interpretation_label.iat[0],
                "n_rows": len(g),
                "mae_interpolation": float(np.nanmean(b)),
                "mae_shift_task": float(g.abs_log_err.mean()),
                "gap": float(g.abs_log_err.mean() - np.nanmean(b)),
            }
        )
    return pd.DataFrame(out)


def extrapolation_profile(cfg: Config, rows: pd.DataFrame, meas: pd.DataFrame, track: str) -> pd.DataFrame:
    """Error and bias against distance beyond the declared condition boundary.

    Discriminates 'extrapolation distance' from 'source-specific behaviour':
    if errors at matched distance agree across sources, distance explains them."""
    midx = meas.set_index("measurement_id")
    out = []
    specs = {
        "temp_low": ("temperature_K", cfg["tasks"]["temp_low"]["train_min_K"], -1, 10.0),
        "temp_high": ("temperature_K", cfg["tasks"]["temp_high"]["train_max_K"], 1, 10.0),
        "conc_high": ("concentration", cfg["tasks"]["conc_high"]["train_max"], 1, 0.3),
    }
    r = rows[(rows.track_id == track) & rows.task_id.isin(list(specs)) & rows.valid]
    for (task, model), g in r.groupby(["task_id", "model"]):
        col, bound, sign, width = specs[task]
        g = g.join(midx[[col, "source_group_id"]], on="measurement_id")
        dist = sign * (g[col] - bound)
        g = g.assign(distance_bin=(np.ceil(dist / width) * width).round(3))
        for (b, src), h in g.groupby(["distance_bin", "source_group_id"]):
            out.append({"task_id": task, "model": model, "variable": col, "distance_beyond_boundary": b, "source_group_id": src, "n": len(h), "mae_log10": h.abs_log_err.mean(), "bias_log10": h.signed_log_err.mean()})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# C: controlled source-familiarity comparison
# --------------------------------------------------------------------------
def _fit_score(model, params, seed, X, meas_idx, tr_ids, te_ids):
    with threadpool_limits(1):
        m = make_model(model, seed, params)
        y = meas_idx.loc[tr_ids, "conductivity_mS_cm"].to_numpy()
        m.fit(X.loc[tr_ids], y, meas_idx.loc[tr_ids])
        p = m.predict(X.loc[te_ids], meas_idx.loc[te_ids])
        return np.abs(np.log10(p.prediction_mS_cm.to_numpy()) - np.log10(meas_idx.loc[te_ids, "conductivity_mS_cm"].to_numpy()))


def _match_rows(pool: pd.DataFrame, target: pd.DataFrame, rng) -> tuple[np.ndarray, float]:
    """Draw one pool row per target row, same chemistry system where possible."""
    chosen, matched = [], 0
    avail = {k: list(rng.permutation(v.to_numpy())) for k, v in pool.groupby("chemistry_system_id").measurement_id}
    rest = list(rng.permutation(pool.measurement_id.to_numpy()))
    used = set()
    for sys_id in target.chemistry_system_id.to_numpy():
        lst = avail.get(sys_id, [])
        while lst and lst[-1] in used:
            lst.pop()
        if lst:
            x = lst.pop()
            matched += 1
        else:
            while rest and rest[-1] in used:
                rest.pop()
            x = rest.pop()
        used.add(x)
        chosen.append(x)
    return np.array(chosen), matched / max(len(target), 1)


def _support_variants(cfg: Config) -> dict[str, dict]:
    """'supported_by_both' uses the task's support; v0.2 adds named sensitivity variants."""
    out = {"supported_by_both": support_spec(cfg, cfg["tasks"]["source_transfer"]["support"])}
    for name in cfg["investigation"].get("sensitivity", {}).get("support_definitions", []):
        out[f"supported_by_both__{name}"] = support_spec(cfg, name)
    return out


def source_familiarity(cfg: Config, meas: pd.DataFrame, comps: pd.DataFrame, X: pd.DataFrame, include_reserved: bool, n_jobs: int, exclude_groups=(), models=None, variants: dict | None = None) -> pd.DataFrame:
    inv = cfg["investigation"]
    sf = inv["source_familiarity"]
    track = inv["primary_track"]
    variants = variants or _support_variants(cfg)
    solv = solvent_matrix(comps)
    df = benchmark_frame(cfg, meas, track, include_reserved=include_reserved)
    df = df[~df.source_group_id.isin(set(exclude_groups))]
    midx = meas.set_index("measurement_id")
    reserved = [k for k, v in cfg["sources"].items() if v.get("role") == "reserved_external_evaluation"]
    targets = []
    for sg, g in df.groupby("source_group_id"):
        is_reserved = g.dataset_id.isin(reserved).all()
        if is_reserved and not include_reserved:
            continue
        if g.formulation_id.nunique() >= sf["min_source_formulations"]:
            targets.append((sg, is_reserved))
    plans, meta = [], []
    for sg, is_reserved in targets:
        rng = np.random.default_rng([cfg["seed"], zlib.crc32(sg.encode())])
        S = df[df.source_group_id == sg]
        units = np.sort(S.split_unit_id.unique())
        test_units = set(rng.choice(units, size=max(1, int(round(sf["test_fraction"] * len(units)))), replace=False))
        te = S[S.split_unit_id.isin(test_units)]
        s_tr = S[~S.split_unit_id.isin(test_units)]
        # background pool: other non-reserved sources only, so confirmation runs
        # leave every non-reserved result unchanged
        P = df[(df.source_group_id != sg) & ~df.dataset_id.isin(reserved)]
        bg_n = min(sf["background_size"], len(P) - len(s_tr))
        if bg_n <= 0:
            continue
        for rep in range(sf["repeats"]):
            r2 = np.random.default_rng([cfg["seed"], zlib.crc32(sg.encode()), rep])
            bg = P.sample(n=bg_n, random_state=int(r2.integers(2**31)))
            Q = P[~P.measurement_id.isin(bg.measurement_id)]
            repl, match_frac = _match_rows(Q, s_tr, r2)
            A = np.r_[s_tr.measurement_id.to_numpy(), bg.measurement_id.to_numpy()]
            B = np.r_[repl, bg.measurement_id.to_numpy()]
            masks, m = {}, {}
            for name, spec in variants.items():
                sup_A = common_support(df[df.measurement_id.isin(A)], te, solv, spec) == "in_support"
                sup_B = common_support(df[df.measurement_id.isin(B)], te, solv, spec) == "in_support"
                masks[name] = (sup_A & sup_B).to_numpy()
                if name == "supported_by_both":
                    m = {"test_in_support_A": float(sup_A.mean()), "test_in_support_B": float(sup_B.mean())}
                m[f"n_test_{name}"] = int(masks[name].sum())
            meta.append(
                {
                    "target_source": sg,
                    "reserved_target": is_reserved,
                    "repeat": rep,
                    "n_test": len(te),
                    "n_test_formulations": te.formulation_id.nunique(),
                    "n_source_train": len(s_tr),
                    "n_background": bg_n,
                    "replacement_same_system_frac": match_frac,
                    **m,
                }
            )
            for model in models or sf["models"]:
                params = inv["fixed_configs"].get(model, {})
                for arm, tr in (("with_source", A), ("without_source", B)):
                    plans.append((sg, rep, model, arm, params, tr, te.measurement_id.to_numpy(), masks, te.formulation_id.to_numpy()))
    errs = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_fit_score)(m, p, rep, X, midx, tr, te) for sg, rep, m, arm, p, tr, te, masks, f in plans)
    rows = []
    for (sg, rep, model, arm, p, tr, te, masks, form), e in zip(plans, errs):
        gm = lambda mask: float(pd.Series(e[mask]).groupby(form[mask]).mean().mean()) if mask.any() else np.nan  # noqa: E731
        rows.append({"target_source": sg, "repeat": rep, "model": model, "arm": arm, "mae_all": gm(np.ones(len(e), bool)), **{f"mae_{n}": gm(mk) for n, mk in masks.items()}})
    res = pd.DataFrame(rows)
    subs = ["all", *variants]
    wide = res.pivot_table(index=["target_source", "repeat", "model"], columns="arm", values=[f"mae_{x}" for x in subs], dropna=False).reset_index()
    wide.columns = ["_".join(c for c in col if c) for col in wide.columns]
    for sub in subs:
        wide[f"effect_{sub}"] = wide[f"mae_{sub}_without_source"] - wide[f"mae_{sub}_with_source"]
    return wide.merge(pd.DataFrame(meta), on=["target_source", "repeat"])


def familiarity_contrast(fam: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Primary contrast: mean effect over target sources, paired bootstrap over sources."""
    pc = cfg["investigation"]["primary_contrast"]
    variants = [c[len("effect_"):] for c in fam.columns if c.startswith("effect_supported_by_both__")]
    out = []
    for (model, reserved), g in fam.groupby(["model", "reserved_target"]):
        cols = [c for c in fam.columns if c.startswith("effect_")]
        per_src = g.groupby("target_source")[cols + ["replacement_same_system_frac"]].mean()
        # exploratory sensitivity (not predeclared): targets whose replacement rows
        # could be drawn from the same chemistry system for >=80% of rows
        well = per_src[per_src.replacement_same_system_frac >= 0.8]
        specs = [("all", per_src, "effect_all"), ("supported_by_both", per_src, "effect_supported_by_both"), ("supported_by_both__well_matched_targets", well, "effect_supported_by_both")]
        specs += [(v, per_src, f"effect_{v}") for v in variants]
        for sub, src_tab, col in specs:
            v = src_tab[col].dropna()
            G = pd.DataFrame({"effect": v})
            b = paired_bootstrap(G, cfg["evaluation"]["n_bootstrap"], cfg["evaluation"]["bootstrap_seed"])
            out.append(
                {
                    "model": model,
                    "reserved_target": reserved,
                    "subset": sub,
                    "is_primary_contrast": bool(model == pc["model"] and sub == pc["subset"] and not reserved),
                    "exploratory": sub != "supported_by_both" and sub != "all",
                    "n_target_sources": len(v),
                    "mean_effect_log10": float(v.mean()) if len(v) else np.nan,
                    "ci_low": float(b.ci_low.iat[0]) if len(v) else np.nan,
                    "ci_high": float(b.ci_high.iat[0]) if len(v) else np.nan,
                    "n_sources_positive": int((v > 0).sum()),
                    "repeat_sd_mean": float(g[g.target_source.isin(v.index)].groupby("target_source")[col].std().mean()) if len(v) else np.nan,
                }
            )
    return pd.DataFrame(out)


def familiarity_logo(cfg: Config, meas, comps, X, fam: pd.DataFrame, n_jobs: int) -> pd.DataFrame:
    """Leave-one-source-group-out sensitivity of the primary contrast.

    target_jackknife: drop one target from the average (no refit).
    full_exclusion: remove the group from the data entirely (as target and as
    background), rebuild every training set and refit."""
    pc = cfg["investigation"]["primary_contrast"]
    col = f"effect_{pc['subset']}"
    f = fam[(fam.model == pc["model"]) & ~fam.reserved_target]
    per = f.groupby("target_source")[col].mean().dropna()
    rows = [{"mode": "none", "left_out_group": "", "n_target_sources": len(per), "mean_effect_log10": per.mean(), "n_sources_positive": int((per > 0).sum())}]
    for g in per.index:
        rest = per.drop(index=g)
        rows.append({"mode": "target_jackknife", "left_out_group": g, "n_target_sources": len(rest), "mean_effect_log10": rest.mean(), "n_sources_positive": int((rest > 0).sum())})
    track = cfg["investigation"]["primary_track"]
    groups = sorted(benchmark_frame(cfg, meas, track).source_group_id.unique())
    primary = {"supported_by_both": support_spec(cfg, cfg["tasks"]["source_transfer"]["support"])}
    for g in groups:
        fx = source_familiarity(cfg, meas, comps, X, False, n_jobs, exclude_groups=[g], models=[pc["model"]], variants=primary)
        v = fx.groupby("target_source")[col].mean().dropna()
        rows.append({"mode": "full_exclusion", "left_out_group": g, "n_target_sources": len(v), "mean_effect_log10": v.mean() if len(v) else np.nan, "n_sources_positive": int((v > 0).sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# D: source diversity at matched budget (exploratory)
# --------------------------------------------------------------------------
def source_diversity(cfg: Config, meas: pd.DataFrame, comps: pd.DataFrame, X: pd.DataFrame, n_jobs: int) -> pd.DataFrame:
    inv = cfg["investigation"]
    sd = inv["source_diversity"]
    track = inv["primary_track"]
    solv = solvent_matrix(comps)
    tol = support_spec(cfg, cfg["tasks"]["source_transfer"]["support"])
    df = benchmark_frame(cfg, meas, track)
    midx = meas.set_index("measurement_id")
    folds = load_folds(cfg)
    st = folds[(folds.task_id == "source_transfer") & (folds.track_id == track)]
    man = load_manifest(cfg, "source_transfer", track)
    plans, meta = [], []
    for fold in st.itertuples():
        te = man[(man.fold_id == fold.fold_id) & (man.role == "test")].measurement_id.to_numpy()
        te_df = df[df.measurement_id.isin(te)]
        pool = df[(df.source_group_id != fold.heldout_unit) & df.chemistry_system_id.isin(te_df.chemistry_system_id.unique())]
        counts = pool.source_group_id.value_counts()
        for k in range(1, sd["max_sources"] + 1):
            per = sd["budget"] // k
            cand = np.sort(counts[counts >= per].index.to_numpy())
            if len(cand) < k:
                continue
            for rep in range(sd["repeats"]):
                rng = np.random.default_rng([cfg["seed"], zlib.crc32(fold.heldout_unit.encode()), k, rep])
                srcs = rng.choice(cand, size=k, replace=False)
                tr = np.concatenate([pool[pool.source_group_id == s].sample(n=per, random_state=int(rng.integers(2**31))).measurement_id.to_numpy() for s in srcs])
                sup = (common_support(df[df.measurement_id.isin(tr)], te_df, solv, tol) == "in_support").mean()
                meta.append({"target_source": fold.heldout_unit, "k_sources": k, "repeat": rep, "train_sources": "|".join(srcs), "n_train": len(tr), "n_candidate_sources": len(cand), "test_in_support": float(sup)})
                plans.append((fold.heldout_unit, k, rep, tr, te))
    errs = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_fit_score)("extra_trees", inv["fixed_configs"]["extra_trees"], rep, X, midx, tr, te) for _, _, rep, tr, te in plans)
    res = pd.DataFrame(meta)
    res["mae_log10"] = [float(e.mean()) for e in errs]
    return res


# --------------------------------------------------------------------------
# v0.2 sensitivity analyses
# --------------------------------------------------------------------------
def _chosen(cfg: Config, task: str, track: str, fold: str, model: str) -> dict:
    runs = pd.read_csv(cfg.p("results") / "train_runs.csv")
    r = runs[(runs.task_id == task) & (runs.track_id == track) & (runs.fold_id == fold) & (runs.model == model)]
    return json.loads(r.chosen_config.iat[0]) if len(r) else cfg["investigation"]["fixed_configs"].get(model, {})


def _fit_predict_ids(model, params, X, midx, tr, te):
    with threadpool_limits(1):
        m = make_model(model, 0, params)
        m.fit(X.loc[tr], midx.loc[tr, "conductivity_mS_cm"].to_numpy(), midx.loc[tr])
        p = m.predict(X.loc[te], midx.loc[te])
        y = midx.loc[te, "conductivity_mS_cm"].to_numpy()
        pr = p.prediction_mS_cm.to_numpy()
        ok = (p.status.to_numpy() == "ok") & (pr > 0)
        return np.where(ok, np.abs(np.log10(np.where(ok, pr, 1.0)) - np.log10(y)), np.nan)


def support_sensitivity(cfg: Config, meas, comps, X, n_jobs: int) -> pd.DataFrame:
    """Source transfer scored under alternative support definitions.

    Every row of each held-out source group is predicted once (the fold's chosen
    configuration, seed 0), then scored on the rows each definition supports.
    'interior_gap' rows are inside the marginal ranges but have no training point
    of a nearby recipe within the local (T, c) window."""
    track = cfg["investigation"]["primary_track"]
    df = benchmark_frame(cfg, meas, track)
    solv = solvent_matrix(comps)
    midx = meas.set_index("measurement_id")
    defs = {"primary": support_spec(cfg, cfg["tasks"]["source_transfer"]["support"])}
    defs.update({n: support_spec(cfg, n) for n in cfg["investigation"]["sensitivity"]["support_definitions"]})
    models = ["extra_trees", "ridge"]
    plans, reasons = [], {}
    for sg in sorted(df.source_group_id.unique()):
        tr, te = df[df.source_group_id != sg], df[df.source_group_id == sg]
        reasons[sg] = {n: common_support(tr, te, solv, d).to_numpy() for n, d in defs.items()}
        fold = f"source/{sg}"
        for m in models:
            plans.append((sg, m, _chosen(cfg, "source_transfer", track, fold, m), tr.measurement_id.to_numpy(), te.measurement_id.to_numpy()))
    errs = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_fit_predict_ids)(m, p, X, midx, tr, te) for sg, m, p, tr, te in plans)
    rows = []
    for (sg, m, p, tr, te), e in zip(plans, errs):
        for n, r in reasons[sg].items():
            sel = r == "in_support"
            rows.append({"definition": n, "source_group_id": sg, "model": m, "n_rows": int(sel.sum()), "mae_log10": float(np.nanmean(e[sel])) if sel.any() else np.nan})
        if "marginal_v01" in reasons[sg]:
            gap = (reasons[sg]["marginal_v01"] == "in_support") & (reasons[sg]["primary"] == "no_training_point_nearby_in_T_and_c")
            rows.append({"definition": "interior_gap(marginal_not_local)", "source_group_id": sg, "model": m, "n_rows": int(gap.sum()), "mae_log10": float(np.nanmean(e[gap])) if gap.any() else np.nan})
    return pd.DataFrame(rows)


def support_sensitivity_summary(ss: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    rep = cfg["reporting"]
    out = []
    for (d, m), g in ss.groupby(["definition", "model"]):
        ok = g[g.n_rows >= rep["min_test_measurements_per_unit"]]
        out.append({"definition": d, "model": m, "n_units_with_min_rows": len(ok), "n_rows": int(g.n_rows.sum()), "group_mean_mae_log10": ok.mae_log10.mean(), "sample_mae_log10": (g.mae_log10 * g.n_rows).sum() / max(g.n_rows[g.mae_log10.notna()].sum(), 1)})
    return pd.DataFrame(out)


def condition_logo(cfg: Config, meas, X, n_jobs: int, task: str = "temp_low") -> pd.DataFrame:
    """Refit with each source group removed from training AND testing; does the
    low-temperature ranking reversal survive?"""
    track = cfg["investigation"]["primary_track"]
    man = load_manifest(cfg, task, track)
    fold = f"{task}/all"
    outer = man[man.fold_id == fold]
    tr_all = outer.measurement_id[outer.role == "train"].to_numpy()
    te_all = outer.measurement_id[outer.role == "test"].to_numpy()
    midx = meas.set_index("measurement_id")
    models = ["extra_trees", "hist_gb", "knn", "ridge", "arrhenius_local"]
    groups = ["", *sorted(midx.loc[te_all].source_group_id.unique())]
    plans = []
    for g in groups:
        tr = tr_all[midx.loc[tr_all].source_group_id.to_numpy() != g]
        te = te_all[midx.loc[te_all].source_group_id.to_numpy() != g]
        for m in models:
            plans.append((g, m, _chosen(cfg, task, track, fold, m), tr, te))
    errs = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_fit_predict_ids)(m, p, X, midx, tr, te) for g, m, p, tr, te in plans)
    rows = []
    for (g, m, p, tr, te), e in zip(plans, errs):
        gm = pd.Series(e).groupby(midx.loc[te].source_group_id.to_numpy()).mean()
        rows.append({"task_id": task, "left_out_group": g or "(none)", "model": m, "n_groups": int(gm.notna().sum()), "group_mean_mae_log10": float(gm.mean())})
    res = pd.DataFrame(rows)
    res["rank"] = res.groupby("left_out_group").group_mean_mae_log10.rank(method="min")
    return res


def ridge_audit(cfg: Config, meas, X, components=("TFP", "EMC", "EC")) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Numerical audit of ridge under component holdout.

    Checks that scaling is fitted on training rows only, reports how far the
    held-out rows' descriptors sit outside the training range (z-scores), the
    per-feature contributions to the log prediction, and the error after
    clipping test features to the training range (a diagnostic, not a model)."""
    from .models import make_model as mk

    track = cfg["investigation"]["primary_track"]
    man = load_manifest(cfg, "component_transfer", track)
    midx = meas.set_index("measurement_id")
    summ, contrib = [], []
    for comp in components:
        fold = f"component/{comp}"
        o = man[man.fold_id == fold]
        if not len(o):
            continue
        tr = o.measurement_id[o.role == "train"].to_numpy()
        te = o.measurement_id[o.role == "test"].to_numpy()
        params = _chosen(cfg, "component_transfer", track, fold, "ridge")
        m = mk("ridge", 0, params).fit(X.loc[tr], midx.loc[tr, "conductivity_mS_cm"].to_numpy(), midx.loc[tr])
        scaler, lin = m.model.steps[0][1], m.model.steps[1][1]
        scaler_ok = bool(np.allclose(scaler.mean_, X.loc[tr].to_numpy().mean(axis=0)))
        Z = (X.loc[te].to_numpy() - scaler.mean_) / scaler.scale_
        C = Z * lin.coef_
        y = np.log10(midx.loc[te, "conductivity_mS_cm"].to_numpy())
        pred = C.sum(axis=1) + lin.intercept_
        Xc = X.loc[te].clip(lower=X.loc[tr].min(), upper=X.loc[tr].max(), axis=1)
        pred_clip = np.log10(m.predict(Xc, midx.loc[te]).prediction_mS_cm.to_numpy())
        zmax = np.abs(Z).max(axis=0)
        summ.append(
            {
                "component": comp,
                "alpha": params.get("alpha"),
                "scaler_fitted_on_training_rows_only": scaler_ok,
                "n_test": len(te),
                "mae_log10": float(np.mean(np.abs(pred - y))),
                "mean_bias_log10": float(np.mean(pred - y)),
                "mae_log10_features_clipped_to_training_range": float(np.mean(np.abs(pred_clip - y))),
                "max_abs_z": float(zmax.max()),
                "feature_with_max_abs_z": X.columns[int(zmax.argmax())],
                "n_features_with_abs_z_gt_5": int((zmax > 5).sum()),
            }
        )
        for j, f in enumerate(X.columns):
            contrib.append({"component": comp, "feature": f, "coef": float(lin.coef_[j]), "train_mean": float(scaler.mean_[j]), "train_sd": float(scaler.scale_[j]), "test_mean_value": float(X.loc[te, f].mean()), "mean_abs_z": float(np.abs(Z[:, j]).mean()), "mean_contribution_log10": float(C[:, j].mean())})
    c = pd.DataFrame(contrib)
    if len(c):
        c = c.reindex(c.mean_contribution_log10.abs().sort_values(ascending=False).index)
    return pd.DataFrame(summ), c


# --------------------------------------------------------------------------
# E: failure cases traced to raw records
# --------------------------------------------------------------------------
def failure_cases(cfg: Config, rows: pd.DataFrame, meas: pd.DataFrame, n_cases: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    track = cfg["investigation"]["primary_track"]
    ref = cfg["models"]["reference_model"]
    midx = meas.set_index("measurement_id")
    r = rows[(rows.track_id == track) & (rows.model == ref) & rows.valid & rows.task_id.isin(["source_transfer", "interp_formulation", "temp_low", "temp_high", "conc_high", "component_transfer"])]
    r = r.join(midx[["formulation_id", "source_group_id", "chemistry_system_id"]], on="measurement_id")
    by_f = r.groupby(["task_id", "fold_id", "formulation_id"]).agg(n=("abs_log_err", "size"), mae=("abs_log_err", "mean"), bias=("signed_log_err", "mean"), source=("source_group_id", "first"), system=("chemistry_system_id", "first")).reset_index()
    by_f = by_f[by_f.n >= 3].sort_values("mae", ascending=False)
    picked, seen = [], set()
    for row in by_f.itertuples():
        if row.source in seen:
            continue
        picked.append(row)
        seen.add(row.source)
        if len(picked) == n_cases:
            break
    cases, records = [], []
    for i, c in enumerate(picked, 1):
        man = load_manifest(cfg, c.task_id, track)
        tr_ids = man[(man.fold_id == c.fold_id) & (man.role == "train")].measurement_id
        tr = midx.loc[tr_ids]
        same = tr[tr.chemistry_system_id == c.system]
        frm = midx[midx.formulation_id == c.formulation_id]
        test_rows = r[(r.fold_id == c.fold_id) & (r.formulation_id == c.formulation_id)]
        # evidence from other source groups: same chemistry, near the same conditions
        T0, c0 = frm.temperature_K.median(), frm.concentration.iat[0]
        ind = midx[(midx.inclusion_status == "included") & (midx.track_id == track) & (midx.chemistry_system_id == c.system) & (midx.source_group_id != c.source) & ((midx.temperature_K - T0).abs() <= 5) & ((midx.concentration - c0).abs() <= 0.15)]
        cases.append(
            {
                "case": i,
                "task_id": c.task_id,
                "fold_id": c.fold_id,
                "formulation_id": c.formulation_id,
                "chemistry_system_id": c.system,
                "source_group_id": c.source,
                "n_rows": c.n,
                "mae_log10": c.mae,
                "bias_log10": c.bias,
                "concentration": float(c0),
                "T_range_K": f"{frm.temperature_K.min():.1f}-{frm.temperature_K.max():.1f}",
                "solvent_fractions_reported": frm.solvent_fractions_reported.iat[0],
                "train_rows_same_system": len(same),
                "train_sources_same_system": "; ".join(sorted(same.source_group_id.unique())),
                "train_T_range_same_system": f"{same.temperature_K.min():.1f}-{same.temperature_K.max():.1f}" if len(same) else "",
                "train_c_range_same_system": f"{same.concentration.min():.2f}-{same.concentration.max():.2f}" if len(same) else "",
                "other_source_points_near_conditions": len(ind),
                "other_source_median_mS_cm": float(ind.conductivity_mS_cm.median()) if len(ind) else np.nan,
                "case_median_mS_cm": float(frm.conductivity_mS_cm.median()),
            }
        )
        tr_rows = test_rows.join(midx[["dataset_id", "original_row_id", "source_doi", "temperature_original", "temperature_unit_original", "concentration_original", "concentration_unit_original", "conductivity_original", "conductivity_unit_original", "source_record_link"]], on="measurement_id")
        tr_rows = tr_rows.assign(case=i)[["case", "measurement_id", "dataset_id", "original_row_id", "source_doi", "temperature_original", "temperature_unit_original", "concentration_original", "concentration_unit_original", "conductivity_original", "conductivity_unit_original", "y_mS_cm", "pred_geomean_mS_cm", "signed_log_err", "source_record_link"]]
        records.append(tr_rows.sort_values("temperature_original"))
    return pd.DataFrame(cases), pd.concat(records, ignore_index=True) if records else pd.DataFrame()


# --------------------------------------------------------------------------
def investigate(cfg: Config, n_jobs: int = -1, include_reserved: bool = False) -> None:
    man = RunManifest("investigate", cfg)
    meas, comps = load_processed(cfg)
    desc = pd.read_parquet(cfg.p("cache") / "descriptors.parquet")
    tables = cfg.p("results") / "tables"
    out = cfg.p("results") / "investigation"
    out.mkdir(parents=True, exist_ok=True)
    track = cfg["investigation"]["primary_track"]
    summary = pd.read_csv(tables / "summary.csv")
    rows = pd.read_parquet(tables / "row_errors.parquet")
    X = track_features(cfg, meas, comps, desc, track)

    res = {}
    res["ranking_shift"] = ranking_shift(summary, track)
    res["extrapolation_profile"] = extrapolation_profile(cfg, rows, meas, track)
    res["transfer_gaps"] = pd.concat([transfer_gaps(rows, track, m).assign(model=m) for m in ["extra_trees", "ridge", "knn"]], ignore_index=True)
    fam = source_familiarity(cfg, meas, comps, X, include_reserved, n_jobs)
    res["source_familiarity"] = fam
    res["familiarity_contrast"] = familiarity_contrast(fam, cfg)
    if include_reserved:
        log_reserved_exposure(cfg, man.run_id, "investigate", "source_familiarity confirmation on reserved source")
    res["source_diversity"] = source_diversity(cfg, meas, comps, X, n_jobs)
    res["failure_cases"], res["failure_records"] = failure_cases(cfg, rows, meas)
    if "sensitivity" in cfg["investigation"] and not include_reserved:
        res["familiarity_logo"] = familiarity_logo(cfg, meas, comps, X, fam, n_jobs)
        res["support_sensitivity"] = support_sensitivity(cfg, meas, comps, X, n_jobs)
        res["support_sensitivity_summary"] = support_sensitivity_summary(res["support_sensitivity"], cfg)
        res["condition_logo"] = pd.concat([condition_logo(cfg, meas, X, n_jobs, t) for t in cfg["investigation"]["sensitivity"].get("condition_logo_tasks", ["temp_low"])], ignore_index=True)
        res["ridge_audit"], res["ridge_audit_contributions"] = ridge_audit(cfg, meas, X)
    if "label_effects" in cfg["investigation"] and not include_reserved:
        from .label_effects import evaluation_filtering, training_cleaning

        res["label_evaluation_filtering"], res["label_cases_removed"] = evaluation_filtering(cfg, meas)
        res["label_training_cleaning"] = training_cleaning(cfg, meas, X, n_jobs)
    if "selective" in cfg["investigation"] and not include_reserved:
        from .selective import run_selective

        res["selective_risk_coverage"], res["selective_dev_threshold"], res["selective_rejection"] = run_selective(cfg, meas, X)
    suffix = "__confirm" if include_reserved else ""
    for name, df in res.items():
        p = out / f"{name}{suffix}.csv" if name in ("source_familiarity", "familiarity_contrast") else out / f"{name}.csv"
        df.to_csv(p, index=False)
        man.add_output(p)
    man.notes = {"include_reserved": include_reserved, "primary_contrast": cfg["investigation"]["primary_contrast"]}
    man.write(cfg.p("results") / "runs")
    (out / "README.json").write_text(json.dumps({"track": track, "analyses": list(res)}, indent=1))
