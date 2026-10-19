"""Step 4: frozen evaluation tasks.

Manifests are keyed by (measurement_id, task_id, track_id, fold_id, role). Outer
folds carry roles train/test; development folds nested in an outer fold carry
roles train/dev and have fold_id "<outer>::dev<j>". A measurement can take part
in many tasks. Every fold is checked against the leakage invariants before
anything is written.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config, RunManifest, sha256_file
from .curate import load_processed

LABEL_SUPPORTED = "supported_comparison"
LABEL_COMBINED = "combined_shift"
LABEL_INSUFFICIENT = "insufficient_support"


@dataclass
class Fold:
    fold_id: str
    train: np.ndarray
    test: np.ndarray
    heldout_unit: str
    meta: dict = field(default_factory=dict)
    dev: list["Fold"] = field(default_factory=list)
    test_tags: dict[str, str] = field(default_factory=dict)  # measurement_id -> subset tag


# --------------------------------------------------------------------------
# helpers shared with the audit
# --------------------------------------------------------------------------
def reserved_datasets(cfg: Config) -> set[str]:
    return {k for k, v in cfg["sources"].items() if v.get("role") == "reserved_external_evaluation"}


def benchmark_frame(cfg: Config, meas: pd.DataFrame, track: str, include_reserved: bool = False) -> pd.DataFrame:
    df = meas[(meas.inclusion_status == "included") & (meas.track_id == track)]
    if not include_reserved:
        df = df[~df.dataset_id.isin(reserved_datasets(cfg))]
    return df.reset_index(drop=True)


def solvent_matrix(comps: pd.DataFrame) -> pd.DataFrame:
    s = comps[comps.role == "solvent"]
    return s.pivot_table(index="formulation_id", columns="component_id", values="amount_value", aggfunc="sum", fill_value=0.0)


def component_sets(df: pd.DataFrame) -> pd.Series:
    """Canonical component identities (salt + solvents) for every row."""
    return pd.Series([frozenset([s, *v.split("+")]) for s, v in zip(df.salt_id, df.solvent_ids)], index=df.index)


def common_support(train: pd.DataFrame, test: pd.DataFrame, solv: pd.DataFrame, tol: dict) -> pd.Series:
    """Support reason per test row: 'in_support' or the first failing criterion.

    A test row is in support when training data with the same chemistry
    system (and track, hence ratio and concentration basis) contains solvent
    recipes within ``ratio_tolerance`` (max abs fraction difference), and the
    test temperature and concentration lie inside the range covered by those
    nearby training recipes (plus optional margins). Only inputs are used.
    """
    reason = pd.Series("no_shared_chemistry_system", index=test.index, dtype=object)
    rt, tm, cm = tol["ratio_tolerance"], tol.get("temperature_margin_K", 0.0), tol.get("concentration_margin", 0.0)
    tr_groups = train.groupby("chemistry_system_id")
    for sys_id, te in test.groupby("chemistry_system_id"):
        if sys_id not in tr_groups.groups:
            continue
        tr = tr_groups.get_group(sys_id)
        rec = tr.groupby("formulation_id").agg(Tmin=("temperature_K", "min"), Tmax=("temperature_K", "max"), c=("concentration", "first"))
        cols = sorted(set(sys_id.split("|")[1].split("+")))
        Xr = solv.reindex(rec.index)[cols].to_numpy()
        Xt = solv.reindex(te.formulation_id)[cols].to_numpy()
        dist = np.abs(Xt[:, None, :] - Xr[None, :, :]).max(axis=2)  # (n_test, n_train_formulations)
        near = dist <= rt + 1e-9
        Tmin = np.where(near, rec.Tmin.to_numpy()[None, :], np.inf).min(axis=1)
        Tmax = np.where(near, rec.Tmax.to_numpy()[None, :], -np.inf).max(axis=1)
        cmin = np.where(near, rec.c.to_numpy()[None, :], np.inf).min(axis=1)
        cmax = np.where(near, rec.c.to_numpy()[None, :], -np.inf).max(axis=1)
        T = te.temperature_K.to_numpy()
        c = te.concentration.to_numpy()
        r = np.where(
            ~near.any(axis=1),
            "no_nearby_solvent_ratio",
            np.where(
                (T < Tmin - tm) | (T > Tmax + tm),
                "temperature_outside_training_range",
                np.where((c < cmin - cm) | (c > cmax + cm), "concentration_outside_training_range", "in_support"),
            ),
        )
        if tol.get("mode", "marginal") == "local":
            # Marginal ranges can enclose combinations that were never measured
            # (e.g. cold AND concentrated). Local support also needs a training
            # point of a nearby recipe within a (T, c) window of the test row.
            rec_pos = pd.Series(np.arange(len(rec)), index=rec.index)
            p_rec = rec_pos.reindex(tr.formulation_id).to_numpy()
            pT, pc = tr.temperature_K.to_numpy(), tr.concentration.to_numpy()
            wT, wc = tol["temperature_window_K"], tol["concentration_window"]
            cand = np.flatnonzero(r == "in_support")
            for start in range(0, len(cand), 512):
                idx = cand[start : start + 512]
                close = (np.abs(pT[None, :] - T[idx, None]) <= wT) & (np.abs(pc[None, :] - c[idx, None]) <= wc) & near[idx][:, p_rec]
                r[idx[~close.any(axis=1)]] = "no_training_point_nearby_in_T_and_c"
        reason.loc[te.index] = r
    return reason


def support_spec(cfg: Config, spec) -> dict:
    """A task's support entry is either a dict or the name of a support definition."""
    if isinstance(spec, str):
        return cfg["support_definitions"][spec]
    return spec


def _unit_counts(df: pd.DataFrame) -> dict:
    return {
        "n_measurements": int(len(df)),
        "n_formulations": int(df.formulation_id.nunique()),
        "n_series": int(df.series_id.nunique()),
        "n_source_groups": int(df.source_group_id.nunique()),
        "n_author_groups": int(df.author_group_id.nunique()) if "author_group_id" in df else 0,
        "n_chemistry_systems": int(df.chemistry_system_id.nunique()),
    }


# --------------------------------------------------------------------------
# task builders
# --------------------------------------------------------------------------
def _round_robin_units(df: pd.DataFrame, n_folds: int, rng: np.random.Generator) -> dict[str, int]:
    """Assign split units to folds, spreading each chemistry system's units so
    every held-out formulation keeps a familiar system in training. Systems
    with a single unit stay in training (-1)."""
    assign: dict[str, int] = {}
    units = df.groupby("split_unit_id").chemistry_system_id.first()
    offset = 0
    for sys_id in sorted(units.unique()):
        u = units[units == sys_id].index.to_numpy()
        if len(u) < 2:
            for x in u:
                assign[x] = -1
            continue
        u = rng.permutation(np.sort(u))
        for j, x in enumerate(u):
            assign[x] = (j + offset) % n_folds
        offset += len(u)
    return assign


def _interp_folds(df: pd.DataFrame, n_folds: int, rng, prefix: str) -> list[Fold]:
    assign = df.split_unit_id.map(_round_robin_units(df, n_folds, rng)).to_numpy()
    folds = []
    for k in range(n_folds):
        te = assign == k
        if not te.any():
            continue
        folds.append(Fold(f"{prefix}{k}", df.measurement_id[~te].to_numpy(), df.measurement_id[te].to_numpy(), heldout_unit=f"fold{k}"))
    return folds


def build_interpolation(df, t, rng, solv, cfg) -> tuple[list[Fold], list[dict]]:
    folds = _interp_folds(df, t["n_outer_folds"], rng, "interp/")
    elig = []
    for f in folds:
        tr = df[df.measurement_id.isin(f.train)]
        f.dev = _interp_folds(tr, t["n_dev_folds"], rng, f"{f.fold_id}::dev")
        te = df[df.measurement_id.isin(f.test)]
        f.meta = {"interpretation_label": LABEL_SUPPORTED, "reason": "held-out formulations; every test chemistry system present in training", **_unit_counts(te)}
        elig.append({"unit": f.heldout_unit, **f.meta})
    return folds, elig


def build_source_transfer(df, t, rng, solv, cfg) -> tuple[list[Fold], list[dict]]:
    """Hold out one group (source group by default; author group for the v0.2
    sensitivity task) and score it on its common support with the rest."""
    rep = cfg["reporting"]
    gcol = t.get("group_column", "source_group_id")
    prefix = t.get("fold_prefix", "source")
    sup = support_spec(cfg, t["support"])
    folds, elig = [], []
    for sg in sorted(df[gcol].unique()):
        tr = df[df[gcol] != sg]
        te_all = df[df[gcol] == sg]
        why = common_support(tr, te_all, solv, sup)
        te = te_all[why == "in_support"]
        counts = {**_unit_counts(te), "n_source_rows": int(len(te_all)), **{f"n_{k}": int(v) for k, v in why.value_counts().items()}}
        if len(te) < rep["min_test_measurements_per_unit"]:
            elig.append({"unit": sg, "interpretation_label": LABEL_INSUFFICIENT, "reason": "too few held-out rows inside common support", **counts})
            continue
        exact = te.formulation_id.isin(set(tr.formulation_id))
        f = Fold(f"{prefix}/{sg}", tr.measurement_id.to_numpy(), te.measurement_id.to_numpy(), heldout_unit=sg)
        f.test_tags = dict(zip(te.measurement_id, np.where(exact, "exact_formulation_match", "near_support")))
        f.meta = {"interpretation_label": LABEL_SUPPORTED, "reason": f"held-out {gcol.replace('_id', '')} scored on common support only", "n_exact_formulation_match": int(exact.sum()), **counts}
        # development: other groups, each scored on its own support within training
        cand = []
        for dg in sorted(tr[gcol].unique()):
            itr = tr[tr[gcol] != dg]
            dv = tr[tr[gcol] == dg]
            dv = dv[common_support(itr, dv, solv, sup) == "in_support"]
            if len(dv) >= rep["min_test_measurements_per_unit"]:
                cand.append((len(dv), dg, itr, dv))
        for _, dg, itr, dv in sorted(cand, key=lambda x: (-x[0], x[1]))[: t["n_dev_sources"]]:
            f.dev.append(Fold(f"{f.fold_id}::dev:{dg}", itr.measurement_id.to_numpy(), dv.measurement_id.to_numpy(), heldout_unit=dg))
        folds.append(f)
        elig.append({"unit": sg, **f.meta})
    return folds, elig


def component_candidates(df: pd.DataFrame, t: dict) -> pd.DataFrame:
    comps = component_sets(df)
    rows = []
    all_c = sorted(set().union(*comps))
    for c in all_c:
        has = comps.map(lambda s: c in s).to_numpy()
        sub = df[has]
        rows.append(
            {
                "component": c,
                "share": float(has.mean()),
                **_unit_counts(sub),
                "eligible": bool(sub.formulation_id.nunique() >= t["min_formulations"] and has.mean() <= t["max_measurement_share"]),
            }
        )
    return pd.DataFrame(rows)


def build_component_transfer(df, t, rng, solv, cfg) -> tuple[list[Fold], list[dict]]:
    comps = component_sets(df)
    cand = component_candidates(df, t)
    folds, elig = [], []
    eligible = cand[cand.eligible].sort_values(["n_measurements", "component"], ascending=[False, True])
    for r in cand.itertuples():
        if not r.eligible:
            why = "fewer formulations than min_formulations" if r.n_formulations < t["min_formulations"] else "component in too large a share of the track (training would collapse)"
            elig.append({"unit": r.component, "interpretation_label": LABEL_INSUFFICIENT, "reason": why, **{k: v for k, v in r._asdict().items() if k.startswith("n_")}})
            continue
        has = comps.map(lambda s, c=r.component: c in s).to_numpy()
        tr, te = df[~has], df[has]
        familiar_src = te.source_group_id.isin(set(tr.source_group_id))
        t_in = te.temperature_K.between(tr.temperature_K.min(), tr.temperature_K.max())
        c_in = te.concentration.between(tr.concentration.min(), tr.concentration.max())
        n_src_total = te.source_group_id.nunique()
        combined = (not familiar_src.all()) or (not t_in.all()) or (not c_in.all())
        reason = []
        if not familiar_src.all():
            reason.append(f"{(~familiar_src).mean():.0%} of held-out rows come from sources absent from training")
        if n_src_total == 1:
            reason.append("component occurs in a single source group: chemistry and source effects cannot be separated")
        if not t_in.all() or not c_in.all():
            reason.append("held-out rows extend beyond the training temperature/concentration range")
        f = Fold(f"component/{r.component}", tr.measurement_id.to_numpy(), te.measurement_id.to_numpy(), heldout_unit=r.component)
        f.test_tags = dict(zip(te.measurement_id, np.where(familiar_src, "familiar_source", "unfamiliar_source")))
        f.meta = {
            "interpretation_label": LABEL_COMBINED if combined or n_src_total == 1 else LABEL_SUPPORTED,
            "reason": "; ".join(reason) or "component unseen; sources and conditions overlap training",
            "frac_rows_familiar_source": float(familiar_src.mean()),
            "n_component_source_groups": int(n_src_total),
            **_unit_counts(te),
        }
        tr_comps = comps[~has]
        dev_c = [c for c in eligible.component if c != r.component and tr_comps.map(lambda s, c=c: c in s).sum() >= cfg["reporting"]["min_test_measurements_per_unit"]]
        for dc in dev_c[: t["n_dev_components"]]:
            dh = tr_comps.map(lambda s, c=dc: c in s).to_numpy()
            f.dev.append(Fold(f"{f.fold_id}::dev:{dc}", tr.measurement_id[~dh].to_numpy(), tr.measurement_id[dh].to_numpy(), heldout_unit=dc))
        folds.append(f)
        elig.append({"unit": r.component, **f.meta})
    return folds, elig


BOUNDARY_EPS = 1e-9


def _boundary_fold(df, key, group_key, lo_train, hi_test, min_train, higher_is_test, fold_id, comparison: str = "rounded_0.1") -> tuple[Fold | None, pd.Series]:
    """Train on one side of a numeric boundary, test on the other side for groups
    that have at least ``min_train`` distinct training-side values.

    comparison="exact" (v0.3+) compares the reported values with the declared
    boundaries; rounding to 0.1 is used only to count distinct values.
    "rounded_0.1" reproduces v0.1/v0.2, which compared rounded values (so e.g.
    313.15 became 313.2 and fell outside "T <= 313.15")."""
    raw = df[key]
    counted = raw.round(1)
    v = raw if comparison == "exact" else counted
    eps = BOUNDARY_EPS if comparison == "exact" else 0.0
    train_side = v <= lo_train + eps if higher_is_test else v >= lo_train - eps
    test_side = v >= hi_test - eps if higher_is_test else v <= hi_test + eps
    n_train_vals = counted[train_side].groupby(df[group_key][train_side]).nunique()
    has_test = df[group_key][test_side].unique()
    eligible = n_train_vals[n_train_vals >= min_train].index.intersection(has_test)
    te_mask = test_side & df[group_key].isin(eligible)
    # a suspected copy of a training-side point never enters the test side
    te_mask &= ~df.duplicate_group_id.isin(set(df.duplicate_group_id[train_side]))
    if not te_mask.any():
        return None, pd.Series(eligible)
    return Fold(fold_id, df.measurement_id[train_side].to_numpy(), df.measurement_id[te_mask].to_numpy(), heldout_unit=fold_id.split("/")[0]), pd.Series(eligible)


def _build_boundary(df, t, kind) -> tuple[list[Fold], list[dict]]:
    if kind == "temperature_transfer":
        args = ("temperature_K", "formulation_id", t["train_max_K"], t["test_min_K"], t["min_train_temperatures"], True)
        dev_args = ("temperature_K", "formulation_id", t["dev_train_max_K"], t["dev_test_min_K"], t["min_train_temperatures"], True)
        name = "temp_high"
    elif kind == "temperature_transfer_low":
        args = ("temperature_K", "formulation_id", t["train_min_K"], t["test_max_K"], t["min_train_temperatures"], False)
        dev_args = ("temperature_K", "formulation_id", t["dev_train_min_K"], t["dev_test_max_K"], t["min_train_temperatures"], False)
        name = "temp_low"
    else:
        args = ("concentration", "solvent_recipe_id", t["train_max"], t["test_min"], t["min_train_concentrations"], True)
        dev_args = ("concentration", "solvent_recipe_id", t["dev_train_max"], t["dev_test_min"], t["min_train_concentrations"], True)
        name = "conc_high"
    cmp = t.get("boundary_comparison", "rounded_0.1")
    f, eligible = _boundary_fold(df, *args, f"{name}/all", comparison=cmp)
    if f is None:
        return [], [{"unit": "all", "interpretation_label": LABEL_INSUFFICIENT, "reason": "no group spans the declared boundary", "n_measurements": 0}]
    tr = df[df.measurement_id.isin(f.train)]
    # dev boundary lies entirely inside the outer training side
    d, _ = _boundary_fold(tr, *dev_args, f"{f.fold_id}::dev0", comparison=cmp)
    if d is not None:
        f.dev = [d]
    te = df[df.measurement_id.isin(f.test)]
    f.meta = {
        "interpretation_label": LABEL_SUPPORTED,
        "reason": f"same {args[1].replace('_id', '')}s on both sides of a predeclared {args[0]} boundary",
        "n_eligible_groups": int(len(eligible)),
        **_unit_counts(te),
    }
    elig = [{"unit": "all", **f.meta}]
    for sg, g in te.groupby("source_group_id"):
        elig.append({"unit": sg, "interpretation_label": LABEL_SUPPORTED, "reason": "per-source share of the boundary test set", **_unit_counts(g)})
    return [f], elig


def build_external(cfg, meas, t, solv, track) -> tuple[list[Fold], list[dict]]:
    tr = benchmark_frame(cfg, meas, track, include_reserved=False)
    full = benchmark_frame(cfg, meas, track, include_reserved=True)
    te = full[full.dataset_id == t["test_dataset"]]
    why = common_support(tr, te, solv, support_spec(cfg, cfg["tasks"]["source_transfer"]["support"]))
    f = Fold(f"external/{t['test_dataset']}", tr.measurement_id.to_numpy(), te.measurement_id.to_numpy(), heldout_unit=t["test_dataset"])
    f.test_tags = dict(zip(te.measurement_id, np.where(why == "in_support", "common_support", "outside_support")))
    f.meta = {
        "interpretation_label": LABEL_COMBINED,
        "reason": "new source; common_support subset is the supported comparison, remaining rows add chemistry/condition shift",
        "n_in_support": int((why == "in_support").sum()),
        **{f"n_{k}": int(v) for k, v in why.value_counts().items()},
        **_unit_counts(te),
    }
    st = cfg["tasks"]["source_transfer"]
    rep = cfg["reporting"]
    cand = []
    for dg in sorted(tr.source_group_id.unique()):
        itr, dv = tr[tr.source_group_id != dg], tr[tr.source_group_id == dg]
        dv = dv[common_support(itr, dv, solv, support_spec(cfg, st["support"])) == "in_support"]
        if len(dv) >= rep["min_test_measurements_per_unit"]:
            cand.append((len(dv), dg, itr, dv))
    for _, dg, itr, dv in sorted(cand, key=lambda x: (-x[0], x[1]))[: st["n_dev_sources"]]:
        f.dev.append(Fold(f"{f.fold_id}::dev:{dg}", itr.measurement_id.to_numpy(), dv.measurement_id.to_numpy(), heldout_unit=dg))
    return [f], [{"unit": t["test_dataset"], **f.meta}]


BUILDERS = {
    "grouped_interpolation": build_interpolation,
    "source_transfer": build_source_transfer,
    "component_transfer": build_component_transfer,
}


# --------------------------------------------------------------------------
# leakage invariants
# --------------------------------------------------------------------------
class LeakageError(AssertionError):
    pass


def check_fold(f: Fold, df: pd.DataFrame, kind: str, cfg: Config, task: dict, test_role: str = "test") -> dict:
    idx = df.set_index("measurement_id")
    tr, te = idx.loc[f.train], idx.loc[f.test]
    checks = {}

    def require(name, ok):
        checks[name] = bool(ok)
        if not ok:
            raise LeakageError(f"{f.fold_id}: {name}")

    require("ids_unique", not pd.Index(f.train).duplicated().any() and not pd.Index(f.test).duplicated().any())
    require("train_test_disjoint", not set(f.train) & set(f.test))
    require("no_copy_group_crosses", not set(tr.duplicate_group_id) & set(te.duplicate_group_id))
    require("no_replicate_group_crosses", not set(tr.replicate_group_id) & set(te.replicate_group_id))
    require("all_included", (idx.loc[np.r_[f.train, f.test], "inclusion_status"] == "included").all())
    if kind == "grouped_interpolation":
        require("no_split_unit_crosses", not set(tr.split_unit_id) & set(te.split_unit_id))
        require("test_systems_familiar", set(te.chemistry_system_id) <= set(tr.chemistry_system_id))
    if kind in ("source_transfer", "external"):
        require("no_source_group_crosses", not set(tr.source_group_id) & set(te.source_group_id))
        gcol = task.get("group_column", "source_group_id") if kind == "source_transfer" else "source_group_id"
        require(f"no_{gcol}_crosses", not set(tr[gcol]) & set(te[gcol]))
    if kind == "component_transfer":
        c = f.heldout_unit
        require("heldout_component_absent_from_training", not any(c in s for s in component_sets(tr.reset_index())))
        require("heldout_component_in_every_test_row", all(c in s for s in component_sets(te.reset_index())))
    if task.get("boundary_comparison") == "exact" and test_role == "test":
        lim = {"temperature_transfer": ("temperature_K", "train_max_K", "test_min_K", True), "temperature_transfer_low": ("temperature_K", "train_min_K", "test_max_K", False), "concentration_transfer": ("concentration", "train_max", "test_min", True)}.get(kind)
        if lim:
            col, a, b, hi = lim
            if hi:
                require("declared_boundaries_applied", tr[col].max() <= task[a] + BOUNDARY_EPS and te[col].min() >= task[b] - BOUNDARY_EPS)
            else:
                require("declared_boundaries_applied", tr[col].min() >= task[a] - BOUNDARY_EPS and te[col].max() <= task[b] + BOUNDARY_EPS)
    if kind == "temperature_transfer":
        require("boundary_respected", tr.temperature_K.max() < te.temperature_K.min())
    if kind == "temperature_transfer_low":
        require("boundary_respected", tr.temperature_K.min() > te.temperature_K.max())
    if kind == "concentration_transfer":
        require("boundary_respected", tr.concentration.max() < te.concentration.min())
    reserved = reserved_datasets(cfg)
    if kind == "external":
        require("reserved_only_in_test", not tr.dataset_id.isin(reserved).any() and te.dataset_id.isin(reserved).all())
    else:
        require("reserved_dataset_untouched", not tr.dataset_id.isin(reserved).any() and not te.dataset_id.isin(reserved).any())
    for d in f.dev:
        dkind = "source_transfer" if kind == "external" else kind
        sub = check_fold(d, df[df.measurement_id.isin(f.train)], dkind, cfg, task, "dev")
        require(f"dev_inside_outer_train:{d.fold_id}", set(d.train) | set(d.test) <= set(f.train))
        checks.update({f"{d.fold_id}:{k}": v for k, v in sub.items()})
    return checks


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def build_all(cfg: Config, meas: pd.DataFrame, comps: pd.DataFrame):
    """Construct every task/track fold from inputs and provenance only."""
    solv = solvent_matrix(comps)
    rng = np.random.default_rng(cfg["seed"])
    built, elig_rows = [], []
    for task_id, t in cfg["tasks"].items():
        for track in t["tracks"]:
            kind = t["kind"]
            if kind == "external":
                df = benchmark_frame(cfg, meas, track, include_reserved=True)
                folds, elig = build_external(cfg, meas, t, solv, track)
            else:
                df = benchmark_frame(cfg, meas, track)
                if not len(df):
                    folds, elig = [], [{"unit": "all", "interpretation_label": LABEL_INSUFFICIENT, "reason": "track has no included rows", "n_measurements": 0, "n_formulations": 0}]
                elif kind in BUILDERS:
                    folds, elig = BUILDERS[kind](df, t, rng, solv, cfg)
                else:
                    folds, elig = _build_boundary(df, t, kind)
            for e in elig:
                elig_rows.append({"task_id": task_id, "track_id": track, **e})
            built.append((task_id, track, kind, t, df, folds))
    fold_rows = [
        {
            "task_id": task_id,
            "track_id": track,
            "kind": kind,
            "fold_id": f.fold_id,
            "heldout_unit": f.heldout_unit,
            "score_group": t["score_group"],
            "reserved": bool(t.get("reserved", False)),
            "n_train": len(f.train),
            "n_test": len(f.test),
            "n_dev_folds": len(f.dev),
            "dev_folds": "|".join(d.fold_id for d in f.dev),
            **f.meta,
        }
        for task_id, track, kind, t, df, folds in built
        for f in folds
    ]
    folds_df = pd.DataFrame(fold_rows)
    elig_df = _reporting_levels(cfg, folds_df, pd.DataFrame(elig_rows))
    return built, folds_df, elig_df


def make_splits(cfg: Config) -> pd.DataFrame:
    man = RunManifest("make-splits", cfg)
    meas, comps = load_processed(cfg)
    man.add_input(cfg.p("processed") / "measurements.parquet")
    man.add_input(cfg.p("processed") / "components.parquet")
    out = cfg.p("splits")
    out.mkdir(parents=True, exist_ok=True)
    built, folds_df, elig_df = build_all(cfg, meas, comps)
    leak = {}
    for task_id, track, kind, t, df, folds in built:
        rows = []
        for f in folds:
            leak[f"{task_id}/{track}/{f.fold_id}"] = check_fold(f, df, kind, cfg, t)
            tags = f.test_tags
            rows.append(pd.DataFrame({"measurement_id": f.train, "fold_id": f.fold_id, "role": "train", "subset": ""}))
            rows.append(pd.DataFrame({"measurement_id": f.test, "fold_id": f.fold_id, "role": "test", "subset": [tags.get(m, "") for m in f.test]}))
            for d in f.dev:
                rows.append(pd.DataFrame({"measurement_id": d.train, "fold_id": d.fold_id, "role": "train", "subset": ""}))
                rows.append(pd.DataFrame({"measurement_id": d.test, "fold_id": d.fold_id, "role": "dev", "subset": ""}))
        if rows:
            mf = pd.concat(rows, ignore_index=True)
            mf.insert(1, "task_id", task_id)
            mf.insert(2, "track_id", track)
            path = out / f"manifest__{task_id}__{track}.csv.gz"
            with gzip.GzipFile(path, "wb", mtime=0) as gz:  # deterministic bytes
                gz.write(mf.to_csv(index=False).encode())
            man.add_output(path)
    folds_df.to_csv(out / "folds.csv", index=False)
    elig_df.to_csv(out / "eligibility.csv", index=False)
    (out / "leakage_checks.json").write_text(json.dumps(leak, indent=1, sort_keys=True))
    index = {p.name: sha256_file(p) for p in sorted(out.glob("*")) if p.is_file() and p.name != "index.json"}
    (out / "index.json").write_text(json.dumps({"benchmark_version": cfg["benchmark_version"], "files": index}, indent=2))
    for name in ["folds.csv", "eligibility.csv", "leakage_checks.json", "index.json"]:
        man.add_output(out / name)
    man.notes = {"n_folds": len(folds_df), "n_leakage_checks": sum(len(v) for v in leak.values()), "all_leakage_checks_passed": True}
    man.write(out / "runs")
    return folds_df


def _reporting_levels(cfg: Config, folds: pd.DataFrame, elig: pd.DataFrame) -> pd.DataFrame:
    """Headline vs case-study per task/track under the predeclared planning rule."""
    rep = cfg["reporting"]
    levels = {}
    for (task, track), g in elig.groupby(["task_id", "track_id"]):
        kind = cfg["tasks"][task]["kind"]
        if kind in ("temperature_transfer", "temperature_transfer_low", "concentration_transfer", "grouped_interpolation"):
            units = g[g.unit != "all"] if kind != "grouped_interpolation" else g
            if kind == "grouped_interpolation":
                # units are source groups present in the test folds
                sel = folds[(folds.task_id == task) & (folds.track_id == track)] if len(folds) else folds
                ok_units = int(sel.n_source_groups.max()) if len(sel) else 0
            else:
                ok_units = int(((units.interpretation_label == LABEL_SUPPORTED) & (units.n_formulations >= rep["headline_min_formulations_per_unit"])).sum())
        elif kind == "external":
            ok_units = 1
        else:
            ok_units = int(((g.interpretation_label == LABEL_SUPPORTED) & (g.n_formulations >= rep["headline_min_formulations_per_unit"])).sum())
        levels[(task, track)] = ("headline" if ok_units >= rep["headline_min_units"] else "case_study", ok_units)
    elig["reporting_level"] = [levels[(a, b)][0] for a, b in zip(elig.task_id, elig.track_id)]
    elig["n_headline_units"] = [levels[(a, b)][1] for a, b in zip(elig.task_id, elig.track_id)]
    return elig


def load_manifest(cfg: Config, task_id: str, track_id: str) -> pd.DataFrame:
    return pd.read_csv(cfg.p("splits") / f"manifest__{task_id}__{track_id}.csv.gz", keep_default_na=False)


def load_folds(cfg: Config) -> pd.DataFrame:
    return pd.read_csv(cfg.p("splits") / "folds.csv")
