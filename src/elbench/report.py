"""Step 6: technical report and reproduction check.

The report joins the predeclared design, the deterministic result tables and
the hand-written findings (reports/findings_v0.1.md). Every number in the
generated sections is read from tables written by earlier commands.
"""
from __future__ import annotations

import html
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import htmlkit as H
from .config import Config, RunManifest
from .curate import load_processed
from .evaluate import DIAGNOSTIC_MODELS, LOCAL_MODELS, MAIN_MODELS

REFERENCE_TABLES = {
    # file -> (source path relative to results/, key columns, value columns compared within tolerance)
    "summary.csv": ("tables/summary.csv", ["task_id", "track_id", "subset", "model"], ["group_mean_mae_log10", "sample_mae_log10", "worst_quartile_mae_log10", "invalid_rate"]),
    "bootstrap.csv": ("tables/bootstrap.csv", ["task_id", "track_id", "subset", "comparison", "model"], ["group_mean_mae_log10", "diff_vs_reference"]),
    "familiarity_contrast.csv": ("investigation/familiarity_contrast.csv", ["model", "reserved_target", "subset"], ["mean_effect_log10"]),
}
# count columns that must match exactly (when present in both tables)
REFERENCE_COUNTS = {
    "summary.csv": ["n_rows", "n_groups"],
    "bootstrap.csv": ["n_groups", "n_common_rows"],
    "familiarity_contrast.csv": ["n_target_sources", "n_sources_positive"],
}


# --------------------------------------------------------------------------
# tiny markdown subset -> HTML (headings, paragraphs, lists, tables, emphasis, code)
# --------------------------------------------------------------------------
def _inline(s: str) -> str:
    s = html.escape(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<![*\w])\*([^*]+)\*(?!\*)", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<a href='\2'>\1</a>", s)
    return s


def markdown(md: str) -> str:
    out, para, lst, tbl = [], [], [], []

    def flush():
        nonlocal para, lst, tbl
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
        if lst:
            tag = "ol" if lst[0][0] else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(x)}</li>" for _, x in lst) + f"</{tag}>")
        if tbl:
            rows = [[c.strip() for c in r.strip().strip("|").split("|")] for r in tbl if not re.match(r"^\s*\|?\s*:?-{2,}", r)]
            head = "".join(f"<th>{_inline(c)}</th>" for c in rows[0])
            body = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>" for r in rows[1:])
            out.append(f"<div class='tbl'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>")
        para, lst, tbl = [], [], []

    for line in md.splitlines():
        if not line.strip():
            flush()
            continue
        if line.startswith("> "):
            flush()
            out.append(f"<p class='note'>{_inline(line[2:])}</p>")
            continue
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            flush()
            lvl = len(m.group(1)) + 1
            out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>")
        elif line.lstrip().startswith("|"):
            if para or lst:
                flush()
            tbl.append(line)
        elif re.match(r"^\s*([-*]|\d+\.)\s+", line):
            if para or tbl:
                flush()
            ordered = bool(re.match(r"^\s*\d+\.", line))
            lst.append((ordered, re.sub(r"^\s*([-*]|\d+\.)\s+", "", line)))
        elif lst and line.startswith("  "):
            lst[-1] = (lst[-1][0], lst[-1][1] + " " + line.strip())
        else:
            if lst or tbl:
                flush()
            para.append(line.strip())
    flush()
    return "\n".join(out)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------
def _group_dotplot(gm: pd.DataFrame, task: str, track: str, models: list[str]):
    import matplotlib.pyplot as plt

    g = gm[(gm.task_id == task) & (gm.track_id == track) & gm.model.isin(models)]
    groups = g.groupby("group_id").n.first().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(7.5, max(2.2, 0.36 * len(groups) + 1)))
    cmap = plt.get_cmap("tab10")
    for j, m in enumerate(models):
        gg = g[g.model == m].set_index("group_id").reindex(groups)
        y = np.arange(len(groups)) + (j - (len(models) - 1) / 2) * 0.12
        ax.scatter(gg.mae_log10, y, s=22, color=cmap(j), label=m, zorder=3)
    ax.axvline(np.log10(2), color="0.5", lw=0.8, ls="--")
    ax.text(np.log10(2), -0.9, " factor 2", fontsize=7, color="0.4")
    labels = []
    for gid in groups:
        n = int(g[g.group_id == gid].n.iat[0])
        lab = g[g.group_id == gid].interpretation_label.iat[0]
        short = {"supported_comparison": "", "combined_shift": " [combined]", "insufficient_support": " [insufficient]"}.get(lab, "")
        labels.append(f"{str(gid).replace('doi:', '')} (n={n}){short}")
    ax.set_yticks(range(len(groups)), labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("MAE log10(conductivity), per group", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", color="0.9", zorder=0)
    ax.legend(fontsize=7, ncol=len(models), loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
    ax.set_title(f"{task} / {track}", fontsize=9, pad=22)
    return fig


def _familiarity_fig(fam: pd.DataFrame, model: str):
    import matplotlib.pyplot as plt

    f = fam[(fam.model == model)]
    per = f.groupby("target_source").agg(with_=("mae_supported_by_both_with_source", "mean"), without=("mae_supported_by_both_without_source", "mean"), sd=("effect_supported_by_both", "std")).sort_values("without")
    fig, ax = plt.subplots(figsize=(7, max(2.2, 0.42 * len(per) + 1)))
    y = np.arange(len(per))
    ax.hlines(y, per.with_, per.without, color="0.7", lw=2, zorder=1)
    ax.scatter(per.with_, y, color="#2f5d8a", label="with other data from target source", zorder=3)
    ax.scatter(per.without, y, color="#c26b2b", label="without (size- and chemistry-matched)", zorder=3)
    ax.set_yticks(y, [s.replace("doi:", "") for s in per.index], fontsize=7)
    ax.set_xlabel("MAE log10 on target test rows supported by both training sets", fontsize=8)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, ncol=2)
    ax.grid(axis="x", color="0.9", zorder=0)
    ax.set_title(f"Source familiarity ({model})", fontsize=9, pad=20)
    return fig


def _risk_coverage_fig(rc: pd.DataFrame):
    import matplotlib.pyplot as plt

    tasks = list(dict.fromkeys(rc.task_id))
    models = list(dict.fromkeys(rc.model))
    fig, axes = plt.subplots(len(tasks), len(models), figsize=(3.4 * len(models), 2.8 * len(tasks)), squeeze=False)
    styles = {"condition_distance": "-o", "feature_distance": "-s", "ensemble_spread": "-^", "random": "--", "oracle": ":"}
    for i, t in enumerate(tasks):
        for j, m in enumerate(models):
            ax = axes[i][j]
            g = rc[(rc.task_id == t) & (rc.model == m)]
            for sgl, h in g.groupby("signal"):
                h = h.sort_values("target_coverage")
                ax.plot(h.target_coverage, h.group_mean_mae_log10, styles.get(sgl, "-"), ms=3, lw=1.1, label=sgl, color="0.5" if sgl in ("random", "oracle") else None)
            ax.invert_xaxis()
            ax.set_title(f"{t} / {m}", fontsize=8)
            ax.set_xlabel("coverage", fontsize=7)
            ax.set_ylabel("MAE log10 (retained)", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(color="0.92")
    axes[0][0].legend(fontsize=6)
    fig.tight_layout()
    return fig


def _diversity_fig(div: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for s, g in div.groupby("target_source"):
        m = g.groupby("k_sources").mae_log10.agg(["mean", "std"])
        ax.errorbar(m.index, m["mean"], yerr=m["std"], marker="o", ms=4, capsize=2, lw=1, label=s.replace("doi:", ""))
    ax.set_xlabel("number of training sources at fixed budget", fontsize=8)
    ax.set_ylabel("MAE log10 on held-out source", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(color="0.92")
    return fig


def _extrapolation_fig(ep: pd.DataFrame, task: str):
    import matplotlib.pyplot as plt

    e = ep[ep.task_id == task]
    models = [m for m in ["extra_trees", "hist_gb", "knn", "ridge", "arrhenius_local"] if m in set(e.model)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    cmap = plt.get_cmap("tab10")
    for j, m in enumerate(models):
        g = e[e.model == m]
        w = g.assign(wm=g.mae_log10 * g.n, wb=g.bias_log10 * g.n).groupby("distance_beyond_boundary")[["wm", "wb", "n"]].sum()
        for ax, col in zip(axes, ["wm", "wb"]):
            ax.plot(w.index, w[col] / w.n, marker="o", ms=3, lw=1.2, color=cmap(j), label=m)
            ax.scatter(g.distance_beyond_boundary, g.mae_log10 if col == "wm" else g.bias_log10, s=8, alpha=0.35, color=cmap(j))
    unit = "K" if e.variable.iat[0] == "temperature_K" else "mol/kg"
    for ax, lab in zip(axes, ["MAE log10", "bias log10 (pred - true)"]):
        ax.set_xlabel(f"distance beyond training boundary ({unit})", fontsize=8)
        ax.set_ylabel(lab, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(color="0.92")
    axes[1].axhline(0, color="0.5", lw=0.8)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{task}: error grows with extrapolation distance (lines: row-weighted mean; dots: one source group)", fontsize=9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
def _fmt_task_table(summary: pd.DataFrame, boot: pd.DataFrame, task: str, track: str) -> pd.DataFrame:
    s = summary[(summary.task_id == task) & (summary.track_id == track)]
    b = boot[(boot.task_id == task) & (boot.track_id == track) & (boot.comparison.isin(["main", "with_local"]))]
    b = b.drop_duplicates(["subset", "model"], keep="first")[["subset", "model", "ci_low", "ci_high", "diff_vs_reference", "diff_ci_low", "diff_ci_high"]]
    t = s.merge(b, on=["subset", "model"], how="left")
    order = {m: i for i, m in enumerate(MAIN_MODELS + LOCAL_MODELS + DIAGNOSTIC_MODELS)}
    t = t.assign(_o=t.model.map(order)).sort_values(["subset", "_o"])
    t["group MAE log10 [95% CI]"] = [f"{a:.3f} [{lo:.3f}, {hi:.3f}]" if pd.notna(lo) else f"{a:.3f}" for a, lo, hi in zip(t.group_mean_mae_log10, t.ci_low, t.ci_high)]
    t["diff vs ref [95% CI]"] = [f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]" if pd.notna(d) else "" for d, lo, hi in zip(t.diff_vs_reference, t.diff_ci_low, t.diff_ci_high)]
    return t[["subset", "model", "n_rows", "n_groups", "group MAE log10 [95% CI]", "diff vs ref [95% CI]", "sample_mae_log10", "group_median_mae_log10", "worst_quartile_mae_log10", "group_mean_mae_mS_cm", "frac_within_factor2", "invalid_rate"]]


def phase2_gate(cfg: Config, elig: pd.DataFrame, summary: pd.DataFrame, gm: pd.DataFrame) -> pd.DataFrame:
    track = cfg["investigation"]["primary_track"]
    ref = cfg["models"]["reference_model"]
    lv = elig[elig.track_id == track].groupby("task_id").reporting_level.first()
    headline = sorted(t for t, v in lv.items() if v == "headline" and t != "interp_formulation")
    g = gm[(gm.track_id == track) & (gm.model == ref) & (gm.task_id != "interp_formulation")]
    spread = g.groupby("task_id").mae_log10.agg(lambda v: float(v.max() / v.min()) if v.min() > 0 else np.nan)
    rows = [
        {"condition": "At least one shift task has enough held-out units (headline rule)", "evidence": ", ".join(headline) or "none", "met": bool(headline)},
        {"condition": "Errors vary enough that rejection could plausibly help", "evidence": "; ".join(f"{t}: max/min group MAE = {v:.1f}" for t, v in spread.items()), "met": bool((spread > 2).any())},
        {"condition": "Separate development / calibration route exists", "evidence": "Development folds are nested inside every outer fold; reserved external dataset untouched except for logged confirmation runs", "met": True},
    ]
    return pd.DataFrame(rows)


def _external_breakdown(cfg: Config, conf: Path) -> pd.DataFrame:
    """Rows are not independent experiments: count formulations and sweeps per subset and chemistry system."""
    rows = pd.read_parquet(conf / "row_errors.parquet")
    rows = rows[rows.task_id == "external_rahmanian"]
    meas = pd.read_parquet(cfg.p("processed") / "measurements.parquet").set_index("measurement_id")
    out = []
    for (sub, sys_id), g in rows.join(meas[["chemistry_system_id", "formulation_id", "series_id"]], on="measurement_id").groupby(["subset", "chemistry_system_id"]):
        r = {"subset": sub, "chemistry_system_id": sys_id}
        et = g[g.model == "extra_trees"]
        r.update({"rows": len(et), "formulations": et.formulation_id.nunique(), "temperature_sweeps": et.series_id.nunique()})
        for m in ["extra_trees", "ridge"]:
            gm = g[(g.model == m) & g.valid]
            r[f"mae_log10_{m}"] = gm.abs_log_err.mean()
        r["frac_within_factor2_extra_trees"] = float((et.abs_log_err <= np.log10(2)).mean())
        out.append(r)
    return pd.DataFrame(out)


def _temperature_paired(boot: pd.DataFrame, track: str) -> pd.DataFrame:
    b = boot[(boot.track_id == track) & (boot.subset == "all") & (boot.comparison == "with_local") & boot.task_id.isin(["temp_low", "temp_high"])]
    b = b[b.model != "extra_trees"]
    return b[["task_id", "model", "n_groups", "n_common_rows", "group_mean_mae_log10", "ci_low", "ci_high", "diff_vs_reference", "diff_ci_low", "diff_ci_high"]].rename(columns={"diff_vs_reference": "diff_vs_extra_trees"})


KEY_TASKS = ["interp_formulation", "source_transfer", "author_transfer", "component_transfer", "temp_high", "temp_low", "conc_high"]


def _version_sections(cfg: Config, track: str) -> list[str]:
    """Side-by-side comparison with earlier versions and sensitivity runs."""
    versions = {**(cfg.get("compare_with") or {}), f"v{cfg['benchmark_version']} (this report)": cfg["paths"]["results"]}
    if len(versions) < 2:
        return []
    B = ["<h2 id='versions'>Versions side by side</h2>",
         "<p>Same code path, different data versions or source definitions. Rows: primary-track tasks, all test rows; values are group-mean MAE log10 for extra trees / ridge, and the number of score groups. "
         "Differences between columns come from curation (R11), source linking (R12) and the support definition, not from model changes.</p>"]
    rows, contrast = [], []
    for name, rel in versions.items():
        root = cfg.root / rel
        sp = root / "tables" / "summary.csv"
        if not sp.exists():
            B.append(f"<p class='muted'>{H.esc(name)}: no results at <code>{H.esc(rel)}</code></p>")
            continue
        s = pd.read_csv(sp)
        s = s[(s.track_id == track) & (s.subset == "all")]
        for task in KEY_TASKS:
            g = s[s.task_id == task].set_index("model")
            if not len(g):
                continue
            best = g.loc[[m for m in MAIN_MODELS if m in g.index]].group_mean_mae_log10.idxmin()
            rows.append({"version": name, "task_id": task, "n_groups": int(g.n_groups.iloc[0]), "extra_trees": g.group_mean_mae_log10.get("extra_trees"), "ridge": g.group_mean_mae_log10.get("ridge"), "best_model": best})
        fc = root / "investigation" / "familiarity_contrast.csv"
        if fc.exists():
            c = pd.read_csv(fc)
            c = c[(c.model == "extra_trees") & ~c.reserved_target & c.subset.isin(["all", "supported_by_both"])]
            for r in c.itertuples():
                contrast.append({"version": name, "subset": r.subset, "n_target_sources": r.n_target_sources, "mean_effect_log10": r.mean_effect_log10, "ci_low": r.ci_low, "ci_high": r.ci_high, "n_sources_positive": r.n_sources_positive})
    t = pd.DataFrame(rows)
    if len(t):
        order = [v for v in versions if v in set(t.version)]
        wide = t.pivot(index="task_id", columns="version", values=["extra_trees", "ridge", "n_groups"]).reindex(KEY_TASKS).dropna(how="all")
        wide = wide[[(m, v) for m in ["extra_trees", "ridge", "n_groups"] for v in order]]
        wide.columns = [f"{a} | {b}" for a, b in wide.columns]
        for c in [c for c in wide.columns if c.startswith("n_groups")]:
            wide[c] = wide[c].map(lambda x: "" if pd.isna(x) else str(int(x)))
        B.append(H.table(wide, index=True))
    if contrast:
        B.append("<h3>Primary contrast (source familiarity, extra trees) by version</h3>")
        B.append(H.table(pd.DataFrame(contrast)))
    from .provenance import paper_count_breakdown, provenance_pairs

    rep = cfg.p("reports")
    counts = paper_count_breakdown(cfg)
    counts.to_csv(rep / "paper_counts.csv", index=False)
    B.append("<h3>From papers to testable units</h3>")
    B.append("<p>Where the reduction happens: curation, track eligibility (concentration and ratio basis), provenance grouping, and common support when a group is held out.</p>")
    B.append(H.table(counts))
    pairs = provenance_pairs(cfg)
    pairs.to_csv(rep / "provenance_pairs.csv", index=False)
    if len(pairs):
        B.append("<h3>Provenance-based grouping: evidence tiers</h3>")
        B.append("<p>Every pair of papers that share a source group, with the strongest evidence available for that pair. "
                 "<code>documented_reuse</code>: a reviewed statement that the papers report the same measurements (none verified yet). "
                 "<code>suspected_reuse</code>: direct numerical evidence between the pair (R09 near-identical points or R12 agreement). "
                 "<code>conservative_grouping</code>: no direct evidence between the pair; they share a group only through other papers, as a leakage safeguard. "
                 "A source group is therefore a conservative grouping, not a demonstrated single experimental campaign. Numerical agreement is evidence of possible reuse; independent measurements can also agree, and reused measurements can differ after digitization.</p>")
        B.append(H.table(pairs))
    camp = cfg.p("processed") / "campaign_links.csv"
    if camp.exists() and len(pd.read_csv(camp)):
        B.append("<p>Campaign-agreement evidence for every compared pair (T ±1 K, c ±0.03, solvent fractions ±0.02; agreement within 5%):</p>")
        B.append(H.table(pd.read_csv(camp)))
    ag = cfg["paths"].get("author_groups")
    if ag:
        meas = pd.read_parquet(cfg.p("processed") / "measurements.parquet")
        inc = meas[(meas.inclusion_status == "included")]
        tab = inc.groupby(["track_id", "author_group_id"]).agg(source_groups=("source_group_id", "nunique"), papers=("source_doi", "nunique"), measurements=("measurement_id", "size")).reset_index()
        B.append("<h3>Author groups (reviewed mapping from Crossref author lists; affiliations not verified)</h3>")
        B.append(H.table(tab))
    return B


def _sensitivity_sections(cfg: Config, track: str) -> list[str]:
    res = cfg.p("results")
    inv = res / "investigation"
    tab = res / "tables"
    B = []
    if (tab / "paired_group_differences.csv").exists():
        pg = pd.read_csv(tab / "paired_group_differences.csv")
        pg = pg[(pg.track_id == track) & (pg.comparison == "with_local") | (pg.track_id == track) & (pg.comparison == "main") & ~pg.task_id.isin(["temp_low", "temp_high"])]
        if len(pg):
            B.append("<h3>E. Per-group paired differences against the reference model</h3>")
            B.append(f"<p>Model MAE minus {H.esc(cfg['models']['reference_model'])} MAE in each score group (positive: reference better). With 5–9 groups the count of groups favouring each model is as informative as the interval.</p>")
            summ = pg.groupby(["task_id", "model"]).agg(groups=("group_id", "size"), reference_better=("diff_vs_reference", lambda v: int((v > 0).sum())), median_diff=("diff_vs_reference", "median"), min_diff=("diff_vs_reference", "min"), max_diff=("diff_vs_reference", "max")).reset_index()
            B.append(H.table(summ))
    if (tab / "leave_one_group_out.csv").exists():
        lo = pd.read_csv(tab / "leave_one_group_out.csv")
        lo = lo[lo.track_id == track]
        if len(lo):
            B.append("<h3>F. Leave-one-group-out (evaluation jackknife)</h3>")
            B.append("<p>The mean difference against the reference recomputed without each group in turn; <code>sign_flips</code> counts groups whose removal reverses the conclusion.</p>")
            B.append(H.table(lo.groupby(["task_id", "comparison", "model"]).agg(full=("diff_vs_reference_full", "first"), min_without_one=("diff_vs_reference_without_group", "min"), max_without_one=("diff_vs_reference_without_group", "max"), sign_flips=("sign_flips", "sum")).reset_index()))
    if (inv / "condition_logo.csv").exists():
        cl = pd.read_csv(inv / "condition_logo.csv")
        B.append("<h3>G. Leave-one-source-group-out with refitting (condition extrapolation)</h3>")
        B.append("<p>Each source group removed from training and test, models refit with the fold's chosen configuration. Does the ranking survive?</p>")
        w = cl.pivot_table(index=["task_id", "left_out_group"], columns="model", values="group_mean_mae_log10").reset_index()
        B.append(H.table(w))
    if (inv / "familiarity_logo.csv").exists():
        B.append("<h3>H. Leave-one-source-group-out: primary contrast</h3>")
        B.append(H.table(pd.read_csv(inv / "familiarity_logo.csv")))
    if (inv / "support_sensitivity_summary.csv").exists():
        B.append("<h3>I. Source transfer under alternative support definitions</h3>")
        B.append("<p>Every row of each held-out source group predicted once, then scored on the rows each definition supports. "
                 "<code>interior_gap</code>: rows inside the marginal (v0.1) ranges but without a nearby training point in (T, c); their error shows what the v0.1 definition admitted.</p>")
        B.append(H.table(pd.read_csv(inv / "support_sensitivity_summary.csv")))
        defs = cfg.get("support_definitions") or {}
        if defs:
            B.append(H.table(pd.DataFrame([{"definition": k, **v} for k, v in defs.items()])))
    if (inv / "label_evaluation_filtering.csv").exists():
        ef = pd.read_csv(inv / "label_evaluation_filtering.csv")
        B.append("<h3>K. Label reliability: evaluation filtering versus training cleaning</h3>")
        B.append("<p><b>Evaluation filtering</b> holds every model's predictions fixed and rescores them on the cases a digitization floor would keep, so any change comes only from the evaluated population. "
                 "<b>Training cleaning</b> refits the same outer folds with and without the floored training rows and scores both on the identical retained (audited) test cases, so any change comes only from training labels. Fold membership is the reference version's.</p>")
        main = ef[ef.model.isin(["extra_trees", "ridge", "arrhenius_local"])]
        B.append(H.table(main.pivot_table(index=["task_id", "model"], columns="definition", values="change").reset_index()))
        B.append("<p>Coverage (share of test cases retained):</p>" + H.table(ef[ef.model == "extra_trees"].pivot_table(index="task_id", columns="definition", values="coverage").reset_index()))
        if (inv / "label_cases_removed.csv").exists():
            B.append("<p>What the floor removes from each task's test set:</p>" + H.table(pd.read_csv(inv / "label_cases_removed.csv")))
        if (inv / "label_training_cleaning.csv").exists():
            tc = pd.read_csv(inv / "label_training_cleaning.csv")
            if len(tc):
                B.append("<p>Training cleaning on identical audited test cases (negative <code>training_effect</code>: cleaning the training labels helped):</p>" + H.table(tc))
    if (inv / "selective_risk_coverage.csv").exists():
        rc = pd.read_csv(inv / "selective_risk_coverage.csv")
        B.append("<h3>L. Selective prediction for condition extrapolation (exploratory)</h3>")
        B.append("<p>Can unreliable predictions be identified before their labels are known? Each base model's test predictions are ranked by a confidence signal and the least confident are rejected. "
                 "<code>oracle</code> sorts by the actual error and is only a bound. All test labels here were inspected in earlier versions, so these results are exploratory.</p>")
        B.append(H.fig_img(_risk_coverage_fig(rc), "risk coverage"))
        if (inv / "selective_dev_threshold.csv").exists():
            B.append("<p>Thresholds fixed on the development fold for 75% coverage, then applied to the test cases:</p>" + H.table(pd.read_csv(inv / "selective_dev_threshold.csv")))
        if (inv / "selective_rejection.csv").exists():
            rj = pd.read_csv(inv / "selective_rejection.csv")
            B.append("<p>Who gets rejected (development-fixed threshold):</p>" + H.table(rj.pivot_table(index=["task_id", "by", "value"], columns=["model", "signal"], values="rejection_rate").reset_index()))
    if (inv / "ridge_audit.csv").exists():
        B.append("<h3>J. Ridge numerical audit (component holdout; post-hoc diagnostic)</h3>")
        B.append("<p>Scaler statistics are checked against the training rows; z-scores show how far the held-out component's descriptors lie outside the training distribution; "
                 "the clipped column is a post-hoc diagnostic of linear extrapolation in this representation, not a proposed model; even clipped, 1.2 log10 is about a 16-fold error.</p>")
        B.append(H.table(pd.read_csv(inv / "ridge_audit.csv")))
        rc = pd.read_csv(inv / "ridge_audit_contributions.csv")
        B.append(H.table(rc.groupby("component").head(6)))
    return B


def report(cfg: Config) -> Path:
    man = RunManifest("report", cfg)
    res = cfg.p("results")
    rep = cfg.p("reports")
    tab_out = rep / "tables"
    tab_out.mkdir(parents=True, exist_ok=True)
    meas, comps = load_processed(cfg)
    summary = pd.read_csv(res / "tables" / "summary.csv")
    boot = pd.read_csv(res / "tables" / "bootstrap.csv")
    gm = pd.read_csv(res / "tables" / "group_metrics.csv")
    ranks = pd.read_csv(res / "tables" / "group_rankings.csv")
    folds = pd.read_csv(cfg.p("splits") / "folds.csv")
    elig = pd.read_csv(cfg.p("splits") / "eligibility.csv")
    sources = pd.read_csv(cfg.p("raw") / "sources.csv")
    runs = pd.read_csv(res / "train_runs.csv")
    inv = res / "investigation"
    rs = pd.read_csv(inv / "ranking_shift.csv")
    gaps = pd.read_csv(inv / "transfer_gaps.csv")
    fam = pd.read_csv(inv / "source_familiarity.csv")
    con = pd.read_csv(inv / "familiarity_contrast.csv")
    div = pd.read_csv(inv / "source_diversity.csv")
    cases = pd.read_csv(inv / "failure_cases.csv")
    recs = pd.read_csv(inv / "failure_records.csv")
    fam_c = pd.read_csv(inv / "source_familiarity__confirm.csv") if (inv / "source_familiarity__confirm.csv").exists() else None
    con_c = pd.read_csv(inv / "familiarity_contrast__confirm.csv") if (inv / "familiarity_contrast__confirm.csv").exists() else None
    exposure = pd.read_csv(res / "reserved_exposure_log.csv") if (res / "reserved_exposure_log.csv").exists() else pd.DataFrame()
    findings_path = cfg.root / cfg["paths"].get("findings", "reports/findings_v0.1.md")
    findings = findings_path.read_text() if findings_path.exists() else "_Findings not yet written._"
    track = cfg["investigation"]["primary_track"]

    # deterministic tables shipped with the report
    for name, (src, keys, _) in REFERENCE_TABLES.items():
        df = pd.read_csv(res / src)
        df.sort_values(keys).round(6).to_csv(tab_out / name, index=False)
        man.add_output(tab_out / name)
    gm.sort_values(["task_id", "track_id", "model", "group_id"]).round(6).to_csv(tab_out / "group_metrics.csv", index=False)

    B = []
    nav = [("findings", "Findings"), ("versions", "Versions"), ("data", "Data"), ("design", "Design"), ("results", "Results"), ("investigation", "Investigation"), ("failures", "Failure cases"), ("compute", "Compute"), ("gate", "Phase 2 gate"), ("repro", "Reproduce")]
    B.append("<nav>" + "".join(f"<a href='#{a}'>{b}</a>" for a, b in nav) + "</nav>")
    B.append("<p class='note'>Research screening benchmark for measured ionic conductivity of liquid Li electrolytes. Not a battery qualification tool. "
             "Labels <b>supported_comparison / combined_shift / insufficient_support</b> describe evaluation support, not causal identification.</p>")
    B.append("<section id='findings'>" + markdown(findings) + "</section>")
    B.extend(_version_sections(cfg, track))

    # ---- data
    inc = meas[meas.inclusion_status == "included"]
    B.append("<h2 id='data'>Data provenance and curation</h2>")
    B.append(H.table(sources[["dataset_id", "file", "release", "retrieved_utc", "licence", "redistribute_raw", "role"]]))
    cc = pd.read_csv(rep / "audit" / "coverage_counts.csv")
    B.append(H.table(cc, floatfmt="{:.0f}"))
    exc = meas[meas.inclusion_status == "excluded"].exclusion_reason.value_counts().rename_axis("first exclusion reason").reset_index(name="rows")
    B.append(H.table(exc))
    B.append("<p>Full coverage and confounding audit: <a href='coverage.html'>coverage.html</a>. "
             f"{len(meas)} reported measurements, {len(inc)} included across three basis-specific tracks.</p>")

    # ---- design
    B.append("<h2 id='design'>Frozen evaluation design</h2>")
    lvl = elig.groupby(["task_id", "track_id"]).agg(reporting_level=("reporting_level", "first"), headline_units=("n_headline_units", "first")).reset_index()
    fs = folds.groupby(["task_id", "track_id"]).agg(folds=("fold_id", "size"), test_rows=("n_test", "sum"), supported=("interpretation_label", lambda s: int((s == "supported_comparison").sum())), combined=("interpretation_label", lambda s: int((s == "combined_shift").sum()))).reset_index()
    B.append(H.table(fs.merge(lvl, on=["task_id", "track_id"], how="left")))

    # ---- results
    B.append("<h2 id='results'>Results</h2>")
    B.append("<p>Primary metric: MAE of log10(conductivity / mS cm⁻¹), averaged within score groups (source groups, i.e. provenance-based groupings, unless stated) and then equally across groups. "
             "Intervals: paired bootstrap obtained by resampling those groups (same resampled groups for every model), on rows valid for every compared model. "
             "They describe variation under the chosen grouping; groups may still share equipment, calibration or protocols, so they are not population-level uncertainty across laboratories. With fewer than about eight groups, read the per-group results first. "
             f"Reference model for differences: <code>{cfg['models']['reference_model']}</code>. 0.30 log10 ≈ factor-of-two error for one prediction.</p>")
    for tr in [track] + [t for t in cfg["tracks"] if t != track]:
        B.append(f"<h3>Track {H.esc(tr)}{' (primary)' if tr == track else ' (secondary, case studies)'}</h3>")
        for task in cfg["tasks"]:
            if not ((summary.task_id == task) & (summary.track_id == tr)).any():
                continue
            lev = lvl[(lvl.task_id == task) & (lvl.track_id == tr)].reporting_level
            B.append(f"<h4>{H.esc(task)} <span class='chip'>{H.esc(lev.iat[0] if len(lev) else '')}</span></h4>")
            B.append(H.table(_fmt_task_table(summary, boot, task, tr)))
            if tr == track:
                models = [m for m in MAIN_MODELS if m != "median"] + ([m for m in LOCAL_MODELS if task.startswith("temp")])
                B.append(H.fig_img(_group_dotplot(gm, task, tr, models), f"group errors {task}"))

    # ---- reserved external confirmation (run once, logged)
    conf = res / "tables_confirm"
    if (conf / "summary.csv").exists():
        sc, bc = pd.read_csv(conf / "summary.csv"), pd.read_csv(conf / "bootstrap.csv")
        B.append("<h3>Reserved external evaluation: Rahmanian et al. (confirmation run)</h3>")
        B.append("<p>Trained on all included CALiSol rows of the primary track, tested on the reserved automated-lab dataset. "
                 "<code>common_support</code> rows have CALiSol training data with the same chemistry within the declared tolerances; the rest add chemistry and condition shift. "
                 "Groups are chemistry systems (two), so intervals are not meaningful; read the rows individually. Exposure is recorded in the log below.</p>")
        B.append(H.table(_fmt_task_table(sc, bc, "external_rahmanian", track)))
        B.append(H.table(_external_breakdown(cfg, conf)))

    # ---- investigation
    B.append("<h2 id='investigation'>Investigation: source and chemistry entanglement</h2>")
    B.append("<h3>A. Model rankings across tasks</h3>")
    B.append(H.table(rs))
    rk = ranks[ranks.track_id == track].groupby(["task_id", "best_model"]).size().unstack(fill_value=0).reset_index()
    B.append("<p>Best model per score group (count of groups):</p>" + H.table(rk))
    tp = _temperature_paired(boot, track)
    if len(tp):
        B.append("<h3>A1. Temperature extrapolation: paired differences against extra trees</h3>")
        B.append("<p>All models are scored on exactly the same rows (<code>n_common_rows</code>). Negative values: better than extra trees. "
                 "The per-formulation Arrhenius fit needs at least three training-side temperatures of the test formulation itself; every temperature-task test formulation meets this by construction, "
                 "and the other models see the same training rows, so it uses no extra information, but it cannot be applied to a formulation without warm-side measurements.</p>")
        B.append(H.table(tp))
    if (inv / "extrapolation_profile.csv").exists():
        ep = pd.read_csv(inv / "extrapolation_profile.csv")
        B.append("<h3>A2. Condition extrapolation: error against distance beyond the boundary</h3>")
        B.append("<p>If failures were source-specific, sources would disagree at matched distance. Dots are individual source groups.</p>")
        for task in ["temp_low", "temp_high", "conc_high"]:
            if (ep.task_id == task).any():
                B.append(H.fig_img(_extrapolation_fig(ep, task), f"extrapolation {task}"))
    B.append("<h3>B. Same rows, different training: shift-task error minus interpolation error</h3>")
    gsum = gaps.groupby(["model", "task_id", "interpretation_label"]).agg(groups=("group", "size"), rows=("n_rows", "sum"), mean_gap=("gap", "mean"), min_gap=("gap", "min"), max_gap=("gap", "max")).reset_index()
    B.append(H.table(gsum))
    B.append(H.table(gaps[gaps.model == cfg["models"]["reference_model"]].sort_values(["task_id", "gap"], ascending=[True, False])))
    B.append("<h3>C. Controlled source-familiarity comparison (primary contrast)</h3>")
    B.append("<p>For each eligible target source, 30% of its formulations are reserved as a test subset. Two equal-size training sets share the same background sample from other sources; "
             "one adds the target source's remaining formulations, the other replaces them with rows from other sources drawn from the same chemistry system wherever possible. "
             "Effect = MAE(without) − MAE(with), positive when familiarity with the source helps. The predeclared primary contrast uses extra_trees on test rows inside the common support of both training sets.</p>")
    B.append(H.table(con))
    B.append(H.fig_img(_familiarity_fig(fam, "extra_trees"), "source familiarity"))
    per = fam.groupby(["target_source", "model"]).agg(
        effect_supported=("effect_supported_by_both", "mean"), effect_sd=("effect_supported_by_both", "std"), effect_all=("effect_all", "mean"),
        n_test=("n_test", "first"), n_supported_by_both=("n_test_supported_by_both", "mean"), same_system_replacement=("replacement_same_system_frac", "mean"),
        support_A=("test_in_support_A", "mean"), support_B=("test_in_support_B", "mean")).reset_index()
    B.append(H.table(per))
    if con_c is not None:
        B.append("<h4>Confirmation on reserved data</h4>")
        B.append(H.table(con_c[con_c.reserved_target]))
        B.append(H.table(fam_c[fam_c.reserved_target].groupby("model")[["effect_all", "effect_supported_by_both", "n_test_supported_by_both", "replacement_same_system_frac", "test_in_support_A", "test_in_support_B"]].agg(["mean", "std"]).round(4), index=True))
    B.append("<h3>D. Source diversity at a matched budget (exploratory)</h3>")
    B.append(H.table(div.groupby(["target_source", "k_sources"]).agg(repeats=("repeat", "size"), mae_mean=("mae_log10", "mean"), mae_sd=("mae_log10", "std"), test_in_support=("test_in_support", "mean"), candidate_sources=("n_candidate_sources", "first")).reset_index()))
    if len(div):
        B.append(H.fig_img(_diversity_fig(div), "source diversity"))

    # ---- failure cases
    B.extend(_sensitivity_sections(cfg, track))
    B.append("<h2 id='failures'>Inspected failure cases</h2>")
    B.append("<p>Worst held-out formulations for the reference model on the primary track (one per source group), traced to their raw records.</p>")
    B.append(H.table(cases))
    for c in cases.case:
        B.append(f"<h4>Case {c}: raw records</h4>")
        B.append(H.table(recs[recs.case == c].drop(columns=["case"])))

    # ---- compute
    B.append("<h2 id='compute'>Compute and run record</h2>")
    comp = runs.groupby("model").agg(jobs=("fold_id", "size"), cpu_s=("cpu_s", "sum"), wall_s=("wall_s", "sum"), max_worker_rss_mb=("worker_peak_rss_mb", "max"), failures=("n_failures", "sum"), failed_seed_fits=("failed_seeds", "sum")).reset_index()
    B.append(H.table(comp))
    B.append("<p>Budget: ≤12 configurations per family chosen on nested development folds with the tuning seed; stochastic families refit with 3 seeds. Peak memory is the worker-process peak RSS. "
             "Fitted models are saved for small families; tree ensembles are refit deterministically from the recorded configuration, seed and manifest (see README).</p>")
    if cfg.get("reserved_note"):
        B.append(f"<p class='note'>{H.esc(cfg['reserved_note'])}</p>")
    if len(exposure):
        B.append("<h3>Reserved-data exposure log</h3>" + H.table(exposure))
    else:
        B.append("<p>Reserved external data: <b>no exposure recorded</b> (labels never scored).</p>")

    B.append("<h2 id='gate'>Phase 2 gate evidence</h2>")
    B.append(H.table(phase2_gate(cfg, elig, summary, gm)))

    B.append("<h2 id='repro'>Reproduce</h2>")
    B.append("<pre><code>uv sync --extra dev\nuv run elbench all\nuv run elbench verify</code></pre>")
    B.append(f"<p class='muted'>Generated {time.strftime('%Y-%m-%d')} by elbench {cfg['benchmark_version']}; config sha256 {cfg.sha256[:16]}.</p>")
    path = rep / "report.html"
    path.write_text(H.page("Electrolyte generalization benchmark v0.1", "\n".join(B), "Phase 1 technical report: data provenance, frozen splits, CPU baselines and a source/chemistry entanglement investigation"))
    man.add_output(path)
    man.write(rep / "runs")
    return path


def compare_tables(a: pd.DataFrame, b: pd.DataFrame, keys: list[str], vals: list[str], counts: list[str], tol: float, name: str) -> tuple[bool, list[str]]:
    """Strict comparison of a reference table (a) and a regenerated table (b).

    Fails on duplicate keys, rows present in only one table, a value missing on
    one side only, a non-finite value that differs, a value differing by more
    than tol, or any count column that differs."""
    ok, lines = True, []
    for side, t in (("reference", a), ("new", b)):
        dup = int(t.duplicated(keys).sum())
        if dup:
            ok = False
            lines.append(f"{name}: {dup} duplicate keys in {side} table")
    m = a.merge(b, on=keys, how="outer", suffixes=("_ref", "_new"), indicator=True)
    only = int((m._merge != "both").sum())
    if only:
        ok = False
        lines.append(f"{name}: {only} rows present in only one table")
    both = m[m._merge == "both"]
    for v in vals:
        x, y = both[f"{v}_ref"].to_numpy(dtype=float), both[f"{v}_new"].to_numpy(dtype=float)
        nx, ny = np.isnan(x), np.isnan(y)
        miss = int((nx ^ ny).sum())
        finite = np.isfinite(x) & np.isfinite(y)
        inf_mismatch = int(((~finite) & ~nx & ~ny & (x != y)).sum())  # e.g. inf vs finite
        d = np.abs(x[finite] - y[finite])
        bad = int((d > tol).sum())
        if miss or inf_mismatch or bad:
            ok = False
            lines.append(f"{name}.{v}: {miss} missing on one side, {inf_mismatch} infinite mismatches, {bad} values differ by more than {tol} (max {d.max() if len(d) else 0:.4g})")
        else:
            lines.append(f"{name}.{v}: OK ({int(nx.sum())} missing on both sides, max abs diff {d.max() if len(d) else 0:.2e})")
    for c in counts:
        if f"{c}_ref" in both and f"{c}_new" in both:
            diff = int((both[f"{c}_ref"].fillna(-1) != both[f"{c}_new"].fillna(-1)).sum())
            if diff:
                ok = False
                lines.append(f"{name}.{c}: {diff} counts differ")
            else:
                lines.append(f"{name}.{c}: counts identical")
    return ok, lines


def verify_reproduction(cfg: Config, write_reference: bool = False) -> bool:
    """Compare the report tables currently on disk with the published reference tables.

    This checks committed or existing tables; it does not rerun anything. To
    verify that a fresh run reproduces the reference, use `elbench reproduce`,
    which rebuilds the benchmark in a clean copy and then runs this check there."""
    tol = cfg["evaluation"]["reproduction_tolerance_log10"]
    cur = cfg.p("reports") / "tables"
    ref = cfg.p("reference_tables")
    if write_reference:
        ref.mkdir(parents=True, exist_ok=True)
        for name in REFERENCE_TABLES:
            (ref / name).write_bytes((cur / name).read_bytes())
        print(f"reference tables written to {ref}")
        return True
    ok = True
    lines = [f"mode: compare {cur} with {ref} (no rerun)"]
    for name, (_, keys, vals) in REFERENCE_TABLES.items():
        good, ls = compare_tables(pd.read_csv(ref / name), pd.read_csv(cur / name), keys, vals, REFERENCE_COUNTS.get(name, []), tol, name)
        ok &= good
        lines.extend(ls)
    print("\n".join(lines))
    print("REPRODUCTION OK" if ok else "REPRODUCTION FAILED")
    (cfg.p("reports") / "verify.json").write_text(json.dumps({"ok": ok, "tolerance": tol, "checks": lines}, indent=1))
    return ok


def reproduce(cfg: Config, workdir: str | None = None) -> bool:
    """Rebuild the benchmark from raw inputs in a clean copy and verify it there."""
    import shutil
    import subprocess
    import sys
    import tempfile

    work = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="elbench-reproduce-"))
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f"{work} is not empty")
    work.mkdir(parents=True, exist_ok=True)
    skip = {".venv", ".git", "results", "splits"}
    for item in cfg.root.iterdir():
        if item.name in skip:
            continue
        if item.is_dir():
            shutil.copytree(item, work / item.name, ignore=shutil.ignore_patterns("processed", "cache", "__pycache__"))
        else:
            shutil.copy2(item, work / item.name)
    # drop generated report folders but keep reference tables and findings
    for d in (work / "reports").glob("v*"):
        if d.is_dir():
            shutil.rmtree(d)
    rel = cfg.path.relative_to(cfg.root)
    print(f"reproducing in {work}", flush=True)
    for cmd in (["all"], ["verify"]):
        r = subprocess.run([sys.executable, "-m", "elbench.cli", "--config", str(work / rel), *cmd], cwd=work)
        if r.returncode:
            print(f"REPRODUCE FAILED at {cmd[0]}")
            return False
    return True
