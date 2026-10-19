"""Step 3: coverage and confounding audit (built before any model is fitted).

Writes coverage.html and eligibility.csv under paths.reports, plus the CSV
tables behind every figure under paths.reports/audit/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import htmlkit as H
from .config import Config, RunManifest
from .curate import load_processed
from .splits import LABEL_COMBINED, LABEL_INSUFFICIENT, LABEL_SUPPORTED, build_all, component_sets, reserved_datasets

COUNT_COLS = dict(
    n_measurements=("measurement_id", "size"),
    n_formulations=("formulation_id", "nunique"),
    n_series=("series_id", "nunique"),
    n_replicate_groups=("replicate_group_id", "nunique"),
    n_source_groups=("source_group_id", "nunique"),
    n_chemistry_systems=("chemistry_system_id", "nunique"),
)


def coverage_counts(meas: pd.DataFrame) -> pd.DataFrame:
    inc = meas[meas.inclusion_status == "included"]
    t = inc.groupby(["track_id", "dataset_id"]).agg(**COUNT_COLS).reset_index()
    tot = inc.groupby("track_id").agg(**COUNT_COLS).reset_index().assign(dataset_id="(all)")
    return pd.concat([t, tot]).sort_values(["track_id", "dataset_id"]).reset_index(drop=True)


def exclusion_impact(meas: pd.DataFrame, exc: pd.DataFrame) -> pd.DataFrame:
    """Marginal effect of each rule: coverage of rows failing only this rule."""
    reasons = exc.groupby("measurement_id").reason.apply(frozenset)
    r = meas.measurement_id.map(reasons).map(lambda x: x if isinstance(x, frozenset) else frozenset())
    kept = meas[r.map(len) == 0]
    rows = []
    for reason in sorted(exc.reason.unique()):
        only = meas[r.map(lambda s, q=reason: s == frozenset([q]))]
        anyr = meas[r.map(lambda s, q=reason: q in s)]
        base = pd.concat([kept, only])
        lost = {}
        for col in ["source_group_id", "chemistry_system_id", "salt_id"]:
            lost[col] = sorted(set(base[col]) - set(kept[col]))
        lost_comp = sorted(set().union(*component_sets(base)) - set().union(*component_sets(kept))) if len(base) else []
        rows.append(
            {
                "reason": reason,
                "rows_with_reason": len(anyr),
                "rows_only_this_reason": len(only),
                "datasets": ",".join(sorted(anyr.dataset_id.unique())),
                "sources_touched": anyr.source_doi.nunique(),
                "source_groups_lost": "; ".join(lost["source_group_id"]),
                "chemistry_systems_lost": "; ".join(lost["chemistry_system_id"]),
                "components_lost": "; ".join(lost_comp),
            }
        )
    return pd.DataFrame(rows).sort_values("rows_with_reason", ascending=False)


def source_component_matrix(df: pd.DataFrame) -> pd.DataFrame:
    comps = component_sets(df)
    long = pd.DataFrame({"source_group_id": df.source_group_id.to_numpy(), "component": comps.to_numpy()}).explode("component")
    return long.groupby(["component", "source_group_id"]).size().unstack(fill_value=0)


def _short_src(s: str) -> str:
    return s.replace("doi:", "")


def _heatmap(mat: pd.DataFrame, title: str):
    import matplotlib.pyplot as plt

    m = mat.loc[mat.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * m.shape[1] + 3), max(3, 0.32 * m.shape[0] + 1.5)))
    v = np.log10(m.to_numpy() + 1)
    ax.imshow(v, aspect="auto", cmap="Blues")
    ax.set_xticks(range(m.shape[1]), [_short_src(c) for c in m.columns], rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(m.shape[0]), m.index, fontsize=8)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m.iat[i, j]:
                ax.text(j, i, str(m.iat[i, j]), ha="center", va="center", fontsize=6, color="black" if v[i, j] < 2.4 else "white")
    ax.set_title(title, fontsize=10)
    return fig


def _tc_scatter(df: pd.DataFrame, systems: list[str], title: str):
    import matplotlib.pyplot as plt

    n = len(systems)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.3 * rows), squeeze=False)
    srcs = sorted(df.source_group_id.unique())
    cmap = plt.get_cmap("tab10")
    color = {s: cmap(i % 10) for i, s in enumerate(srcs)}
    for ax, sys_id in zip(axes.flat, systems):
        g = df[df.chemistry_system_id == sys_id]
        for s, gs in g.groupby("source_group_id"):
            ax.scatter(gs.concentration, gs.temperature_K, s=6, alpha=0.6, color=color[s], label=_short_src(s))
        ax.set_title(sys_id, fontsize=9)
        ax.set_xlabel("concentration", fontsize=8)
        ax.set_ylabel("T / K", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc="best", markerscale=2)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def secondary_overlap(cfg: Config, meas: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Does the reserved secondary dataset already appear in the backbone?"""
    res = reserved_datasets(cfg)
    inc = meas[meas.inclusion_status == "included"]
    sec = inc[inc.dataset_id.isin(res)]
    bb = inc[~inc.dataset_id.isin(res) & inc.track_id.isin(sec.track_id.unique())]
    rows = []
    for sys_id, g in sec.groupby("chemistry_system_id"):
        b = bb[bb.chemistry_system_id == sys_id]
        rows.append(
            {
                "chemistry_system_id": sys_id,
                "secondary_rows": len(g),
                "secondary_formulations": g.formulation_id.nunique(),
                "backbone_rows": len(b),
                "backbone_source_groups": "; ".join(sorted(_short_src(s) for s in b.source_group_id.unique())),
                "shared_formulation_ids": len(set(g.formulation_id) & set(b.formulation_id)),
                "secondary_T_range": f"{g.temperature_K.min():.1f}-{g.temperature_K.max():.1f}",
                "backbone_T_range": f"{b.temperature_K.min():.1f}-{b.temperature_K.max():.1f}" if len(b) else "",
                "secondary_c_range": f"{g.concentration.min():.2f}-{g.concentration.max():.2f}",
                "backbone_c_range": f"{b.concentration.min():.2f}-{b.concentration.max():.2f}" if len(b) else "",
            }
        )
    flags = meas["quality_flags"].fillna("")
    copies = meas[flags.str.contains("cross_source_copy") & meas.dataset_id.isin(res)]
    shared_doi = set(sec.source_doi) & set(bb.source_doi)
    return pd.DataFrame(rows), {"copies_with_backbone": int(len(copies)), "shared_source_dois": sorted(shared_doi)}


def audit(cfg: Config) -> None:
    man = RunManifest("audit", cfg)
    meas, comps = load_processed(cfg)
    exc = pd.read_csv(cfg.p("processed") / "exclusions.csv")
    links = pd.read_csv(cfg.p("processed") / "copy_links.csv")
    sources = pd.read_csv(cfg.p("raw") / "sources.csv")
    for f in ["measurements.parquet", "components.parquet", "exclusions.csv", "copy_links.csv"]:
        man.add_input(cfg.p("processed") / f)
    out = cfg.p("reports")
    tab = out / "audit"
    tab.mkdir(parents=True, exist_ok=True)

    counts = coverage_counts(meas)
    impact = exclusion_impact(meas, exc)
    _, folds, elig = build_all(cfg, meas, comps)
    overlap, overlap_meta = secondary_overlap(cfg, meas)
    cross = links[links.source_doi_a != links.source_doi_b]
    if len(cross):
        pair = cross[["source_doi_a", "source_doi_b"]].apply(lambda r: pd.Series(sorted(r), index=["source_doi_a", "source_doi_b"]), axis=1)
        pair["status"] = cross.status.to_numpy()
        link_summary = pair.groupby(["source_doi_a", "source_doi_b", "status"]).size().rename("n_matched_points").reset_index()
    else:
        link_summary = pd.DataFrame(columns=["source_doi_a", "source_doi_b", "status", "n_matched_points"])

    counts.to_csv(tab / "coverage_counts.csv", index=False)
    impact.to_csv(tab / "exclusion_impact.csv", index=False)
    overlap.to_csv(tab / "secondary_overlap.csv", index=False)
    link_summary.to_csv(tab / "source_links.csv", index=False)
    elig.to_csv(out / "eligibility.csv", index=False)

    res = reserved_datasets(cfg)
    inc = meas[(meas.inclusion_status == "included") & ~meas.dataset_id.isin(res)]
    body = ["<nav>" + "".join(f"<a href='#{a}'>{b}</a>" for a, b in [("prov", "Provenance"), ("counts", "Counts"), ("excl", "Exclusions"), ("matrix", "Source x component"), ("tc", "Conditions"), ("elig", "Eligibility"), ("second", "Secondary data")]) + "</nav>"]

    body.append("<h2 id='prov'>Provenance</h2>")
    body.append(H.table(sources[["dataset_id", "file", "release", "retrieved_utc", "sha256", "licence", "redistribute_raw", "role"]].assign(sha256=lambda d: d.sha256.str[:16] + "…")))
    body.append(
        "<p>Counts below are measurements after curation. They are <em>not</em> independent formulations: "
        "a formulation is measured at many temperatures, a series is one sweep in one paper, and a source group is one paper "
        "plus any paper shown to reuse its points. Lab identity is never inferred from a DOI.</p>"
    )
    body.append("<h2 id='counts'>Counts by track</h2>")
    body.append(H.table(counts, floatfmt="{:.0f}"))
    if len(link_summary):
        body.append("<h3>Cross-source agreement and suspected reuse</h3>")
        body.append("<p>Points that match another paper's point within the copy tolerance (same chemistry, ratio ±0.005, concentration ±0.001, T ±0.05 K, conductivity ±0.5%). "
                    "When two papers share at least three such points (<code>suspected_reuse_linked</code>) they are treated as one source group, so they never sit on both sides of a source split. "
                    "Isolated agreements are listed but not linked.</p>")
        body.append(H.table(link_summary))

    body.append("<h2 id='excl'>Exclusions and their effect on coverage</h2>")
    body.append("<p><code>rows_only_this_reason</code> is the marginal effect of a rule; the <code>…_lost</code> columns list entities that disappear from the curated set because of that rule alone.</p>")
    body.append(H.table(impact))

    body.append("<h2 id='matrix'>Source by component coverage</h2>")
    single_rows = []
    for track in sorted(inc.track_id.dropna().unique()):
        df = inc[inc.track_id == track]
        mat = source_component_matrix(df)
        mat.to_csv(tab / f"source_component__{track}.csv")
        body.append(f"<h3>{H.esc(track)}</h3>")
        body.append(H.fig_img(_heatmap(mat, f"{track}: measurements per component and source group (log colour)"), f"source by component, {track}"))
        nsrc = (mat > 0).sum(axis=1)
        for comp, n in nsrc.items():
            single_rows.append({"track_id": track, "component": comp, "n_source_groups": int(n), "n_measurements": int(mat.loc[comp].sum()), "only_source": _short_src(mat.columns[(mat.loc[comp] > 0).to_numpy()][0]) if n == 1 else ""})
    single = pd.DataFrame(single_rows)
    single.to_csv(tab / "component_source_counts.csv", index=False)
    body.append("<h3>Components that occur in only one source group</h3>")
    body.append("<p>For these components, chemical novelty and source change cannot be separated in any holdout.</p>")
    body.append(H.table(single[single.n_source_groups == 1].sort_values(["track_id", "n_measurements"], ascending=[True, False])))

    body.append("<h2 id='tc'>Temperature versus concentration within chemistry systems</h2>")
    for track in sorted(inc.track_id.dropna().unique()):
        df = inc[inc.track_id == track]
        nsrc = df.groupby("chemistry_system_id").source_group_id.nunique()
        multi = nsrc[nsrc >= 2].index.tolist()
        top = df.chemistry_system_id.value_counts().index.tolist()
        systems = (multi + [s for s in top if s not in multi])[:9]
        body.append(f"<h3>{H.esc(track)}</h3><p class='muted'>Systems shared by ≥2 source groups first: {H.esc(', '.join(multi) or 'none')}</p>")
        body.append(H.fig_img(_tc_scatter(df, systems, f"{track}: conditions by source group"), f"T vs c, {track}"))

    body.append("<h2 id='elig'>Task eligibility and interpretation labels</h2>")
    body.append(
        f"<p>Labels: <span class='chip'>{LABEL_SUPPORTED}</span> intended shift present with documented overlap of nuisance dimensions; "
        f"<span class='chip'>{LABEL_COMBINED}</span> chemistry, source or conditions change together; "
        f"<span class='chip'>{LABEL_INSUFFICIENT}</span> too few groups or no credible overlap. These describe evaluation support, not causal identification.</p>"
    )
    lvl = elig.groupby(["task_id", "track_id"]).agg(
        reporting_level=("reporting_level", "first"),
        headline_units=("n_headline_units", "first"),
        supported=("interpretation_label", lambda s: int((s == LABEL_SUPPORTED).sum())),
        combined=("interpretation_label", lambda s: int((s == LABEL_COMBINED).sum())),
        insufficient=("interpretation_label", lambda s: int((s == LABEL_INSUFFICIENT).sum())),
    ).reset_index()
    body.append(H.table(lvl))
    cols = ["task_id", "track_id", "unit", "interpretation_label", "n_measurements", "n_formulations", "n_source_groups", "reason"]
    body.append(H.table(elig[[c for c in cols if c in elig.columns]]))

    body.append("<h2 id='second'>Secondary dataset overlap audit</h2>")
    body.append(
        f"<p>Reserved dataset(s): {H.esc(', '.join(sorted(res)))}. Copies matched against the backbone: <b>{overlap_meta['copies_with_backbone']}</b>; "
        f"shared source DOIs: <b>{H.esc(overlap_meta['shared_source_dois'] or 'none')}</b>. Name-level independence is not assumed: "
        "the copy rule was run across datasets, and shared formulation identities are counted below.</p>"
    )
    body.append(H.table(overlap))
    html_path = out / "coverage.html"
    html_path.write_text(H.page("Coverage and confounding audit", "".join(body), f"benchmark {cfg['benchmark_version']} — built from inputs and provenance only, before any model was fitted"))
    for p in [html_path, out / "eligibility.csv", *sorted(tab.glob("*.csv"))]:
        man.add_output(p)
    man.write(out / "runs")
