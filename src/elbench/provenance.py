"""Provenance tiers and count breakdowns for the report (descriptive only).

Nothing here changes rows, groups or splits. Every pair of papers that end up
in the same source group is assigned one tier:

  documented_reuse      a reviewed statement (quotation and page) that the two
                        papers report the same measurements
  suspected_reuse       direct numerical evidence between the two papers: >=3
                        near-identical points (R09) or campaign agreement (R12)
  conservative_grouping no direct evidence between this pair; they share a
                        source group only through other papers (connected
                        component), as a leakage safeguard

Grouping into one source group is therefore a conservative leakage choice; it
does not establish that all member papers belong to one experimental campaign.
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd

from .config import Config

TIERS = ["documented_reuse", "suspected_reuse", "conservative_grouping"]


def _pair(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))


def provenance_pairs(cfg: Config) -> pd.DataFrame:
    proc = cfg.p("processed")
    meas = pd.read_parquet(proc / "measurements.parquet")
    if "author_group_id" not in meas:  # tables written before v0.2
        meas["author_group_id"] = meas.source_group_id
    groups = meas.drop_duplicates("source_doi").set_index("source_doi")
    evidence: dict[tuple[str, str], list[str]] = {}
    links = pd.read_csv(proc / "copy_links.csv")
    r09 = links[(links.status == "suspected_reuse_linked") & (links.source_doi_a != links.source_doi_b)]
    for (a, b), g in r09.groupby([r09[["source_doi_a", "source_doi_b"]].min(axis=1), r09[["source_doi_a", "source_doi_b"]].max(axis=1)]):
        evidence.setdefault((a, b), []).append(f"R09: {len(g)} near-identical points")
    camp_path = proc / "campaign_links.csv"
    camp = pd.read_csv(camp_path) if camp_path.exists() else pd.DataFrame()
    for r in camp.itertuples() if len(camp) else []:
        if r.linked:
            evidence.setdefault(_pair(r.source_doi_a, r.source_doi_b), []).append(f"R12: {r.n_agree}/{r.n_comparable} comparable points within 5%")
    doc_path = cfg.root / "data" / "mappings" / "source_reuse_documented.csv"
    documented, notes = set(), {}
    if doc_path.exists():
        d = pd.read_csv(doc_path)
        for r in d.itertuples():
            key = _pair(r.source_doi_a, r.source_doi_b)
            notes[key] = f"{r.status}: {r.evidence}"
            if r.status == "documented":
                documented.add(key)
    rows = []
    for sg, g in groups.groupby("source_group_id"):
        dois = sorted(g.index)
        for a, b in combinations(dois, 2):
            key = (a, b)
            tier = "documented_reuse" if key in documented else ("suspected_reuse" if key in evidence else "conservative_grouping")
            rows.append(
                {
                    "source_group_id": sg,
                    "source_doi_a": a,
                    "source_doi_b": b,
                    "tier": tier,
                    "numerical_evidence": "; ".join(evidence.get(key, [])),
                    "bibliographic_note": notes.get(key, ""),
                    "same_author_group": bool(groups.at[a, "author_group_id"] == groups.at[b, "author_group_id"]),
                }
            )
    return pd.DataFrame(rows, columns=["source_group_id", "source_doi_a", "source_doi_b", "tier", "numerical_evidence", "bibliographic_note", "same_author_group"])


def paper_count_breakdown(cfg: Config) -> pd.DataFrame:
    """How many papers survive each step, per track: eligibility vs linking vs support."""
    meas = pd.read_parquet(cfg.p("processed") / "measurements.parquet")
    if "author_group_id" not in meas:
        meas["author_group_id"] = meas.source_group_id
    reserved = {k for k, v in cfg["sources"].items() if v.get("role") == "reserved_external_evaluation"}
    bb = meas[~meas.dataset_id.isin(reserved)]
    elig = pd.read_csv(cfg.p("splits") / "eligibility.csv")
    rows = [{"track_id": "(all)", "step": "papers in the backbone collection", "count": bb.source_doi.nunique(), "cause": "starting material"}]
    inc = bb[bb.inclusion_status == "included"]
    rows.append({"track_id": "(all)", "step": "papers with >=1 included row", "count": inc.source_doi.nunique(), "cause": "curation exclusions"})
    for track in sorted(inc.track_id.dropna().unique()):
        t = inc[inc.track_id == track]
        rows.append({"track_id": track, "step": "papers in this track", "count": t.source_doi.nunique(), "cause": "concentration and ratio basis (track eligibility)"})
        rows.append({"track_id": track, "step": "source groups", "count": t.source_group_id.nunique(), "cause": "provenance grouping (R09/R12 links)"})
        rows.append({"track_id": track, "step": "author groups", "count": t.author_group_id.nunique(), "cause": "reviewed author grouping (sensitivity)"})
        for task, label in [("source_transfer", "source groups testable in source transfer"), ("author_transfer", "author groups testable in author transfer")]:
            e = elig[(elig.task_id == task) & (elig.track_id == track)]
            if len(e):
                rows.append({"track_id": track, "step": label, "count": int((e.interpretation_label == "supported_comparison").sum()), "cause": "rows inside common support when held out"})
    return pd.DataFrame(rows)
