"""Human-readable display names for public outputs (report, overview, workbench).

Internal identifiers stay unchanged in tables and the API; these labels are
used only where people read results.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

MODEL_LABELS = {
    "extra_trees": "Extra Trees",
    "hist_gb": "Gradient boosting",
    "knn": "k-nearest neighbours",
    "ridge": "Ridge (log target)",
    "ridge_raw": "Ridge, raw target (diagnostic)",
    "median": "Training median",
    "arrhenius_local": "Per-formulation Arrhenius",
    "submission": "Your submission",
}

TASK_LABELS = {
    "interp_formulation": "New formulations in familiar chemistry",
    "source_transfer": "New source (held-out paper group)",
    "author_transfer": "New author group",
    "component_transfer": "New solvent or salt",
    "temp_high": "Warm extrapolation (≥ 323 K)",
    "temp_low": "Cold extrapolation (≤ 253 K)",
    "conc_high": "High-concentration extrapolation",
    "external_rahmanian": "External automated laboratory",
}

TRACK_LABELS = {
    "molal_w": "molality, mass-fraction solvent recipes (primary track)",
    "molar_x": "molarity, mole-fraction recipes",
    "molar_v": "molarity, volume-fraction recipes",
}

SIGNAL_LABELS = {
    "condition_distance": "Distance beyond the measured temperature range",
    "feature_distance": "Distance to nearest training example",
    "ensemble_spread": "Disagreement between trees",
    "random": "Random rejection (reference)",
    "oracle": "Oracle: sorted by actual error (bound only)",
}

METRIC_LABELS = {
    "group_mean_mae_log10": "Mean error (log10, averaged over source groups)",
    "group_median_mae_log10": "Median group error (log10)",
    "worst_quartile_mae_log10": "Worst-quartile group error (log10)",
    "sample_mae_log10": "Mean error over all cases (log10)",
    "frac_within_factor2": "Share within a factor of 2",
    "n_rows": "Test cases",
    "n_groups": "Groups",
    "n_common_rows": "Common cases",
    "ci_low": "95% interval, low",
    "ci_high": "95% interval, high",
    "diff_vs_reference": "Difference vs reference model",
    "diff_ci_low": "Difference, interval low",
    "diff_ci_high": "Difference, interval high",
    "invalid_rate": "Share of invalid predictions",
    "coverage": "Coverage (share of cases that receive a prediction)",
    "target_coverage": "Target coverage",
    "mae_log10": "Mean error (log10)",
}


def model_label(m: str) -> str:
    return MODEL_LABELS.get(m, m)


def task_label(t: str) -> str:
    return TASK_LABELS.get(t, t)


def friendly_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(columns={**METRIC_LABELS, "model": "Model", "task_id": "Setting", "group_id": "Source group", "signal": "Confidence signal", "note": "Note"})
    for col, fn in (("Model", model_label), ("Setting", task_label), ("Confidence signal", lambda s: SIGNAL_LABELS.get(s, s))):
        if col in out:
            out[col] = out[col].map(fn)
    return out


@lru_cache(maxsize=None)
def _source_meta(root: str) -> dict[str, str]:
    """DOI -> 'Ding et al. 2005a' style labels, unique per DOI."""
    p = Path(root) / "data" / "mappings" / "source_metadata.csv"
    if not p.exists():
        return {}
    m = pd.read_csv(p)
    base = {}
    for r in m.itertuples():
        authors = str(r.authors) if isinstance(r.authors, str) else ""
        first = authors.split(";")[0].split(",")[0].strip() or r.source_doi
        n = len([a for a in authors.split(";") if a.strip()])
        year = "" if pd.isna(r.year) else str(int(r.year))
        base[r.source_doi] = f"{first}{' et al.' if n > 2 else (' & ' + authors.split(';')[1].split(',')[0].strip() if n == 2 else '')} {year}".strip()
    out, seen = {}, {}
    for doi in sorted(base):
        seen.setdefault(base[doi], []).append(doi)
    for lab, dois in seen.items():
        for i, doi in enumerate(sorted(dois)):
            out[doi] = lab + ("abcdefghij"[i] if len(dois) > 1 else "")
    return out


def source_label(group_id: str, root: str | Path) -> str:
    """Author/year label for a source group id ('doi:...'); unknown ids pass through."""
    doi = str(group_id).replace("doi:", "")
    return _source_meta(str(root)).get(doi, str(group_id))


def group_label(group_id: str, root: str | Path, n_papers: int = 1) -> str:
    """Source-group label that shows when papers were merged by provenance linking."""
    base = source_label(group_id, root)
    return base if n_papers <= 1 else f"{base} + {n_papers - 1} linked paper{'s' if n_papers > 2 else ''}"
