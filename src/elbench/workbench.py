"""Local evaluation workbench (Streamlit).

    uv sync --extra workbench
    uv run elbench workbench            # after `uv run elbench all`

Runs on your machine against the locally rebuilt benchmark. It is not meant to
be hosted publicly: scoring uploads needs the CALiSol labels, which are not
redistributed. The question it supports:

    Given the measurements I already have, which modeling approach deserves
    further validation?

It starts with the best-supported setting, cold-temperature prediction for
formulations that already have warm-side measurements.
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SETTINGS = {
    "temp_low": {
        "title": "Cold-temperature prediction (formulations with warm measurements)",
        "requires": "At least three measured temperatures at or above 263.15 K for the same formulation. Predictions are for 253.15 K and below.",
        "question": "You have warm-side conductivity for a formulation and need values below the measured range.",
    },
    "temp_high": {
        "title": "Warm-temperature prediction (formulations with cold and room-temperature measurements)",
        "requires": "At least three measured temperatures at or below 313.15 K for the same formulation. Predictions are for 323.15 K and above.",
        "question": "You have conductivity up to about 40 °C and need values above it.",
    },
    "interp_formulation": {
        "title": "New formulation in a familiar chemistry",
        "requires": "The salt and solvents (the chemistry system) appear in training; only the ratios or concentration are new.",
        "question": "You are screening new ratios or concentrations of a known salt/solvent system.",
    },
    "source_transfer": {
        "title": "Measurements from a new laboratory or paper",
        "requires": "Chemistry and conditions inside the common support of other sources (case study only: 4 testable source groups).",
        "question": "You want to know how well models trained on published data match a new lab's measurements.",
    },
}


def main(config_path: str) -> None:  # pragma: no cover - UI
    import streamlit as st

    from elbench.config import load_config
    from elbench.curate import load_processed
    from elbench.evaluate import MAIN_MODELS, PredictionIDError
    from elbench.splits import load_folds, load_manifest
    from elbench.submission import score_predictions

    st.set_page_config(page_title="Electrolyte benchmark workbench", layout="wide")
    cfg = load_config(config_path)
    res, inv = cfg.p("results"), cfg.p("results") / "investigation"
    track = cfg["investigation"]["primary_track"]

    @st.cache_data
    def tables():
        t = {n: pd.read_csv(res / "tables" / f"{n}.csv") for n in ["summary", "group_metrics", "bootstrap"]}
        t["elig"] = pd.read_csv(cfg.p("splits") / "eligibility.csv")
        for n in ["failure_records", "label_evaluation_filtering", "label_cases_removed", "selective_risk_coverage", "selective_dev_threshold", "condition_logo"]:
            p = inv / f"{n}.csv"
            t[n] = pd.read_csv(p) if p.exists() else pd.DataFrame()
        return t

    @st.cache_data
    def data():
        meas, comps = load_processed(cfg)
        meta = pd.read_csv(cfg.root / "data/mappings/source_metadata.csv") if (cfg.root / "data/mappings/source_metadata.csv").exists() else pd.DataFrame()
        pres = pd.read_csv(cfg.root / "data/mappings/source_presentation.csv") if (cfg.root / "data/mappings/source_presentation.csv").exists() else pd.DataFrame()
        return meas, meta, pres

    T = tables()
    meas, srcmeta, pres = data()
    st.sidebar.title("Workbench")
    st.sidebar.caption(f"benchmark {cfg['benchmark_version']} · config {cfg.sha256[:12]}")
    task = st.sidebar.selectbox("Your situation", list(SETTINGS), format_func=lambda k: SETTINGS[k]["title"])
    page = st.sidebar.radio("Step", ["1 · What can be established", "2 · Published baselines", "3 · Evaluate your predictions", "4 · Inspect a result", "5 · Export"])
    S = SETTINGS[task]
    elig = T["elig"][(T["elig"].task_id == task) & (T["elig"].track_id == track)]
    summ = T["summary"][(T["summary"].task_id == task) & (T["summary"].track_id == track) & (T["summary"].subset == "all")]

    if page.startswith("1"):
        st.header(S["title"])
        st.markdown(f"**Situation.** {S['question']}\n\n**Data requirement.** {S['requires']}")
        lvl = elig.reporting_level.iat[0] if len(elig) else "n/a"
        units = int(elig.n_headline_units.iat[0]) if len(elig) else 0
        scored = int(summ.n_groups.max()) if len(summ) else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Groups scored", scored)
        c2.metric("Groups meeting the headline rule", units)
        c3.metric("Reporting level", lvl)
        st.markdown(
            "**What this benchmark can establish here.** Errors are averaged within source groups (provenance-based groupings of papers), "
            "then across groups; intervals resample those groups and are conditional on that grouping, not population-level uncertainty across laboratories."
        )
        if task == "temp_low":
            st.info(
                "Across all evaluated groups, temperature-aware baselines (a per-formulation Arrhenius fit and ridge with a 1/T term) outperform tree ensembles below the coldest training temperature. "
                "The Arrhenius baseline needs at least three warm-side measurements of the same formulation; without them this setting does not apply. "
                "Values below ~0.2 mS/cm read off linear-axis figures carry large digitization error (see step 4)."
            )
        st.subheader("Which approach deserves further validation?")
        ranked = summ[summ.complete.astype(bool) & summ.model.isin(MAIN_MODELS + ["arrhenius_local"])].sort_values("group_mean_mae_log10") if "complete" in summ else summ
        show(st, ranked[["model", "group_mean_mae_log10", "group_median_mae_log10", "worst_quartile_mae_log10", "frac_within_factor2", "n_rows", "n_groups"]])
        if len(ranked):
            st.success(recommendation(ranked))

    elif page.startswith("2"):
        st.header("Published baselines")
        b = T["bootstrap"][(T["bootstrap"].task_id == task) & (T["bootstrap"].track_id == track) & (T["bootstrap"].subset == "all")]
        comp = "with_local" if (b.comparison == "with_local").any() else "main"
        st.subheader("Ranking with paired intervals (difference against the reference model)")
        show(st, b[b.comparison == comp][["model", "n_groups", "n_common_rows", "group_mean_mae_log10", "ci_low", "ci_high", "diff_vs_reference", "diff_ci_low", "diff_ci_high", "note"]])
        st.subheader("Per source group")
        gm = T["group_metrics"][(T["group_metrics"].task_id == task) & (T["group_metrics"].track_id == track)]
        show(st, gm.pivot_table(index="group_id", columns="model", values="mae_log10").round(3), cfg.root)
        if len(T["condition_logo"]) and task in set(T["condition_logo"].task_id):
            st.subheader("Removing one source group at a time (refit)")
            cl = T["condition_logo"][T["condition_logo"].task_id == task]
            show(st, cl.pivot_table(index="left_out_group", columns="model", values="group_mean_mae_log10").round(3), cfg.root)
        if len(T["selective_risk_coverage"]) and task in set(T["selective_risk_coverage"].task_id):
            st.subheader("Selective prediction (exploratory): error when the least confident predictions are rejected")
            rc = T["selective_risk_coverage"][T["selective_risk_coverage"].task_id == task]
            st.caption("Coverage = share of test cases that still receive a prediction after the least confident are rejected.")
            show(st, rc.pivot_table(index=["model", "signal"], columns="target_coverage", values="group_mean_mae_log10").round(3))
        fr = T["failure_records"]
        if len(fr):
            st.subheader("Traced failure examples")
            st.dataframe(fr, hide_index=True)

    elif page.startswith("3"):
        st.header("Evaluate your predictions on frozen cases")
        folds = load_folds(cfg)
        fs = folds[(folds.task_id == task) & (folds.track_id == track) & ~folds.reserved]
        man = load_manifest(cfg, task, track)
        cases = man[man.fold_id.isin(fs.fold_id) & (man.role == "test")][["fold_id", "measurement_id"]]
        inputs = cases.merge(meas[["measurement_id", "formulation_id", "chemistry_system_id", "solvent_fractions_reported", "concentration", "concentration_basis", "temperature_K"]], on="measurement_id")
        st.markdown(
            "1. Download the case template (inputs only, no labels).  \n"
            "2. Train only on each fold's training rows (`elbench.submission.iter_folds`), tune on its development folds.  \n"
            "3. Upload a CSV with `fold_id, measurement_id, prediction_mS_cm` and optionally `status` (`ok`, `abstain`, or a failure) and `confidence`."
        )
        st.download_button("Download case template", inputs.to_csv(index=False), f"cases_{task}.csv", "text/csv")
        mode = st.radio("Scoring mode", ["full_coverage", "selective"], horizontal=True, help="Full coverage: every case needs a prediction; incomplete submissions get diagnostics only. Selective: intentional abstentions with a confidence column, thresholds fixed on development data.")
        up = st.file_uploader("Predictions CSV", type="csv")
        if up is not None:
            pred = pd.read_csv(up)
            missing_cols = {"fold_id", "measurement_id", "prediction_mS_cm"} - set(pred.columns)
            if missing_cols:
                st.error(f"Missing columns: {sorted(missing_cols)}")
                st.stop()
            checks = unit_check(pred, cases, meas)
            for level, msg in checks:
                getattr(st, level)(msg)
            try:
                out = score_predictions(cfg, task, track, pred, mode=mode)
            except (PredictionIDError, ValueError) as e:
                st.error(f"Rejected: {e}")
                st.stop()
            st.subheader("Diagnostics")
            show(st, out["diagnostics"])
            st.write(out.get("comparison_status", ""))
            st.subheader("Score")
            show(st, out["summary"])
            if "risk_coverage" in out:
                st.subheader("Error versus coverage")
                show(st, out["risk_coverage"])
            if "comparison" in out:
                st.subheader("Paired comparison with eligible baselines (same cases)")
                show(st, out["comparison"])
                show(st, out["comparison_cases"])
            st.session_state["last_eval"] = {"task": task, "mode": mode, "file": up.name, "out": {k: v for k, v in out.items() if isinstance(v, pd.DataFrame)}, "status": out.get("comparison_status", ""), "checks": checks}

    elif page.startswith("4"):
        st.header("Inspect a result")
        gm = T["group_metrics"][(T["group_metrics"].task_id == task) & (T["group_metrics"].track_id == track)]
        groups = sorted(gm.group_id.unique())
        g = st.selectbox("Source group (or held-out unit)", groups)
        sub = meas[meas.source_group_id == g] if g in set(meas.source_group_id) else meas.iloc[0:0]
        if len(sub):
            dois = sorted(sub.source_doi.unique())
            st.subheader("Provenance")
            m = srcmeta[srcmeta.source_doi.isin(dois)] if len(srcmeta) else pd.DataFrame()
            st.dataframe(m if len(m) else pd.DataFrame({"source_doi": dois}), hide_index=True)
            st.write("Author group:", ", ".join(sorted(sub.author_group_id.unique())))
            pr = cfg.p("reports") / "provenance_pairs.csv"
            if pr.exists():
                pp = pd.read_csv(pr)
                pp = pp[pp.source_group_id == g]
                if len(pp):
                    st.write("Why these papers form one group (evidence tier per pair):")
                    st.dataframe(pp, hide_index=True)
            st.subheader("Label-quality flags")
            pr_rows = pres[pres.source_doi.isin(dois)] if len(pres) else pres
            if len(pr_rows):
                st.dataframe(pr_rows, hide_index=True)
            else:
                st.caption("No data-presentation review recorded for these papers (not known to be digitized from linear axes).")
            inc = sub[sub.inclusion_status == "included"]
            st.write(f"{len(inc)} included measurements; {int((inc.conductivity_mS_cm < 0.2).sum())} below 0.2 mS/cm.")
            exc = sub[sub.inclusion_status == "excluded"].exclusion_reason.value_counts().rename_axis("reason").reset_index(name="rows")
            if len(exc):
                st.write("Excluded rows by reason:")
                st.dataframe(exc, hide_index=True)
            else:
                st.caption("No excluded rows from this group.")
        st.subheader("Sensitivity of this setting's scores to the digitization floor")
        ef = T["label_evaluation_filtering"]
        if len(ef):
            st.dataframe(ef[ef.task_id == task].pivot_table(index="model", columns="definition", values="group_mean_mae_retained").round(3))
            st.caption("Predictions fixed; only the evaluated cases change. See the report for the training-cleaning comparison.")

    else:
        st.header("Export an evaluation summary")
        last = st.session_state.get("last_eval")
        md = export_markdown(cfg, task, summ, last)
        st.markdown(md)
        st.download_button("Download (Markdown)", md, f"evaluation_{task}.md", "text/markdown")


def recommendation(ranked: pd.DataFrame, tie_log10: float = 0.05) -> str:
    """Shortlist the lowest-error approaches without overstating small differences.

    The rule is practical, not statistical: every model within ``tie_log10``
    (0.05 log10, about 12%) of the lowest group-mean error is shortlisted."""
    from elbench.labels import model_label

    best = ranked.iloc[0]
    close = ranked[ranked.group_mean_mae_log10 <= best.group_mean_mae_log10 + tie_log10]
    if len(close) > 1:
        names = ", ".join(f"**{model_label(m)}** ({v:.3f})" for m, v in zip(close.model, close.group_mean_mae_log10))
        return (f"Shortlist for further validation: {names} (mean error, log10). They are within {tie_log10} log10 (about 12%) of the lowest error. "
                "This is a practical shortlist rule, not a statistical equivalence test; validate them on your own data.")
    return (f"Shortlist for further validation: **{model_label(best.model)}** (mean error {best.group_mean_mae_log10:.3f} log10, about a factor of "
            f"{10 ** best.group_mean_mae_log10:.2f}). Every other model is more than {tie_log10} log10 worse. This is evidence for validation, not a guarantee.")


def show(st, df: pd.DataFrame, root=None, **kw) -> None:
    """Display a table with readable column, model and source-group names."""
    from elbench.labels import MODEL_LABELS, friendly_columns, source_label

    out = friendly_columns(df.reset_index() if df.index.name or isinstance(df.index, pd.MultiIndex) else df)
    out = out.rename(columns={c: MODEL_LABELS[c] for c in out.columns if c in MODEL_LABELS})
    for c in ("Source group", "group_id", "left_out_group"):
        if c in out and root is not None:
            out[c] = out[c].map(lambda g: source_label(g, root) if str(g).startswith("doi:") else g)
    st.dataframe(out, hide_index=True, **kw)


def unit_check(pred: pd.DataFrame, cases: pd.DataFrame, meas: pd.DataFrame) -> list[tuple[str, str]]:
    """Catch likely unit mistakes before scoring (S/cm or S/m instead of mS/cm)."""
    out = []
    p = pd.to_numeric(pred.prediction_mS_cm, errors="coerce")
    ok = p[(p > 0) & np.isfinite(p)]
    if not len(ok):
        return [("error", "No positive finite predictions.")]
    typical = meas[meas.measurement_id.isin(cases.measurement_id)].conductivity_mS_cm
    ratio = float(np.median(ok)) / float(np.median(typical[typical > 0]))
    if ratio < 0.003:
        out.append(("warning", f"Predictions are ~{1 / ratio:.0f}x smaller than typical values for these cases: are they in S/cm? Expected mS/cm."))
    elif ratio < 0.3:
        out.append(("warning", f"Predictions are ~{1 / ratio:.1f}x smaller than typical: are they in S/m? Expected mS/cm."))
    elif ratio > 300:
        out.append(("warning", f"Predictions are ~{ratio:.0f}x larger than typical: are they in uS/cm? Expected mS/cm."))
    else:
        out.append(("success", "Prediction magnitudes are consistent with mS/cm."))
    n_bad = int(len(p) - len(ok))
    if n_bad:
        out.append(("warning", f"{n_bad} predictions are missing, nonpositive or non-finite; they count as failures unless marked 'abstain' in selective mode."))
    return out


def md_table(df: pd.DataFrame) -> str:
    def fmt(v):
        return f"{v:.4g}" if isinstance(v, float) else str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def export_markdown(cfg, task: str, summ: pd.DataFrame, last) -> str:
    lines = [
        f"# Evaluation summary: {SETTINGS[task]['title']}",
        "",
        f"- Benchmark version {cfg['benchmark_version']}, config sha256 {cfg.sha256[:16]}, generated {time.strftime('%Y-%m-%d')}",
        f"- Data requirement: {SETTINGS[task]['requires']}",
        "- Errors: MAE of log10(conductivity / mS cm^-1), averaged within source groups then across groups; intervals resample source groups and are conditional on that grouping.",
        "- Reference data apply confirmed corrections only; the digitization floor is a sensitivity assumption, not applied here.",
        "",
        "## Published baselines (complete predictions only)",
        "",
        md_table(summ[summ.complete.astype(bool)][["model", "group_mean_mae_log10", "n_groups", "n_rows"]].sort_values("group_mean_mae_log10")) if len(summ) and "complete" in summ else "(none)",
    ]
    if len(summ) and "complete" in summ and (~summ.complete.astype(bool)).any():
        inc = summ[~summ.complete.astype(bool)]
        lines += ["", "Not ranked (incomplete predictions): " + ", ".join(f"{m} ({r:.0%} invalid)" for m, r in zip(inc.model, inc.invalid_rate))]
    if last and last.get("task") == task:
        lines += ["", f"## Your submission ({last['file']}, mode {last['mode']})", "", f"Status: {last['status']}", ""]
        for name in ["diagnostics", "summary", "risk_coverage", "comparison"]:
            if name in last["out"]:
                lines += [f"### {name}", "", md_table(last["out"][name]), ""]
        lines += ["Checks:", *[f"- {lvl}: {msg}" for lvl, msg in last["checks"]]]
    else:
        lines += ["", "_No submission evaluated for this setting in this session._"]
    lines += ["", "This is research screening support, not battery qualification."]
    return "\n".join(lines)


def launch(config_path: str) -> None:
    """Start Streamlit on this module."""
    import subprocess

    try:
        import streamlit  # noqa: F401
    except ImportError:
        sys.exit("The workbench needs the optional dependency: uv sync --extra workbench")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(Path(__file__)), "--", "--config", str(Path(config_path).resolve())], check=False)


if __name__ == "__main__":  # executed by `streamlit run`
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
