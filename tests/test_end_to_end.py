"""One fresh end-to-end run recreates the reference report tables."""
import json

import pandas as pd

from elbench.config import load_config
from elbench.curate import curate
from elbench.evaluate import evaluate
from elbench.ingest import ingest
from elbench.investigate import investigate
from elbench.overview import overview
from elbench.report import report, verify_reproduction
from elbench.splits import make_splits
from elbench.audit import audit
from elbench.train import train

from .conftest import write_project


def _run(root):
    cfg = load_config(write_project(root))
    ingest(cfg, local_dir=str(root / "local_raw"))
    curate(cfg)
    audit(cfg)
    make_splits(cfg)
    train(cfg, n_jobs=2)
    evaluate(cfg)
    investigate(cfg, n_jobs=2)
    report(cfg)
    return cfg


def test_fresh_run_reproduces_reference(tmp_path):
    a = _run(tmp_path / "a")
    verify_reproduction(a, write_reference=True)
    s = pd.read_csv(a.p("results") / "tables" / "summary.csv")
    # every frozen task produced scores, and the trivial baseline is beaten on interpolation
    assert {"interp_formulation", "source_transfer", "temp_high", "temp_low"} <= set(s.task_id)
    interp = s[(s.task_id == "interp_formulation") & (s.track_id == "molal_w") & (s.subset == "all")].set_index("model")
    assert interp.at["extra_trees", "group_mean_mae_log10"] < interp.at["median", "group_mean_mae_log10"]
    assert (a.p("reports") / "report.html").stat().st_size > 10_000
    assert not (a.p("results") / "reserved_exposure_log.csv").exists()
    # v0.2 robustness outputs
    for rel in ["tables/paired_group_differences.csv", "tables/leave_one_group_out.csv", "investigation/support_sensitivity_summary.csv",
                "investigation/condition_logo.csv", "investigation/familiarity_logo.csv", "investigation/ridge_audit_contributions.csv"]:
        assert (a.p("results") / rel).exists(), rel
    for rel in ["investigation/label_evaluation_filtering.csv", "investigation/label_training_cleaning.csv", "investigation/selective_risk_coverage.csv", "investigation/selective_dev_threshold.csv"]:
        assert (a.p("results") / rel).exists(), rel
    lf = pd.read_csv(a.p("results") / "investigation" / "label_evaluation_filtering.csv")
    assert (lf.coverage <= 1).all() and (lf.cases_retained <= lf.cases_all).all()
    tc = pd.read_csv(a.p("results") / "investigation" / "label_training_cleaning.csv")
    assert len(tc) and {"trained_on_all", "trained_on_cleaned"} <= set(tc.columns)
    rc = pd.read_csv(a.p("results") / "investigation" / "selective_risk_coverage.csv")
    assert {"condition_distance", "random", "oracle"} <= set(rc.signal)
    cl = pd.read_csv(a.p("results") / "investigation" / "condition_logo.csv")
    assert set(cl.left_out_group) > {"(none)"}
    ab = pd.read_csv(a.p("results") / "investigation" / "temperature_ablation.csv")
    assert {"all features", "T only (no 1/T)", "1/T only (no T)", "no temperature"} == set(ab.variant)
    assert ab.group_mean_mae_log10.notna().all()

    # the public overview page is built from aggregate tables only: no measurement ids or raw values
    page = overview(a).read_text()
    meas_ids = pd.read_parquet(a.p("processed") / "measurements.parquet").measurement_id
    assert not any(m in page for m in meas_ids)
    assert (a.root / "docs" / "settings_overview.png").stat().st_size > 10_000

    # cached re-run of train is a no-op that keeps results identical
    before = (a.p("results") / "tables" / "summary.csv").read_bytes()
    train(a, n_jobs=2)
    evaluate(a)
    assert (a.p("results") / "tables" / "summary.csv").read_bytes() == before

    # a completely fresh project reproduces the reference tables
    b = _run(tmp_path / "b")
    ref = a.p("reference_tables")
    b_ref = b.p("reference_tables")
    b_ref.mkdir(parents=True, exist_ok=True)
    for p in ref.glob("*.csv"):
        (b_ref / p.name).write_bytes(p.read_bytes())
    assert verify_reproduction(b)
    assert json.loads((b.p("reports") / "verify.json").read_text())["ok"]


def test_confirmation_run_logs_reserved_exposure(tmp_path):
    cfg = _run(tmp_path / "c")
    train(cfg, tasks=["external_rahmanian"], n_jobs=2, include_reserved=True)
    tables = evaluate(cfg, include_reserved=True)
    assert "external_rahmanian" in set(tables["summary"].task_id)
    log = pd.read_csv(cfg.p("results") / "reserved_exposure_log.csv")
    assert {"train", "evaluate"} <= set(log.command)
