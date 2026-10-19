"""v0.5: submission scoring policy (full coverage vs selective prediction)."""
import numpy as np
import pandas as pd
import pytest

from elbench.curate import load_processed
from elbench.splits import load_folds, load_manifest
from elbench.submission import risk_coverage, score_predictions

TASK, TRACK = "interp_formulation", "molal_w"


def _perfect(project):
    meas, _ = load_processed(project)
    truth = meas.set_index("measurement_id").conductivity_mS_cm
    man = load_manifest(project, TASK, TRACK)
    folds = load_folds(project)
    outer = set(folds[(folds.task_id == TASK) & (folds.track_id == TRACK)].fold_id)
    t = man[(man.role == "test") & man.fold_id.isin(outer)][["fold_id", "measurement_id"]].copy()
    t["prediction_mS_cm"] = truth.loc[t.measurement_id].to_numpy() * 1.1
    t["status"] = "ok"
    return t.reset_index(drop=True)


def test_complete_submission_is_eligible_for_ranking(project):
    out = score_predictions(project, TASK, TRACK, _perfect(project), compare_to_baselines=False)
    assert out["diagnostics"].coverage.iat[0] == 1.0
    assert "not ranked" not in out.get("comparison_status", "")


@pytest.mark.parametrize("status", ["abstain", "failed"])
def test_incomplete_submission_gets_diagnostics_but_no_ranking(project, status):
    p = _perfect(project)
    p.loc[:4, "status"] = status
    p.loc[:4, "prediction_mS_cm"] = np.nan
    out = score_predictions(project, TASK, TRACK, p)  # compare_to_baselines=True by default
    assert out["comparison_status"].startswith("not ranked")
    assert "comparison" not in out
    d = out["diagnostics"].iloc[0]
    assert d.coverage < 1.0 and (d.abstained if status == "abstain" else d.failed) == 5


def test_selective_mode_separates_abstentions_from_failures(project):
    p = _perfect(project)
    p["confidence"] = np.linspace(1, 0, len(p))
    p.loc[len(p) - 10 :, "status"] = "abstain"
    out = score_predictions(project, TASK, TRACK, p, mode="selective", compare_to_baselines=False)
    assert out["diagnostics"].abstained.iat[0] == 10 and out["diagnostics"].failed.iat[0] == 0
    rc = out["risk_coverage"]
    assert not bool(rc[rc.target_coverage == 1.0].achieved.iat[0])  # cannot reach full coverage after abstaining
    p.loc[0, "status"] = "failed"
    out = score_predictions(project, TASK, TRACK, p, mode="selective")
    assert out["comparison_status"].startswith("not ranked")


def test_selective_mode_requires_confidence(project):
    with pytest.raises(ValueError):
        score_predictions(project, TASK, TRACK, _perfect(project), mode="selective")


def test_risk_coverage_rejects_least_confident_first():
    e = pd.DataFrame({"valid": True, "abs_log_err": [0.1, 0.2, 0.3, 1.0], "confidence": [0.9, 0.8, 0.7, 0.1], "group": "g"})
    rc = risk_coverage(e, levels=(1.0, 0.75, 0.5)).set_index("target_coverage")
    assert rc.at[1.0, "sample_mae_log10"] == pytest.approx(0.4)
    assert rc.at[0.75, "sample_mae_log10"] == pytest.approx(0.2)
    assert rc.at[0.5, "sample_mae_log10"] == pytest.approx(0.15)


def test_workbench_unit_check_flags_common_unit_mistakes():
    from elbench.workbench import unit_check

    meas = pd.DataFrame({"measurement_id": ["a", "b", "c"], "conductivity_mS_cm": [5.0, 8.0, 10.0]})
    cases = meas[["measurement_id"]]
    ok = unit_check(pd.DataFrame({"prediction_mS_cm": [5.0, 7.0, 9.0]}), cases, meas)
    s_cm = unit_check(pd.DataFrame({"prediction_mS_cm": [0.005, 0.007, 0.009]}), cases, meas)
    s_m = unit_check(pd.DataFrame({"prediction_mS_cm": [0.5, 0.7, 0.9]}), cases, meas)
    assert ok[0][0] == "success"
    assert s_cm[0][0] == "warning" and "S/cm" in s_cm[0][1]
    assert s_m[0][0] == "warning" and "S/m" in s_m[0][1]


def test_workbench_markdown_table():
    from elbench.workbench import md_table

    t = md_table(pd.DataFrame({"model": ["ridge"], "mae": [0.12345]}))
    assert t.splitlines()[0] == "| model | mae |" and "0.1235" in t


def test_workbench_recommendation_does_not_separate_near_ties():
    from elbench.workbench import recommendation

    tie = recommendation(pd.DataFrame({"model": ["arrhenius_local", "ridge", "extra_trees"], "group_mean_mae_log10": [0.229, 0.249, 0.536]}))
    assert "arrhenius_local" in tie and "ridge" in tie and "extra_trees" not in tie and "does not separate" in tie
    clear = recommendation(pd.DataFrame({"model": ["extra_trees", "ridge"], "group_mean_mae_log10": [0.05, 0.16]}))
    assert "extra_trees" in clear and "ridge" not in clear
