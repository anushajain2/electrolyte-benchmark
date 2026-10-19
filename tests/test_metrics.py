import numpy as np
import pandas as pd
import pytest

from elbench.evaluate import PredictionIDError, check_prediction_ids
from elbench.metrics import group_table, paired_bootstrap, row_errors, summarize


def _pred(ids, p, status=None):
    return pd.DataFrame({"measurement_id": ids, "prediction_mS_cm": p, "status": status or ["ok"] * len(ids)})


def test_log_errors_in_mS_cm_units():
    truth = pd.Series([10.0, 1.0], index=["a", "b"])
    e = row_errors(_pred(["a", "b"], [20.0, 1.0]), truth)
    assert e.abs_log_err.tolist() == pytest.approx([np.log10(2), 0.0])
    assert e.abs_err_mS_cm.tolist() == pytest.approx([10.0, 0.0])


def test_invalid_predictions_counted_not_dropped():
    truth = pd.Series([10.0, 1.0, 5.0], index=["a", "b", "c"])
    e = row_errors(_pred(["a", "b", "c"], [10.0, -0.5, 5.0]), truth)
    assert e.valid.tolist() == [True, False, True]
    s = summarize(e, group_table(e, pd.Series(["g1", "g1", "g2"])))
    assert s["invalid_rate"] == pytest.approx(1 / 3)
    assert s["n_rows"] == 3


def test_group_weighting_equal_across_groups():
    truth = pd.Series(1.0, index=list("abcde"))
    # group A: four perfect rows; group B: one row off by a factor of 10
    e = row_errors(_pred(list("abcde"), [1, 1, 1, 1, 10.0]), truth)
    gt = group_table(e, pd.Series(["A"] * 4 + ["B"]))
    s = summarize(e, gt)
    assert s["group_mean_mae_log10"] == pytest.approx(0.5)  # (0 + 1) / 2
    assert s["sample_mae_log10"] == pytest.approx(0.2)  # 1 / 5
    assert s["worst_quartile_mae_log10"] == pytest.approx(1.0)


def test_paired_bootstrap_uses_identical_resamples():
    G = pd.DataFrame({"m1": [0.1, 0.2, 0.3, 0.4], "m2": [0.1, 0.2, 0.3, 0.4], "m3": [0.2, 0.3, 0.4, 0.5]}, index=list("wxyz"))
    b = paired_bootstrap(G, 500, 0, reference="m1").set_index("model")
    # identical models: every paired difference is exactly zero
    assert b.loc["m2", "diff_ci_low"] == 0 and b.loc["m2", "diff_ci_high"] == 0
    # constant offset: the paired interval collapses to the offset
    assert b.loc["m3", "diff_ci_low"] == pytest.approx(0.1) and b.loc["m3", "diff_ci_high"] == pytest.approx(0.1)
    # unpaired spread is wide by comparison
    assert b.loc["m1", "ci_high"] - b.loc["m1", "ci_low"] > 0.1


def test_bootstrap_resamples_groups_not_rows():
    G = pd.DataFrame({"m": [0.0, 1.0]}, index=["g1", "g2"])
    b = paired_bootstrap(G, 2000, 1)
    assert b.n_groups.iat[0] == 2
    assert b.ci_low.iat[0] == 0.0 and b.ci_high.iat[0] == 1.0


@pytest.mark.parametrize(
    "ids",
    [["a", "b", "b"], ["a"], ["a", "b", "zzz"]],
    ids=["duplicate", "missing", "unexpected"],
)
def test_evaluator_rejects_bad_prediction_ids(ids):
    pred = pd.DataFrame({"measurement_id": ids, "seed": 0, "prediction_mS_cm": 1.0, "status": "ok"})
    with pytest.raises(PredictionIDError):
        check_prediction_ids(pred, np.array(["a", "b"]), "test")


def test_evaluator_accepts_any_row_order():
    pred = pd.DataFrame({"measurement_id": ["b", "a"], "seed": 0, "prediction_mS_cm": 1.0, "status": "ok"})
    check_prediction_ids(pred, np.array(["a", "b"]), "test")
