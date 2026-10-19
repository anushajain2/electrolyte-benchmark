"""v0.3 corrections: exact boundaries, aligned paired comparisons, strict
verification, data-sensitive cache keys and deterministic k-NN ties."""
import json

import numpy as np
import pandas as pd
import pytest

from elbench.models import make_model
from elbench.report import compare_tables
from elbench.splits import _boundary_fold, load_folds, load_manifest
from elbench.submission import compare_on_common_cases
from elbench.train import data_hash


def _frame(temps, concs=None, form="F1"):
    n = len(temps)
    return pd.DataFrame(
        {
            "measurement_id": [f"m{i}" for i in range(n)],
            "temperature_K": temps,
            "concentration": concs if concs is not None else [1.0] * n,
            "formulation_id": [form] * n,
            "solvent_recipe_id": ["R1"] * n,
            "duplicate_group_id": [f"D{i}" for i in range(n)],
        }
    )


def test_exact_boundary_keeps_declared_endpoints():
    # 313.15 must be training for "T <= 313.15" and 323.15 test for "T >= 323.15"
    df = _frame([293.15, 303.15, 313.15, 323.15])
    exact, _ = _boundary_fold(df, "temperature_K", "formulation_id", 313.15, 323.15, 3, True, "temp_high/all", comparison="exact")
    assert set(exact.train) == {"m0", "m1", "m2"} and set(exact.test) == {"m3"}
    # v0.1/v0.2 behaviour: 313.15 rounds to 313.2, drops out of training, and the
    # formulation loses eligibility (only two training temperatures remain)
    legacy, eligible = _boundary_fold(df, "temperature_K", "formulation_id", 313.15, 323.15, 3, True, "temp_high/all", comparison="rounded_0.1")
    assert legacy is None and len(eligible) == 0


def test_exact_boundary_low_temperature_and_concentration():
    low = _frame([253.15, 263.15, 273.15, 283.15])
    f, _ = _boundary_fold(low, "temperature_K", "formulation_id", 263.15, 253.15, 3, False, "temp_low/all", comparison="exact")
    assert set(f.test) == {"m0"}  # 253.15 belongs to the "T <= 253.15" test side
    conc = _frame([298.15] * 5, concs=[0.5, 0.8, 1.2, 1.24, 1.5])
    conc["solvent_recipe_id"] = "R1"
    f, _ = _boundary_fold(conc, "concentration", "solvent_recipe_id", 1.2, 1.5, 3, True, "conc_high/all", comparison="exact")
    assert set(f.train) == {"m0", "m1", "m2"}  # 1.24 is inside the declared buffer
    assert set(f.test) == {"m4"}


def test_project_boundaries_and_leakage_record(project):
    folds = load_folds(project)
    leak = json.loads((project.p("splits") / "leakage_checks.json").read_text())
    for task in ["temp_high", "temp_low", "conc_high"]:
        keys = [k for k in leak if k.startswith(f"{task}/") and "::" not in k.split("/", 2)[2]]
        assert keys, task
        assert all(leak[k].get("declared_boundaries_applied") for k in keys)
    assert len(folds)


class _Cfg:
    def __init__(self, root, data):
        self._root, self.data = root, data

    def p(self, key):
        return self._root

    def __getitem__(self, k):
        return self.data[k]


def _rows(model, cases, errs, valid=None):
    return pd.DataFrame(
        {
            "task_id": "t",
            "track_id": "k",
            "model": model,
            "fold_id": [c[0] for c in cases],
            "measurement_id": [c[1] for c in cases],
            "group": "g1",
            "abs_log_err": errs,
            "valid": valid if valid is not None else [True] * len(cases),
        }
    )


def test_paired_comparison_uses_identical_cases(tmp_path):
    # Every model has zero error on the one case all of them predict validly;
    # baseline A also has an extra case with error 1.0 that the submission did not
    # predict validly. A paired comparison must report zero differences.
    (tmp_path / "tables").mkdir()
    cases = [("f1", "m1"), ("f1", "m2")]
    rows = pd.concat([_rows("A", cases, [0.0, 1.0]), _rows("B", cases, [0.0, 0.5])])
    rows.to_parquet(tmp_path / "tables" / "row_errors.parquet")
    sub = _rows("submission", cases, [0.0, np.nan], valid=[True, False])
    cfg = _Cfg(tmp_path, {"evaluation": {"n_bootstrap": 50, "bootstrap_seed": 0}})
    boot, counts = compare_on_common_cases(cfg, "t", "k", sub, ("A", "B"))
    assert (counts.common_cases == 1).all() and (counts.cases_used == 1).all()
    assert np.allclose(boot.diff_vs_reference, 0.0)


def test_same_measurement_in_two_folds_is_two_cases(tmp_path):
    (tmp_path / "tables").mkdir()
    cases = [("f1", "m1"), ("f2", "m1")]
    _rows("A", cases, [0.0, 0.2]).to_parquet(tmp_path / "tables" / "row_errors.parquet")
    sub = _rows("submission", cases, [0.0, 0.2])
    cfg = _Cfg(tmp_path, {"evaluation": {"n_bootstrap": 50, "bootstrap_seed": 0}})
    _, counts = compare_on_common_cases(cfg, "t", "k", sub, ("A",))
    assert (counts.common_cases == 2).all()


def test_incomplete_baseline_is_left_out_of_comparison(tmp_path):
    (tmp_path / "tables").mkdir()
    cases = [("f1", "m1"), ("f1", "m2")]
    rows = pd.concat([_rows("A", cases, [0.1, 0.1]), _rows("B", cases, [0.0, np.nan], valid=[True, False])])
    rows.to_parquet(tmp_path / "tables" / "row_errors.parquet")
    sub = _rows("submission", cases, [0.1, 0.1])
    cfg = _Cfg(tmp_path, {"evaluation": {"n_bootstrap": 50, "bootstrap_seed": 0}})
    boot, counts = compare_on_common_cases(cfg, "t", "k", sub, ("A", "B"))
    assert "B" not in set(boot.model) and counts.excluded_incomplete_baselines.iat[0] == "B"


def test_verification_rejects_missing_values_counts_and_duplicates():
    a = pd.DataFrame({"k": [1, 2, 3], "v": [0.1, np.nan, 0.3], "n": [5, 5, 5]})
    assert compare_tables(a, a.copy(), ["k"], ["v"], ["n"], 0.005, "t")[0]
    b = a.copy()
    b.loc[0, "v"] = np.nan  # a metric disappears
    assert not compare_tables(a, b, ["k"], ["v"], ["n"], 0.005, "t")[0]
    b = a.copy()
    b.loc[0, "v"] = np.inf
    assert not compare_tables(a, b, ["k"], ["v"], ["n"], 0.005, "t")[0]
    b = a.copy()
    b.loc[2, "n"] = 6
    assert not compare_tables(a, b, ["k"], ["v"], ["n"], 0.005, "t")[0]
    assert not compare_tables(a, pd.concat([a, a.iloc[:1]]), ["k"], ["v"], ["n"], 0.005, "t")[0]


def test_cache_key_changes_when_a_label_changes():
    X = pd.DataFrame({"a": [1.0, 2.0]}, index=["m1", "m2"])
    meta = pd.DataFrame({"conductivity_mS_cm": [1.0, 2.0]}, index=["m1", "m2"])
    h1 = data_hash(X, meta)
    meta2 = meta.copy()
    meta2.loc["m2", "conductivity_mS_cm"] = 2.5
    X2 = X.copy()
    X2.loc["m1", "a"] = 1.5
    assert h1 == data_hash(X.copy(), meta.copy())
    assert h1 != data_hash(X, meta2) and h1 != data_hash(X2, meta)


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_knn_ties_are_resolved_independently_of_row_order(weights):
    rng = np.random.default_rng(0)
    # four replicates at the origin and four equidistant neighbours: exact ties
    pts = np.r_[np.zeros((4, 2)), [[1, 0], [-1, 0], [0, 1], [0, -1]], rng.normal(size=(40, 2))]
    X = pd.DataFrame(pts, columns=["a", "b"], index=[f"m{i:03d}" for i in range(len(pts))])
    y = np.r_[[1, 2, 3, 4], [5, 6, 7, 8], rng.uniform(1, 10, 40)]
    Xt = pd.DataFrame([[0.0, 0.0], [0.5, 0.0], [0.2, 0.2]], columns=["a", "b"], index=["t0", "t1", "t2"])
    preds = []
    for _ in range(15):
        p = rng.permutation(len(X))
        m = make_model("knn", 0, {"n_neighbors": 3, "weights": weights, "tie_policy": "deterministic"}).fit(X.iloc[p], y[p], None)
        preds.append(m.predict(Xt.iloc[rng.permutation(3)], None).set_index("measurement_id").loc[["t0", "t1", "t2"]].prediction_mS_cm.to_numpy())
    assert np.ptp(np.array(preds), axis=0).max() == 0.0
