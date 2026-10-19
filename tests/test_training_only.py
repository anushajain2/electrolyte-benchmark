"""A test-only extreme value must not change anything that is fitted."""
import numpy as np
import pandas as pd
import pytest

from elbench.curate import load_processed
from elbench.features import track_features
from elbench.models import make_model
from elbench.splits import load_manifest
from elbench.train import _run_job


@pytest.fixture(scope="module")
def setup(project):
    meas, comps = load_processed(project)
    desc = pd.read_parquet(project.p("cache") / "descriptors.parquet")
    X = track_features(project, meas, comps, desc, "molal_w")
    meta = meas.set_index("measurement_id").loc[X.index]
    man = load_manifest(project, "interp_formulation", "molal_w")
    fold = "interp/0"
    sub = man[(man.fold_id == fold) | man.fold_id.str.startswith(fold + "::")]
    return project, X, meta, sub, fold


@pytest.mark.parametrize("model", ["knn", "ridge", "extra_trees", "hist_gb"])
def test_extreme_test_value_does_not_change_fit(setup, model, tmp_path):
    cfg, X, meta, man, fold = setup
    outer = man[man.fold_id == fold]
    te = outer.measurement_id[outer.role == "test"].to_numpy()
    victim, others = te[0], te[1:]
    X2 = X.copy()
    X2.loc[victim, "T_K"] = 1e9
    X2.loc[victim, "solv_mean_mw"] = -1e9
    meta2 = meta.copy()
    meta2.loc[victim, "conductivity_mS_cm"] = 1e12  # an absurd test label must not matter either
    job = {"run_id": "t", "task_id": "interp_formulation", "track_id": "molal_w", "fold_id": fold, "model": model, "score_group": "source_group_id", "cache_key": "x"}
    r1 = _run_job(job, X, meta, man, cfg["models"], tmp_path / "a", tmp_path / "ma")
    r2 = _run_job(job, X2, meta2, man, cfg["models"], tmp_path / "b", tmp_path / "mb")
    assert r1["chosen_config"] == r2["chosen_config"]
    assert [d["dev_scores"] for d in r1["dev_scores"]] == [d["dev_scores"] for d in r2["dev_scores"]]
    p1 = pd.read_parquet(tmp_path / "a" / f"{model}.parquet").set_index(["seed", "measurement_id"]).prediction_mS_cm
    p2 = pd.read_parquet(tmp_path / "b" / f"{model}.parquet").set_index(["seed", "measurement_id"]).prediction_mS_cm
    keep = p1.index.get_level_values(1).isin(others)
    np.testing.assert_allclose(p1[keep].to_numpy(), p2[keep].to_numpy(), rtol=1e-12)


def test_scaler_statistics_come_from_training_rows_only(setup):
    cfg, X, meta, man, fold = setup
    outer = man[man.fold_id == fold]
    tr = outer.measurement_id[outer.role == "train"].to_numpy()
    m = make_model("ridge", 0, {"alpha": 1.0}).fit(X.loc[tr], meta.loc[tr, "conductivity_mS_cm"].to_numpy(), meta.loc[tr])
    scaler = m.model.steps[0][1]
    np.testing.assert_allclose(scaler.mean_, X.loc[tr].to_numpy().mean(axis=0))
