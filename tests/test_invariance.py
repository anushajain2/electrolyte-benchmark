"""Reordering rows or mixture components must preserve IDs and predictions."""
import numpy as np
import pandas as pd
import pytest

from elbench.config import load_config
from elbench.curate import curate, load_processed
from elbench.features import track_features
from elbench.ingest import ingest
from elbench.models import make_model

from .conftest import write_project


def test_component_column_order_does_not_change_identities(project, tmp_path):
    meas1, comps1 = load_processed(project)
    cfg_path = write_project(tmp_path)
    raw = tmp_path / "local_raw" / "calisol23" / "CALiSol-23 Dataset.csv"
    df = pd.read_csv(raw)
    meta_cols = list(df.columns[:7])
    solv = list(df.columns[7:])
    df[meta_cols + solv[::-1]].to_csv(raw, index=False)  # reverse the mixture component columns
    cfg = load_config(cfg_path)
    ingest(cfg, local_dir=str(tmp_path / "local_raw"))
    curate(cfg)
    meas2, comps2 = load_processed(cfg)
    cols = ["measurement_id", "formulation_id", "chemistry_system_id", "solvent_recipe_id", "split_unit_id", "inclusion_status"]
    pd.testing.assert_frame_equal(meas1[cols].reset_index(drop=True), meas2[cols].reset_index(drop=True))
    desc = pd.read_parquet(project.p("cache") / "descriptors.parquet")
    X1 = track_features(project, meas1, comps1, desc, "molal_w")
    X2 = track_features(cfg, meas2, comps2, desc, "molal_w")
    pd.testing.assert_frame_equal(X1, X2.loc[X1.index], rtol=1e-12)


@pytest.mark.parametrize("model,params", [("ridge", {"alpha": 1.0}), ("knn", {"n_neighbors": 5}), ("median", {}), ("extra_trees", {"n_estimators": 20}), ("hist_gb", {"max_iter": 30})])
def test_row_order_does_not_change_predictions(project, model, params):
    meas, comps = load_processed(project)
    desc = pd.read_parquet(project.p("cache") / "descriptors.parquet")
    X = track_features(project, meas, comps, desc, "molal_w")
    meta = meas.set_index("measurement_id").loc[X.index]
    y = meta.conductivity_mS_cm.to_numpy()
    n = len(X) // 2
    rng = np.random.default_rng(0)
    p1 = make_model(model, 0, params).fit(X.iloc[:n], y[:n], meta.iloc[:n]).predict(X.iloc[n:], meta.iloc[n:]).set_index("measurement_id")
    perm_tr, perm_te = rng.permutation(n), n + rng.permutation(len(X) - n)
    p2 = make_model(model, 0, params).fit(X.iloc[perm_tr], y[perm_tr], meta.iloc[perm_tr]).predict(X.iloc[perm_te], meta.iloc[perm_te]).set_index("measurement_id")
    np.testing.assert_allclose(np.log10(p1.prediction_mS_cm), np.log10(p2.loc[p1.index].prediction_mS_cm), atol=1e-6)
