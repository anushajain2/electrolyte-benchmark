"""v0.2 additions: campaign linking, author groups, local support."""
import json

import numpy as np
import pandas as pd

from elbench.curate import load_processed
from elbench.splits import common_support, load_folds


def test_digitized_republication_is_linked_as_one_campaign(project):
    meas, _ = load_processed(project)
    g = meas.groupby("source_doi").source_group_id.first()
    assert g["10.1/srcA"] == g["10.1/srcI"]
    camp = pd.read_csv(project.p("processed") / "campaign_links.csv")
    row = camp[(camp.source_doi_a == "10.1/srcA") & (camp.source_doi_b == "10.1/srcI")].iloc[0]
    assert row.linked and row.frac_agree >= 0.95  # one deliberate 5% repeated reading in srcA
    # srcC/srcD differ by ~8%: comparable but not a campaign
    cd = camp[(camp.source_doi_a == "10.1/srcC") & (camp.source_doi_b == "10.1/srcD")]
    assert len(cd) == 0 or not cd.linked.iat[0]


def test_campaign_near_matches_are_grouped_as_suspected_copies(project):
    meas, _ = load_processed(project)
    links = pd.read_csv(project.p("processed") / "copy_links.csv")
    near = links[links.status == "campaign_near_match"]
    assert len(near) >= 60
    idx = meas.set_index("measurement_id")
    for a, b in zip(near.measurement_id_a, near.measurement_id_b):
        assert idx.at[a, "duplicate_group_id"] == idx.at[b, "duplicate_group_id"]
        assert idx.at[a, "split_unit_id"] == idx.at[b, "split_unit_id"]


def test_author_group_holdout_never_leaks(project):
    folds = load_folds(project)
    a = folds[folds.task_id == "author_transfer"]
    assert len(a) and a.fold_id.str.startswith("authors/").all()
    leak = json.loads((project.p("splits") / "leakage_checks.json").read_text())
    keys = [k for k in leak if k.startswith("author_transfer/")]
    assert keys and all(v.get("no_author_group_id_crosses") for k, v in leak.items() if k in keys)


def test_local_support_rejects_unmeasured_interior_combinations():
    solv = pd.DataFrame({"EC": [0.5] * 4, "DMC": [0.5] * 4}, index=["F1", "F2", "F3", "F4"])
    base = {"chemistry_system_id": "LiPF6|DMC+EC"}
    # training covers cold+dilute and warm+concentrated only (same solvent recipe)
    train = pd.DataFrame([{**base, "formulation_id": "F1", "temperature_K": 240.0, "concentration": 0.5}, {**base, "formulation_id": "F2", "temperature_K": 330.0, "concentration": 2.0}])
    test = pd.DataFrame([{**base, "formulation_id": "F3", "temperature_K": 240.0, "concentration": 2.0}, {**base, "formulation_id": "F4", "temperature_K": 245.0, "concentration": 0.6}])
    marginal = common_support(train, test, solv, {"mode": "marginal", "ratio_tolerance": 0.1})
    local = common_support(train, test, solv, {"mode": "local", "ratio_tolerance": 0.1, "temperature_window_K": 10.0, "concentration_window": 0.25})
    assert list(marginal) == ["in_support", "in_support"]
    assert list(local) == ["no_training_point_nearby_in_T_and_c", "in_support"]
