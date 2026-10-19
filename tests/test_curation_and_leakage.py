"""Identity, duplicate and split-leakage invariants on the fixture project."""
import json

import pandas as pd
import pytest

from elbench.curate import load_processed
from elbench.splits import component_sets, load_folds, load_manifest


@pytest.fixture(scope="module")
def tables(project):
    meas, comps = load_processed(project)
    return project, meas, comps


def test_raw_rows_all_visible(tables):
    cfg, meas, _ = tables
    raw = pd.read_csv(cfg.p("raw") / "calisol23" / "commit-02334f51d581" / "CALiSol-23 Dataset.csv")
    assert (meas.dataset_id == "calisol23").sum() == len(raw)
    assert meas.measurement_id.is_unique


def test_exact_copy_collapsed_and_traceable(tables):
    _, meas, _ = tables
    dup = meas[meas.exclusion_reason == "duplicate_copy_collapsed"]
    assert len(dup) == 1
    assert dup.duplicate_of.iat[0] == "calisol23:0"
    canon = meas.set_index("measurement_id").loc["calisol23:0"]
    assert canon.inclusion_status == "included"
    assert canon.duplicate_group_id == dup.duplicate_group_id.iat[0]


def test_cross_source_copy_links_source_groups(tables):
    _, meas, _ = tables
    flagged = meas[meas.quality_flags.fillna("").str.contains("suspected_cross_source_copy")]
    assert set(flagged.source_doi) == {"10.1/srcC", "10.1/srcD"}
    assert flagged.source_group_id.nunique() == 1
    assert (flagged.inclusion_status == "included").all()  # kept, not collapsed
    # an isolated agreement between srcA and srcB is logged but does not merge them
    groups = meas.groupby("source_doi").source_group_id.first()
    assert groups["10.1/srcA"] != groups["10.1/srcB"]
    links = pd.read_csv(tables[0].p("processed") / "copy_links.csv")
    assert (links.status == "isolated_agreement_not_linked").sum() == 1
    assert (links.status == "suspected_reuse_linked").sum() == 3


def test_replicates_and_repeated_readings_are_kept(tables):
    _, meas, _ = tables
    rah = meas[(meas.dataset_id == "rahmanian2023") & (meas.inclusion_status == "included")]
    # two channels per sample -> replicate groups of size 2 at each temperature
    assert rah.groupby("replicate_group_id").size().max() == 2
    rep = meas.set_index("measurement_id")
    a, b = rep.loc["calisol23:0"], rep[rep.temperature_K == 299.0].iloc[0]
    assert a.formulation_id == b.formulation_id
    assert b.inclusion_status == "included"


def test_nonpositive_and_missing_targets_preserved_but_excluded(tables):
    _, meas, _ = tables
    reasons = meas.exclusion_reason.value_counts()
    assert reasons["missing_conductivity"] == 1
    assert reasons["negative_conductivity_nonphysical"] == 1
    assert reasons["reported_zero_conductivity"] == 2
    z = meas[meas.exclusion_reason == "reported_zero_conductivity"].sort_values("conductivity_mS_cm")
    assert z.conductivity_mS_cm.iat[0] == 0.0 and pd.isna(z.log10_conductivity.iat[0])
    assert "floating_point_zero" in z.quality_flags.iat[1]  # 1.11e-16 is a zero, not a measurement


def test_unresolved_component_quarantined(tables):
    _, meas, _ = tables
    q = meas[meas.exclusion_reason == "unresolved_component"]
    assert len(q) == 1 and "DMM" in q.quality_flags.iat[0]


def test_suspected_measurement_error_flag_respected(tables):
    _, meas, _ = tables
    s = meas[meas.exclusion_reason == "suspected_measurement_error_in_source"]
    assert len(s) == 5 and (s.dataset_id == "rahmanian2023").all()


def test_rahmanian_units_converted_exactly(tables):
    _, meas, _ = tables
    r = meas[meas.dataset_id == "rahmanian2023"].iloc[0]
    assert r.temperature_K == pytest.approx(r.temperature_original + 273.15)
    assert r.conductivity_mS_cm == pytest.approx(r.conductivity_original * 1000)
    assert r.concentration == pytest.approx(1.2 / 151.903 / 0.010, rel=1e-3)
    assert r.concentration_conversion == "computed_from_weighed_masses"


def test_molarity_and_molality_never_share_a_track(tables):
    _, meas, _ = tables
    inc = meas[meas.inclusion_status == "included"]
    assert inc.groupby("track_id").concentration_basis.nunique().max() == 1


def test_all_leakage_checks_recorded_and_passed(project):
    leak = json.loads((project.p("splits") / "leakage_checks.json").read_text())
    assert leak and all(all(v.values()) for v in leak.values())


def _fold_frames(cfg, meas):
    idx = meas.set_index("measurement_id")
    folds = load_folds(cfg)
    for (task, track), fg in folds.groupby(["task_id", "track_id"]):
        man = load_manifest(cfg, task, track)
        for fid in man.fold_id.unique():
            m = man[man.fold_id == fid]
            tr = idx.loc[m.measurement_id[m.role == "train"]]
            te = idx.loc[m.measurement_id[m.role.isin(["test", "dev"])]]
            yield task, fid, tr, te


def test_copies_and_replicates_never_cross_any_split(tables):
    cfg, meas, _ = tables
    n = 0
    for task, fid, tr, te in _fold_frames(cfg, meas):
        assert not set(tr.duplicate_group_id) & set(te.duplicate_group_id), fid
        assert not set(tr.replicate_group_id) & set(te.replicate_group_id), fid
        assert not set(tr.index) & set(te.index), fid
        n += 1
    assert n > 10


def test_component_holdouts_remove_every_occurrence(tables):
    cfg, meas, _ = tables
    seen = 0
    for task, fid, tr, te in _fold_frames(cfg, meas):
        if task != "component_transfer":
            continue
        comp = fid.split("::dev:")[-1] if "::dev:" in fid else fid.split("/", 1)[1]
        assert not any(comp in s for s in component_sets(tr.reset_index())), fid
        assert all(comp in s for s in component_sets(te.reset_index())), fid
        seen += 1
    assert seen > 0


def test_reserved_dataset_only_in_external_test(tables):
    cfg, meas, _ = tables
    for task, fid, tr, te in _fold_frames(cfg, meas):
        assert not (tr.dataset_id == "rahmanian2023").any(), fid
        if task != "external_rahmanian":
            assert not (te.dataset_id == "rahmanian2023").any(), fid


def test_source_holdout_has_no_linked_source_in_training(tables):
    cfg, meas, _ = tables
    for task, fid, tr, te in _fold_frames(cfg, meas):
        if task == "source_transfer":
            assert not set(tr.source_group_id) & set(te.source_group_id)
            assert not set(tr.source_doi) & set(te.source_doi)


def test_manifests_are_deterministic(project, tmp_path):
    """Rebuilding the splits from the same inputs yields byte-identical manifests."""
    import shutil

    from elbench.config import load_config
    from elbench.splits import make_splits

    root2 = tmp_path / "copy"
    shutil.copytree(project.root, root2, ignore=shutil.ignore_patterns("splits", "results"))
    cfg2 = load_config(root2 / "configs" / "benchmark_v0.3.yaml")
    make_splits(cfg2)
    for p in sorted(project.p("splits").glob("manifest__*.csv.gz")):
        assert p.read_bytes() == (cfg2.p("splits") / p.name).read_bytes(), p.name
