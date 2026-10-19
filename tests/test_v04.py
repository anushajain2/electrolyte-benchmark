"""v0.4 label-reliability rules: confirmed row errors (R13) and the
digitization floor for linear-axis sources (R14)."""
import pandas as pd
import yaml

from elbench.config import load_config
from elbench.curate import curate, load_processed
from elbench.ingest import ingest

from .conftest import write_project


def test_confirmed_rows_and_digitization_floor(tmp_path):
    cfg_path = write_project(tmp_path)
    cfg = yaml.safe_load(cfg_path.read_text())
    raw = pd.read_csv(tmp_path / "local_raw" / "calisol23" / "CALiSol-23 Dataset.csv")
    # a low value in a linear-axis source and the same value in a non-linear one
    low_a = raw.index[(raw.doi == "10.1/srcA") & (raw.k > 0) & (raw.k < 1.0)]
    low_c = raw.index[(raw.doi == "10.1/srcC") & (raw.k > 0) & (raw.k < 1.0)]
    assert len(low_a) and len(low_c)
    pd.DataFrame(
        [
            {"source_doi": "10.1/srcA", "presentation": "linear_axis_figure", "status": "fixture", "evidence": ""},
            {"source_doi": "10.1/srcC", "presentation": "log_axis_or_table", "status": "fixture", "evidence": ""},
        ]
    ).to_csv(tmp_path / "data" / "mappings" / "source_presentation.csv", index=False)
    cfg["curation"]["quarantine_rows"] = [{"dataset_id": "calisol23", "original_row_ids": ["5"], "reason": "fixture_confirmed"}]
    cfg["curation"]["digitization_floor"] = {"threshold_mS_cm": 1.0, "presentation_map": "data/mappings/source_presentation.csv", "apply_to_presentation": ["linear_axis_figure"]}
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    c = load_config(cfg_path)
    ingest(c, local_dir=str(tmp_path / "local_raw"))
    curate(c)
    meas, _ = load_processed(c)
    idx = meas.set_index("measurement_id")
    assert idx.at["calisol23:5", "exclusion_reason"] == "confirmed_compilation_error"
    exc = pd.read_csv(c.p("processed") / "exclusions.csv")
    floored = set(exc.measurement_id[exc.reason == "below_digitization_floor"])
    a_ids = {f"calisol23:{i}" for i in low_a}
    assert a_ids <= floored  # every low value of the linear-axis source, whatever else it failed
    assert idx.loc[sorted(a_ids)].quality_flags.str.contains("linear_axis_low_value").all()
    # the floor applies only to linear-axis sources
    assert not ({f"calisol23:{i}" for i in low_c} & floored)
    # nothing at or above the floor is touched
    hi = set(idx.index[(idx.source_doi == "10.1/srcA") & (idx.conductivity_mS_cm >= 1.0)])
    assert not (hi & floored)
