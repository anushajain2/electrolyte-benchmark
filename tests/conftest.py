"""Small synthetic fixtures in the raw formats of both sources.

Values are generated from a smooth conductivity model so baselines can learn,
and deliberately include the curation edge cases: a missing target, negative
and zero targets, an exact in-source copy, a cross-source copy, an
unresolved component, explicit replicates, and a molarity/volume track.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOLVENTS = ["EC", "PC", "DMC", "EMC", "DEC", "EA", "DMM"]


def kappa(salt, fr, c, T, offset=0.0):
    base = {"LiPF6": 11.0, "LiBF4": 5.0, "LiBOB": 7.0}[salt]
    visc = 1.0 + 0.6 * fr.get("EC", 0) + 0.9 * fr.get("PC", 0) - 0.3 * fr.get("DMC", 0) - 0.2 * fr.get("EA", 0)
    return base / visc * c * np.exp(-c / 1.6) * np.exp(-1800 * (1 / T - 1 / 298.15)) * (1 + offset)


def make_calisol(rng) -> pd.DataFrame:
    recs = []
    plan = [
        # doi, salt, solvents, offset (shared systems differ by >=5% so 1% noise
        # cannot create coincidental cross-source matches)
        ("10.1/srcA", "LiPF6", ["EC", "DMC"], 0.00),
        ("10.1/srcB", "LiPF6", ["EC", "DMC"], 0.12),
        ("10.1/srcB", "LiPF6", ["EC", "EMC"], 0.12),
        ("10.1/srcC", "LiPF6", ["PC", "DEC"], -0.04),
        ("10.1/srcD", "LiPF6", ["PC", "DEC"], 0.04),
        ("10.1/srcC", "LiBF4", ["PC", "DEC"], -0.04),
        ("10.1/srcE", "LiBF4", ["PC", "DEC"], 0.04),
        ("10.1/srcE", "LiPF6", ["EC", "EMC"], -0.03),
        ("10.1/srcF", "LiBOB", ["EC", "PC"], 0.00),
        ("10.1/srcF", "LiBOB", ["PC", "EA"], 0.00),
        ("10.1/srcA", "LiBOB", ["EC", "PC"], 0.08),
    ]
    for doi, salt, solv, off in plan:
        for w0 in (0.3, 0.5, 0.7):
            fr = {solv[0]: w0, solv[1]: round(1 - w0, 2)}
            for c in (0.4, 0.8, 1.2, 1.6, 2.0):
                for T in (243.15, 263.15, 283.15, 298.15, 313.15, 333.15):
                    k = kappa(salt, fr, c, T, off) * (1 + 0.01 * rng.standard_normal())
                    recs.append({"doi": doi, "k": k, "T": T, "c": c, "salt": salt, "c units": "mol/kg", "solvent ratio type": "w", **fr})
    # molarity / volume track (two sources, shared chemistry)
    for doi, off in (("10.1/srcG", 0.0), ("10.1/srcH", 0.10)):
        for v0 in (0.3, 0.5, 0.7):
            fr = {"EC": v0, "DMC": round(1 - v0, 2)}
            for c in (0.5, 1.0, 1.5):
                for T in (263.15, 298.15, 323.15):
                    recs.append({"doi": doi, "k": kappa("LiPF6", fr, c, T, off), "T": T, "c": c, "salt": "LiPF6", "c units": "mol/l", "solvent ratio type": "v", **fr})
    # srcI republishes srcA's EC+DMC sweep read off a plot: every point 1-3% off
    # (never within the 0.5% copy tolerance, always within the 5% campaign tolerance)
    for r in [x for x in recs if x["doi"] == "10.1/srcA" and x["salt"] == "LiPF6"][:60]:
        recs.append({**r, "doi": "10.1/srcI", "k": r["k"] * (1.01 + 0.02 * rng.random())})
    df = pd.DataFrame(recs)
    # edge cases
    extra = []
    base = df.iloc[0].to_dict()
    extra.append({**base})  # exact in-source copy of row 0
    extra.append({**base, "k": np.nan, "T": 300.0})  # missing target
    extra.append({**base, "k": -0.01, "T": 233.15})  # negative
    extra.append({**base, "k": 0.0, "T": 228.15})  # reported zero
    extra.append({**base, "k": 1.11e-16, "T": 223.15})  # digitized zero stored as float noise
    for j in (3, 4, 5):  # three points of srcC republished in srcD -> suspected reuse
        copy_src = df[(df.doi == "10.1/srcC") & (df.salt == "LiPF6")].iloc[j].to_dict()
        extra.append({**copy_src, "doi": "10.1/srcD", "k": copy_src["k"] * 1.002})
    lone = df[(df.doi == "10.1/srcA") & (df.salt == "LiPF6")].iloc[7].to_dict()
    extra.append({**lone, "doi": "10.1/srcB", "k": lone["k"] * 0.999})  # isolated agreement, not linked
    extra.append({**base, "EC": 0.5, "DMC": 0.0, "DMM": 0.5})  # unresolved component
    extra.append({**base, "T": 299.0, "k": base["k"] * 1.05})  # repeated reading, kept
    df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    for s in SOLVENTS:
        df[s] = df.get(s, 0.0)
    df[SOLVENTS] = df[SOLVENTS].fillna(0.0)
    return df[["doi", "k", "T", "c", "salt", "c units", "solvent ratio type", *SOLVENTS]]


def make_rahmanian(rng) -> str:
    cols = ["label", "experimentID", "temperature", "frequency", "real_impedance", "imaginary_impedance", "cell_constant,_standard_deviation", "PC", "EC", "EMC", "LiPF_6", "metadata", "EIS_conductivity"]
    sym = ["symbol", "-", "T", "f", "Z'", "Z''", "-", "-", "-", "-", "-", "-", "-"]
    unit = ["unit", "-", "°C", "Hz", "Ohm", "Ohm", "-", "g", "g", "g", "g", "-", "S/cm"]
    lines = [";".join(cols), ";".join(sym), ";".join(unit)]
    label = 0
    for b, (pc, ec, emc, salt_g) in enumerate([(1.5, 1.5, 7.0, 1.2), (2.5, 1.0, 6.5, 1.8), (1.0, 2.0, 7.0, 0.8)]):
        for ch in (1, 2):
            for tc in (-20, 0, 20, 40, 60):
                tot = pc + ec + emc
                fr = {"PC": pc / tot, "EC": ec / tot, "EMC": emc / tot}
                c = salt_g / 151.904 / (tot / 1000)
                k = kappa("LiPF6", fr, c, tc + 273.15) * (1 + 0.005 * rng.standard_normal()) / 1000
                meta = "{'experimentDate': '01.01.2021', 'channel': 'n%d', 'suspectedMeasurementError': %s, 'comment': nan}" % (ch, "True" if (b == 2 and ch == 2) else "False")
                lines.append(";".join(map(str, [label, f"WOC_01012021_BM{b:03d}_{ch}", float(tc), "[1.0]", "[1.0]", "[1.0]", "(4.7, 0.003)", pc, ec, emc, salt_g, meta, k])))
                label += 1
    return "\n".join(lines) + "\n"


def write_project(root: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    local = root / "local_raw"
    (local / "calisol23").mkdir(parents=True)
    (local / "rahmanian2023").mkdir(parents=True)
    make_calisol(rng).to_csv(local / "calisol23" / "CALiSol-23 Dataset.csv", index=False)
    (local / "calisol23" / "calisolsmile.csv").write_text("placeholder\n")
    (local / "calisol23" / "CALiSol-23.ipynb").write_text("{}\n")
    (local / "rahmanian2023" / "Conductivtiy_experiment.csv").write_text(make_rahmanian(rng))
    (root / "data" / "mappings").mkdir(parents=True)
    shutil.copy(ROOT / "data" / "mappings" / "species.csv", root / "data" / "mappings" / "species.csv")
    (root / "reports").mkdir()
    (root / "reports" / "findings_v0.3.md").write_text("# Findings\n\nFixture run.\n")
    groups = {"10.1/srcA": "AG1", "10.1/srcB": "AG1", "10.1/srcI": "AG1", "10.1/srcC": "AG2", "10.1/srcD": "AG2", "10.1/srcE": "AG3", "10.1/srcF": "AG4", "10.1/srcG": "AG5", "10.1/srcH": "AG6", "10.1038/s41597-023-01936-3": "AG7"}
    pd.DataFrame({"source_doi": list(groups), "author_group_id": list(groups.values()), "status": "fixture", "evidence": ""}).to_csv(root / "data" / "mappings" / "source_author_groups.csv", index=False)
    cfg = yaml.safe_load((ROOT / "configs" / "benchmark_v0.3.yaml").read_text())
    cfg.pop("compare_with", None)
    cfg["sources"]["rahmanian2023"].pop("expected_md5")
    cfg["curation"]["quarantine_sources"] = []
    cfg["reporting"].update({"min_test_measurements_per_unit": 5, "headline_min_units": 3, "headline_min_formulations_per_unit": 3})
    cfg["tasks"]["component_transfer"].update({"min_formulations": 3, "max_measurement_share": 0.45})
    cfg["models"]["seeds"] = [0, 1]
    fam = cfg["models"]["families"]
    fam["extra_trees"]["fixed"] = {"n_estimators": 20}
    fam["extra_trees"]["grid"] = {"max_features": [0.5, 1.0], "min_samples_leaf": [1]}
    fam["hist_gb"]["fixed"] = {"max_iter": 30}
    fam["hist_gb"]["grid"] = {"learning_rate": [0.1], "max_leaf_nodes": [7, 15], "l2_regularization": [0.0]}
    cfg["evaluation"]["n_bootstrap"] = 200
    inv = cfg["investigation"]
    inv["fixed_configs"]["extra_trees"]["n_estimators"] = 20
    inv["source_familiarity"].update({"min_source_formulations": 5, "background_size": 150, "repeats": 2})
    inv["source_diversity"].update({"budget": 40, "max_sources": 2, "repeats": 2})
    (root / "configs").mkdir()
    path = root / "configs" / "benchmark_v0.3.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


@pytest.fixture(scope="session")
def project(tmp_path_factory):
    """A fixture project taken through ingest -> curate -> audit -> make-splits."""
    from elbench.audit import audit
    from elbench.config import load_config
    from elbench.curate import curate
    from elbench.ingest import ingest
    from elbench.splits import make_splits

    root = tmp_path_factory.mktemp("proj")
    cfg_path = write_project(root)
    cfg = load_config(cfg_path)
    ingest(cfg, local_dir=str(root / "local_raw"))
    curate(cfg)
    audit(cfg)
    make_splits(cfg)
    return cfg
