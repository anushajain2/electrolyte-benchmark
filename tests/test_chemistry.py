import pytest

from elbench import chemistry
from elbench.features import FEATURE_COLUMNS

from .conftest import ROOT


@pytest.fixture(scope="module")
def species():
    return chemistry.load_species(ROOT / "data" / "mappings" / "species.csv")


def test_reviewed_mapping_is_consistent(species):
    v = species[species.review_status == "verified"]
    for r in v.itertuples():
        if r.role == "solvent":
            chemistry.mol(r.smiles)
        else:
            chemistry.mol(r.cation_smiles)
            chemistry.mol(r.anion_smiles)
    # disputed identities carry no structure, so they can never produce descriptors
    d = species[species.review_status == "disputed"]
    assert len(d) >= 3 and (d.smiles == "").all()


def test_tfsi_labels_merge_to_one_identity(species):
    ids = species[species.raw_name.isin(["LiTFSI", "LiN(CF3SO2)2"])].canonical_id.unique()
    assert list(ids) == ["LiTFSI"]


def test_salts_are_ions_with_charges(species):
    from rdkit import Chem

    for r in species[(species.role == "salt") & (species.review_status == "verified")].itertuples():
        cat, an = chemistry.mol(r.cation_smiles), chemistry.mol(r.anion_smiles)
        assert Chem.GetFormalCharge(cat) * r.cation_stoich + Chem.GetFormalCharge(an) * r.anion_stoich == 0


def test_salt_molar_mass_matches_source_metadata(species):
    row = species[(species.dataset_id == "rahmanian2023") & (species.raw_name == "LiPF_6")].iloc[0]
    assert chemistry.salt_molar_mass(row) == pytest.approx(151.904, abs=0.01)


def test_feature_set_is_fixed():
    assert len(FEATURE_COLUMNS) == 32
    assert len(set(FEATURE_COLUMNS)) == 32
    assert not any("source" in c or "doi" in c for c in FEATURE_COLUMNS)
