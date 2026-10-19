"""Canonical table schemas and typed validation."""
from __future__ import annotations

import numpy as np
import pandas as pd

# column -> (dtype kind, nullable, description)
MEASUREMENT_COLUMNS: dict[str, tuple[str, bool, str]] = {
    # identity and provenance
    "measurement_id": ("str", False, "Stable ID: <dataset_id>:<original_row_id>"),
    "dataset_id": ("str", False, "Imported collection"),
    "original_row_id": ("str", False, "Row identifier in the raw file (0-based data row or source label)"),
    "source_doi": ("str", False, "DOI of the originating paper (a source proxy, not a lab identifier)"),
    "source_group_id": ("str", False, "Originating paper plus any source proven to reuse its measurements"),
    "series_id": ("str", False, "Experimental sweep / batch inside one source"),
    "duplicate_group_id": ("str", False, "Proven copies of the same original measurement"),
    "replicate_group_id": ("str", False, "Repeated readings of one formulation at one temperature bin inside one source group"),
    "split_unit_id": ("str", False, "Union of formulation and copy groups; the atomic unit for formulation-level splits"),
    "author_group_id": ("str", False, "Reviewed grouping of sources by shared authors (v0.2+); equals source_group_id when no mapping is configured. Not a verified lab id"),
    "lab_id": ("str", True, "Only when explicitly verified; empty in v0.1"),
    # formulation
    "formulation_id": ("str", False, "Chemistry + solvent ratios + concentration (no temperature)"),
    "solvent_recipe_id": ("str", False, "Chemistry + solvent ratios (no concentration, no temperature)"),
    "chemistry_system_id": ("str", False, "Salt | solvent identities, no amounts"),
    "salt_id": ("str", False, "Canonical salt identity"),
    "solvent_ids": ("str", False, "'+'-joined canonical solvent identities"),
    "additive_ids": ("str", True, "Additives (none recorded separately in v0.1 sources)"),
    "n_solvents": ("int", False, "Number of solvent components"),
    "ratio_basis": ("str", False, "w (mass), v (volume) or mol (mole) solvent fractions"),
    "solvent_fractions_reported": ("str", False, "JSON of reported solvent fractions (raw names, reported basis)"),
    # conditions
    "temperature_K": ("float", False, "Temperature in kelvin"),
    "temperature_original": ("float", False, "Reported temperature value"),
    "temperature_unit_original": ("str", False, "Reported temperature unit"),
    "concentration_original": ("float", False, "Reported salt concentration value"),
    "concentration_unit_original": ("str", False, "Reported concentration unit"),
    "concentration_basis": ("str", False, "molality or molarity"),
    "concentration": ("float", False, "Canonical value in the basis unit (mol/kg or mol/L)"),
    "concentration_conversion": ("str", False, "as_reported | computed_from_weighed_masses"),
    # target
    "conductivity_mS_cm": ("float", True, "Measured ionic conductivity in mS/cm"),
    "conductivity_original": ("float", True, "Reported value"),
    "conductivity_unit_original": ("str", False, "Reported unit"),
    "log10_conductivity": ("float", True, "log10 of conductivity in mS/cm; NaN when nonpositive"),
    "conductivity_uncertainty_mS_cm": ("float", True, "Only if reported"),
    "measurement_method": ("str", True, "Only if reported"),
    # quality and traceability
    "inclusion_status": ("str", False, "included | excluded"),
    "exclusion_reason": ("str", True, "First failing rule; all reasons in exclusions.csv"),
    "quality_flags": ("str", True, "';'-joined quality flags"),
    "conversion_provenance": ("str", False, "';'-joined rule ids applied to this row"),
    "source_record_link": ("str", False, "Pointer to the raw record"),
    "duplicate_of": ("str", True, "Canonical measurement_id for collapsed copies"),
    "track_id": ("str", True, "Basis-specific benchmark track for included rows"),
    "has_uncertainty": ("bool", False, "Metadata completeness"),
    "has_method": ("bool", False, "Metadata completeness"),
    "has_lab_id": ("bool", False, "Metadata completeness"),
}

COMPONENT_COLUMNS: dict[str, tuple[str, bool, str]] = {
    "formulation_id": ("str", False, ""),
    "component_id": ("str", False, "Canonical identity"),
    "role": ("str", False, "solvent | salt"),
    "smiles": ("str", True, "Canonical SMILES (solvent) or anion SMILES (salt)"),
    "cation_smiles": ("str", True, ""),
    "cation_stoich": ("float", True, ""),
    "anion_smiles": ("str", True, ""),
    "anion_stoich": ("float", True, ""),
    "amount_value": ("float", False, "Solvent fraction in the reported basis, or salt concentration"),
    "amount_basis": ("str", False, "mass_fraction | volume_fraction | mole_fraction | molality | molarity"),
    "mole_fraction": ("float", True, "Solvent mole fraction when exactly convertible"),
    "molar_mass": ("float", True, "g/mol (whole salt for salts)"),
}

CURATION_RULES = {
    "R01": "Temperature reported in K kept as is; degC converted by +273.15 exactly.",
    "R02": "Conductivity reported in mS/cm kept as is; S/cm multiplied by 1000 exactly.",
    "R03": "Concentration kept in its reported basis; molality and molarity are separate tracks, never converted into each other (no density data).",
    "R04": "Salt molality computed from weighed salt and solvent masses with the RDKit molecular weight of the salt.",
    "R05": "Solvent mass fractions computed from weighed solvent masses.",
    "R06": "Reported solvent fractions renormalized to sum to 1 when within 0.02 of 1 (rounding, e.g. 0.33 x 3).",
    "R07": "Solvent mass fractions converted to mole fractions exactly from molecular weights; volume fractions are not converted (no densities).",
    "R08": "Raw species labels mapped through data/mappings/species.csv (reviewed). Two CALiSol labels for LiTFSI merged.",
    "R09": "Exact copies inside one source (identical chemistry, ratio within 0.005, concentration within 0.001, T within 0.05 K, identical conductivity) are collapsed to the first row. Across sources, matches within 0.5% conductivity are suspected reuse when a pair of papers shares at least 3 such points: rows are kept, share a duplicate group, and the two source groups are merged. Isolated agreements are logged only.",
    "R10": "Nonpositive reported conductivities preserved as excluded, flagged records; no positive value is invented.",
    "R12": "v0.2+: two papers are linked into one source group (a conservative provenance-based grouping, not proof of one campaign) when at least min_comparable point pairs at comparable conditions exist (T within 1 K, c within 0.03, fractions within 0.02) and at least min_agreement of them agree within 5% in conductivity; agreeing points become one duplicate group.",
    "R13": "v0.4+: rows confirmed wrong against the original paper are excluded (reason confirmed_compilation_error); each list in the config cites its evidence.",
    "R14": "v0.4+: for sources whose conductivities were digitized from linear-axis figures (data/mappings/source_presentation.csv), values below the digitization floor (0.2 mS/cm) are excluded (below_digitization_floor): against one paper's own fit they differ by a median factor of 1.4-3.7, versus about 2% above 1 mS/cm.",
    "R11": "Reported conductivities below 1e-9 mS/cm (e.g. 1.11e-16) are treated as suspected numerical artifacts of a digitized zero and excluded like reported zeros (R10). Added in v0.1 after failure-case tracing; affects 3 rows; not yet traced to the original figures.",
}


def _kind_ok(series: pd.Series, kind: str) -> bool:
    if kind == "str":
        return pd.api.types.is_string_dtype(series) or series.dtype == object
    if kind == "float":
        return pd.api.types.is_float_dtype(series)
    if kind == "int":
        return pd.api.types.is_integer_dtype(series)
    if kind == "bool":
        return pd.api.types.is_bool_dtype(series)
    raise ValueError(kind)


def validate(df: pd.DataFrame, spec: dict[str, tuple[str, bool, str]], name: str) -> None:
    missing = [c for c in spec if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    for col, (kind, nullable, _) in spec.items():
        s = df[col]
        if not _kind_ok(s, kind):
            raise TypeError(f"{name}.{col}: expected {kind}, got {s.dtype}")
        if not nullable and s.isna().any():
            raise ValueError(f"{name}.{col}: {int(s.isna().sum())} nulls in non-nullable column")


def validate_measurements(df: pd.DataFrame) -> None:
    validate(df, MEASUREMENT_COLUMNS, "measurements")
    if df.measurement_id.duplicated().any():
        raise ValueError("measurement_id not unique")
    inc = df[df.inclusion_status == "included"]
    if (inc.conductivity_mS_cm <= 0).any() or inc.conductivity_mS_cm.isna().any():
        raise ValueError("included rows must have positive conductivity")
    if inc.track_id.isna().any():
        raise ValueError("included rows must have a track")
    if not np.allclose(inc.log10_conductivity, np.log10(inc.conductivity_mS_cm)):
        raise ValueError("log10 column inconsistent")
    exc = df[df.inclusion_status == "excluded"]
    if exc.exclusion_reason.isna().any():
        raise ValueError("excluded rows need a reason")


def validate_components(df: pd.DataFrame) -> None:
    validate(df, COMPONENT_COLUMNS, "components")
    solv = df[df.role == "solvent"]
    sums = solv.groupby("formulation_id").amount_value.sum()
    if not np.allclose(sums, 1.0, atol=1e-6):
        raise ValueError("solvent fractions per formulation must sum to 1")
