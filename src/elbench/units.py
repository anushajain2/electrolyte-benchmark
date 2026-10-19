"""Exact unit and composition conversions.

Every function either converts exactly with the information it is given or
raises ``ConversionUnsupported``. Nothing here guesses a density, a ratio
basis or a molecular weight.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np


class ConversionUnsupported(ValueError):
    """Raised when a conversion would need information that is not available."""


ZERO_CELSIUS_K = 273.15

_CONDUCTIVITY_TO_MS_CM = {
    "mS/cm": 1.0,
    "S/cm": 1000.0,
    "S/m": 10.0,
    "mS/m": 0.01,
    "uS/cm": 0.001,
}


def temperature_to_K(value: float, unit: str) -> float:
    if unit == "K":
        return float(value)
    if unit in ("°C", "C", "degC"):
        return float(value) + ZERO_CELSIUS_K
    raise ConversionUnsupported(f"unknown temperature unit {unit!r}")


def conductivity_to_mS_cm(value: float, unit: str) -> float:
    try:
        return float(value) * _CONDUCTIVITY_TO_MS_CM[unit]
    except KeyError:
        raise ConversionUnsupported(f"unknown conductivity unit {unit!r}") from None


def mS_cm_to(value: float, unit: str) -> float:
    try:
        return float(value) / _CONDUCTIVITY_TO_MS_CM[unit]
    except KeyError:
        raise ConversionUnsupported(f"unknown conductivity unit {unit!r}") from None


def normalize_fractions(amounts: Mapping[str, float]) -> dict[str, float]:
    total = float(sum(amounts.values()))
    if not total > 0:
        raise ConversionUnsupported("fractions sum to zero")
    return {k: float(v) / total for k, v in amounts.items()}


def mass_to_mole_fractions(mass_fractions: Mapping[str, float], molar_mass: Mapping[str, float]) -> dict[str, float]:
    """w_i -> x_i = (w_i / M_i) / sum_j (w_j / M_j). Exact given molecular weights."""
    missing = [k for k in mass_fractions if k not in molar_mass or not molar_mass[k] > 0]
    if missing:
        raise ConversionUnsupported(f"molecular weight missing for {missing}")
    moles = {k: float(w) / float(molar_mass[k]) for k, w in mass_fractions.items()}
    return normalize_fractions(moles)


def mole_to_mass_fractions(mole_fractions: Mapping[str, float], molar_mass: Mapping[str, float]) -> dict[str, float]:
    missing = [k for k in mole_fractions if k not in molar_mass or not molar_mass[k] > 0]
    if missing:
        raise ConversionUnsupported(f"molecular weight missing for {missing}")
    return normalize_fractions({k: float(x) * float(molar_mass[k]) for k, x in mole_fractions.items()})


def volume_to_mole_fractions(volume_fractions: Mapping[str, float], molar_mass: Mapping[str, float], density: Mapping[str, float] | None = None) -> dict[str, float]:
    """Needs pure-component densities *and* an ideal-mixing convention; refused without them."""
    if not density or any(k not in density for k in volume_fractions):
        raise ConversionUnsupported("volume -> mole fraction needs pure-component densities and a mixing convention")
    masses = {k: float(v) * float(density[k]) for k, v in volume_fractions.items()}
    return mass_to_mole_fractions(normalize_fractions(masses), molar_mass)


def molality_from_masses(salt_mass_g: float, salt_molar_mass: float, solvent_mass_g: float) -> float:
    """mol salt per kg solvent, exact from weighed masses."""
    if not solvent_mass_g > 0:
        raise ConversionUnsupported("solvent mass must be positive")
    if not salt_molar_mass > 0:
        raise ConversionUnsupported("salt molar mass must be positive")
    return (float(salt_mass_g) / float(salt_molar_mass)) / (float(solvent_mass_g) / 1000.0)


def molarity_to_molality(molarity: float, *, density_g_per_ml: float | None = None, salt_molar_mass: float | None = None) -> float:
    """c[mol/L] -> b[mol/kg solvent] = c / (rho - c*M/1000). Needs solution density."""
    if density_g_per_ml is None or salt_molar_mass is None:
        raise ConversionUnsupported("molarity <-> molality needs solution density and salt molar mass")
    solvent_kg_per_l = float(density_g_per_ml) - float(molarity) * float(salt_molar_mass) / 1000.0
    if not solvent_kg_per_l > 0:
        raise ConversionUnsupported("nonphysical density for this molarity")
    return float(molarity) / solvent_kg_per_l


def molality_to_molarity(molality: float, *, density_g_per_ml: float | None = None, salt_molar_mass: float | None = None) -> float:
    if density_g_per_ml is None or salt_molar_mass is None:
        raise ConversionUnsupported("molarity <-> molality needs solution density and salt molar mass")
    b, M, rho = float(molality), float(salt_molar_mass), float(density_g_per_ml)
    return b * rho / (1.0 + b * M / 1000.0)


def safe_log10(values: np.ndarray) -> np.ndarray:
    """log10 that returns NaN for nonpositive inputs instead of inventing a value."""
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan)
    ok = values > 0
    out[ok] = np.log10(values[ok])
    return out
