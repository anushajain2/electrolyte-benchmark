import numpy as np
import pytest

from elbench import units
from elbench.units import ConversionUnsupported


def test_known_unit_examples():
    assert units.temperature_to_K(25.0, "°C") == pytest.approx(298.15)
    assert units.temperature_to_K(298.15, "K") == 298.15
    assert units.conductivity_to_mS_cm(0.0105, "S/cm") == pytest.approx(10.5)
    assert units.conductivity_to_mS_cm(1.05, "S/m") == pytest.approx(10.5)
    assert units.mS_cm_to(10.5, "S/cm") == pytest.approx(0.0105)
    with pytest.raises(ConversionUnsupported):
        units.temperature_to_K(77, "F")
    with pytest.raises(ConversionUnsupported):
        units.conductivity_to_mS_cm(1.0, "mho")


def test_molality_from_weighed_masses():
    # 0.01 mol LiPF6 (1.51904 g) in 10 g solvent -> 1.0 mol/kg
    assert units.molality_from_masses(1.51904, 151.904, 10.0) == pytest.approx(1.0)
    with pytest.raises(ConversionUnsupported):
        units.molality_from_masses(1.0, 151.9, 0.0)


def test_mass_mole_fraction_round_trip():
    mw = {"EC": 88.06, "DMC": 90.08, "TFP": 344.06}
    w = {"EC": 0.3, "DMC": 0.5, "TFP": 0.2}
    x = units.mass_to_mole_fractions(w, mw)
    assert sum(x.values()) == pytest.approx(1.0)
    assert x["TFP"] < w["TFP"]  # heavy component has a smaller mole fraction
    back = units.mole_to_mass_fractions(x, mw)
    for k in w:
        assert back[k] == pytest.approx(w[k], rel=1e-12)


def test_conversions_refused_without_information():
    with pytest.raises(ConversionUnsupported):
        units.mass_to_mole_fractions({"EC": 0.5, "XYZ": 0.5}, {"EC": 88.06})
    with pytest.raises(ConversionUnsupported):
        units.volume_to_mole_fractions({"EC": 0.5, "DMC": 0.5}, {"EC": 88.06, "DMC": 90.08})
    with pytest.raises(ConversionUnsupported):
        units.molarity_to_molality(1.0)
    with pytest.raises(ConversionUnsupported):
        units.molality_to_molarity(1.0, density_g_per_ml=1.2)


def test_molarity_molality_round_trip_with_density():
    b = units.molarity_to_molality(1.0, density_g_per_ml=1.25, salt_molar_mass=151.9)
    assert units.molality_to_molarity(b, density_g_per_ml=1.25, salt_molar_mass=151.9) == pytest.approx(1.0)


def test_safe_log_never_invents_values():
    out = units.safe_log10(np.array([10.0, 0.0, -1.0, np.nan]))
    assert out[0] == pytest.approx(1.0)
    assert np.isnan(out[1:]).all()
