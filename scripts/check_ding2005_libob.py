"""Check CALiSol rows of 10.1149/1.1833611 against the paper's own fitted functions.

Ding, Xu & Jow, "Conductivity and Viscosity of PC-DEC and PC-EC Solutions of
LiBOB", J. Electrochem. Soc. 152 (2005) A132 (open access at the publisher).
Table II gives, for each measurement temperature theta, the fit

    ln(kappa / uS cm^-1) = (a0 + a1 w + a2 w^2) ln m + (b0 + b1 w + b2 w^2)
                           + (c0 + c1 w + c2 w^2) m + (d0 + d1 w + d2 w^2) m^2

for (LiBOB)m-PC(1-w)EC(w), m in mol/kg (0.1-1.1), w = EC weight fraction (0-0.7);
coefficients transcribed in data/evidence/ding2005_libob_pc_ec_tableII.csv.
The paper's data appear only as figures on a linear 0-10 mS/cm axis, so this
measures CALiSol's digitization error and finds mis-assigned temperatures.

    uv run python scripts/check_ding2005_libob.py [processed_dir]

Writes reports/evidence/ding2005_libob_rows.csv (per row), _by_level.csv
(error by conductivity level), _temperature_mislabels.csv and
_internal_consistency.csv, plus data/evidence/corrections_review_record.csv.

The PC-DEC half cannot be checked against the paper's Table I: as printed, its
"-" entries are lost numbers (the negative temperatures in its first column are
also printed as "-"), and the transcribed table disagrees with the paper's own
Fig. 2 by orders of magnitude (data/evidence/ding2005_libob_pc_dec_tableI_as_printed.csv).
Both halves are therefore also checked by internal consistency, which needs no fit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DOI = "10.1149/1.1833611"


def fit_kappa(tab: pd.DataFrame, theta: float, m: float, w: float) -> float:
    r = tab.loc[theta]
    lnk = (r.a0 + r.a1 * w + r.a2 * w * w) * np.log(m) + (r.b0 + r.b1 * w + r.b2 * w * w) + (r.c0 + r.c1 * w + r.c2 * w * w) * m + (r.d0 + r.d1 * w + r.d2 * w * w) * m * m
    return float(np.exp(lnk) / 1000.0)  # mS/cm


def main(processed: Path) -> None:
    tab = pd.read_csv(ROOT / "data/evidence/ding2005_libob_pc_ec_tableII.csv").set_index("theta_C")
    meas = pd.read_parquet(processed / "measurements.parquet")
    x = meas[(meas.source_doi == DOI) & meas.conductivity_mS_cm.notna()].copy()
    fr = x.solvent_fractions_reported.map(json.loads)
    x = x[fr.map(lambda f: set(f) <= {"PC", "EC"})]
    x["w_EC"] = fr.loc[x.index].map(lambda f: f.get("EC", 0.0))
    x["theta_C"] = x.temperature_K.map(lambda T: min(tab.index, key=lambda t: abs(t - (T - 273.15))))
    x = x[x.concentration.between(0.1, 1.1) & (x.w_EC <= 0.7)]
    x["kappa_fit_mS_cm"] = [fit_kappa(tab, t, c, w) for t, c, w in zip(x.theta_C, x.concentration, x.w_EC)]
    pos = x.conductivity_mS_cm > 0
    x["log10_ratio"] = np.where(pos, np.log10(x.conductivity_mS_cm.where(pos, 1.0) / x.kappa_fit_mS_cm), np.nan)
    # temperature that best explains each value (same m and w)
    best = []
    for t, c, w, k in zip(x.theta_C, x.concentration, x.w_EC, x.conductivity_mS_cm):
        if k > 0:
            best.append(min(tab.index, key=lambda th: abs(np.log(fit_kappa(tab, th, c, w)) - np.log(k))))
        else:
            best.append(np.nan)
    x["best_matching_theta_C"] = best
    out = ROOT / "reports" / "evidence"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["measurement_id", "original_row_id", "temperature_K", "theta_C", "concentration", "w_EC", "conductivity_original", "kappa_fit_mS_cm", "log10_ratio", "best_matching_theta_C", "inclusion_status"]
    x[cols].to_csv(out / "ding2005_libob_rows.csv", index=False)
    lv = x[pos].assign(level=pd.cut(x.kappa_fit_mS_cm[pos], [0, 0.05, 0.1, 0.2, 0.5, 1, 2, 10]))
    by = lv.groupby("level", observed=True).log10_ratio.agg(n="size", median_abs=lambda v: v.abs().median(), p90_abs=lambda v: v.abs().quantile(0.9), median_signed="median").reset_index()
    by["level"] = by.level.astype(str)
    by.to_csv(out / "ding2005_libob_by_level.csv", index=False)
    # rule 1: a value off by > 0.5 log10 where the fit is well determined (> 1 mS/cm)
    #         and another measured temperature explains it
    shifted = (x.log10_ratio.abs() > 0.5) & (x.best_matching_theta_C != x.theta_C) & (x.kappa_fit_mS_cm > 1.0)
    # rule 2: an exact copy of another row's value at a different temperature
    allx = meas[meas.source_doi == DOI]
    copies = set()
    for _, g in allx.groupby("conductivity_original"):
        if len(g) > 1 and g.temperature_K.round(1).nunique() > 1:
            # keep the row whose value matches the fit at its own temperature; flag the others
            for mid in g.measurement_id:
                if mid in set(x.measurement_id):
                    r = x.set_index("measurement_id").loc[mid]
                    # only where the fit is well determined; low values carry digitization noise
                    if abs(r.log10_ratio) > 0.1 and r.kappa_fit_mS_cm > 1.0:
                        copies.add(mid)
    x["mislabel_rule"] = np.where(shifted, "value_matches_other_temperature", np.where(x.measurement_id.isin(copies), "exact_copy_at_other_temperature", ""))
    mis = x[x.mislabel_rule != ""]
    mis[cols + ["mislabel_rule"]].to_csv(out / "ding2005_libob_temperature_mislabels.csv", index=False)
    print(by.round(3).to_string(index=False))
    ic = internal_consistency(meas)
    ic.to_csv(out / "ding2005_libob_internal_consistency.csv", index=False)
    rec = write_review_record(mis, ic, meas)
    rec.to_csv(ROOT / "data" / "evidence" / "corrections_review_record.csv", index=False)
    print("internal-consistency flags by half:", ic.groupby("half").size().to_dict(), "| recall of fit-confirmed rows:", int(mis.measurement_id.isin(ic.measurement_id).sum()), "/", len(mis))
    print(f"rows compared: {len(x)}; nonpositive: {int((~pos).sum())}; temperature mislabels: {len(mis)}")
    print(mis[["original_row_id", "temperature_K", "concentration", "conductivity_original", "kappa_fit_mS_cm", "best_matching_theta_C", "mislabel_rule"]].round(3).to_string(index=False))


def internal_consistency(meas: pd.DataFrame) -> pd.DataFrame:
    """Checks that need no fitted equation, applied to both halves of the paper.

    (1) conductivity must rise with temperature at (nearly) fixed composition:
        flag a row whose value is > 2x below a colder row within 0.015 mol/kg;
    (2) a value copied exactly from another temperature, where rows at its own
        temperature and composition are > 1.4x higher.
    Values below 0.02 mS/cm are not judged (digitization noise)."""
    x = meas[(meas.source_doi == DOI) & (meas.conductivity_mS_cm > 0)].copy()
    fr = x.solvent_fractions_reported.map(json.loads)
    x["half"] = fr.map(lambda f: "PC-DEC" if "DEC" in f else ("PC-EC" if "EC" in f else "PC"))
    x["w"] = fr.map(lambda f: f.get("DEC", f.get("EC", 0.0)))
    flags: dict = {}
    for _, g in x.groupby(["half", "w"]):
        for i, r in g.iterrows():
            colder = g[(abs(g.concentration - r.concentration) <= 0.015) & (g.temperature_K < r.temperature_K - 5)]
            if len(colder) and r.conductivity_mS_cm > 0.02 and np.log10(colder.conductivity_mS_cm.max() / r.conductivity_mS_cm) > 0.3:
                flags.setdefault(i, []).append("lower_than_a_colder_row")
    for _, g in x.groupby("conductivity_original"):
        if len(g) > 1 and g.temperature_K.round(1).nunique() > 1:
            for i, r in g.iterrows():
                same = x[(x.half == r.half) & (x.w == r.w) & (abs(x.concentration - r.concentration) <= 0.015) & (abs(x.temperature_K - r.temperature_K) < 1) & (x.index != i)]
                if len(same) and np.log10(same.conductivity_mS_cm.median() / r.conductivity_mS_cm) > 0.15:
                    flags.setdefault(i, []).append("exact_copy_inconsistent_with_same_temperature")
    out = x.loc[sorted(flags), ["measurement_id", "original_row_id", "half", "w", "temperature_K", "concentration", "conductivity_original"]].copy()
    out["flags"] = [";".join(flags[i]) for i in out.index]
    return out


def write_review_record(mis: pd.DataFrame, ic: pd.DataFrame, meas: pd.DataFrame) -> pd.DataFrame:
    """One row per correction or trace, with the paper reference and review status."""
    rows = []
    ic_ids = set(ic.measurement_id)
    for r in mis.itertuples():
        rows.append(
            {
                "correction_id": f"C-{r.original_row_id}",
                "dataset_id": "calisol23",
                "original_row_id": r.original_row_id,
                "source_doi": DOI,
                "issue": "temperature mislabel" if r.mislabel_rule == "value_matches_other_temperature" else "value copied from another temperature",
                "stored_temperature_K": r.temperature_K,
                "stored_value_mS_cm": r.conductivity_original,
                "paper_fit_at_stored_temperature_mS_cm": round(r.kappa_fit_mS_cm, 4),
                "best_matching_temperature_C": r.best_matching_theta_C,
                "paper_reference": "Ding, Xu & Jow, J. Electrochem. Soc. 152, A132 (2005): Fig. 2 panel (LiBOB)m-PC0.9EC0.1; Table II fitted functions",
                "independent_check": "flagged by internal-consistency check" if r.measurement_id in ic_ids else "not flagged",
                "action": "exclude (rule R13, v0.5 reference)",
                "status": "confirmed",
                "reviewed_by": "automated check (scripts/check_ding2005_libob.py); human review pending",
                "review_date": "2026-10-20",
            }
        )
    tr = meas[(meas.source_doi == DOI) & (meas.original_row_id == "9968")]
    if len(tr):
        rows.append(
            {
                "correction_id": "T-9968",
                "dataset_id": "calisol23",
                "original_row_id": 9968,
                "source_doi": DOI,
                "issue": "near-axis reading stored as 1.78e-15 (suspected numerical artifact, R11)",
                "stored_temperature_K": float(tr.temperature_K.iat[0]),
                "stored_value_mS_cm": float(tr.conductivity_original.iat[0]),
                "paper_fit_at_stored_temperature_mS_cm": 0.0214,
                "best_matching_temperature_C": -39.4,
                "paper_reference": "Fig. 2 panel (LiBOB)m-PC on a linear 0-10 mS/cm axis; Table II at -39.4 C, w = 0",
                "independent_check": "value is 2^-49, a floating-point artifact",
                "action": "exclude (rule R11, since v0.1)",
                "status": "traced",
                "reviewed_by": "automated check; human review pending",
                "review_date": "2026-10-20",
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/processed/v0.3")
