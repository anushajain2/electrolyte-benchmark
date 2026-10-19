# Draft issue for the CALiSol-23 maintainers (not yet submitted)

Proposed target: https://github.com/Pele0599/CALiSol-23/issues (author repository).

---

**Title:** Ten LiBOB rows from 10.1149/1.1833611 appear to carry wrong temperatures

Thank you for compiling and releasing CALiSol-23. While building an evaluation benchmark on it, we checked a sample of rows against their original paper and found ten that appear to be assigned the wrong temperature. We are reporting them with a reproducible check, in case they are useful for a corrected release.

**Source.** Ding, Xu & Jow, "Conductivity and Viscosity of PC-DEC and PC-EC Solutions of LiBOB", *J. Electrochem. Soc.* 152, A132 (2005), DOI 10.1149/1.1833611 (open access at the publisher). Conductivity is given only in Fig. 2 (linear axis) and as fitted functions κ(m, w) per temperature in Table II.

**Rows** (0-based data row index in `CALiSol-23 Dataset.csv` at commit `02334f51d581b45d6760fa02048ffa0d84c056d8`; EC:PC = 10:90 by weight, LiBOB):

| Row | T in CALiSol (K) | m (mol/kg) | k in CALiSol (mS/cm) | Paper's Table II fit at that T | Temperature whose curve matches the value |
| --- | --- | --- | --- | --- | --- |
| 10075 | 302.55 | 0.719 | 0.087 | 3.96 | −39.4 °C |
| 10076 | 312.45 | 0.719 | 0.242 | 5.08 | −29.5 °C |
| 10077 | 322.35 | 0.718 | 0.484 | 6.31 | −19.8 °C |
| 10078 | 332.15 | 0.711 | 2.953 | 7.62 | 19.5 °C (identical to row 10070) |
| 10107 | 282.85 | 1.043 | 0.065 | 1.55 | −29.5 °C |
| 10108 | 292.65 | 1.043 | 0.175 | 2.34 | −19.8 °C |
| 10109 | 302.55 | 1.043 | 0.264 | 3.31 | −19.8 °C |
| 10110 | 312.45 | 1.042 | 0.506 | 4.44 | −10.0 °C |
| 10111 | 322.35 | 1.043 | 0.925 | 5.72 | −0.2 °C |
| 10112 | 332.15 | 1.034 | 3.328 | 7.14 | 29.4 °C (identical to row 10103) |

For comparison, the neighbouring rows of the same series at the same temperatures (e.g. rows 10071–10074, 10103–10106) agree with the paper's fit to within about 1%.

**Independent check without the fitted equation.** Conductivity rises with temperature at fixed composition. All ten rows are lower than rows of the same composition measured at colder temperatures, or are exact copies of another temperature's value.

**Reproduce.** `scripts/check_ding2005_libob.py` in https://github.com/anushajain2/electrolyte-benchmark (coefficients transcribed in `data/evidence/ding2005_libob_pc_ec_tableII.csv`).

**Also noticed:**
- Row 9968 stores 1.78e-15 (= 2⁻⁴⁹) for a point the paper's fit puts at about 0.02 mS/cm. This is likely a reading at the axis of the linear plot.
- Several low-temperature values of this paper are slightly negative, for the same reason.
- Below about 0.2 mS/cm, values read off this paper's linear axis differ from its fit by a median factor of 1.4–3.7. This is expected for digitization and is not an error in the dataset.

We have not modified CALiSol; our benchmark excludes the ten rows and documents the reason.
