## Findings (v0.4, label-reliability sensitivity release)

> v0.4 adds two curation rules derived from reading an original source paper, and reruns the benchmark. Both rules were added after v0.3 results had been seen, and the presentation of most sources is inferred rather than verified. **v0.3 remains the reference benchmark**; v0.4 shows how its conclusions move when known and likely label errors are removed. The reserved external data were not rescored.

All numbers refer to the primary track. Errors are source-group-weighted MAE of log10(conductivity / mS cm⁻¹). Intervals come from a paired bootstrap that resamples source groups, so they are conditional on that grouping.

### What reading the original paper showed

Of the five papers needed for open data issues, only one was freely readable: Ding, Xu & Jow, *J. Electrochem. Soc.* 152, A132 (2005), `10.1149/1.1833611` (LiBOB in PC-DEC and PC-EC).

**How the data were presented.** The paper gives conductivity only in figures, plotted on a linear 0–10 mS/cm axis at 11 temperatures from 59.0 to −39.4 °C. CALiSol digitized those figures. The paper also publishes fitted functions κ(m, w) at each temperature (Table II), which makes the digitized values checkable (`scripts/check_ding2005_libob.py`, `reports/evidence/`).

**Agreement between CALiSol and the paper's fit, by conductivity level (1,139 PC-EC rows).**

| Fit value (mS/cm) | Rows | Median absolute difference (log10) | 90th percentile |
| --- | --- | --- | --- |
| > 2 | 628 | 0.006 | 0.018 |
| 1–2 | 199 | 0.011 | 0.038 |
| 0.5–1 | 112 | 0.029 | 0.085 |
| 0.2–0.5 | 94 | 0.064 | 0.219 |
| 0.1–0.2 | 62 | 0.143 | 0.435 |
| 0.05–0.1 | 15 | 0.379 | 0.554 |
| < 0.05 | 13 | 0.565 | 1.077 |

Below about 0.2 mS/cm, values read off a linear axis carry factor-of-1.4-to-3.7 noise. The direction is mixed, and at these levels the fit itself is also less certain.

**Ten rows carry wrong temperatures (now confirmed; issue I-11).**
- Rows 10075–10077 and 10107–10111 match the −39.4 to −0.2 °C curves but are labeled 283–322 K, where the paper's fit gives 1.6–6.3 mS/cm.
- Rows 10078 and 10112 are exact copies of the 19.5 °C and 29.4 °C values, assigned to 332 K.

**The 1.78e-15 value (R11) is a point read at the axis.** The paper's fit puts it near 0.021 mS/cm. The same mechanism explains the paper's 20 negative values. It is a digitization artifact, not a reported zero.

### v0.4 rules

- **R13.** Exclude the 10 confirmed rows.
- **R14.** Exclude values below 0.2 mS/cm from sources digitized from linear axes: 269 values across 8 sources. The source classes are in `data/mappings/source_presentation.csv`:
  - 1 verified (the LiBOB paper);
  - 2 inferred from that paper's statement that it parallels them "in both structure and content";
  - 5 inferred because they contain zero or negative digitized values, which a log axis or a table cannot produce.
- **Not floored.** `10.1021/JE020219O` reports values down to ~1e-6 mS/cm, so it cannot have been read off a linear axis.

### How the conclusions move

| | v0.3 (reference) | v0.4, R13 only | v0.4 (R13 + R14) |
| --- | --- | --- | --- |
| temp_low: extra trees / Arrhenius / ridge | 0.531 / 0.229 / 0.247 | 0.536 / 0.229 / 0.249 | 0.511 / 0.215 / 0.224 |
| temp_low: ridge minus extra trees | −0.284 [−0.412, −0.196] | n/a | −0.287 [−0.419, −0.201] |
| temp_low: Arrhenius minus extra trees | −0.302 [−0.394, −0.223] | n/a | −0.296 [−0.393, −0.216] |
| temp_high: extra trees / Arrhenius / ridge | 0.115 / 0.160 / 0.197 | 0.114 / 0.159 / 0.195 | 0.117 / 0.150 / 0.195 |
| interpolation: extra trees | 0.049 | 0.047 | 0.042 |
| source transfer: extra trees (4 groups) | 0.074 | 0.075 | 0.048 |
| primary contrast (4 targets) | 0.034 [0.007, 0.061] | 0.036 | 0.020 [0.007, 0.033] |

- **The cold-extrapolation result is unchanged.** Both temperature-aware baselines beat extra trees in all 6 groups. Removing any one source group and refitting leaves Arrhenius at 0.13–0.25, ridge at 0.17–0.25 and extra trees at 0.36–0.56.
- **The low-temperature bias claim needed revising (I-20).** v0.3 said every model is biased high under cold extrapolation. With unreliable low labels removed, ridge's mean bias is small (+0.05 log10 averaged over groups; −0.02 to −0.09 when restricted to labels ≥ 0.2 or 0.5 mS/cm in v0.3). Tree ensembles (+0.51) and the per-formulation Arrhenius fit (+0.21) remain biased high.
- **The source-familiarity effect shrinks but keeps its sign.** It drops from 0.034 to 0.020 and stays positive for all four targets. Part of the v0.3 estimate came from low-conductivity rows whose labels are least reliable.
- **Source-transfer error drops by a third** (0.074 to 0.048) when unreliable low labels are removed. Some apparent transfer error was label noise.
- **Ridge's component-transfer error is unstable** (5.50, 7.10, 4.61 across the three runs). Small data changes alter the chosen regularization, and the TFP descriptor extrapolation dominates. This reinforces that the 43-order figure is a diagnostic, not a stable benchmark number.

### What remains open

- **Paywalled papers.** Four papers could not be read: `S0378-7753(03)00153-8` (two R11 rows; the overview linking four Ding papers), `10.1007/s10008-002-0300-9` (quarantined for a suspected unit error), `10.1021/je020219o` (reported companion-paper statement), and `10.1016/0378-7753(91)80004-H` (disputed solvent names).
- **Inferred presentation.** The presentation of seven floored sources is inferred, not verified.
- **Unchecked half of the LiBOB paper.** Only its PC-EC data were checked; the PC-DEC half was not.
- **Low values in the cold-extrapolation task.** `JE020219O`, on a log axis or in tables, still contributes many very low labels there; they are physically plausible but were not checked against the paper.
