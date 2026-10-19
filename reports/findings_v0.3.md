## Findings (v0.3)

> v0.3 is a correction release. A code review of v0.2 found implementation defects. v0.3 fixes them, reruns every affected experiment, and keeps v0.1 and v0.2 as historical versions (git tag `v0.2`; earlier results in `results/`). Like v0.2, it was made after earlier test results had been seen, so it is a revision, not a blind re-evaluation. Reserved labels were scored in v0.1, v0.2 and again here (post-correction rerun).

Working title: *What can public electrolyte conductivity data tell us about generalization? A coverage-aware benchmark and audit.*

**Scope of the numbers.**
- All numbers refer to the primary track (`molal_w`: molality with mass-fraction recipes, 10,481 CALiSol measurements) unless stated otherwise.
- Errors are **source-group-weighted**: MAE of log10(conductivity / mS cm⁻¹) is averaged within each source group (a provenance-based grouping of papers), then equally across groups.
- Brackets are 95% intervals from a paired bootstrap that resamples those groups. They are conditional on the grouping, not population-level uncertainty across laboratories.
- With few groups, per-group results and removal checks carry more weight than any interval.
- 0.30 log10 is roughly a factor of two for one prediction.

### What v0.3 corrected, and whether the conclusions survive

| Defect in v0.2 | Correction | Effect |
| --- | --- | --- |
| Temperature and concentration boundaries compared values rounded to 0.1, so 313.15 K and 253.15 K fell on the wrong side and concentrations up to 1.249 or down to 1.45 mol/kg crossed the declared buffer | Boundaries applied to reported values; a leakage check now asserts the declared boundaries | temp_high training +161 rows, test +17; temp_low test +74; conc_high training −267, test −239. Temperature conclusions unchanged; numbers below are v0.3 |
| Submission comparison could score models on different cases while calling the comparison paired | One common set of (fold, measurement) cases; incomplete baselines left out and listed | Affects external users of `score_predictions`; published tables already used common rows |
| A model with invalid predictions could appear in rankings with an MAE over its valid half | Models with any invalid prediction in a subset are reported but excluded from comparisons and rankings | Only the diagnostic raw-scale ridge is affected (invalid on 2–78% of rows); it was never ranked in the findings |
| `verify` passed when a metric became missing | Missingness, finiteness, duplicate keys and counts are checked; `elbench reproduce` rebuilds from raw files in a clean copy | v0.2 and v0.3 both reproduce |
| Training cache ignored data values | Cache key includes a hash of features and targets | No published result affected |
| k-NN could change under row reordering (distance ties among replicates) | Replicate inputs merged; direct distances; fixed tie order | Predictions identical across row orders |

**What survives the corrections.**
- **Temperature reversal:** survives in every version and variant.
- **Primary source-familiarity contrast:** unchanged at 0.034 (it does not use the boundary tasks).
- **Concentration transfer:** the error rises (extra trees 0.171 → 0.198) once the declared 0.3 mol/kg buffer is enforced. It remains a case study.

### 1. Apparent breadth overstates the number of testable transfer settings

The 27 CALiSol papers shrink in steps:

| Step | Papers or groups left | Cause |
| --- | --- | --- |
| Curation | 26 papers | Exclusions |
| Primary-track eligibility | **12 papers** | Concentration and ratio basis |
| Provenance grouping | 10 groups under the v0.1 rule; **7** under v0.2/v0.3 campaign agreement | Linking papers that appear to share measurements |
| Common support when held out | **4** source groups | Only these keep rows inside common support |

**Provenance tiers** (`reports/v0.3/provenance_pairs.csv`).
- **Documented reuse:** none yet.
- **Suspected reuse:** 7 pairs with direct numerical evidence.
- **Conservative grouping:** 4 pairs linked only through the 2003 overview paper. This merging is a leakage safeguard, not proof of one experimental campaign.
- **Alternatives:** the v0.1 rule, the campaign rule and author-group holdouts are alternative provenance-based grouping schemes, not statistical bounds. The campaign thresholds were chosen after seeing agreement rates.

**Author groups.** The 12 papers fall into 3 author groups, and one (Ding/Jow/Xu) supplies 10,108 of 10,481 rows. Holding out a whole author group leaves 2 testable units.

**Components.** All 9 eligible component holdouts are combined shifts.

**What qualifies as headline.** These comparisons meet the predeclared headline rule (≥5 groups with ≥10 formulations each):
- grouped interpolation (7 groups scored, 7 qualifying);
- warm extrapolation (5 scored, 5 qualifying);
- cold extrapolation (**6 scored, 5 qualifying**: `10.1016/S0378-7753(01)00697-8` contributes only 6 formulations).

Source transfer and concentration transfer are case studies.

### 2. Temperature-aware baselines outperform extra trees under cold-temperature extrapolation

All models are scored on the same 1,787 cold-side cases (6 groups scored). Differences are paired against extra trees.

| Model | interpolation | temp_high | temp_low | temp_low: difference vs extra trees |
| --- | --- | --- | --- | --- |
| extra trees | 0.049 [0.037, 0.063] | 0.115 | 0.531 | n/a |
| per-formulation Arrhenius | n/a | 0.160 | 0.229 | −0.302 [−0.394, −0.223] |
| ridge (log target) | 0.156 | 0.197 | 0.247 | −0.284 [−0.412, −0.196] |
| gradient boosting | 0.057 | 0.122 | 0.503 | −0.028 [−0.066, −0.001] |
| k-NN | 0.067 | 0.122 | 0.499 | −0.033 [−0.100, 0.001] |

- **Both temperature-aware baselines beat extra trees in every one of the 6 groups.** Arrhenius is lowest in 4 groups and ridge in 2, so neither is uniquely best.
- **Removing any one source group and refitting does not change the ranking.** Arrhenius stays at 0.15–0.27, ridge at 0.20–0.28, and extra trees at 0.39–0.58.
- **The Arrhenius baseline's information requirement.** It fits each test formulation's own warm-side measurements (at least three, guaranteed by the task's eligibility rule). The other models see the same training rows, so it uses no extra information, but it cannot be applied to a formulation with no warm-side data.
- **What the evidence supports.** The reversal persists across source groups and at matched extrapolation distances. This does not fully separate distance, chemistry and source.
- **The physics is documented, not discovered here.** The coldest data come from a paper that measured glass-transition temperatures (`10.1021/JE020219O`). The finding is that tree ensembles flatten below the coldest training point. (v0.3 originally added that every model class is biased high there; v0.4 shows that ridge is not once labels below the digitization floor are removed, see `findings_v0.4.md` and I-20.)
- **Above the training range** extra trees is best or tied (ridge +0.082 [0.065, 0.110], Arrhenius +0.045 [−0.005, 0.123]).

### 3. Matching coverage changes the interpretation of source transfer

- **Same rows, different training.** On identical rows, holding out a source costs extra trees far less than holding out a component or extrapolating in temperature.
- **Primary contrast, predeclared in v0.1.** Equal-size training sets with and without the target source's other formulations, scored on jointly supported rows. Four target source groups qualify:

| Target source group | Effect on jointly supported rows | Same-system replacement |
| --- | --- | --- |
| `10.1149/1.1833611` | +0.075 | 5% |
| `10.1149/1.1914757` | +0.046 | 10% |
| `10.1149/1.1403730` | +0.010 | 29% |
| `10.1149/2.0271802jes` | +0.004 | 22% |

**Reading the four effects.**
- Two are near zero and two are moderate.
- Removing any one source group and refitting moves the mean between 0.009 and 0.060.
- Across support definitions it stays between 0.026 and 0.034.
- The mean (0.034) is reported for completeness; with four groups its interval carries little of the argument.
- No target had chemistry-matched replacement data for more than 29% of rows, so local data density remains an alternative explanation.

**Further cautions.**
- **The unrestricted contrast mixes two things.** The estimated benefit is substantially smaller on the jointly supported subset (0.034) than on all test rows (0.146, six targets). The unrestricted contrast mixes source familiarity with coverage differences; the two numbers are computed on different populations.
- **The external case study did not provide convincing confirmation:** +0.008 for extra trees on 36 jointly supported rows.

### 4. Narrow external transfer (v0.3 post-correction rerun)

These are temperature sweeps of a few formulations, not thousands of independent validation experiments.

| Subset | Chemistry system | Rows | Formulations | Temperature sweeps | Extra trees | Ridge |
| --- | --- | --- | --- | --- | --- | --- |
| common support | LiPF6, EC+EMC+PC | 153 | 5 | 17 | 0.039 | 0.094 |
| outside support | LiPF6, EC+EMC+PC | 4,432 | 94 | 459 | 0.088 | 0.194 |
| outside support | LiPF6, EMC+PC | 450 | 10 | 45 | 0.130 | 0.177 |

The supported score is a 5-formulation case study within one chemistry system.

### 5. Sensitivity to consequential exclusions

| Run | temp_low: extra trees / Arrhenius / ridge | temp_high: extra trees / Arrhenius / ridge / k-NN | Primary contrast |
| --- | --- | --- | --- |
| v0.3 | 0.531 / 0.229 / 0.247 (6 groups) | 0.115 / 0.160 / 0.197 / 0.122 (5) | 0.034 (4 targets) |
| quarantined source included (I-01) | 0.649 / 0.220 / 0.267 (7) | 0.211 / 0.155 / 0.358 / 0.113 (6) | 0.043 |
| R11 disabled (suspected artifacts kept) | 0.538 / 0.236 / 0.254 (6) | 0.118 / 0.164 / 0.198 / 0.122 (5) | 0.044 |
| v0.1 source linking | 0.581 / 0.268 / 0.242 (9) | 0.119 / 0.183 / 0.213 / 0.127 (7) | 0.024 (8 targets) |

- **The cold-side reversal holds in every run.**
- **Including the quarantined paper changes the warm-side ranking.** Its values look about 100× too low. It adds one small group with 2 warm-side and 5 cold-side test rows, but group weighting gives that group 1/6 to 1/7 of the score: extra trees errs by 0.68 and 1.30 log10 on it, while k-NN, which finds the paper's own warm points, errs by 0.07. That is a consequence of the unresolved unit question, not evidence about warm extrapolation.
- **R11 moves the primary contrast by about +0.01** under v0.3 grouping (it moved it by 0.017 under v0.1 grouping). The direction is unchanged.

### 6. A descriptor-extrapolation diagnosis (component holdout)

Ridge's frozen component-transfer error is 5.50 (TFP: 43.6 log10). The audit traces it to the mole-weighted variance of molecular weight, which places TFP mixtures on average 184 training standard deviations away. Clipping features to the training range, as a post-hoc diagnostic, gives 1.2 log10, still about a 16-fold error. This concerns this representation and fitted model, not linear models in general.

### Data issues still open

- **Suspected numerical artifacts (R11)** are not yet traced to the original figures (I-15).
- **Compilation error, confirmed in v0.4:** `10.1149/1.1833611` rows 10075–10078 and 10107–10112 carry wrong temperatures (I-11).
- **Quarantined source** with a suspected unit error (I-01); the sensitivity run above shows what including it does.
- **Reported companion-paper statement** for `JE020219O` is unverified (I-17).

Checking these against the original papers is the most valuable remaining step.

### Practical consequences

- Count provenance-based groups and author groups, not papers or rows, before claiming source transfer.
- Under cold-temperature extrapolation, prefer temperature-aware baselines to tree ensembles. Treat labels below ~0.2 mS/cm read from linear-axis figures as unreliable (v0.4).
- Treat "generalization to new solvents or salts" on CALiSol-derived data as a combined chemistry and source result.
- Check very small conductivities, and suspected unit errors, before log-scale training. Report how the main results move when they are included.

### Positioning

This is a careful empirical audit with a reusable evaluation package. Its central findings are the cold-extrapolation reversal and the limits of identifiable transfer in public conductivity data. Its supporting material is:
- the provenance tiers and grouping decisions;
- exact split and support definitions;
- version-to-version comparisons, including this correction release;
- per-group results and traced failure cases.

No "first benchmark" claim is made.

### Phase 2 decision

The gate conditions remain met through the temperature tasks (5 qualifying groups each), not through source transfer. The recommended next experiment is selective prediction for condition extrapolation, compared with a distance-to-training-range rejection rule at matched retained coverage. A molecular representation model is not yet justified.
