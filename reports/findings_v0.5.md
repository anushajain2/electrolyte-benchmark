## Findings (v0.5, reference)

> v0.5 is the reference release. It is v0.3 plus **confirmed corrections only**: 10 CALiSol rows whose temperatures were checked against the original paper (rule R13). The 0.2 mS/cm digitization floor is a **sensitivity assumption**, run separately for the verified source and for verified plus inferred sources. Like v0.2–v0.4, v0.5 was made after earlier test results had been seen. All numbers are from the primary track. Errors are source-group-weighted MAE of log10(conductivity / mS cm⁻¹); intervals come from a paired bootstrap that resamples source groups and are conditional on that grouping.

### 1. Label reliability: a small, checkable correction rather than a broad cleaning rule

**Confirmed corrections (R13; `data/evidence/corrections_review_record.csv`).** Ten rows of Ding, Xu & Jow, *J. Electrochem. Soc.* 152, A132 (2005) carry wrong temperatures. Each row has a review record: figure and table reference, stored value, the paper's fitted value at the stored temperature, and the temperature whose curve matches the value. Two independent checks both identify exactly these ten rows:
- the paper's own fitted functions (Table II, PC-EC half);
- internal consistency, which needs no fit: conductivity must rise with temperature at fixed composition, and exact copies of another temperature's value stand out.

**PC-DEC half of the paper.**
- **The printed fit table can't be used.** The paper's Table I has "-" entries that are lost numbers, not zeros: its own temperature column prints −10.0 to −39.4 °C as "-". Read with blanks as zero, it predicts 17.7 mS/cm for pure PC at 59 °C, while the paper's Fig. 2 shows about 7.3. Table I is transcribed as printed and marked unusable (`data/evidence/ding2005_libob_pc_dec_tableI_as_printed.csv`).
- **The internal-consistency check finds no mislabels** in the PC-DEC half. The same check recovers all 10 confirmed PC-EC errors, with one false alarm at a very low value.

**Upstream report.** A reproducible issue for the CALiSol maintainers is drafted in `reports/evidence/calisol_upstream_report.md`. It has not been submitted.

**Digitization floor, by evidence tier (R14; sensitivity only).**

| Run | temp_low: extra trees / Arrhenius / ridge | source transfer: extra trees | interpolation: extra trees | primary contrast |
| --- | --- | --- | --- | --- |
| v0.5 reference (confirmed corrections only) | 0.536 / 0.229 / 0.249 | 0.075 | 0.047 | 0.036 [0.007, 0.064] |
| + floor on the verified source | 0.514 / 0.216 / 0.230 | 0.049 | 0.046 | 0.021 [0.010, 0.033] |
| + floor on verified and inferred sources | 0.511 / 0.215 / 0.224 | 0.048 | 0.042 | 0.020 [0.007, 0.033] |

Almost all of the floor's effect comes from the one verified source. Adding the seven inferred sources changes little.

### 2. Better labels versus an easier test set

The two comparisons keep the reference version's fold membership fixed (`results/v0.5/investigation/label_*.csv`):
- **Evaluation filtering.** Predictions are held fixed and rescored on the cases the floor would keep.
- **Training cleaning.** The same outer folds are refit with and without the floored training rows, and both are scored on identical audited test cases.

| Task (extra trees, floor 0.2 mS/cm) | Change from evaluation filtering | Change from training cleaning | Test coverage retained |
| --- | --- | --- | --- |
| source transfer, verified tier | −0.016 | −0.012 | 94% |
| source transfer, verified + inferred | −0.016 | −0.009 | 94% |
| temp_low, verified + inferred | −0.024 | +0.001 (13 training rows removed) | 89% |
| interpolation, verified + inferred | −0.003 | −0.001 | 98% |
| conc_high, verified + inferred | −0.010 | +0.000 | 93% |

- **v0.4's interpretation needs splitting.** v0.4 said source-transfer error falling from 0.074 to 0.048 meant some apparent transfer error was label noise. About half of that drop is cleaner training labels (−0.009 to −0.012 on identical cases). The other half comes from evaluating an easier set of cases (−0.016).
- **For cold extrapolation, the improvement is entirely an easier test set.** At 0.5 mS/cm the verified-and-inferred floor would keep only 64% of cold test cases and remove 225 formulations entirely, too large a population change to call cleaning.

### 3. Cold-temperature extrapolation: temperature-aware baselines outperform extra trees

| Model | temp_low (6 groups scored, 5 meet the headline rule) | Difference vs extra trees |
| --- | --- | --- |
| per-formulation Arrhenius | 0.229 | −0.306 [−0.399, −0.226] |
| ridge (log target) | 0.249 | −0.287 [−0.416, −0.198] |
| gradient boosting | 0.499 | −0.037 [−0.107, 0.001] |
| extra trees | 0.536 | n/a |

- **All 6 groups agree.** Both temperature-aware baselines beat extra trees in each of the 6 groups; Arrhenius is lowest in 4 and ridge in 2.
- **Removing any one source group and refitting changes little:** Arrhenius 0.15–0.27, ridge 0.20–0.28, extra trees 0.39–0.59.
- **The Arrhenius fit has a data requirement.** It needs at least three warm-side measurements of the same formulation.
- **The result holds in every floor variant above.**

### 4. Selective prediction for cold extrapolation (exploratory)

These results are exploratory because every test label here was inspected in earlier versions (`results/v0.5/investigation/selective_*.csv`).

**Group-mean MAE log10 when the least confident predictions are rejected:**

| Base model / signal | 100% | 75% | 50% |
| --- | --- | --- | --- |
| Arrhenius / distance below coldest training temperature | 0.229 | 0.168 | 0.140 |
| Arrhenius / random | 0.229 | 0.230 | 0.231 |
| Arrhenius / oracle (bound) | 0.229 | 0.150 | 0.114 |
| extra trees / distance below coldest training temperature | 0.536 | 0.444 | 0.385 |
| extra trees / ensemble spread across trees | 0.536 | 0.569 | 0.658 |
| extra trees / random | 0.536 | 0.536 | 0.536 |

- **The distance rule works.** A simple distance-to-training-range rule captures most of what an oracle could gain for the Arrhenius baseline.
- **Ensemble disagreement is worse than random.** Below the coldest training point the trees agree with each other while all being wrong, so their spread does not flag the failures that matter.
- **Thresholds fixed on the development fold transfer poorly.** The development boundary sits only 10 K below its training data, so a 75% coverage target became 98% on development and 71% on test (distance rule), or 99% on test (ensemble spread).
- **Rejection concentrates on one group and the coldest temperatures.** It falls almost entirely on cases at or below 233 K (100% rejected) and on one source group (`S0378-7753(03)00153-8`, 41% rejected). That limited usable domain should be visible to anyone deploying the rule.

### 5. Scoring policy for external models

`score_predictions` has two explicit modes:
- **`full_coverage` (default).** Any abstention or failure leaves the submission with diagnostics only, the same rule applied to baselines.
- **`selective`.** Intentional abstentions plus a `confidence` column give an error-versus-coverage table. Execution failures are reported separately and block a ranked comparison. Comparisons on the submission's retained cases are labelled as favouring the submission.

### 6. Workbench

`uv run elbench workbench` runs a local Streamlit app built around one question: *given the measurements I already have, which modeling approach deserves further validation?* It starts with cold-temperature prediction for formulations with warm measurements. It covers:
- what the benchmark can establish for each setting, including its data requirements;
- published baselines with per-group and leave-one-group-out results;
- upload and validation of predictions, with ID checks, a unit check and both scoring modes;
- provenance and label-quality inspection;
- a Markdown export.

Every baseline within 0.05 log10 (about 12%) of the lowest error is shortlisted. This is a practical shortlist rule, not a statistical equivalence test. It runs locally because scoring needs CALiSol labels, which are not redistributed. A hosted version could show only aggregate results. It has not yet been tested with researchers.

### 7. Temperature-representation ablation (added in v0.5.1)

Each model is refit on the same outer folds, with the configurations chosen for the full feature set and one seed, after removing 1/T, T, or both (`results/v0.5/investigation/temperature_ablation.csv`).

| Task | Model | all features | T only | 1/T only | no temperature |
| --- | --- | --- | --- | --- | --- |
| temp_low | extra trees | 0.536 | 0.540 | 0.539 | 0.747 |
| temp_low | ridge | 0.249 | 0.334 | 0.267 | 0.845 |
| temp_high | extra trees | 0.114 | 0.121 | 0.120 | 0.465 |
| temp_high | ridge | 0.195 | 0.514 | 0.351 | 0.568 |
| interp_formulation | extra trees | 0.047 | 0.062 | 0.062 | 0.340 |
| interp_formulation | ridge | 0.155 | 0.199 | 0.183 | 0.325 |

- **The cold-extrapolation reversal is mainly a model-family effect.** Extra trees barely changes between T and 1/T, because trees predict a constant beyond the training range of any feature. Ridge with T alone (0.334) still beats extra trees with every feature (0.536).
- **1/T strengthens the linear model.** Under cold extrapolation, ridge with 1/T alone (0.267) is close to its full-feature error.
- **Ridge needs both columns above the training range.** Under warm extrapolation, dropping either column more than doubles its error: together, T and 1/T let a linear model bend.
- **These are temperature tasks.** Removing temperature makes every model fail.

This is a diagnostic, not a retuned comparison. `METHODS.md` explains the inputs and the Arrhenius baseline's limits.

### Carried over

The conclusions on provenance and grouping (3 author groups, 7 source groups, 4 testable in source transfer), the component-transfer limits and the ridge descriptor diagnosis are unchanged from v0.3; see `findings_v0.3.md`. External numbers are from the v0.3 post-correction rerun, and the reserved data were not rescored.
