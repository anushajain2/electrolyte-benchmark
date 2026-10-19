## Findings (v0.2, report revision 2)

> v0.2 revises v0.1 after external review and after v0.1 test results had been seen, so it is a revision, not a blind re-evaluation. Earlier versions are kept and shown side by side: `v0.1 pre-R11` (the original run, regenerated), `v0.1`, and `v0.2 with v0.1 source linking`. Reserved labels were first scored in v0.1; v0.2 external numbers are a post-correction rerun. Report revision 2 changes wording and adds descriptive tables only; no rows, groups, splits or scores changed.

Working title: *What can public electrolyte conductivity data tell us about generalization? A coverage-aware benchmark and audit.*

**Scope of the numbers.**
- All numbers refer to the primary track (`molal_w`: molality with mass-fraction recipes, 10,481 CALiSol measurements) unless stated otherwise.
- Errors are **source-group-weighted**: MAE of log10(conductivity / mS cm⁻¹) is averaged within each source group (a provenance-based grouping of papers), then equally across groups.
- Brackets are 95% intervals from a paired bootstrap that resamples those groups. They describe variation under the chosen grouping. Groups may still share equipment, calibration practice or protocols, so the intervals are not population-level uncertainty across laboratories.
- With few groups, the per-group results and removal checks carry more weight than any interval.
- 0.30 log10 corresponds to roughly a factor of two for a single prediction.

### 1. Apparent breadth overstates the number of testable transfer settings

The 27 CALiSol papers shrink in four distinct steps (report table *From papers to testable units*):

| Step | Papers or groups left | Cause |
| --- | --- | --- |
| Curation | 26 papers | Exclusions |
| Primary-track eligibility | **12 papers** | Concentration and ratio basis (molality with mass fractions) |
| Provenance grouping | 10 groups under the v0.1 rule (≥3 near-identical points); **7** under v0.2 campaign agreement | Linking papers that appear to share measurements |
| Common support when held out | **4** source groups | Only these keep rows inside common support |

Most of the reduction from 27 to 12 is basis eligibility, not deduplication.

**Provenance-based grouping, in three evidence tiers** (report table *Provenance-based grouping*):

- **Documented reuse.** None verified yet. A reviewer reports that `10.1021/JE020219O` states it shares measurements with a companion paper; this stays unverified until checked against the text.
- **Suspected reuse.** 7 paper pairs with direct numerical evidence. Examples: `10.1021/je0498863` and `10.1149/1.1630593` agree within 5% on 723 of 1,022 comparable points; the 2003 overview `10.1016/S0378-7753(03)00153-8` agrees with four Ding papers on 54–80% of comparable points.
- **Conservative grouping.** 4 pairs have no direct evidence and share a group only through the overview paper (e.g. `JE020219O` and `je0498863`). Merging the whole connected component is a leakage safeguard. It does not establish that these papers form one experimental campaign; the overview may reproduce several distinct experiments.

**Cautions on the grouping evidence.**
- Numerical agreement is evidence of possible reuse, not proof. Independent measurements can agree, and reused measurements can differ after digitization.
- The v0.1 rule, the v0.2 campaign rule and the author-group holdout are alternative provenance-based grouping schemes. They are not statistically established bounds.
- The v0.2 thresholds were chosen after seeing the agreement rates.

**Author groups.** By Crossref author lists, the 12 primary-track papers fall into 3 author groups, and one group (Ding/Jow/Xu) supplies 10,108 of the 10,481 rows. Holding out a whole author group leaves 2 testable units (26 and 11 formulations). Affiliations are not verified; this is a reviewed author grouping, not a lab identifier. Seven source groups drawn largely from three author groups are not seven statistically independent sources.

**What else limits testing.**
- **Components.** All 9 eligible component holdouts remain `combined_shift`: each component occurs in one source group, or most of its rows come from sources then absent from training.
- **Molarity tracks and reserved data.** These support case studies only.

Comparisons with enough groups for the predeclared headline rule (5 or more groups with at least 10 formulations each): grouped interpolation (7 groups) and temperature extrapolation (6 cold, 5 warm). Source transfer is a case study in v0.2.

### 2. Temperature-aware baselines outperform extra trees under cold-temperature extrapolation

All models are scored on exactly the same 1,713 cold-side rows (6 groups). Differences are paired against extra trees:

| Model | interpolation (7) | temp_high (5) | temp_low (6) | temp_low: difference vs extra trees |
| --- | --- | --- | --- | --- |
| extra trees | 0.049 [0.037, 0.063] | 0.138 [0.123, 0.152] | 0.589 [0.436, 0.812] | n/a |
| per-formulation Arrhenius | n/a | 0.177 | 0.248 [0.125, 0.418] | −0.341 [−0.431, −0.258] |
| ridge (log target) | 0.156 | 0.205 | 0.257 [0.170, 0.378] | −0.332 [−0.442, −0.255] |
| gradient boosting | 0.057 | 0.145 | 0.546 | −0.043 [−0.126, 0.003] |

- **Both temperature-aware baselines beat extra trees in every one of the 6 groups.** Arrhenius is lowest in 4 groups and ridge in 2. Neither is uniquely best.
- **The result is stable when any one group is removed.** With each source group removed from training and testing and the models refit, ridge stays at 0.21–0.29 and extra trees at 0.47–0.63.
- **The Arrhenius baseline's information requirement.** It fits each test formulation's own warm-side measurements (at least three training temperatures, guaranteed for every test formulation by the task's eligibility rule). The other models see the same training rows, so it uses no extra information, but it cannot be applied to a formulation that has no warm-side measurements.
- **The reversal persists across source groups and at matched extrapolation distances.** This does not fully separate distance, chemistry and source.
- **The physics is documented, not discovered here.** The coldest data come from a paper that also measured glass-transition temperatures (`10.1021/JE020219O`, per its title). The evaluation finding is that tree ensembles flatten below the coldest training point and every model class tested is biased high there.

Above the training range, extra trees remains best or tied (temp_high: ridge +0.067 [0.030, 0.107], Arrhenius +0.039 [−0.002, 0.115] relative to extra trees).

### 3. Matching coverage changes the interpretation of source transfer

- **Same rows, different training.** On identical rows, holding out a source costs extra trees far less than holding out a component or extrapolating in temperature (v0.1: 0.028 vs 0.273 and 0.543 log10).
- **Primary contrast, predeclared in v0.1.** Two equal-size training sets, with and without the target source's other formulations, are scored on rows both sets cover. In v0.2 only four target source groups qualify.

| Target source group | Effect on jointly supported rows | Same-system replacement |
| --- | --- | --- |
| `10.1149/1.1833611` | +0.075 | 5% |
| `10.1149/1.1914757` | +0.046 | 10% |
| `10.1149/1.1403730` | +0.010 | 29% |
| `10.1149/2.0271802jes` | +0.004 | 22% |

**Reading the four effects.**
- Two are near zero and two are moderate.
- Removing any one source group from the data and refitting moves the mean between 0.009 and 0.060.
- Across support definitions it stays between 0.026 and 0.034.
- The mean (0.034) is shown for completeness; with four groups its interval should carry little of the argument.
- In none of the four targets could the replacement rows match the chemistry system for more than 29% of rows, so local data density remains an alternative explanation.

**How the estimate moved across versions.**

| Version | Targets | Mean effect (extra trees) |
| --- | --- | --- |
| v0.1 pre-R11 (original) | 8 | 0.047 |
| v0.1 (suspected numerical artifacts excluded) | 8 | 0.030 |
| v0.2 with v0.1 linking (local support) | 8 | 0.024 |
| v0.2 (campaign linking) | 4 | 0.034 |

**Further cautions.**
- **The unrestricted contrast mixes two things.** The estimated benefit is substantially smaller on the jointly supported subset (0.034) than on all test rows (0.146 over six targets). The unrestricted contrast mixes source familiarity with coverage differences. The two numbers are computed on different populations, so their gap does not measure how much of the effect is due to coverage.
- **The external case study did not provide convincing confirmation.** The estimate was +0.008 for extra trees on 36 jointly supported rows of the reserved dataset, too few to be informative either way.
- **Local support matters under the v0.1 grouping.** With v0.1 source groups, the marginal support definition admitted 875 of 4,344 source-transfer test rows (20%, 834 of them from `10.1021/JE020219O`). These rows had no training point of a nearby recipe within 10 K and 0.25 mol/kg, and extra trees erred more on them (0.161 vs 0.104 on locally supported rows). Under v0.2 grouping, local and marginal support select the same 552 rows.
- **Which units remain testable drives the v0.2 source-transfer score.** In v0.1, the four units v0.2 can still test averaged 0.081, while the four merged into the Ding/Jow group (now untestable) averaged 0.118. Transfer did not become easier.

### 4. Narrow external transfer

Trained on CALiSol only, extra trees was scored on the reserved automated-laboratory data (v0.2 post-correction rerun). These are temperature sweeps of a small number of formulations, not thousands of independent validation experiments.

| Subset | Chemistry system | Rows | Formulations | Temperature sweeps | Extra trees | Ridge |
| --- | --- | --- | --- | --- | --- | --- |
| common support | LiPF6, EC+EMC+PC | 153 | 5 | 17 | 0.039 | 0.094 |
| outside support | LiPF6, EC+EMC+PC | 4,432 | 94 | 459 | 0.088 | 0.194 |
| outside support | LiPF6, EMC+PC | 450 | 10 | 45 | 0.130 | 0.177 |

- **The supported score is a 5-formulation case study.** It shows transfer within one chemistry system covered by a single CALiSol paper.
- **The outside-support rows are a different population.** They add composition and condition shift and should be read separately. The pooled figure (0.108; v0.1: 0.119) is not a summary of either subset.

### 5. A descriptor-extrapolation diagnosis (component holdout)

Ridge's frozen component-transfer error is 5.50 (TFP holdout: 43.6 log10). The audit traces it to this representation:
- Scaling is fitted on training rows only.
- The mole-weighted variance of molecular weight places TFP-containing mixtures on average 184 training standard deviations from the training mean, contributing −118 log units.
- As a post-hoc diagnostic, clipping features to the training range gives 1.2 log10, still about a 16-fold error.

This is a lesson about descriptor extrapolation in this feature set and fitted model, not a general statement about linear models. Clipping mitigates the catastrophe without establishing reliable component transfer (issue I-16).

### Data issues found by tracing

- **Suspected numerical artifacts (R11).** Three values stored as 1.11e-16 (2⁻⁵³) and 1.78e-15 (2⁻⁴⁹) match machine-precision scales. That is a strong warning sign but does not prove the original reported a zero. They are excluded (never entered into the log target), and both versions are reported because the exclusion moves the primary contrast (0.047 to 0.030). The 1e-9 mS/cm threshold is a policy choice. Its distance from the smallest smooth-sweep values (~1e-6 mS/cm) does not validate it independently. Tracing to the original figures is open (I-15).
- **Suspected compilation error.** `10.1149/1.1833611` rows 10075–10078 look temperature-shifted by about 70 K (row 10078 equals row 10070 exactly). They stay suspected until the original figure is checked, and the rows are kept (I-11).

### Practical consequences

- Count provenance-based groups and author groups, not papers or rows, before claiming source transfer. In this collection one author group supplies most of the primary track.
- Under cold-temperature extrapolation, prefer temperature-aware baselines to tree ensembles, and expect all tested models to be biased high near the glass transition.
- Treat "generalization to new solvents or salts" on CALiSol-derived data as a combined chemistry and source result.
- Check very small conductivities before log-scale training.

### Positioning

The contribution is the combined, reproducible audit and deployment-focused evaluation. The ingredients are familiar (tree extrapolation failure, distribution shift, duplicated data); what is shown here is their interaction in a widely used electrolyte dataset. The supporting release material is:
- the provenance tiers and grouping decisions;
- exact split and support definitions;
- version-to-version comparisons;
- per-group results and traced failure cases.

No "first benchmark" claim is made.

### Phase 2 decision

The gate conditions remain met through the temperature tasks, not through source transfer. The recommended Phase 2 experiment is selective prediction for condition extrapolation, compared with a simple distance-to-training-range rejection rule at matched retained coverage. The current results motivate that experiment; they do not show that a learned uncertainty method adds value. A molecular representation model is not yet justified, because no component task is a supported comparison.
