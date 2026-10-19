## Findings (v0.1)

> Superseded by `reports/findings_v0.2.md`. Kept as the record of what v0.1 reported; wording corrected after review (suspected rather than confirmed data errors; glass-transition behaviour credited to the source paper).

All numbers are from the primary track (`molal_w`: molality with mass-fraction recipes, 10,481 CALiSol measurements, 1,648 formulations, 10 source groups, 20 chemistry systems), unless stated otherwise. Errors are MAE of log10(conductivity / mS cm⁻¹), averaged within independent groups and then across groups; brackets are 95% paired-bootstrap intervals over groups. 0.30 corresponds to roughly a factor of two for a single prediction.

### What this dataset can and cannot evaluate

- **Source transfer is supported, but only on a minority of each source.** Eight of ten source groups have at least 10 rows inside the common support of the remaining sources (same chemistry system, a training recipe within 0.10 in every solvent fraction, temperature and concentration inside the covered range). Together they contribute 4,344 rows. Two sources have no supported rows at all.
- **Component transfer cannot be separated from source transfer.** Every one of the 9 eligible component holdouts is a `combined_shift`. Five components (DMC, MA, EA, TFP, DME) and one salt (LiAsF6) occur in a single source group, so holding out the component also removes the source. Of the remaining holdouts, LiBF4 and EMC take 97% and 85% of their rows from sources that are then absent from training, and EC (38%) and EMC also extend beyond the training condition range. No headline component-transfer score is published.
- **The molarity tracks support case studies only.** Source transfer there has no unit with ten or more formulations.
- **The reserved secondary dataset barely overlaps the backbone.** Only one Rahmanian et al. chemistry system (LiPF6 in EC+EMC+PC) occurs in CALiSol, from one paper. 153 of 5,035 reserved rows fall inside CALiSol common support. No copied measurements were found between the two collections.

### Result 1: model rankings reverse under low-temperature extrapolation

Extra trees is best or tied for best in interpolation (0.061 [0.041, 0.085]), source transfer (0.099 [0.076, 0.122]) and high-temperature transfer (0.139 [0.127, 0.150]). Below the training temperature range the ranking inverts (Kendall τ against the interpolation ranking = 0.0):

| Model | temp_low (9 source groups) | temp_high (7) | component_transfer (9, all combined) |
| --- | --- | --- | --- |
| ridge (log target) | **0.248** [0.158, 0.383] | 0.219 | 5.50 (TFP 43.6, EMC 1.74, EC 1.38) |
| per-formulation Arrhenius | 0.280 [0.166, 0.432] | 0.196 | n/a |
| extra trees | 0.613 [0.461, 0.838] | **0.139** | **0.323** [0.179, 0.491] |
| gradient boosting | 0.586 | 0.143 | 0.347 |
| k-NN | 0.606 | 0.149 | 0.375 |

Competing explanations tested:

- *Source-specific behaviour.* At matched distance below the 263 K boundary, sources agree: extra trees errs by 0.30–0.51 at 20 K and 0.51–0.83 at 30 K across sources. Ridge stays at 0.10–0.28 and 0.10–0.40 at the same distances. Only one source group (`10.1021/JE020219O`) extends 40–70 K below the boundary, and its error grows with that distance for every model. The ranking change follows extrapolation distance, not source.
- *Unequal tuning.* All families used nested development folds with an analogous boundary (train ≥ 283 K, develop ≤ 273 K) and the same budget.
- *Documented physics the models miss.* Predictions are biased high in 95% of (model, distance, source) cells. The source paper for the coldest data measured glass-transition temperatures alongside conductivity (its title says so), so super-Arrhenius collapse near Tg is documented behaviour, not a discovery here. The evaluation finding is that these model classes fail to reproduce it under this split. In two traced cases, PC-rich LiPF6 at 1.7–2.3 mol/kg, measured conductivity falls to 10⁻⁵–10⁻⁶ mS/cm near 200 K.

The same linear model that extrapolates best in temperature is the most dangerous under component novelty. Ridge predicts TFP-containing electrolytes 43 orders of magnitude too low, because a halogen-rich phosphate pushes its descriptors far outside the training range. **No single baseline is safe under all shifts, and a ranking from an interpolation split would recommend the wrong model for low-temperature screening.**

### Result 2: holding out a source costs little once chemistry is matched (primary contrast)

Scoring the same rows under interpolation and under source transfer, extra trees loses on average 0.028 log10 per source (range −0.029 to +0.119). Holding out a component costs 0.273, low-temperature extrapolation 0.543 and high concentration 0.154. Within common support, source change is the mildest of the shifts studied.

The predeclared controlled comparison builds, for each of 8 eligible target sources, two equal-size training sets that share a background sample. One adds the target source's other formulations; the other replaces them with rows from other sources of the same chemistry system where possible. **Primary contrast** (extra trees, test rows inside the support of both training sets): including the target source lowers error by **0.030 [0.003, 0.059]** log10, positive for 5 of 8 sources. Ridge gives 0.029 [0.010, 0.051].

The effect is entangled with chemistry coverage:

- Without restricting to rows both sets cover, the apparent effect is 0.106 [0.047, 0.172], three times larger. Most of the apparent "source familiarity" is chemistry or condition coverage that only the target source supplied.
- The replacement step matched chemistry for ≥80% of rows in only 3 targets. For those, the effect is 0.006 for both models (exploratory, chosen after seeing the matching rates). Where matching failed (`JE020219O`: 6% matched; `1.1833611`: 5%), local data density remains a live alternative explanation.
- **Confirmation on reserved data** (run once, logged): with the Rahmanian dataset as target, the effect on doubly supported rows was +0.008 for extra trees and +0.117 for ridge. Only 36 test rows qualified and 3% of replacement rows matched chemistry. The confirmation is underpowered and does not establish the effect.

Conclusion: source familiarity yields a small benefit (≈0.03 log10, about 7%) that cannot be separated from local chemistry density with the public data. It is far smaller than the cost of chemical or condition novelty.

### Result 3: transfer to an independent automated laboratory works inside support

Trained on CALiSol only, extra trees predicts the reserved Rahmanian measurements with 0.037 log10 error on the 153 common-support rows. Overall (5,035 rows, both chemistry systems) the error is 0.119, with 95% of predictions within a factor of two (k-NN: 0.106). This was computed once, before the R11 curation fix described below. That fix changed 3 CALiSol training rows and no reserved row.

### Inspected failure cases

1. **Glass-like collapse at low temperature** (`10.1021/JE020219O`, rows 981, 995, 1009; LiPF6 2.32 mol/kg in PC/DEC 90:10, 224–244 K). Measured 4.5×10⁻³ down to 1.4×10⁻⁵ mS/cm; every model predicts ~0.24. The sweep is smooth and the paper reports glass-transition measurements for this system, so the rows stay. The error is a model-form limitation, not a data error.
2. **Same formulation, two papers** (rows 984–1040 of `JE020219O` plus row 415 of `S0378-7753(03)00153-8`; 1.69 mol/kg, 204–244 K). At 233 K the two papers agree (0.0166 vs 0.0152 mS/cm), which supports the low values being real.
3. **A suspected compilation error in the source table** (`10.1149/1.1833611`, rows 10075–10078; LiBOB in EC/PC 10:90, ~0.72 mol/kg). These rows report 0.087, 0.24, 0.48 and 2.95 mS/cm at 302–332 K. The neighbouring series rows 10067–10074 (0.70–0.71 mol/kg) report 3.97–7.60 at the same temperatures. Row 10078 (2.953263 at 332 K) equals row 10070 (2.953263 at 293 K) exactly. The block appears temperature-shifted by roughly 70 K. The model may be disagreeing with a wrong label, but that is not confirmed until the original figure is checked (logged as I-11; rows kept because no label-independent rule identifies them).

A fourth issue was found and fixed during tracing: three values stored as 1.11×10⁻¹⁶ and 1.78×10⁻¹⁵ mS/cm are floating-point zeros from digitization. They produced 13-order-of-magnitude "errors". Rule R11 now treats values below 10⁻⁹ mS/cm as reported zeros. This change was made after inspecting test errors. Before it, the primary contrast read 0.047 [0.018, 0.076]; the conclusion did not change.

### Practical consequences

- Do not pick a conductivity model from an interpolation split if it will be used outside the training temperature range. Tree ensembles flatten below the coldest training point. Prefer models with an explicit temperature dependence, and expect all of them to overpredict near the glass transition.
- Treat any claim of "generalization to new solvents or salts" on CALiSol-derived data as a combined chemistry and source result. The public data cannot isolate chemical transfer.
- Before training, check that very small conductivities are real measurements, not digitized zeros: log-scale objectives amplify them by many orders of magnitude.
- More data from new laboratories helps mostly by covering new chemistry and conditions. The incremental value of seeing a lab's other formulations is small and not robustly confirmed.

### Phase 2 decision

The gate's three conditions are met: four shift tasks have headline support, group errors vary several-fold within tasks, and nested development folds keep a separate tuning route. **Selective prediction is justified** for condition extrapolation, where errors are large, systematic and predictable from distance to the training range. A molecular representation model is **not yet justified**: there is no `supported_comparison` component task, so a gain could not be attributed to chemistry. Expand component coverage across sources first.
