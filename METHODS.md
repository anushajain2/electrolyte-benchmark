# What the models see, and why the baselines behave as they do

This page explains the inputs of every baseline, what those inputs cannot represent, and why a simple per-formulation Arrhenius fit beats a flexible tree ensemble when predicting colder temperatures. Numbers are from the v0.5 reference, primary track (molality recipes).

## Inputs: 32 fixed columns, no source identifiers

Every learned baseline (Ridge, Extra Trees, gradient boosting, k-nearest neighbours) sees the same 32 columns, built in `src/elbench/features.py` from the curated component table:

| Block | Columns | How it is built |
| --- | --- | --- |
| Solvent mixture, mean | 10 | Composition-weighted mean of 10 RDKit descriptors per solvent: molecular weight, Crippen logP, topological polar surface area, H-bond acceptors, H-bond donors, rotatable bonds, ring count, fraction sp³ carbons, heavy-atom count, halogen fraction. Weights are mole fractions (mass fractions converted exactly in the primary track). |
| Solvent mixture, spread | 10 | Composition-weighted variance of the same 10 descriptors (zero for a single solvent). |
| Mixture size | 1 | Number of solvents. |
| Salt anion | 6 | Anion molecular weight, heavy atoms, fluorine count, polar surface area, rings, rotatable bonds. The cation is always Li⁺. |
| Conditions | 5 | Temperature T (K), 1000/T, concentration c, √c, log c. |

No column identifies the paper, laboratory or author, so a model cannot memorise a source. Units are never mixed: molality and molarity tracks are kept apart because converting between them needs densities that the data does not report.

## What these descriptors do not represent

The features describe each molecule on its own and average them. They do not encode:

- **Ion pairing and aggregation.** Contact ion pairs and larger aggregates lower conductivity at high salt concentration. That is why conductivity peaks near 1 mol/kg and then falls, and concentration terms can capture the curve only for chemistries already seen.
- **Specific mixture interactions.** Two solvents with the same average descriptors can solvate Li⁺ very differently (for example, cyclic versus linear carbonates). A weighted mean and variance cannot express preferential solvation.
- **Viscosity directly.** Conductivity tracks viscosity closely, but viscosity is not an input. Descriptors such as molecular weight and polar surface area are only weak proxies for it.
- **Transport mechanism.** Nothing separates vehicular from hopping transport, or Li⁺ from anion contributions.
- **Phase state.** Nothing tells a model that a mixture is close to freezing, crystallising or glass-forming.

These are deliberate limits of a fixed, cheap baseline representation, not claims that such effects are unimportant. A model that represents them (molecular graphs, physics-informed viscosity terms) is outside this release.

One numerical consequence matters in practice. The variance of molecular weight is quadratic in MW, so a heavy co-solvent (TFP, 344 g/mol) puts held-out mixtures about 180 standard deviations outside training. A linear model extrapolates that blindly; this alone explains Ridge's 43-order failure on one component holdout (ISSUES I-16).

## The per-formulation Arrhenius baseline

`arrhenius_local` (`src/elbench/models.py`) fits, separately for each formulation, a straight line

    log10 κ = a + b · (1000 / T)

through that formulation's own training temperatures, then evaluates it at the test temperature. It uses no descriptors and no other formulation.

**Why it is a useful local baseline.** Over a modest temperature window, ion transport in liquid electrolytes is thermally activated, and log κ against 1/T is close to a straight line. A formulation measured at, say, 263–313 K already contains most of the information needed to estimate 253 K, as long as the mixture stays liquid and nothing changes regime. The fit is also local by construction: it cannot be misled by other chemistries.

**Data requirement.** It needs at least three distinct training temperatures for the same formulation. Formulations with fewer are returned as unsupported, never imputed. In this benchmark every cold-extrapolation test formulation meets that requirement, which is why the Arrhenius fit is scored as complete. In practice, the requirement is the first question to ask: *do you have three or more warm measurements of this exact formulation?*

**Where it breaks.**

- **Curvature near the glass transition.** Many carbonate electrolytes follow Vogel–Tammann–Fulcher (VTF) behaviour, in which log κ bends downward as T approaches the glass temperature. A straight Arrhenius line then overestimates conductivity, and the error grows with distance below the coldest measurement. In the selective-prediction results, the cases at or below 233 K are exactly those a distance rule rejects first.
- **Phase changes.** Crystallisation of EC-rich mixtures or salt precipitation at low temperature causes abrupt drops that no smooth fit predicts.
- **Data, not mechanism.** A good fit is empirical. It does not show that a single activation energy describes the electrolyte, and it says nothing about formulations that were never measured.

## Why the cold-extrapolation ranking flips

| Model | Cold extrapolation (≤ 253 K), mean error log10 | New formulations in familiar chemistry |
| --- | --- | --- |
| Per-formulation Arrhenius | 0.229 | not applicable |
| Ridge (log target) | 0.249 | 0.155 |
| Extra Trees | 0.536 | 0.047 |

Tree ensembles split on feature values seen in training. Beyond the coldest training temperature every tree lands in the same leaf, so the prediction stays flat at the value for the coldest training temperature, while real conductivity keeps falling. Extra Trees is therefore biased high by about 0.54 log10 (roughly 3.4× too high). Linear models in 1/T keep the downward slope.

### Temperature-representation ablation

To check whether this is about the model family or about how temperature is represented, each model is refit on the same outer folds with temperature columns removed (`results/v0.5/investigation/temperature_ablation.csv`, `src/elbench/ablation.py`):

| Task | Model | all features | T only (no 1/T) | 1/T only (no T) | no temperature |
| --- | --- | --- | --- | --- | --- |
| Cold extrapolation | Extra Trees | 0.536 | 0.540 | 0.539 | 0.747 |
| Cold extrapolation | Ridge | 0.249 | 0.334 | 0.267 | 0.845 |
| Warm extrapolation | Extra Trees | 0.114 | 0.121 | 0.120 | 0.465 |
| Warm extrapolation | Ridge | 0.195 | 0.514 | 0.351 | 0.568 |
| New formulations | Extra Trees | 0.047 | 0.062 | 0.062 | 0.340 |
| New formulations | Ridge | 0.155 | 0.199 | 0.183 | 0.325 |

- **Mainly a model-family effect.** Extra Trees barely changes between T and 1/T, and Ridge with T alone (0.334) still beats Extra Trees with every feature (0.536).
- **1/T helps linear models.** Under cold extrapolation, 1/T alone gives Ridge 0.267, close to its full-feature 0.249.
- **Ridge needs both columns for warm extrapolation.** Together, T and 1/T let a linear model bend; dropping either more than doubles its warm-side error.
- **Temperature is essential.** Without any temperature column every model fails, which confirms these tasks measure temperature transfer and not a shortcut.

The ablation reuses the configurations selected for the full feature set and a single seed. It is a diagnostic, not a retuned comparison.

## Metrics in one paragraph

The headline error is the mean absolute error of log10(conductivity / mS cm⁻¹), averaged within each source group and then equally across groups, so one large paper cannot dominate. 0.1 log10 is about 26%; 0.3 log10 is a factor of 2. Intervals come from a paired bootstrap that resamples source groups and are conditional on that grouping. Seven source groups from three author groups are not independent laboratories.
