# Electrolyte generalization benchmark (v0.2, Phase 1)

An open benchmark and evaluation tool for predicting the **measured ionic conductivity of liquid lithium electrolytes**, built to test transfer across **experimental sources**, **chemical components** and **operating conditions** separately wherever the data supports that separation, and to say plainly where it does not.

The release contains data provenance, curated tables (rebuilt locally from pinned raw files), frozen split manifests with leakage checks, CPU baselines with matched tuning budgets, per-group results with paired bootstrap uncertainty, a source and chemistry entanglement investigation, and a technical report.

> Scope: research screening support. Version 0.1 does not predict SEI chemistry, cycle life, safety or overall battery suitability, and is not battery qualification. Phase 2 (selective prediction / rejection and an optional molecular representation model) is **not** part of this release.

## Versions

| Version | Config | Results | What changed |
| --- | --- | --- | --- |
| **v0.2** (current) | `configs/benchmark_v0.2.yaml` | `results/v0.2`, `reports/v0.2/report.html` | Provenance-based source grouping (R12) with evidence tiers, author-group holdout task, local (T, c) support, robustness analyses (per-group paired differences, leave-one-group-out with refits, support-definition sensitivity), ridge audit |
| v0.2, v0.1 linking | `configs/benchmark_v0.2_linking_v01.yaml` | `results/v0.2-linking-v01` | Sensitivity run: v0.2 with campaign linking disabled |
| v0.1 | `configs/benchmark_v0.1.yaml` | `results/v0.1` | Original release plus rule R11 (floating-point zeros) |
| v0.1 pre-R11 | `configs/benchmark_v0.1_preR11.yaml` | `results/v0.1-preR11` | The original v0.1 run, regenerated without R11 |

v0.2 was made after an external review and after v0.1 test results had been seen, so it is a revision, not a blind re-evaluation. The v0.2 report shows all four versions side by side. Findings: `reports/findings_v0.2.md` (current) and `reports/findings_v0.1.md` (superseded).

## Quick start

```bash
uv sync --extra dev          # Python 3.12, versions pinned in uv.lock
uv run elbench all           # ingest -> curate -> audit -> make-splits -> train -> evaluate -> investigate -> report
uv run elbench verify        # compare regenerated tables with the published reference tables
uv run pytest                # scientific-invariant tests and a small end-to-end fixture run
```

A full run takes about 20-30 minutes on a 12-core laptop CPU (no GPU). Every command accepts `--config` (default `configs/benchmark_v0.2.yaml`) and writes a machine-readable run manifest (`runs/run_<command>_<id>.json`: config hash, input and output checksums, package versions, wall/CPU time, peak memory).

Outputs to read first:

| File | What it is |
| --- | --- |
| `reports/v0.2/report.html` | Technical report: findings, versions side by side, results per task, investigation and robustness checks, failure cases, Phase 2 gate evidence |
| `reports/v0.2/coverage.html` | Coverage and confounding audit, built before any model was fitted |
| `reports/v0.2/eligibility.csv` | Every candidate evaluation unit with counts and its interpretation label |
| `reports/findings_v0.2.md` | The written research findings (included in the report) |
| `notebooks/example_usage.ipynb` | How to evaluate your own model on a frozen task |
| `DATA.md` / `ISSUES.md` | Data card and open issue log |

## Commands

| Command | Guide step | Writes |
| --- | --- | --- |
| `elbench ingest` | 1 Acquire and inventory | `data/raw/<dataset>/<release>/…` (read-only), `data/raw/sources.csv` |
| `elbench curate` | 2 Normalize and audit | `data/processed/v0.2/{measurements,components}.parquet`, `exclusions.csv`, `copy_links.csv`, `curation_rules.csv` |
| `elbench audit` | 3 Map coverage | `reports/v0.2/coverage.html`, `reports/v0.2/eligibility.csv`, `reports/v0.2/audit/*.csv` |
| `elbench make-splits` | 4 Freeze evaluations | `splits/v0.2/manifest__<task>__<track>.csv.gz`, `folds.csv`, `leakage_checks.json`, `index.json` |
| `elbench train` | 5 Fit baselines | `results/v0.2/predictions/…`, `results/v0.2/models/…`, `train_runs.csv` |
| `elbench evaluate` | 5 Score | `results/v0.2/tables/{summary,group_metrics,bootstrap,group_rankings}.csv`, `row_errors.parquet` |
| `elbench investigate` | 6-7 Investigate | `results/v0.2/investigation/*.csv` |
| `elbench report` | 6 Explain and release | `reports/v0.2/report.html`, `reports/v0.2/tables/*.csv` |
| `elbench verify` | 6 Release check | `reports/v0.2/verify.json` |

`--confirm` on `train`, `evaluate` and `investigate` runs the **reserved** evaluation (the Rahmanian et al. dataset). Each use is appended to `results/v0.2/reserved_exposure_log.csv`; do not call repeatedly inspected labels a blind test.

## Benchmark design in one page

**Data.** CALiSol-23 (27 papers, 13,825 rows) is the backbone; the Rahmanian et al. automated-lab dataset (5,035 EIS measurements) is reserved for external confirmation. Raw files are pinned (git commit / Zenodo record) and checksummed.

**Canonical records.** One row per reported measurement and a separate component table. Original values and units are kept next to canonical ones; every transformation cites a rule id (`curation_rules.csv`). Missing metadata stays missing. Excluded rows stay visible with a reason.

**Identifiers.** `chemistry_system_id` (salt and solvent identities), `formulation_id` (plus ratios and concentration), `series_id` (one sweep in one paper), `source_group_id` (a paper plus any paper shown to reuse its points), `duplicate_group_id` (copies), `replicate_group_id` (repeated readings at one condition). A DOI is a source proxy, never a lab identifier.

**Tracks.** Molality and molarity are different quantities and are never converted into each other without densities: `molal_w` (primary; mol/kg with mass fractions, converted exactly to mole fractions), `molar_x` (mol/L with mole fractions), `molar_v` (mol/L with volume fractions).

**Source groups (v0.2).** Papers are linked when they appear to share measurements (rule R12: comparable points agree within 5% for at least half of ≥20 comparable pairs); a source group is the connected component, i.e. a conservative provenance-based grouping rather than a demonstrated single campaign. `reports/v0.2/provenance_pairs.csv` gives each pair's evidence tier (documented reuse, suspected reuse, conservative grouping). A reviewed author-group mapping (`data/mappings/source_author_groups.csv`, from Crossref author lists in `data/mappings/source_metadata.csv`) defines the stricter `author_transfer` task.

**Tasks** (split membership in manifests keyed by `measurement_id, task_id, track_id, fold_id, role`):

| Task | Construction | Permitted conclusion |
| --- | --- | --- |
| `interp_formulation` | whole formulations held out; every test chemistry system stays in training | new formulations in familiar chemistry |
| `source_transfer` | one source group held out; scored only on its common support with training (v0.2: local support) | transfer across sources within observed support |
| `author_transfer` (v0.2) | one author group held out | transfer across research groups (2 testable units in the primary track) |
| `component_transfer` | every record containing a solvent or salt held out | transfer to that component, with concurrent source/condition shifts reported |
| `temp_high`, `temp_low` | same formulations on both sides of a predeclared temperature boundary | condition extrapolation in familiar chemistry |
| `conc_high` | fixed solvent recipe, train c ≤ 1.2, test c ≥ 1.5 | concentration extrapolation |
| `external_rahmanian` | reserved; train on CALiSol, test on a new automated lab | source transfer on untouched data |

Each fold has an interpretation label (`supported_comparison`, `combined_shift`, `insufficient_support`) and each task a reporting level (`headline` when it has at least 5 held-out units, i.e. source groups or components, with at least 10 formulations each, else `case_study`; units are provenance-based groupings, not established independent laboratories).

**Baselines.** training median, k-nearest neighbours, ridge (log target, plus a raw-target diagnostic whose nonpositive predictions are counted as invalid), extra trees, histogram gradient boosting, and a per-formulation Arrhenius fit for temperature tasks. All use the same fixed 32-column feature set (mole-fraction-weighted mean and variance of 10 RDKit solvent descriptors, 6 anion descriptors, temperature and concentration terms); no source identifiers. Up to 12 configurations per family chosen on nested development folds that mimic each task; 3 seeds for stochastic models.

**Metrics.** Primary: MAE of log10(conductivity / mS cm⁻¹) averaged within score groups (source groups unless stated), then equally across groups. Also sample-weighted MAE, MAE in mS/cm, per-group median and worst quartile, fraction within a factor of two, and the invalid-prediction rate. Uncertainty: paired bootstrap that resamples groups, never rows; intervals are conditional on the grouping, not population-level uncertainty across laboratories.

## Evaluate your own model

```python
from elbench.config import load_config
from elbench.submission import iter_folds, score_predictions

cfg = load_config("configs/benchmark_v0.2.yaml")
preds = []
for fold in iter_folds(cfg, "source_transfer", "molal_w"):
    model = my_fit(fold.X_train, fold.y_train, dev_folds=fold.dev_folds)   # tune on dev folds only
    preds.append(fold.prediction_frame(my_predict(model, fold.X_test)))     # mS/cm
result = score_predictions(cfg, "source_transfer", "molal_w", pd.concat(preds))
```

`fold.rows_train` / `fold.rows_test` give the canonical inputs (components, temperature, concentration basis, provenance) if you want your own representation. Predictions are joined by ID; duplicate, missing or unexpected IDs are rejected. Declare your model's training data: anything pretrained on conductivity data may have seen these labels.

## Repository layout

```
configs/benchmark_v0.2.yaml   every threshold, tolerance, grid and seed (frozen before test errors were seen)
data/mappings/species.csv     reviewed name -> structure mapping (salts as ions), with review notes
data/raw/sources.csv          release ids, URLs, retrieval dates, checksums, licences
src/elbench/                  ingest, curate, schema, units, chemistry, audit, splits, features, models,
                              train, metrics, evaluate, investigate, report, submission, cli
splits/v0.2/                  frozen manifests + leakage check record
results/v0.2/                 predictions, saved models, run records, tables, investigation
reports/                      findings, report, coverage audit, reference tables for `verify`
tests/                        unit, invariant and end-to-end tests
```

Fitted models are saved for the small families (median, ridge, gradient boosting, Arrhenius). Extra trees and k-NN are not stored (hundreds of MB per fold); they refit exactly from the recorded configuration, seed and manifest in `results/v0.2/predictions/<task>/<track>/<fold>/<model>.json`.

## Data licences

CALiSol-23's author repository has no licence file, so raw CALiSol rows are **not redistributed**: `elbench ingest` downloads the pinned commit and `elbench curate` rebuilds the curated tables. The Rahmanian et al. dataset is CC BY 4.0. See `DATA.md`.

## References

1. De Blasio et al., CALiSol-23, *Scientific Data* 11, 750 (2024). https://doi.org/10.1038/s41597-024-03575-8 — data: https://github.com/Pele0599/CALiSol-23
2. Rahmanian et al., *Scientific Data* 10, 43 (2023). https://doi.org/10.1038/s41597-023-01936-3 — data: https://zenodo.org/records/7244939

Code: MIT. Version 0.1. Changes to rows, unit rules, source groups or split membership require a new benchmark version.
