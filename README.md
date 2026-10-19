# Electrolyte generalization benchmark (v0.5)

[![tests](https://github.com/anushajain2/electrolyte-benchmark/actions/workflows/tests.yml/badge.svg)](https://github.com/anushajain2/electrolyte-benchmark/actions/workflows/tests.yml)

An open benchmark and evaluation tool for predicting the **measured ionic conductivity of liquid lithium electrolytes**, built to test transfer across **experimental sources**, **chemical components** and **operating conditions** separately wherever the data supports that separation, and to say plainly where it does not.

The release contains data provenance, curated tables (rebuilt locally from pinned raw files), frozen split manifests with leakage checks, CPU baselines with matched tuning budgets, per-group results with paired bootstrap uncertainty, a source and chemistry entanglement investigation, and a technical report.

> Scope: research screening support. This release does not predict SEI chemistry, cycle life, safety or overall battery suitability, and is not battery qualification. Phase 2 (selective prediction / rejection and an optional molecular representation model) is **not** part of this release.

## Versions

| Version | Config | Results | What changed |
| --- | --- | --- | --- |
| **v0.5** (current reference) | `configs/benchmark_v0.5.yaml` | `results/v0.5`, `reports/v0.5/report.html` | v0.3 + confirmed corrections only (R13, 10 rows checked against the original paper, with a review record); controlled label-effect comparisons; exploratory selective prediction for cold extrapolation; explicit full-coverage / selective scoring modes; local workbench |
| v0.5 floor sensitivity | `configs/benchmark_v0.5_floor_verified.yaml`, `_floor_inferred.yaml` | `results/v0.5-floor-*` | Digitization floor (R14) applied to the verified source only, and to verified + inferred sources |
| v0.3 | `configs/benchmark_v0.3.yaml` | `results/v0.3`, `reports/v0.3/report.html` | Correction release: temperature and concentration boundaries applied to reported values (v0.2 compared values rounded to 0.1); paired comparisons on one common set of (fold, measurement) cases; incomplete models excluded from rankings; deterministic k-NN ties; stricter `verify` and a `reproduce` command; cache keyed on data values |
| v0.4 (superseded by v0.5) | `configs/benchmark_v0.4.yaml`, `_rows_only.yaml` | `results/v0.4`, `results/v0.4-rows-only`, `reports/v0.4/report.html` | Adds R13 (10 rows with confirmed wrong temperatures) and R14 (values below 0.2 mS/cm excluded for linear-axis sources), based on reading one original paper. Reported as a sensitivity release; v0.3 remains the reference |
| v0.3 sensitivity runs | `configs/benchmark_v0.3_include_quarantine.yaml`, `_noR11.yaml`, `_linking_v01.yaml` | `results/v0.3-*` | Quarantined source included; R11 disabled; v0.1 source linking |
| v0.2 (superseded, tag `v0.2`) | `configs/benchmark_v0.2.yaml` | `results/v0.2` | Provenance-based source grouping (R12) with evidence tiers, author-group holdout, local (T, c) support, robustness analyses, ridge audit |
| v0.2, v0.1 linking | `configs/benchmark_v0.2_linking_v01.yaml` | `results/v0.2-linking-v01` | Sensitivity run for v0.2 |
| v0.1 | `configs/benchmark_v0.1.yaml` | `results/v0.1` | Original release plus rule R11 (suspected numerical artifacts) |
| v0.1 pre-R11 | `configs/benchmark_v0.1_preR11.yaml` | `results/v0.1-preR11` | The original v0.1 run, regenerated without R11 |

Only the v0.1 thresholds were fixed before any test error was seen. v0.2 (after an external review) and v0.3 (after a code review) were made after earlier test results had been seen, so they are revisions, not blind re-evaluations. Every earlier version is kept and rebuilds from its own config. The v0.3 report shows the versions side by side. Findings: `reports/findings_v0.5.md` (current); `findings_v0.3.md` holds the provenance, component and ridge-audit findings that v0.5 carries over; `findings_v0.4.md`, `findings_v0.2.md` and `findings_v0.1.md` are superseded records. Checks against original papers: `DATA.md`, `reports/evidence/`.

## Quick start

```bash
uv sync --extra dev          # Python 3.12, versions pinned in uv.lock
uv run elbench all           # ingest -> curate -> audit -> make-splits -> train -> evaluate -> investigate -> report
uv run elbench verify        # compare the report tables on disk with the published reference tables (no rerun)
uv run elbench reproduce     # rebuild everything from the raw files in a clean copy, then verify there
uv sync --extra workbench && uv run elbench workbench   # local evaluation workbench (after `elbench all`)
uv run pytest                # scientific-invariant tests and a small end-to-end fixture run
```

A full run takes about 20-30 minutes on a 12-core laptop CPU (no GPU). Every command accepts `--config` (default `configs/benchmark_v0.5.yaml`) and writes a machine-readable run manifest (`runs/run_<command>_<id>.json`: config hash, input and output checksums, package versions, wall/CPU time, peak memory).

Outputs to read first:

| File | What it is |
| --- | --- |
| `reports/v0.3/report.html` | Technical report: findings, versions side by side, results per task, investigation and robustness checks, failure cases, Phase 2 gate evidence |
| `reports/v0.3/coverage.html` | Coverage and confounding audit, built before any model was fitted |
| `reports/v0.3/eligibility.csv` | Every candidate evaluation unit with counts and its interpretation label |
| `reports/findings_v0.3.md` | The written research findings (included in the report) |
| `notebooks/example_usage.ipynb` | How to evaluate your own model on a frozen task |
| `DATA.md` / `ISSUES.md` | Data card and open issue log |

## Commands

| Command | Guide step | Writes |
| --- | --- | --- |
| `elbench ingest` | 1 Acquire and inventory | `data/raw/<dataset>/<release>/…` (read-only), `data/raw/sources.csv` |
| `elbench curate` | 2 Normalize and audit | `data/processed/v0.3/{measurements,components}.parquet`, `exclusions.csv`, `copy_links.csv`, `curation_rules.csv` |
| `elbench audit` | 3 Map coverage | `reports/v0.3/coverage.html`, `reports/v0.3/eligibility.csv`, `reports/v0.3/audit/*.csv` |
| `elbench make-splits` | 4 Freeze evaluations | `splits/v0.3/manifest__<task>__<track>.csv.gz`, `folds.csv`, `leakage_checks.json`, `index.json` |
| `elbench train` | 5 Fit baselines | `results/v0.3/predictions/…`, `results/v0.3/models/…`, `train_runs.csv` |
| `elbench evaluate` | 5 Score | `results/v0.3/tables/{summary,group_metrics,bootstrap,group_rankings}.csv`, `row_errors.parquet` |
| `elbench investigate` | 6-7 Investigate | `results/v0.3/investigation/*.csv` |
| `elbench report` | 6 Explain and release | `reports/v0.3/report.html`, `reports/v0.3/tables/*.csv` |
| `elbench verify` | 6 Release check | `reports/v0.3/verify.json` |

`--confirm` on `train`, `evaluate` and `investigate` runs the **reserved** evaluation (the Rahmanian et al. dataset). Each use is appended to `results/v0.3/reserved_exposure_log.csv`; do not call repeatedly inspected labels a blind test.

## Benchmark design in one page

**Data.** CALiSol-23 (27 papers, 13,825 rows) is the backbone; the Rahmanian et al. automated-lab dataset (5,035 EIS measurements) is reserved for external confirmation. Raw files are pinned (git commit / Zenodo record) and checksummed.

**Canonical records.** One row per reported measurement and a separate component table. Original values and units are kept next to canonical ones; every transformation cites a rule id (`curation_rules.csv`). Missing metadata stays missing. Excluded rows stay visible with a reason.

**Identifiers.** `chemistry_system_id` (salt and solvent identities), `formulation_id` (plus ratios and concentration), `series_id` (one sweep in one paper), `source_group_id` (a paper plus any paper shown to reuse its points), `duplicate_group_id` (copies), `replicate_group_id` (repeated readings at one condition). A DOI is a source proxy, never a lab identifier.

**Tracks.** Molality and molarity are different quantities and are never converted into each other without densities: `molal_w` (primary; mol/kg with mass fractions, converted exactly to mole fractions), `molar_x` (mol/L with mole fractions), `molar_v` (mol/L with volume fractions).

**Source groups (since v0.2).** Papers are linked when they appear to share measurements (rule R12: comparable points agree within 5% for at least half of ≥20 comparable pairs); a source group is the connected component, i.e. a conservative provenance-based grouping rather than a demonstrated single campaign. `reports/v0.3/provenance_pairs.csv` gives each pair's evidence tier (documented reuse, suspected reuse, conservative grouping). A reviewed author-group mapping (`data/mappings/source_author_groups.csv`, from Crossref author lists in `data/mappings/source_metadata.csv`) defines the stricter `author_transfer` task.

**Tasks** (split membership in manifests keyed by `measurement_id, task_id, track_id, fold_id, role`):

| Task | Construction | Permitted conclusion |
| --- | --- | --- |
| `interp_formulation` | whole formulations held out; every test chemistry system stays in training | new formulations in familiar chemistry |
| `source_transfer` | one source group held out; scored only on its common support with training (local support since v0.2) | transfer across sources within observed support |
| `author_transfer` (since v0.2) | one author group held out | transfer across research groups (2 testable units in the primary track) |
| `component_transfer` | every record containing a solvent or salt held out | transfer to that component, with concurrent source/condition shifts reported |
| `temp_high`, `temp_low` | same formulations on both sides of a predeclared temperature boundary (train T ≤ 313.15 K / test ≥ 323.15 K; train ≥ 263.15 K / test ≤ 253.15 K), applied to reported values | condition extrapolation in familiar chemistry |
| `conc_high` | fixed solvent recipe, train c ≤ 1.2, test c ≥ 1.5 mol/kg, applied to reported values | concentration extrapolation |
| `external_rahmanian` | reserved; train on CALiSol, test on a new automated lab | narrow source transfer; never used for training or tuning, but its labels have been scored once per version (v0.1, v0.2, v0.3), so it is no longer untouched |

Each fold has an interpretation label (`supported_comparison`, `combined_shift`, `insufficient_support`) and each task a reporting level (`headline` when it has at least 5 held-out units, i.e. source groups or components, with at least 10 formulations each, else `case_study`; units are provenance-based groupings, not established independent laboratories).

**Baselines.** training median, k-nearest neighbours, ridge (log target, plus a raw-target diagnostic whose nonpositive predictions are counted as invalid), extra trees, histogram gradient boosting, and a per-formulation Arrhenius fit for temperature tasks. All use the same fixed 32-column feature set (mole-fraction-weighted mean and variance of 10 RDKit solvent descriptors, 6 anion descriptors, temperature and concentration terms); no source identifiers. Up to 12 configurations per family chosen on nested development folds that mimic each task; 3 seeds for stochastic models. From v0.3, k-NN merges replicate inputs and breaks distance ties in a fixed order, so predictions do not depend on row order or the linear-algebra library.

**Metrics.** Primary: MAE of log10(conductivity / mS cm⁻¹) averaged within score groups (source groups unless stated), then equally across groups. Also sample-weighted MAE, MAE in mS/cm, per-group median and worst quartile, fraction within a factor of two, and the invalid-prediction rate. Uncertainty: paired bootstrap that resamples groups, never rows; intervals are conditional on the grouping, not population-level uncertainty across laboratories. Paired comparisons use one common set of (fold, measurement) cases; a model with any invalid prediction in a subset is reported but excluded from comparisons and rankings.

## Evaluate your own model

Scoring modes (`score_predictions(..., mode=...)`):

* `full_coverage` (default): every frozen case needs a valid prediction. A submission with any abstention or failure gets diagnostics but no ranked comparison, the same policy applied to the baselines.
* `selective`: intentional abstentions (`status="abstain"`) with a `confidence` column (thresholds fixed on development folds). Returns error versus coverage; execution failures are reported separately and block a ranked comparison; comparisons on the retained cases are labelled as favouring the submission.

The workbench (`elbench workbench`) wraps the same API: what each setting can establish, published baselines, upload and validation (IDs, units, failures), provenance and label-quality inspection, and an exportable summary. It runs locally because scoring needs CALiSol labels, which are not redistributed.


```python
from elbench.config import load_config
from elbench.submission import iter_folds, score_predictions

cfg = load_config("configs/benchmark_v0.3.yaml")
preds = []
for fold in iter_folds(cfg, "source_transfer", "molal_w"):
    model = my_fit(fold.X_train, fold.y_train, dev_folds=fold.dev_folds)   # tune on dev folds only
    preds.append(fold.prediction_frame(my_predict(model, fold.X_test)))     # mS/cm
result = score_predictions(cfg, "source_transfer", "molal_w", pd.concat(preds))
```

`fold.rows_train` / `fold.rows_test` give the canonical inputs (components, temperature, concentration basis, provenance) if you want your own representation. Predictions are joined by ID; duplicate, missing or unexpected IDs are rejected. Declare your model's training data: anything pretrained on conductivity data may have seen these labels.

## Repository layout

```
configs/benchmark_v0.*.yaml   every threshold, tolerance, grid and seed for each version (see Versions for what was fixed when)
data/mappings/species.csv     reviewed name -> structure mapping (salts as ions), with review notes
data/raw/sources.csv          release ids, URLs, retrieval dates, checksums, licences
src/elbench/                  ingest, curate, schema, units, chemistry, audit, splits, features, models,
                              train, metrics, evaluate, investigate, report, submission, cli
splits/v0.*/                  frozen manifests + leakage check record, per version
results/v0.*/                 run records, tables, investigation (predictions and models are rebuilt, not committed)
reports/                      findings, report, coverage audit, reference tables for `verify`
tests/                        unit, invariant and end-to-end tests
```

Fitted models are saved for the small families (median, ridge, gradient boosting, Arrhenius). Extra trees and k-NN are not stored (hundreds of MB per fold); they refit exactly from the recorded configuration, seed and manifest in `results/<version>/predictions/<task>/<track>/<fold>/<model>.json`.

## Data licences

CALiSol-23's author repository has no licence file, so raw CALiSol rows are **not redistributed**: `elbench ingest` downloads the pinned commit and `elbench curate` rebuilds the curated tables. The Rahmanian et al. dataset is CC BY 4.0. See `DATA.md`.

## References

1. De Blasio et al., CALiSol-23, *Scientific Data* 11, 750 (2024). https://doi.org/10.1038/s41597-024-03575-8 — data: https://github.com/Pele0599/CALiSol-23
2. Rahmanian et al., *Scientific Data* 10, 43 (2023). https://doi.org/10.1038/s41597-023-01936-3 — data: https://zenodo.org/records/7244939

Code: MIT (see `LICENSE`); cite with `CITATION.cff`. Version 0.3. Changes to rows, unit rules, source groups or split membership require a new benchmark version.
