# Changelog

Every version is kept and rebuilds from its own config (`uv run elbench --config <config> all`). Only the v0.1 thresholds were fixed before any test error was seen. Later versions were made after earlier test results had been seen (v0.2 after an external review, v0.3 after a code review, v0.4–v0.5 after reading an original paper), so they are revisions, not blind re-evaluations.

## v0.5.1 (2026-10-20): presentation and diagnostics

No change to rows, rules, groups, splits or reference metrics; `elbench verify` passes against the v0.5 reference tables.

- Temperature-representation ablation (`src/elbench/ablation.py`, `results/v0.5/investigation/temperature_ablation.csv`; report section M).
- `METHODS.md`: what the 32 input columns are, what they cannot represent, and where the per-formulation Arrhenius fit works and fails.
- Aggregate-only overview page (`docs/index.html`, `elbench overview`) and a README figure. Served with the report on GitHub Pages: https://anushajain2.github.io/electrolyte-benchmark/docs/
- README reorganised around the findings; version history moved here.
- Readable model, task and source names in report figures and workbench tables (`src/elbench/labels.py`). The workbench shortlist wording now says it is a practical rule, not a statistical equivalence test.
- Fixes to stale metadata: report title, package and citation version, data-card title, notebook config.

## v0.5 (reference)

Config `configs/benchmark_v0.5.yaml`; results `results/v0.5`; report `reports/v0.5/report.html`; findings `reports/findings_v0.5.md`.

- v0.3 plus **confirmed corrections only**: rule R13 excludes 10 CALiSol rows whose temperatures were checked against the original paper, each with a review record (`data/evidence/corrections_review_record.csv`).
- Digitization floor (R14) moved to a sensitivity assumption, run by evidence tier: `configs/benchmark_v0.5_floor_verified.yaml` and `_floor_inferred.yaml` (`results/v0.5-floor-*`).
- Controlled label-effect comparisons that separate evaluation filtering from training cleaning.
- Exploratory selective prediction for cold extrapolation (distance rule vs ensemble spread).
- Explicit `full_coverage` / `selective` scoring modes for external submissions.
- Local Streamlit workbench (`elbench workbench`).

## v0.4 (superseded by v0.5)

Configs `configs/benchmark_v0.4.yaml`, `_rows_only.yaml`; findings `reports/findings_v0.4.md`.

- Adds R13 (10 rows with confirmed wrong temperatures) and R14 (values below 0.2 mS/cm excluded for linear-axis sources), from reading one original paper (Ding, Xu & Jow 2005). Reported as a sensitivity release.

## v0.3 (correction release after code review)

Config `configs/benchmark_v0.3.yaml`; findings `reports/findings_v0.3.md` (still the source for provenance, component-transfer and ridge-audit findings).

- Temperature and concentration boundaries applied to reported values (v0.2 compared values rounded to 0.1).
- Paired comparisons on one common set of (fold, measurement) cases; incomplete models excluded from rankings.
- Deterministic k-NN ties; cache keyed on data values; stricter `verify` and a `reproduce` command.
- MIT licence and `CITATION.cff`.
- Sensitivity runs: quarantined source included, R11 disabled, v0.1 source linking (`configs/benchmark_v0.3_*.yaml`).

See `ISSUES.md` (F-01 to F-07) for the defects fixed.

## v0.2 (tag `v0.2`)

Config `configs/benchmark_v0.2.yaml`; findings `reports/findings_v0.2.md`.

- Provenance-based source grouping (R12) with evidence tiers.
- Author-group holdout task.
- Local (T, c) support check.
- Tolerance and leave-one-group-out robustness analyses.
- Ridge numerical audit.
- Sensitivity run with v0.1 linking (`configs/benchmark_v0.2_linking_v01.yaml`).

## v0.1

Config `configs/benchmark_v0.1.yaml`; findings `reports/findings_v0.1.md`.

- Original release plus rule R11 (suspected floating-point zeros).
- `configs/benchmark_v0.1_preR11.yaml` regenerates the original run without R11.
