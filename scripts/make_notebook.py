"""Build notebooks/example_usage.ipynb (run with: uv run python scripts/make_notebook.py)."""
from pathlib import Path

import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
code = lambda s: cells.append(nbf.v4.new_code_cell(s))  # noqa: E731

md("""# Electrolyte generalization benchmark v0.3: example usage

This notebook shows how to (1) inspect the curated data and its provenance, (2) read the frozen evaluation design,
(3) evaluate **your own model** on a frozen task with the benchmark's metrics, and (4) trace an error back to the raw record.

Run `uv run elbench all` first so the curated tables, splits and baseline results exist.""")
code("""import os, pathlib
import numpy as np
import pandas as pd
from elbench.config import load_config
from elbench.curate import load_processed

root = pathlib.Path.cwd() if (pathlib.Path.cwd() / "configs").exists() else pathlib.Path.cwd().parent
cfg = load_config(root / "configs" / "benchmark_v0.3.yaml")
meas, comps = load_processed(cfg)
pd.set_option("display.width", 160)
meas.inclusion_status.value_counts()""")
md("## 1. Canonical records and provenance\nOne row per reported measurement. Excluded rows stay visible with a reason; every value keeps its original unit and a link to the raw record.")
code("""cols = ["measurement_id", "source_doi", "chemistry_system_id", "ratio_basis", "concentration", "concentration_basis",
        "temperature_K", "conductivity_mS_cm", "conductivity_original", "conductivity_unit_original", "track_id", "source_record_link"]
meas[meas.inclusion_status == "included"][cols].sample(5, random_state=0)""")
code("""meas[meas.inclusion_status == "excluded"].exclusion_reason.value_counts()""")
code("""inc = meas[meas.inclusion_status == "included"]
inc.groupby("track_id").agg(measurements=("measurement_id", "size"), formulations=("formulation_id", "nunique"),
                            source_groups=("source_group_id", "nunique"), chemistry_systems=("chemistry_system_id", "nunique"))""")
md("## 2. Frozen evaluation design\nEach fold carries an interpretation label. `combined_shift` means chemistry, source or conditions change together, so a failure cannot be attributed to one factor.")
code("""folds = pd.read_csv(cfg.p("splits") / "folds.csv")
folds[folds.track_id == "molal_w"][["task_id", "fold_id", "n_train", "n_test", "interpretation_label", "n_formulations", "n_source_groups"]]""")
md("""## 3. Evaluate your own model
`iter_folds` yields features and training labels for each outer fold, plus nested development folds for tuning.
Test labels are never handed out; `score_predictions` joins them by ID and compares against the stored baselines with a paired bootstrap over groups.

The example model below is a random forest that tunes `min_samples_leaf` on the development folds only.""")
code("""from sklearn.ensemble import RandomForestRegressor
from elbench.submission import iter_folds, score_predictions

TASK, TRACK = "source_transfer", "molal_w"

def fit(X, y, leaf):
    return RandomForestRegressor(n_estimators=100, min_samples_leaf=leaf, random_state=0, n_jobs=-1).fit(X, np.log10(y))

preds = []
for fold in iter_folds(cfg, TASK, TRACK):
    # choose the leaf size on development folds (other held-out sources inside the training data)
    scores = {}
    for leaf in (1, 5, 20):
        dev_err = [np.mean(np.abs(fit(d.X_train, d.y_train, leaf).predict(d.X_test) - np.log10(d.y_dev))) for d in fold.dev_folds]
        scores[leaf] = np.mean(dev_err)
    best = min(scores, key=scores.get)
    model = fit(fold.X_train, fold.y_train, best)
    preds.append(fold.prediction_frame(10 ** model.predict(fold.X_test)))
    print(f"{fold.fold_id:45s} leaf={best:2d}  n_test={len(fold.X_test)}")
submission = pd.concat(preds)""")
code("""result = score_predictions(cfg, TASK, TRACK, submission)
result["summary"][["n_rows", "n_groups", "group_mean_mae_log10", "sample_mae_log10", "worst_quartile_mae_log10", "invalid_rate"]]""")
code("""# paired bootstrap over held-out sources; diff_vs_reference is baseline minus submission (positive = submission better)
result["comparison"][["model", "n_groups", "group_mean_mae_log10", "ci_low", "ci_high", "diff_vs_reference", "diff_ci_low", "diff_ci_high"]]""")
md("## 4. Trace an error back to its raw record")
code("""g = result["groups"].sort_values("mae_log10", ascending=False)
worst_source = g.group.iat[0]
print("worst held-out source:", worst_source, "MAE log10 =", round(g.mae_log10.iat[0], 3))
sub = submission.merge(meas[["measurement_id", "source_group_id", "conductivity_mS_cm", "temperature_original", "concentration_original",
                             "conductivity_original", "conductivity_unit_original", "source_record_link"]], on="measurement_id")
sub = sub[sub.source_group_id == worst_source]
sub["abs_log_err"] = np.abs(np.log10(sub.prediction_mS_cm) - np.log10(sub.conductivity_mS_cm))
sub.sort_values("abs_log_err", ascending=False).head(5)""")
md("""The baseline report (`reports/v0.3/report.html`) contains the full results, the source-familiarity investigation and three inspected failure cases.
The coverage audit (`reports/v0.3/coverage.html`) shows which comparisons the data can support.""")

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = Path(__file__).resolve().parents[1] / "notebooks" / "example_usage.ipynb"
nbf.write(nb, out)
print(out)
