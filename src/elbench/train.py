"""Step 5: fit baselines on frozen manifests.

For every outer fold and model family: choose one configuration on the
fold's development folds (tuning seed only), refit on the outer training rows
with every declared seed, predict the outer test rows, and save predictions
plus run metadata (configuration, dev scores, CPU time, wall time, peak RSS,
failures). Completed (fold, model) jobs are cached and skipped on re-runs.
"""
from __future__ import annotations

import hashlib
import json
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from . import __version__
from .config import Config, RunManifest, peak_rss_mb, sha256_file
from .curate import load_processed
from .features import track_features
from .metrics import group_mean_mae, row_errors
from .models import config_grid, make_model
from .splits import load_folds, load_manifest

SAVE_MODEL_FAMILIES = {"median", "ridge", "ridge_raw", "hist_gb", "arrhenius_local"}


def safe(s: str) -> str:
    return s.replace("/", "__").replace(":", "_")


def families_for(cfg: Config, task_id: str, only: list[str] | None) -> list[str]:
    fams = []
    for name, fc in cfg["models"]["families"].items():
        if only and name not in only:
            continue
        if fc.get("only_tasks") and task_id not in fc["only_tasks"]:
            continue
        fams.append(name)
    return fams


def _fit_predict(name, seed, params, X, y, meta, Xt, mt):
    m = make_model(name, seed, params)
    m.fit(X, y, meta)
    return m, m.predict(Xt, mt)


def run_job(job: dict, X: pd.DataFrame, meta: pd.DataFrame, manifest: pd.DataFrame, cfg_models: dict, out_dir: Path, models_dir: Path) -> dict:
    with threadpool_limits(1):
        return _run_job(job, X, meta, manifest, cfg_models, out_dir, models_dir)


def _run_job(job, X, meta, manifest, cfg_models, out_dir, models_dir):
    name, fold_id = job["model"], job["fold_id"]
    pred_path = out_dir / f"{name}.parquet"
    meta_path = out_dir / f"{name}.json"
    if pred_path.exists() and meta_path.exists() and json.loads(meta_path.read_text()).get("cache_key") == job["cache_key"]:
        return json.loads(meta_path.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    fam = cfg_models["families"][name]
    configs = config_grid(fam, cfg_models["max_configs_per_family"])
    seeds = cfg_models["seeds"] if fam.get("stochastic") else [0]
    y_all = meta.conductivity_mS_cm
    score_col = job["score_group"] if job["score_group"] in meta.columns else None
    t0, c0 = time.time(), time.process_time()

    outer = manifest[manifest.fold_id == fold_id]
    tr_ids = outer.measurement_id[outer.role == "train"].to_numpy()
    te_ids = outer.measurement_id[outer.role == "test"].to_numpy()
    dev_ids = sorted(f for f in manifest.fold_id.unique() if f.startswith(fold_id + "::"))

    # ---- tuning on development folds (tuning seed only)
    dev_scores = []
    failures = []
    chosen = configs[0]
    if len(configs) > 1 and dev_ids:
        for ci, params in enumerate(configs):
            scores = []
            for d in dev_ids:
                dm = manifest[manifest.fold_id == d]
                a = dm.measurement_id[dm.role == "train"].to_numpy()
                b = dm.measurement_id[dm.role == "dev"].to_numpy()
                try:
                    _, p = _fit_predict(name, cfg_models["tuning_seed"], params, X.loc[a], y_all.loc[a].to_numpy(), meta.loc[a], X.loc[b], meta.loc[b])
                    err = row_errors(p, y_all)
                    scores.append(group_mean_mae(err, meta.loc[b, score_col].to_numpy() if score_col else None))
                except Exception as e:  # recorded, never hidden
                    failures.append({"stage": "tuning", "config": ci, "dev_fold": d, "error": repr(e)})
                    scores.append(np.nan)
            dev_scores.append({"config": params, "dev_scores": scores, "mean": float(np.nanmean(scores)) if np.isfinite(scores).any() else None})
        valid = [(s["mean"], i) for i, s in enumerate(dev_scores) if s["mean"] is not None]
        chosen = configs[min(valid)[1]] if valid else configs[0]
    elif len(configs) > 1:
        failures.append({"stage": "tuning", "error": "no development folds; first configuration used"})

    # ---- final fits
    preds = []
    seed_runs = []
    for seed in seeds:
        s0, sc0 = time.time(), time.process_time()
        try:
            model, p = _fit_predict(name, seed, chosen, X.loc[tr_ids], y_all.loc[tr_ids].to_numpy(), meta.loc[tr_ids], X.loc[te_ids], meta.loc[te_ids])
            if seed == seeds[0] and name in SAVE_MODEL_FAMILIES:
                models_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, models_dir / f"{name}__seed{seed}.joblib", compress=3)
            status = "ok"
        except Exception as e:
            p = pd.DataFrame({"measurement_id": te_ids, "prediction_mS_cm": np.nan, "status": "failed", "failure_reason": repr(e)})
            failures.append({"stage": "final", "seed": seed, "error": repr(e), "traceback": traceback.format_exc(limit=3)})
            status = "failed"
        p.insert(0, "seed", seed)
        preds.append(p)
        seed_runs.append({"seed": seed, "status": status, "wall_s": round(time.time() - s0, 3), "cpu_s": round(time.process_time() - sc0, 3)})
    pred = pd.concat(preds, ignore_index=True)
    for k in ["run_id", "task_id", "track_id", "fold_id", "model"]:
        pred.insert(0, k, job[k] if k != "run_id" else job["run_id"])
    pred.to_parquet(pred_path, index=False)
    record = {
        **{k: job[k] for k in ["run_id", "task_id", "track_id", "fold_id", "model", "cache_key"]},
        "training_data": "benchmark training rows of this fold only (no pretrained components)",
        "n_train": int(len(tr_ids)),
        "n_test": int(len(te_ids)),
        "n_dev_folds": len(dev_ids),
        "n_configs": len(configs),
        "chosen_config": chosen,
        "dev_scores": dev_scores,
        "seeds": seed_runs,
        "wall_s": round(time.time() - t0, 3),
        "cpu_s": round(time.process_time() - c0, 3),
        "worker_peak_rss_mb": round(peak_rss_mb(), 1),
        "failures": failures,
        "model_saved": name in SAVE_MODEL_FAMILIES,
    }
    meta_path.write_text(json.dumps(record, indent=1, default=str))
    return record


def train(cfg: Config, tasks: list[str] | None = None, models: list[str] | None = None, n_jobs: int = -1, include_reserved: bool = False) -> pd.DataFrame:
    man = RunManifest("train", cfg)
    meas, comps = load_processed(cfg)
    desc = pd.read_parquet(cfg.p("cache") / "descriptors.parquet")
    folds = load_folds(cfg)
    if tasks:
        folds = folds[folds.task_id.isin(tasks)]
    if not include_reserved:
        folds = folds[~folds.reserved]
    res_dir = cfg.p("results")
    pred_root = res_dir / "predictions"
    models_root = res_dir / "models"
    meta_cols = ["measurement_id", "formulation_id", "temperature_K", "concentration", "conductivity_mS_cm", "source_group_id", "chemistry_system_id", "dataset_id"]
    code_hash = hashlib.sha256("".join(Path(__file__).with_name(f).read_text() for f in ["models.py", "train.py", "features.py", "metrics.py"]).encode()).hexdigest()[:12]
    records = []
    for (task_id, track), fg in folds.groupby(["task_id", "track_id"], sort=True):
        X = track_features(cfg, meas, comps, desc, track)
        meta = meas.set_index("measurement_id").loc[X.index, meta_cols[1:]]
        manifest = load_manifest(cfg, task_id, track)
        man_sha = sha256_file(cfg.p("splits") / f"manifest__{task_id}__{track}.csv.gz")
        fams = families_for(cfg, task_id, models)
        jobs = []
        for fold_id, score_group in zip(fg.fold_id, fg.score_group):
            for fam in fams:
                key = hashlib.sha256(json.dumps([cfg["models"], man_sha, code_hash, __version__, fold_id, fam], sort_keys=True, default=str).encode()).hexdigest()[:16]
                jobs.append({"run_id": man.run_id, "task_id": task_id, "track_id": track, "fold_id": fold_id, "model": fam, "score_group": score_group, "cache_key": key})
        print(f"  train {task_id}/{track}: {len(fg)} folds x {len(fams)} families", flush=True)
        recs = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(run_job)(
                j,
                X,
                meta,
                manifest[(manifest.fold_id == j["fold_id"]) | manifest.fold_id.str.startswith(j["fold_id"] + "::")],
                cfg["models"],
                pred_root / task_id / track / safe(j["fold_id"]),
                models_root / task_id / track / safe(j["fold_id"]),
            )
            for j in jobs
        )
        records.extend(recs)
        if fg.reserved.any():
            log_reserved_exposure(cfg, man.run_id, "train", f"{task_id}/{track}: predictions produced for reserved rows (labels not read by train)")
    runs = pd.DataFrame(
        [
            {
                **{k: r[k] for k in ["run_id", "task_id", "track_id", "fold_id", "model", "n_train", "n_test", "n_configs", "wall_s", "cpu_s", "worker_peak_rss_mb"]},
                "chosen_config": json.dumps(r["chosen_config"], sort_keys=True),
                "n_failures": len(r["failures"]),
                "failed_seeds": sum(s["status"] != "ok" for s in r["seeds"]),
            }
            for r in records
        ]
    )
    res_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(res_dir / "train_runs.csv", index=False)
    man.add_output(res_dir / "train_runs.csv")
    man.notes = {"n_jobs": len(records), "total_cpu_s": float(runs.cpu_s.sum()), "failed_seed_fits": int(runs.failed_seeds.sum())}
    man.write(res_dir / "runs")
    return runs


def log_reserved_exposure(cfg: Config, run_id: str, command: str, what: str) -> None:
    path = cfg.p("results") / "reserved_exposure_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "run_id": run_id, "command": command, "what": what}])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)
