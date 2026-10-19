"""Command-line entry points: ingest, curate, audit, make-splits, train, evaluate, investigate, report, verify, reproduce, all."""
from __future__ import annotations

import argparse
import sys
import time

from .config import load_config

DEFAULT_CONFIG = "configs/benchmark_v0.3.yaml"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cmd_ingest(cfg, args):
    from .ingest import ingest

    df = ingest(cfg, force=args.force, local_dir=args.local_dir)
    _log(f"ingest: {len(df)} raw files recorded in {cfg.p('raw') / 'sources.csv'}")


def cmd_curate(cfg, args):
    from .curate import curate

    out = curate(cfg)
    m = out["measurements"]
    _log(f"curate: {len(m)} measurements, {(m.inclusion_status == 'included').sum()} included")


def cmd_audit(cfg, args):
    from .audit import audit

    audit(cfg)
    _log(f"audit: wrote {cfg.p('reports') / 'coverage.html'} and eligibility.csv")


def cmd_make_splits(cfg, args):
    from .splits import make_splits

    summary = make_splits(cfg)
    _log(f"make-splits: {len(summary)} task/track/fold entries frozen in {cfg.p('splits')}")


def cmd_train(cfg, args):
    from .train import train

    train(cfg, tasks=args.tasks, models=args.models, n_jobs=args.n_jobs, include_reserved=args.confirm)
    _log("train: done")


def cmd_evaluate(cfg, args):
    from .evaluate import evaluate

    evaluate(cfg, include_reserved=args.confirm)
    _log("evaluate: done")


def cmd_investigate(cfg, args):
    from .investigate import investigate

    investigate(cfg, n_jobs=args.n_jobs, include_reserved=args.confirm)
    _log("investigate: done")


def cmd_report(cfg, args):
    from .report import report

    path = report(cfg)
    _log(f"report: {path}")


def cmd_verify(cfg, args):
    from .report import verify_reproduction

    ok = verify_reproduction(cfg, write_reference=args.write_reference)
    if not ok:
        sys.exit(1)


def cmd_reproduce(cfg, args):
    from .report import reproduce

    if not reproduce(cfg, workdir=args.workdir):
        sys.exit(1)


def cmd_all(cfg, args):
    for fn in (cmd_ingest, cmd_curate, cmd_audit, cmd_make_splits, cmd_train, cmd_evaluate, cmd_investigate, cmd_report):
        fn(cfg, args)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="elbench", description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--root", default=None, help="project root (default: parent of configs/)")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, fn, **extra):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--force", action="store_true", help=argparse.SUPPRESS if name != "ingest" else "re-download raw files")
        p.add_argument("--local-dir", default=None, help=argparse.SUPPRESS if name != "ingest" else "copy raw files from <dir>/<dataset_id>/ instead of downloading")
        p.add_argument("--tasks", nargs="*", default=None)
        p.add_argument("--models", nargs="*", default=None)
        p.add_argument("--n-jobs", type=int, default=-1)
        p.add_argument(
            "--confirm",
            action="store_true",
            help="also run the reserved external evaluation (logged as an exposure of reserved labels)",
        )
        p.add_argument("--write-reference", action="store_true", help=argparse.SUPPRESS)
        p.add_argument("--workdir", default=None, help=argparse.SUPPRESS if name != "reproduce" else "empty directory for the clean rebuild (default: a new temp dir)")
        return p

    add("ingest", cmd_ingest)
    add("curate", cmd_curate)
    add("audit", cmd_audit)
    add("make-splits", cmd_make_splits)
    add("train", cmd_train)
    add("evaluate", cmd_evaluate)
    add("investigate", cmd_investigate)
    add("report", cmd_report)
    add("verify", cmd_verify)
    add("reproduce", cmd_reproduce)
    add("all", cmd_all)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, root=args.root)
    args.fn(cfg, args)


if __name__ == "__main__":
    main()
