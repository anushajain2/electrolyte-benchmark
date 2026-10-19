"""Step 1: acquire immutable raw files and record provenance in sources.csv."""
from __future__ import annotations

import json
import os
import stat
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from .config import Config, RunManifest, md5_file, sha256_file

USER_AGENT = "elbench/0.1 (research benchmark; contact via repository)"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _source_urls(spec: dict) -> dict[str, str]:
    if spec["kind"] == "github":
        base = f"https://raw.githubusercontent.com/{spec['repo']}/{spec['commit']}/"
        return {f: base + urllib.parse.quote(f) for f in spec["files"]}
    if spec["kind"] == "zenodo":
        base = f"https://zenodo.org/api/records/{spec['record']}/files/"
        return {f: base + urllib.parse.quote(f) + "/content" for f in spec["files"]}
    raise ValueError(f"unknown source kind {spec['kind']}")


def _release_label(spec: dict) -> str:
    return f"commit-{spec['commit'][:12]}" if spec["kind"] == "github" else f"zenodo-{spec['record']}"


def raw_dir(cfg: Config, dataset_id: str) -> Path:
    return cfg.p("raw") / dataset_id / _release_label(cfg["sources"][dataset_id])


def raw_data_file(cfg: Config, dataset_id: str) -> Path:
    return raw_dir(cfg, dataset_id) / cfg["sources"][dataset_id]["data_file"]


def ingest(cfg: Config, force: bool = False, local_dir: str | None = None) -> pd.DataFrame:
    """Download (or copy from ``local_dir``) every configured raw file.

    Raw files are written once and made read-only. A second run verifies
    checksums against the existing download record instead of re-downloading.
    """
    man = RunManifest("ingest", cfg)
    rows = []
    for dataset_id, spec in cfg["sources"].items():
        target = raw_dir(cfg, dataset_id)
        target.mkdir(parents=True, exist_ok=True)
        record_path = target / "download_record.json"
        previous = json.loads(record_path.read_text()) if record_path.exists() else {}
        urls = _source_urls(spec)
        record = {}
        for fname, url in urls.items():
            dest = target / fname
            if dest.exists() and not force:
                retrieved = previous.get(fname, {}).get("retrieved_utc", "unknown")
            else:
                if local_dir:
                    data = (Path(local_dir) / dataset_id / fname).read_bytes()
                    url = f"local:{Path(local_dir) / dataset_id / fname}"
                else:
                    data = _get(url)
                if dest.exists():
                    os.chmod(dest, stat.S_IWUSR | stat.S_IRUSR)
                dest.write_bytes(data)
                os.chmod(dest, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)  # immutable by convention
                retrieved = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            sha = sha256_file(dest)
            md5 = md5_file(dest)
            expected = spec.get("expected_md5", {}).get(fname)
            if expected and expected != md5:
                raise RuntimeError(f"{dataset_id}/{fname}: md5 {md5} != published {expected}")
            if fname in previous and previous[fname]["sha256"] != sha:
                raise RuntimeError(f"{dataset_id}/{fname}: raw file changed since first download")
            record[fname] = {"url": url, "retrieved_utc": retrieved, "sha256": sha, "md5": md5, "bytes": dest.stat().st_size}
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "file": fname,
                    "is_data_file": fname == spec["data_file"],
                    "release": _release_label(spec),
                    "url": url,
                    "retrieved_utc": retrieved,
                    "sha256": sha,
                    "md5": md5,
                    "md5_published": expected or "",
                    "bytes": dest.stat().st_size,
                    "article_doi": spec.get("article_doi", ""),
                    "dataset_doi": spec.get("dataset_doi", ""),
                    "licence": spec["licence"],
                    "redistribute_raw": spec["redistribute_raw"],
                    "role": spec["role"],
                    "original_fields": _original_fields(dest) if fname == spec["data_file"] else "",
                }
            )
            man.add_output(dest)
        if not record_path.exists() or force or record != previous:
            record_path.write_text(json.dumps(record, indent=2))
    df = pd.DataFrame(rows)
    out = cfg.p("raw") / "sources.csv"
    df.to_csv(out, index=False)
    man.add_output(out)
    man.write(cfg.p("raw") / "runs")
    return df


def _original_fields(path: Path) -> str:
    with open(path, encoding="utf-8-sig") as f:
        header = f.readline().strip()
    sep = ";" if header.count(";") > header.count(",") else ","
    return "|".join(h.strip() for h in header.split(sep))
