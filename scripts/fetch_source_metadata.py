"""Record Crossref metadata (title, authors, venue, year) for every source DOI.

Output: data/mappings/source_metadata.csv. Used as documentary evidence when
reviewing whether two papers belong to one experimental campaign; it never
changes source groups by itself.

    uv run python scripts/fetch_source_metadata.py
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def crossref(doi: str) -> dict:
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi)
    req = urllib.request.Request(url, headers={"User-Agent": "elbench/0.2 (metadata for provenance review)"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["message"]


def main() -> None:
    m = pd.read_parquet(ROOT / "data/processed/v0.1/measurements.parquet", columns=["source_doi", "dataset_id"])
    rows = []
    for doi in sorted(m.source_doi.unique()):
        try:
            d = crossref(doi)
            title = re.sub(r"<[^>]+>|\s+", " ", " ".join(d.get("title", []))).strip()
            authors = "; ".join(f"{a.get('family', '')}, {a.get('given', '')}".strip(", ") for a in d.get("author", []))
            rows.append(
                {
                    "source_doi": doi,
                    "title": title,
                    "authors": authors,
                    "venue": " ".join(d.get("container-title", [])),
                    "year": (d.get("issued", {}).get("date-parts") or [[None]])[0][0],
                    "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
        except Exception as e:  # recorded, not hidden
            rows.append({"source_doi": doi, "title": "", "authors": "", "venue": "", "year": None, "retrieved_utc": f"error: {e!r}"})
        time.sleep(0.3)
    out = ROOT / "data/mappings/source_metadata.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(out, len(rows))


if __name__ == "__main__":
    main()
