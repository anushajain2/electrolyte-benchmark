"""Step 2: normalize raw records into canonical measurement and component tables.

Outputs (under paths.processed):
  measurements.parquet  one row per reported measurement, excluded rows included
  components.parquet    one row per (formulation, component)
  exclusions.csv        every failed rule for every excluded row
  curation_rules.csv    the rule ids referenced in conversion_provenance
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import chemistry, units
from .config import Config, RunManifest
from .ingest import raw_data_file
from .schema import CURATION_RULES, MEASUREMENT_COLUMNS, validate_components, validate_measurements

CALISOL_META_COLS = ["doi", "k", "T", "c", "salt", "c units", "solvent ratio type"]
RAHMANIAN_DOI = "10.1038/s41597-023-01936-3"


def _hid(prefix: str, *parts: object) -> str:
    h = hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:10]
    return f"{prefix}{h}"


# --------------------------------------------------------------------------
# dataset readers -> "long" intermediate records
# --------------------------------------------------------------------------
def read_calisol(path: Path, commit: str, repo: str, data_file: str) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="utf-8-sig")
    solvent_cols = [c for c in raw.columns if c not in CALISOL_META_COLS]
    recs = []
    fr = raw[solvent_cols].to_numpy(dtype=float)
    for i, row in enumerate(raw.itertuples(index=False)):
        vals = dict(zip(raw.columns, row))
        fracs = {solvent_cols[j]: float(fr[i, j]) for j in np.flatnonzero(fr[i] > 0)}
        recs.append(
            {
                "dataset_id": "calisol23",
                "original_row_id": str(i),
                "source_doi": str(vals["doi"]).strip(),
                "salt_raw": str(vals["salt"]),
                "solvent_fractions_raw": fracs,
                "ratio_basis": str(vals["solvent ratio type"]).strip(),
                "temperature_original": float(vals["T"]),
                "temperature_unit_original": "K",
                "concentration_original": float(vals["c"]),
                "concentration_unit_original": str(vals["c units"]).strip(),
                "conductivity_original": float(vals["k"]) if pd.notna(vals["k"]) else np.nan,
                "conductivity_unit_original": "mS/cm",
                "measurement_method": None,
                "conductivity_uncertainty_mS_cm": np.nan,
                "series_hint": None,
                "source_flags": [],
                "source_record_link": f"https://github.com/{repo}/blob/{commit}/{data_file.replace(' ', '%20')}#L{i + 2}",
            }
        )
    return pd.DataFrame(recs)


def _parse_metadata(text: str) -> dict:
    # Python-literal dict with bare nan values.
    return ast.literal_eval(re.sub(r"\bnan\b", "None", text))


def read_rahmanian(path: Path, record: int, data_file: str) -> pd.DataFrame:
    head = pd.read_csv(path, sep=";", nrows=2, header=None)
    names = list(head.iloc[0])
    unit_row = dict(zip(names, pd.read_csv(path, sep=";", skiprows=2, nrows=1, header=None).iloc[0]))
    expected = {"temperature": "°C", "PC": "g", "EC": "g", "EMC": "g", "LiPF_6": "g", "EIS_conductivity": "S/cm"}
    for col, unit in expected.items():
        if unit_row[col] != unit:
            raise ValueError(f"Rahmanian unit for {col} is {unit_row[col]!r}, expected {unit!r}")
    raw = pd.read_csv(path, sep=";", skiprows=[1, 2], usecols=["label", "experimentID", "temperature", "PC", "EC", "EMC", "LiPF_6", "metadata", "EIS_conductivity"])
    recs = []
    for row in raw.itertuples(index=False):
        meta = _parse_metadata(row.metadata)
        masses = {"PC": float(row.PC), "EC": float(row.EC), "EMC": float(row.EMC)}
        recs.append(
            {
                "dataset_id": "rahmanian2023",
                "original_row_id": str(row.label),
                "source_doi": RAHMANIAN_DOI,
                "salt_raw": "LiPF_6",
                "solvent_masses_g": {k: v for k, v in masses.items() if v > 0},
                "salt_mass_g": float(row.LiPF_6),
                "solvent_fractions_raw": None,
                "ratio_basis": "w",
                "temperature_original": float(row.temperature),
                "temperature_unit_original": "°C",
                "concentration_original": np.nan,
                "concentration_unit_original": "computed from weighed masses (g)",
                "conductivity_original": float(row.EIS_conductivity),
                "conductivity_unit_original": "S/cm",
                "measurement_method": "EIS with automated equivalent-circuit fit",
                "conductivity_uncertainty_mS_cm": np.nan,
                "series_hint": str(row.experimentID),
                "source_flags": ["suspected_measurement_error"] if meta.get("suspectedMeasurementError") else [],
                "metadata_channel": meta.get("channel"),
                "source_record_link": f"https://zenodo.org/records/{record}/files/{data_file}#label={row.label}",
            }
        )
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------
# normalization
# --------------------------------------------------------------------------
class Normalizer:
    def __init__(self, cfg: Config, species: pd.DataFrame):
        self.cfg = cfg
        self.cur = cfg["curation"]
        self.species = {(r.dataset_id, r.raw_name.strip()): r for r in species.itertuples(index=False)}
        self.quarantine = {q["doi"]: q["reason"] for q in self.cur.get("quarantine_sources", [])}

    def _sp(self, dataset_id: str, raw: str):
        return self.species.get((dataset_id, raw.strip()))

    def normalize(self, rec: pd.Series) -> dict:
        out: dict = {}
        reasons: list[str] = []
        flags: list[str] = list(rec.source_flags)
        prov: list[str] = ["R08"]
        ds = rec.dataset_id

        # --- temperature
        out["temperature_K"] = units.temperature_to_K(rec.temperature_original, rec.temperature_unit_original)
        prov.append("R01")
        lo, hi = self.cur["temperature_K_range"]
        if not lo <= out["temperature_K"] <= hi:
            reasons.append("temperature_out_of_range")

        # --- target
        k = rec.conductivity_original
        out["conductivity_mS_cm"] = units.conductivity_to_mS_cm(k, rec.conductivity_unit_original) if pd.notna(k) else np.nan
        prov.append("R02")
        if pd.isna(k):
            reasons.append("missing_conductivity")
        elif k == 0:
            reasons.append("reported_zero_conductivity")
            flags.append("zero_or_below_detection")
            prov.append("R10")
        elif 0 < out["conductivity_mS_cm"] < self.cur["float_zero_threshold_mS_cm"]:
            # e.g. 1.11e-16: a zero digitized from a linear axis, stored as float noise
            reasons.append("reported_zero_conductivity")
            flags.append("floating_point_zero")
            prov.append("R11")
        elif k < 0:
            reasons.append("negative_conductivity_nonphysical")
            flags.append("negative_reported_value")
            prov.append("R10")

        # --- species
        salt = self._sp(ds, rec.salt_raw)
        solvent_amounts = rec.solvent_masses_g if ds == "rahmanian2023" else rec.solvent_fractions_raw
        solv = {raw: self._sp(ds, raw) for raw in solvent_amounts}
        unresolved = [raw for raw, s in [(rec.salt_raw, salt), *solv.items()] if s is None or s.review_status != "verified"]
        if unresolved:
            reasons.append("unresolved_component")
            flags.append("unresolved:" + ",".join(sorted(x.strip() for x in unresolved)))
        out["salt_id"] = salt.canonical_id if salt is not None else rec.salt_raw.strip()

        # --- solvent fractions (canonical ids; duplicates of a canonical id are summed)
        frac: dict[str, float] = {}
        if ds == "rahmanian2023":
            total = sum(solvent_amounts.values())
            for raw, m in solvent_amounts.items():
                frac[solv[raw].canonical_id] = frac.get(solv[raw].canonical_id, 0.0) + m / total
            prov.append("R05")
        else:
            for raw, f in solvent_amounts.items():
                cid = solv[raw].canonical_id if solv[raw] is not None else raw
                frac[cid] = frac.get(cid, 0.0) + f
            s = sum(frac.values())
            if abs(s - 1.0) > 0.02:
                reasons.append("solvent_fractions_do_not_sum_to_one")
            elif abs(s - 1.0) > 1e-9:
                prov.append("R06")
            if s > 0:
                frac = {k_: v / s for k_, v in frac.items()}
        if not frac:
            reasons.append("no_solvent")
        out["solvent_fractions"] = frac

        # --- concentration
        if ds == "rahmanian2023":
            mw_salt = chemistry.salt_molar_mass(pd.Series(salt._asdict()))
            out["concentration"] = units.molality_from_masses(rec.salt_mass_g, mw_salt, sum(solvent_amounts.values()))
            out["concentration_original"] = out["concentration"]
            out["concentration_basis"] = "molality"
            out["concentration_conversion"] = "computed_from_weighed_masses"
            prov.append("R04")
        else:
            unit = rec.concentration_unit_original
            basis = {"mol/kg": "molality", "mol/l": "molarity", "mol/L": "molarity"}.get(unit)
            if basis is None:
                reasons.append("unknown_concentration_unit")
                basis = "unknown"
            out["concentration"] = rec.concentration_original
            out["concentration_original"] = rec.concentration_original
            out["concentration_basis"] = basis
            out["concentration_conversion"] = "as_reported"
            prov.append("R03")
        c = out["concentration"]
        if c < 0:
            reasons.append("negative_concentration_nonphysical")
        elif c == 0:
            reasons.append("zero_salt_concentration")
        elif c > self.cur["max_concentration"]:
            reasons.append("concentration_out_of_range")

        # --- source-level quarantine and source flags
        if rec.source_doi in self.quarantine:
            reasons.append("source_quarantine")
            flags.append(self.quarantine[rec.source_doi].split(":")[0])
        if "suspected_measurement_error" in rec.source_flags and self.cur.get("rahmanian_exclude_suspected_error", True):
            reasons.append("suspected_measurement_error_in_source")

        if rec.ratio_basis == "w":
            prov.append("R07")
        out["reasons"] = reasons
        out["flags"] = flags
        out["prov"] = prov
        return out


def _assign_track(cfg: Config, basis: str, ratio_basis: str) -> str | None:
    for tid, t in cfg["tracks"].items():
        if t["concentration_basis"] == basis and ratio_basis in t["ratio_bases"]:
            return tid
    return None


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def find_copies(df: pd.DataFrame, tol: dict) -> list[tuple[int, int]]:
    """Pairs of row positions that satisfy the proven-copy rule (R09)."""
    pairs = []
    ok = df.conductivity_mS_cm.notna() & (df.conductivity_mS_cm > 0)
    sub = df[ok]
    for _, g in sub.groupby(["chemistry_system_id", "ratio_basis", "concentration_basis"], sort=False):
        if len(g) < 2:
            continue
        g = g.sort_values("temperature_K")
        T = g.temperature_K.to_numpy()
        c = g.concentration.to_numpy()
        k = g.conductivity_mS_cm.to_numpy()
        fr = list(g.solvent_fractions)
        pos = g.index.to_numpy()
        hi = np.searchsorted(T, T + tol["temperature_abs_K"], side="right")
        for i in range(len(g)):
            for j in range(i + 1, hi[i]):
                if abs(c[i] - c[j]) > tol["concentration_abs"]:
                    continue
                if abs(k[i] - k[j]) > tol["conductivity_rel"] * max(k[i], k[j]):
                    continue
                if any(abs(fr[i].get(s, 0) - fr[j].get(s, 0)) > tol["ratio_abs"] for s in set(fr[i]) | set(fr[j])):
                    continue
                pairs.append((pos[i], pos[j]))
    return pairs


def condition_pairs(df: pd.DataFrame, tol: dict) -> tuple[list[tuple[int, int]], list[bool]]:
    """Cross-paper pairs at comparable conditions, and whether their conductivities agree.

    Comparable: same chemistry, ratio basis and concentration basis; every solvent
    fraction within ratio_abs; concentration within concentration_abs; temperature
    within temperature_abs_K. Agree: conductivity within conductivity_rel."""
    pairs, agree = [], []
    ok = df.conductivity_mS_cm.notna() & (df.conductivity_mS_cm > 0)
    for _, g in df[ok].groupby(["chemistry_system_id", "ratio_basis", "concentration_basis"], sort=False):
        if g.source_doi.nunique() < 2:
            continue
        g = g.sort_values("temperature_K")
        T, c, k = g.temperature_K.to_numpy(), g.concentration.to_numpy(), g.conductivity_mS_cm.to_numpy()
        d, fr, pos = g.source_doi.to_numpy(), list(g.solvent_fractions), g.index.to_numpy()
        hi = np.searchsorted(T, T + tol["temperature_abs_K"], side="right")
        for i in range(len(g)):
            for j in range(i + 1, hi[i]):
                if d[i] == d[j] or abs(c[i] - c[j]) > tol["concentration_abs"]:
                    continue
                if any(abs(fr[i].get(s, 0) - fr[j].get(s, 0)) > tol["ratio_abs"] for s in set(fr[i]) | set(fr[j])):
                    continue
                pairs.append((pos[i], pos[j]))
                agree.append(abs(k[i] - k[j]) <= tol["conductivity_rel"] * max(k[i], k[j]))
    return pairs, agree


def curate(cfg: Config) -> dict[str, pd.DataFrame]:
    man = RunManifest("curate", cfg)
    species = chemistry.load_species(cfg.p("mappings"))
    man.add_input(cfg.p("mappings"))
    srcs = cfg["sources"]
    parts = []
    for ds, spec in srcs.items():
        path = raw_data_file(cfg, ds)
        man.add_input(path)
        if ds == "calisol23":
            parts.append(read_calisol(path, spec["commit"], spec["repo"], spec["data_file"]))
        elif ds == "rahmanian2023":
            parts.append(read_rahmanian(path, spec["record"], spec["data_file"]))
        else:  # pragma: no cover
            raise ValueError(f"no reader for {ds}")
    raw = pd.concat(parts, ignore_index=True)
    norm = Normalizer(cfg, species)
    normed = pd.DataFrame([norm.normalize(r) for r in raw.itertuples(index=False)])
    df = pd.concat([raw.drop(columns=["concentration_original"]), normed], axis=1)
    df["measurement_id"] = df.dataset_id + ":" + df.original_row_id

    # ---- identities
    rd = cfg["curation"]["formulation_ratio_decimals"]
    cd = cfg["curation"]["formulation_concentration_decimals"]
    df["solvent_ids"] = df.solvent_fractions.map(lambda f: "+".join(sorted(f)))
    df["n_solvents"] = df.solvent_fractions.map(len).astype("int64")
    df["chemistry_system_id"] = df.salt_id + "|" + df.solvent_ids
    df["_recipe"] = [
        f"{rb}:" + ",".join(f"{k}={v:.{rd}f}" for k, v in sorted(f.items())) for rb, f in zip(df.ratio_basis, df.solvent_fractions)
    ]
    df["solvent_recipe_id"] = [_hid("R", s, r) for s, r in zip(df.chemistry_system_id, df._recipe)]
    df["formulation_id"] = [
        _hid("F", s, r, b, f"{c:.{cd}f}") for s, r, b, c in zip(df.chemistry_system_id, df._recipe, df.concentration_basis, df.concentration)
    ]
    df["series_id"] = np.where(
        df.series_hint.notna(),
        df.dataset_id + ":" + df.series_hint.fillna(""),
        df.dataset_id + ":" + df.source_doi + ":" + df.solvent_recipe_id,
    )
    df["solvent_fractions_reported"] = [
        json.dumps(r if r is not None else {k: round(v, 6) for k, v in f.items()}, sort_keys=True)
        for r, f in zip(df.solvent_fractions_raw, df.solvent_fractions)
    ]

    # ---- proven copies -> duplicate groups, collapse within source, merge source groups across
    df["source_group_id"] = "doi:" + df.source_doi
    tol = cfg["curation"]["copy_tolerance"]
    pairs = find_copies(df, tol)
    # Inside one source only exact copies are proven duplicates; tolerance matches
    # there are explicit replicates or repeated readings (kept, grouped below).
    # Across sources a tolerance match is a suspected reuse: kept, but linked.
    # An isolated cross-source agreement can be coincidence, so a source pair is
    # only linked when it shares at least min_cross_source_matches points.
    k = df.conductivity_mS_cm.to_numpy()
    doi = df.source_doi.to_numpy()
    pair_counts: dict[tuple[str, str], int] = {}
    for a, b in pairs:
        if doi[a] != doi[b]:
            key = tuple(sorted((doi[a], doi[b])))
            pair_counts[key] = pair_counts.get(key, 0) + 1
    kept_pairs, link_status = [], []
    for a, b in pairs:
        if doi[a] == doi[b]:
            exact = abs(k[a] - k[b]) <= tol["within_source_exact_rel"] * max(k[a], k[b])
            status = "exact_copy_collapsed" if exact else "within_source_near_match_kept_as_repeat"
        else:
            n = pair_counts[tuple(sorted((doi[a], doi[b])))]
            status = "suspected_reuse_linked" if n >= tol["min_cross_source_matches"] else "isolated_agreement_not_linked"
        link_status.append(status)
        if status in ("exact_copy_collapsed", "suspected_reuse_linked"):
            kept_pairs.append((a, b))
    all_pairs, pairs = pairs, kept_pairs
    uf = UnionFind(df.index)
    for a, b in pairs:
        uf.union(a, b)
    root = np.array([uf.find(i) for i in df.index])
    df["duplicate_group_id"] = ["D" + df.measurement_id.iat[r] for r in root]
    df["duplicate_of"] = None
    src_uf = UnionFind(df.source_group_id.unique())
    for gid, g in df[df.duplicate_group_id.duplicated(keep=False)].groupby("duplicate_group_id"):
        canonical = g.index.min()
        for i in g.index:
            if i == canonical:
                continue
            if df.at[i, "source_doi"] == df.at[canonical, "source_doi"]:
                if abs(k[i] - k[canonical]) > tol["within_source_exact_rel"] * max(k[i], k[canonical]):
                    continue  # same paper reached only through another source: keep both
                df.at[i, "duplicate_of"] = df.at[canonical, "measurement_id"]
                df.at[i, "reasons"] = df.at[i, "reasons"] + ["duplicate_copy_collapsed"]
            else:
                df.at[i, "flags"] = df.at[i, "flags"] + ["suspected_cross_source_copy"]
                df.at[canonical, "flags"] = df.at[canonical, "flags"] + ["suspected_cross_source_copy"]
                src_uf.union(df.at[i, "source_group_id"], df.at[canonical, "source_group_id"])
        for i in g.index:
            df.at[i, "prov"] = df.at[i, "prov"] + ["R09"]
    # ---- campaign-level linking (v0.2+): the same measurement can be tabulated in
    # one paper and plotted (then digitized) in another, so agreement is tested
    # with looser tolerances over every comparable pair of points.
    camp = cfg["curation"].get("campaign_linking", {})
    campaign = pd.DataFrame(columns=["source_doi_a", "source_doi_b", "n_comparable", "n_agree", "frac_agree", "linked"])
    campaign_pairs: list[tuple[int, int]] = []
    if camp.get("enabled"):
        cpairs, agree = condition_pairs(df, camp)
        ev: dict[tuple[str, str], list[int]] = {}
        for (a, b), ok in zip(cpairs, agree):
            key = tuple(sorted((doi[a], doi[b])))
            e = ev.setdefault(key, [0, 0])
            e[0] += 1
            e[1] += int(ok)
        rows = []
        for (da, db), (n, n_ok) in sorted(ev.items()):
            linked = n >= camp["min_comparable"] and n_ok / n >= camp["min_agreement"]
            rows.append({"source_doi_a": da, "source_doi_b": db, "n_comparable": n, "n_agree": n_ok, "frac_agree": n_ok / n, "linked": linked})
            if linked:
                src_uf.union("doi:" + da, "doi:" + db)
        campaign = pd.DataFrame(rows, columns=campaign.columns)
        linked_keys = {(r.source_doi_a, r.source_doi_b) for r in campaign.itertuples() if r.linked}
        campaign_pairs = [(a, b) for (a, b), ok in zip(cpairs, agree) if ok and tuple(sorted((doi[a], doi[b]))) in linked_keys]
    df["source_group_id"] = df.source_group_id.map(src_uf.find)
    if campaign_pairs:
        # agreeing points of linked papers are treated as suspected copies, so they
        # can never sit on opposite sides of any split
        duf = UnionFind(df.duplicate_group_id.unique())
        for a, b in campaign_pairs:
            duf.union(df.at[a, "duplicate_group_id"], df.at[b, "duplicate_group_id"])
            for i in (a, b):
                df.at[i, "prov"] = df.at[i, "prov"] + ["R12"]
        df["duplicate_group_id"] = df.duplicate_group_id.map(duf.find)
    links = pd.DataFrame(
        [(df.at[a, "measurement_id"], df.at[b, "measurement_id"], df.at[a, "source_doi"], df.at[b, "source_doi"], st) for (a, b), st in zip(all_pairs, link_status)]
        + [(df.at[a, "measurement_id"], df.at[b, "measurement_id"], df.at[a, "source_doi"], df.at[b, "source_doi"], "campaign_near_match") for a, b in campaign_pairs],
        columns=["measurement_id_a", "measurement_id_b", "source_doi_a", "source_doi_b", "status"],
    )

    # ---- author groups (reviewed mapping from author lists; not a verified lab id)
    ag_path = cfg.data["paths"].get("author_groups")
    if ag_path:
        ag = pd.read_csv(cfg.root / ag_path).set_index("source_doi").author_group_id
        missing = sorted(set(df.source_doi) - set(ag.index))
        if missing:
            raise ValueError(f"author-group mapping lacks {missing}")
        # an author group must contain whole source groups
        df["author_group_id"] = df.source_doi.map(ag)
        bad = df.groupby("source_group_id").author_group_id.nunique()
        if (bad > 1).any():
            raise ValueError(f"source groups spanning author groups: {bad[bad > 1].index.tolist()}")
    else:
        df["author_group_id"] = df.source_group_id

    # ---- replicate groups: same source group, formulation and temperature bin
    tb = cfg["curation"]["replicate_temperature_bin_K"]
    tbin = np.round(df.temperature_K / tb).astype("int64")
    df["replicate_group_id"] = [_hid("P", s, f, t) for s, f, t in zip(df.source_group_id, df.formulation_id, tbin)]

    # ---- split unit: formulation joined with any copy group spanning formulations
    su = UnionFind(df.formulation_id.unique())
    for _, g in df.groupby("duplicate_group_id"):
        fids = g.formulation_id.unique()
        for f in fids[1:]:
            su.union(fids[0], f)
    df["split_unit_id"] = df.formulation_id.map(lambda f: "U" + su.find(f)[1:])

    # ---- tracks, inclusion
    df["track_id"] = [_assign_track(cfg, b, rb) for b, rb in zip(df.concentration_basis, df.ratio_basis)]
    for i in df.index[df.track_id.isna()]:
        df.at[i, "reasons"] = df.at[i, "reasons"] + ["no_basis_specific_track"]
    df["inclusion_status"] = np.where(df.reasons.map(len) == 0, "included", "excluded")
    df.loc[df.inclusion_status == "excluded", "track_id"] = None
    df["exclusion_reason"] = df.reasons.map(lambda r: r[0] if r else None)
    df["flags"] = df["flags"].map(lambda f: ";".join(dict.fromkeys(f)) if f else None)
    df["conversion_provenance"] = df.prov.map(lambda p: ";".join(dict.fromkeys(p)))
    df["log10_conductivity"] = units.safe_log10(df.conductivity_mS_cm.to_numpy())
    df["lab_id"] = None
    df["additive_ids"] = None
    df["has_uncertainty"] = df.conductivity_uncertainty_mS_cm.notna()
    df["has_method"] = df.measurement_method.notna()
    df["has_lab_id"] = False

    # ---- components table (first occurrence of each formulation defines amounts)
    sp_by_id = species[species.review_status == "verified"].drop_duplicates("canonical_id").set_index("canonical_id")
    comp_rows = []
    for _, r in df.drop_duplicates("formulation_id").iterrows():
        resolved = all(cid in sp_by_id.index for cid in [r.salt_id, *r.solvent_fractions])
        mw = {cid: chemistry.molar_mass(sp_by_id.at[cid, "smiles"]) for cid in r.solvent_fractions if cid in sp_by_id.index}
        xf = None
        if resolved and r.ratio_basis == "w":
            xf = units.mass_to_mole_fractions(r.solvent_fractions, mw)
        elif resolved and r.ratio_basis == "mol":
            xf = dict(r.solvent_fractions)
        basis_name = {"w": "mass_fraction", "v": "volume_fraction", "mol": "mole_fraction"}[r.ratio_basis]
        for cid, f in sorted(r.solvent_fractions.items()):
            comp_rows.append(
                {
                    "formulation_id": r.formulation_id,
                    "component_id": cid,
                    "role": "solvent",
                    "smiles": chemistry.canonical_smiles(sp_by_id.at[cid, "smiles"]) if cid in sp_by_id.index else None,
                    "cation_smiles": None,
                    "cation_stoich": np.nan,
                    "anion_smiles": None,
                    "anion_stoich": np.nan,
                    "amount_value": f,
                    "amount_basis": basis_name,
                    "mole_fraction": xf[cid] if xf else np.nan,
                    "molar_mass": mw.get(cid, np.nan),
                }
            )
        s = sp_by_id.loc[r.salt_id] if r.salt_id in sp_by_id.index else None
        comp_rows.append(
            {
                "formulation_id": r.formulation_id,
                "component_id": r.salt_id,
                "role": "salt",
                "smiles": chemistry.canonical_smiles(s.anion_smiles) if s is not None else None,
                "cation_smiles": s.cation_smiles if s is not None else None,
                "cation_stoich": float(s.cation_stoich) if s is not None else np.nan,
                "anion_smiles": s.anion_smiles if s is not None else None,
                "anion_stoich": float(s.anion_stoich) if s is not None else np.nan,
                "amount_value": float(r.concentration),
                "amount_basis": r.concentration_basis,
                "mole_fraction": np.nan,
                "molar_mass": chemistry.salt_molar_mass(s) if s is not None else np.nan,
            }
        )
    comps = pd.DataFrame(comp_rows)

    # ---- exclusions log
    exc = df[df.inclusion_status == "excluded"][["measurement_id", "dataset_id", "original_row_id", "source_doi", "reasons", "flags", "duplicate_of"]].copy().rename(columns={"flags": "quality_flags"})
    exc = exc.explode("reasons").rename(columns={"reasons": "reason"})
    exc["rule_order"] = exc.groupby("measurement_id").cumcount()

    df = df.rename(columns={"flags": "quality_flags"})
    meas = df[list(MEASUREMENT_COLUMNS)].copy()
    for col in ["duplicate_of", "track_id", "exclusion_reason", "quality_flags", "lab_id", "additive_ids", "measurement_method"]:
        meas[col] = meas[col].astype("string")
    for col in ["conductivity_mS_cm", "conductivity_original", "conductivity_uncertainty_mS_cm", "concentration_original"]:
        meas[col] = meas[col].astype(float)
    validate_measurements(meas)
    comps_inc = comps[comps.formulation_id.isin(meas.loc[meas.inclusion_status == "included", "formulation_id"])]
    validate_components(comps_inc)

    out = cfg.p("processed")
    out.mkdir(parents=True, exist_ok=True)
    meas.to_parquet(out / "measurements.parquet", index=False)
    comps.to_parquet(out / "components.parquet", index=False)
    exc.to_csv(out / "exclusions.csv", index=False)
    links.to_csv(out / "copy_links.csv", index=False)
    campaign.to_csv(out / "campaign_links.csv", index=False)
    pd.DataFrame([{"rule_id": k, "rule": v} for k, v in CURATION_RULES.items()]).to_csv(out / "curation_rules.csv", index=False)
    desc = chemistry.descriptor_table(species)
    cache = cfg.p("cache")
    cache.mkdir(parents=True, exist_ok=True)
    desc.to_parquet(cache / "descriptors.parquet", index=False)
    for f in ["measurements.parquet", "components.parquet", "exclusions.csv", "copy_links.csv", "campaign_links.csv", "curation_rules.csv"]:
        man.add_output(out / f)
    man.add_output(cache / "descriptors.parquet")
    man.notes = {
        "n_raw": int(len(meas)),
        "n_included": int((meas.inclusion_status == "included").sum()),
        "n_copy_pairs": len(pairs),
        "included_by_track": meas[meas.inclusion_status == "included"].track_id.value_counts().to_dict(),
        "exclusion_reasons": meas.exclusion_reason.value_counts().to_dict(),
    }
    man.write(out / "runs")
    return {"measurements": meas, "components": comps, "exclusions": exc}


def load_processed(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = cfg.p("processed")
    return pd.read_parquet(out / "measurements.parquet"), pd.read_parquet(out / "components.parquet")
