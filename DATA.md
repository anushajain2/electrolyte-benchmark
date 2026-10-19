# Data card: electrolyte generalization benchmark (v0.1 and v0.2)

## Sources

| dataset_id | Release pinned | Role | Licence / redistribution |
| --- | --- | --- | --- |
| `calisol23` | GitHub `Pele0599/CALiSol-23` @ `02334f51d581b45d6760fa02048ffa0d84c056d8` (`CALiSol-23 Dataset.csv`) | backbone | No licence file in the repository; the article is CC BY 4.0 but redistribution of the compiled rows is not stated. **Raw rows are not redistributed**; `elbench ingest` downloads the pinned commit and `elbench curate` rebuilds every derived table. |
| `rahmanian2023` | Zenodo record 7244939 (`Conductivtiy_experiment.csv`, md5 `f84cfeb764e5d4db22cdb8b1253a0935`) | reserved external evaluation | CC BY 4.0 |

Retrieval date, URL, SHA-256, MD5, byte size and the original field names of every raw file are in `data/raw/sources.csv`. Raw files are written read-only, and `ingest` refuses to continue if a file changes after its first download.

Published counts (13,825 CALiSol observations from 27 DOIs; 5,035 Rahmanian experiments) are starting material, not independent examples. The independent counts are below.

## Canonical tables

`data/processed/v0.1/measurements.parquet`: one row per reported measurement (18,860 rows, including the 678 excluded ones). Column definitions are in `src/elbench/schema.py` (`MEASUREMENT_COLUMNS`) and are validated on write. Groups:

* **Identity and provenance**: `measurement_id` (`<dataset>:<raw row>`), `dataset_id`, `original_row_id`, `source_doi`, `source_group_id`, `series_id`, `duplicate_group_id`, `replicate_group_id`, `split_unit_id`, `lab_id` (always empty: never inferred from a DOI)
* **Formulation**: `formulation_id`, `solvent_recipe_id`, `chemistry_system_id`, `salt_id`, `solvent_ids`, `n_solvents`, `ratio_basis`, `solvent_fractions_reported`, `additive_ids` (none recorded separately by either source)
* **Conditions**: `temperature_K`, `temperature_original`, `temperature_unit_original`, `concentration_original`, `concentration_unit_original`, `concentration_basis`, `concentration`, `concentration_conversion`
* **Target**: `conductivity_mS_cm`, `conductivity_original`, `conductivity_unit_original`, `log10_conductivity`, `conductivity_uncertainty_mS_cm` (never reported by these sources), `measurement_method`
* **Quality and traceability**: `inclusion_status`, `exclusion_reason`, `quality_flags`, `conversion_provenance` (rule ids), `source_record_link` (GitHub line or Zenodo label), `duplicate_of`, `track_id`, `has_uncertainty`, `has_method`, `has_lab_id`

`components.parquet`: one row per (formulation, component). Solvents carry the reported fraction and basis plus an exact mole fraction where convertible; salts are represented as ions (`cation_smiles`, `anion_smiles`, stoichiometry, whole-salt molar mass).

`exclusions.csv` lists every failed rule for every excluded row (a row can fail several). `copy_links.csv` lists every pair matched by the copy rule and what was done with it. `curation_rules.csv` defines the rule ids.

## Curation rules (summary)

| Rule | Effect |
| --- | --- |
| R01, R02 | Temperature to K and conductivity to mS/cm, exact (Rahmanian: °C + 273.15, S/cm × 1000). CALiSol `k` is in mS/cm per the article. |
| R03 | Concentration kept in its reported basis. Molality and molarity are separate tracks; no density data, so no conversion between them. |
| R04, R05 | Rahmanian molality and solvent mass fractions computed exactly from the weighed masses (salt molar mass from RDKit, 151.90 g/mol for LiPF6, matching the source metadata). |
| R06 | Reported solvent fractions renormalized when within 0.02 of 1 (e.g. 0.33 × 3). Larger deviations would be quarantined (none occurred). |
| R07 | Mass fractions converted exactly to mole fractions; volume fractions are not converted. |
| R08 | Species mapped through the reviewed `data/mappings/species.csv`. |
| R09 | Exact in-source copies collapsed (none found). Cross-source matches (same chemistry, ratio ±0.005, c ±0.001, T ±0.05 K, κ ±0.5%) mark suspected reuse; a pair of papers sharing ≥3 such points becomes one source group. |
| R10 | Nonpositive reported conductivities kept as excluded, flagged records; no positive value is invented. |
| R11 | Values below 1e-9 mS/cm are treated as **suspected numerical artifacts** and **excluded** (not censored, never entered into the log target). The three affected values are 1.11e-16 (= 2⁻⁵³) twice and 1.78e-15 (= 2⁻⁴⁹): powers of two at machine-precision scale, a strong warning sign of arithmetic on a zero, but not proof that the original reported zero. The threshold is a policy choice; its distance from the smallest smooth-sweep values (~1e-6 mS/cm near 200 K) does not validate it independently. Added after failure-case tracing; not yet traced to the original figures. Both pre- and post-R11 results are kept. |
| R12 (v0.2) | Provenance-based grouping: two papers are linked when ≥20 point pairs are comparable (T ±1 K, c ±0.03, solvent fractions ±0.02) and ≥50% agree within 5% in conductivity; linked papers and their connected components form one source group, and agreeing points become one duplicate group so they cannot straddle any split. This is a conservative leakage grouping, not proof of one experimental campaign; each pair's evidence tier (documented / suspected reuse / conservative grouping) is in `reports/v0.2/provenance_pairs.csv`. Thresholds set after the v0.1 review; the v0.1 rule and author-group holdouts are reported as alternative grouping schemes. |

### Exclusions (first failing rule)

| Reason | Rows | Effect on coverage |
| --- | --- | --- |
| `missing_conductivity` | 523 | 3 papers; no chemistry system lost |
| `unresolved_component` | 96 | 11 chemistry systems of the 1991 J. Power Sources survey lost (molar_v track only) |
| `negative_conductivity_nonphysical` | 27 | low-temperature points from 5 papers |
| `source_quarantine` | 23 | one paper (`10.1007/s10008-002-0300-9`) and its 2 chemistry systems lost |
| `reported_zero_conductivity` | 8 | 5 reported zeros + 3 floating-point zeros (R11) |
| `negative_concentration_nonphysical` | 1 | |

Rahmanian's `suspectedMeasurementError` metadata flag is honoured, but no row in the release carries it. The full table, including the chemistry systems, sources and components each rule removes, is in `reports/v0.1/audit/exclusion_impact.csv`.

### Molecular identity review

Disputed CALiSol mappings are quarantined, not guessed (`review_status = disputed`): `DMM` (CALiSol: dipropylene glycol dimethyl ether; the abbreviation more often means dimethoxymethane), `Ethyldiglyme` (CALiSol: 2-(2-ethoxyethoxy)ethanol, an alcohol; usually diethylene glycol diethyl ether) and `Ethylmonoglyme` (CALiSol: 2-methoxyethanol; usually 1,2-diethoxyethane). Corrected author SMILES: LiPDI (missing Li and charge), LiTDI and LiBMB (invalid or neutral forms), LiTFSI (covalent Li–N bond). `LiTFSI` and `LiN(CF3SO2)2` are the same compound and share one identity, so a component holdout cannot leak it.

## Counts by track (included rows; source groups under the v0.1 rule)

| Track | Dataset | Measurements | Formulations | Series | Replicate groups | Source groups | Chemistry systems |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| molal_w | calisol23 | 10,481 | 1,648 | 97 | 9,942 | 10 | 20 |
| molal_w | rahmanian2023 (reserved) | 5,035 | 104 | 504 | 1,039 | 1 | 2 |
| molar_x | calisol23 | 1,397 | 105 | 44 | 1,379 | 9 | 28 |
| molar_v | calisol23 | 1,269 | 212 | 142 | 1,241 | 5 | 88 |

**v0.2 grouping.** Of 27 CALiSol papers, 26 keep at least one included row and 12 fall in the primary track (basis eligibility). With provenance-based grouping (R12) those 12 papers form 7 source groups (v0.1 rule: 10), 4 of which are testable in source transfer, and 3 author groups. One author group (Ding/Jow/Xu, 8 of the 10 v0.1 source groups) supplies 10,108 of 10,481 rows. Campaign evidence is in `data/processed/v0.2/campaign_links.csv`; author metadata in `data/mappings/source_metadata.csv` and the reviewed grouping in `data/mappings/source_author_groups.csv`.

Most CALiSol values are digitized from figures, so temperatures and concentrations often carry digitization noise (e.g. 577 distinct concentration values in one paper). A formulation identity rounds solvent fractions and concentration to 0.01 for the key only; reported values are untouched.

## Known limitations

* Source groups are DOIs merged by detected reuse. Papers from the same laboratory without shared points stay separate groups. Several CALiSol papers on LiPF6, LiBF4 and LiBOB in PC/EC/DEC appear to come from one research programme, so source independence is likely overstated rather than understated.
* No measurement uncertainty or measurement method is reported for CALiSol rows.
* The reserved dataset overlaps the backbone in chemistry only through LiPF6 in EC+EMC+PC (110 CALiSol rows from one paper). Of its 5,035 rows, 153 fall inside the CALiSol common support; most of the remainder have no CALiSol recipe within 0.10 in solvent fractions.

See `ISSUES.md` for the open issue log.
