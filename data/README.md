# Data

## Processed feature tables

`processed/` contains the paper-ready derived features compressed as `.csv.gz`. Pandas reads these files directly.

Physical files:

- `clean_reextract`
- vibration/electrical gain: 0.75, 0.50, 0.25
- vibration/electrical drift: 0.25, 0.50, 1.00 RMS

The two missing-modality conditions are generated after fold-specific standardization by zeroing all 41 vibration features or all 55 electrical features.

Each table has 4,752 rows and 102 columns:

- 41 vibration features
- 55 electrical features
- 6 metadata columns: `label`, `label_id`, `load`, `freq`, `source_file`, `window_id`

Use `checksums.sha256` to verify file integrity.

## Raw data

The 13 GB compressed inverter-fed raw subset is not duplicated in this repository. Download it from the official ESTOGU Zenodo record and follow `raw/README.md`.

## Provenance

Derived from ESTOGU, DOI 10.5281/zenodo.18222578, licensed CC BY 4.0. See `DATA_LICENSE.md`.
