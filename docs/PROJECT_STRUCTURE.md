# Project Structure

This repository contains a focused SPLT 63-class traffic classification package.
It separates Python training code from future eBPF/XDP export code.

```text
.
├── src/
│   ├── ml_pipeline/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── ebpf_export/
│       └── __init__.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── artifacts/
├── docs/
│   └── PROJECT_STRUCTURE.md
├── setup_data.py
├── pyproject.toml
└── README.md
```

## Source Extraction

The clean package was extracted from these original workspace files:

- `act2-project/src/act2_project/pipeline/splt_features.py`
- `act2-project/src/act2_project/pipeline/act2_prepare.py`
- `act2-project/src/act2_project/pipeline/train.py`
- `act2-project/src/act2_project/pipeline/evaluate.py`
- `act2-project/tools/splt_width_experiments.py`
- `act2-project/data/artifacts/application_63_classes_splt_train_eval/run_metadata.json`

The 63-class model metadata shows `target_column = application_name` and
`feature_width = 25`, producing 75 flattened SPLT features.

## Data Policy

Datasets, packet captures, GeoIP databases, trained model binaries and generated
artifacts are excluded from Git. Restore them from external storage into:

```text
data/raw/final_dataset_63_classes_splt.parquet
data/artifacts/application_63_classes_splt_train_eval/
```

The `.gitignore` blocks `.parquet`, `.csv`, `.pcap`, `.joblib`, `.pkl` and other
large binary formats.
