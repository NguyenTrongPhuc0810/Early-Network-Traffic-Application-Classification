# Project Structure

This repository is being refactored for an Early Traffic Classification workflow that can move from Python training to C code generation for eBPF/XDP deployment.

## Target Layout

```text
.
├── src/
│   ├── ml_pipeline/
│   │   ├── __init__.py
│   │   ├── data_loader.py      # Load and preprocess SPLT parquet datasets
│   │   ├── train.py            # Train Random Forest models
│   │   └── evaluate.py         # Reports, confusion matrices, metrics
│   └── ebpf_export/
│       ├── __init__.py
│       └── model_to_c.py       # Convert trained models into C functions
├── data/
│   ├── raw/                    # Local-only raw datasets
│   ├── processed/              # Local-only feature datasets
│   └── artifacts/              # Local-only models, metrics, exported files
├── docs/
│   └── PROJECT_STRUCTURE.md
├── setup_data.py               # Restore local data from external storage
├── requirements.txt            # Added in Step 4
└── README.md                   # Updated in Step 4
```

## Data Policy

Datasets, packet captures, GeoIP databases, model binaries, and generated artifacts are intentionally excluded from Git. Store them in an external location such as Google Drive, S3, GitHub Releases, or a private object store.

Expected local paths after restore:

```text
02-app-classification/data/final_dataset_63_classes_splt.parquet
act2-project/data/artifacts/application_63_classes_splt_train_eval/
```

The `.gitignore` blocks `*.parquet`, `*.csv`, `*.pcap`, `*.joblib`, `*.pkl`, and related heavy binary formats. If any heavy file was already tracked before this refactor, remove it from the Git index with:

```bash
git rm --cached path/to/file.parquet
git rm --cached path/to/model.joblib
```

Do not delete the local files unless you intentionally want to remove your working copy.
