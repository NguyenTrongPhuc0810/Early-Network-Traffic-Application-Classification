# Early Network Traffic Application Classification

Clean Python package for early network traffic application classification with
SPLT features and a 63-class Random Forest model.

The code in `src/ml_pipeline` was extracted from the original tutorial
workspace and refactored into a small package that is easier to test, retrain
and later bridge into C/eBPF export code.

## Package Layout

```text
src/
  ml_pipeline/
    data_loader.py   # parquet loading, SPLT parsing, feature matrix building
    train.py         # Random Forest train/eval pipeline
    evaluate.py      # classification report and confusion matrix generation
  ebpf_export/
    __init__.py      # reserved for model-to-C export modules
data/
  raw/               # local-only datasets, ignored by Git
  processed/         # local-only prepared data, ignored by Git
  artifacts/         # local-only models/reports, ignored by Git
```

## Data Policy

Large files are not stored in Git:

- parquet/csv datasets
- packet captures
- `model.joblib` and other trained model binaries

Place the 63-class SPLT dataset at:

```text
data/raw/final_dataset_63_classes_splt.parquet
```

Place trained artifacts at:

```text
data/artifacts/application_63_classes_splt_train_eval/
```

Use `setup_data.py` to restore a zip bundle from local storage or an external
URL.

## Train

```bash
python -m pip install -e .
python -m ml_pipeline.train ^
  --data data/raw/final_dataset_63_classes_splt.parquet ^
  --out-dir data/artifacts/application_63_classes_splt_train_eval ^
  --target-column application_name ^
  --feature-width 25
```

The output bundle contains the fitted Random Forest plus stable feature
metadata needed by future C/eBPF export code:

```text
model.joblib
classification_report.txt
classification_report.json
confusion_matrix.png
run_metadata.json
```

## Evaluate Existing Model

```bash
python -m ml_pipeline.evaluate ^
  --model data/artifacts/application_63_classes_splt_train_eval/model.joblib ^
  --data data/raw/final_dataset_63_classes_splt.parquet ^
  --out-dir data/artifacts/application_63_classes_splt_eval
```

## Why SPLT

SPLT features use early packet-level sequences: direction, packet size and
packet inter-arrival time. This keeps the model close to what can be collected
early in a flow and prepares the project for an AI-driven QoS eBPF/XDP path,
where the trained model can later be translated into C-compatible decision
logic.
