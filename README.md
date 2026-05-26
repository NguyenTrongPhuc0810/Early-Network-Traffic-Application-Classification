# Early Network Traffic Application Classification

End-to-end SPLT pipeline for early network traffic application classification.
The repository is structured so a new user can clone it, install dependencies,
train the 63-class Random Forest model, and then run prediction/evaluation.

The included dataset is:

```text
data/raw/final_dataset_63_classes_splt.parquet
```

It contains curated SPLT features for 63 `application_name` classes. The trained
`model.joblib` is not tracked because the reference model is about 4 GB; clone
the repo and retrain it locally.

## Pipeline

```text
PCAP/PCAPNG
  -> NFStream flow extraction
  -> SPLT dataset preparation
  -> Random Forest train/evaluate
  -> prediction artifacts
```

Main package:

```text
src/ml_pipeline/
  nfstream_ingest.py  # PCAP -> flow parquet using NFStream
  prepare.py          # raw NFStream parquet -> curated SPLT parquet
  data_loader.py      # SPLT parsing and feature matrix construction
  train.py            # Random Forest training/evaluation
  evaluate.py         # reports and confusion matrix
  predict.py          # batch prediction with a trained model bundle
  cli.py              # single traffic-clf command
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -e .
```

For PCAP extraction with NFStream:

```bash
python -m pip install -e .[pcap]
```

## Train the Bundled 63-Class SPLT Model

```bash
python -m ml_pipeline.train ^
  --data data/raw/final_dataset_63_classes_splt.parquet ^
  --out-dir data/artifacts/application_63_classes_splt_train_eval ^
  --target-column application_name ^
  --feature-width 25 ^
  --n-estimators 100 ^
  --class-weight balanced
```

Equivalent installed CLI:

```bash
traffic-clf train ^
  --data data/raw/final_dataset_63_classes_splt.parquet ^
  --out-dir data/artifacts/application_63_classes_splt_train_eval
```

Outputs:

```text
data/artifacts/application_63_classes_splt_train_eval/
  model.joblib
  classification_report.txt
  classification_report.json
  confusion_matrix.png
  run_metadata.json
```

## Predict

```bash
python -m ml_pipeline.predict ^
  --model data/artifacts/application_63_classes_splt_train_eval/model.joblib ^
  --data data/raw/final_dataset_63_classes_splt.parquet ^
  --out data/artifacts/predictions.parquet
```

## Test a Local PCAP with the Trained Model

Use this when you only want to run a local `.pcap` or `.pcapng` file through the
trained model and see model predictions. This path applies the same inference
filter used by the model pipeline: `bidirectional_packets >= 10`.

Example for:

```text
D:\CCNA\raw_pcap\ytb_full_hd.pcapng
```

Step 1: extract SPLT flows with NFStream.

```bash
python -m ml_pipeline.nfstream_ingest ^
  --pcap D:\CCNA\raw_pcap\ytb_full_hd.pcapng ^
  --out data/interim/ytb_full_hd_flows.parquet ^
  --n-dissections 20 ^
  --splt-analysis 25 ^
  --no-statistical-analysis
```

Step 2: run model prediction only.

```bash
python -m ml_pipeline.predict ^
  --model data/artifacts/application_63_classes_splt_train_eval/model.joblib ^
  --data data/interim/ytb_full_hd_flows.parquet ^
  --out data/artifacts/pcap_ytb_full_hd/predictions.parquet ^
  --min-packets 10
```

The command prints `prediction_counts` and writes the full per-flow output to:

```text
data/artifacts/pcap_ytb_full_hd/predictions.parquet
```

For another PCAP, replace the `--pcap`, `--out`, and prediction output paths.

## Evaluate Existing Model

```bash
python -m ml_pipeline.evaluate ^
  --model data/artifacts/application_63_classes_splt_train_eval/model.joblib ^
  --data data/raw/final_dataset_63_classes_splt.parquet ^
  --out-dir data/artifacts/application_63_classes_splt_eval
```

## Start from a PCAP

Install NFStream first:

```bash
python -m pip install -e .[pcap]
```

Extract flows:

```bash
python -m ml_pipeline.nfstream_ingest ^
  --pcap samples/input.pcap ^
  --out data/interim/flows.parquet ^
  --n-dissections 20 ^
  --splt-analysis 25
```

Prepare a curated SPLT dataset:

```bash
python -m ml_pipeline.prepare ^
  --input data/interim/flows.parquet ^
  --out data/processed/splt_dataset.parquet ^
  --target-column application_name
```

Then train or predict using the prepared parquet.

## PowerShell Shortcuts

```powershell
.\scripts\01_extract_pcap.ps1 -Pcap samples\input.pcap
.\scripts\02_prepare_dataset.ps1
.\scripts\03_train_eval.ps1
.\scripts\04_predict.ps1
```

## Data and Artifacts

Tracked:

- `data/raw/final_dataset_63_classes_splt.parquet`
- reference report files under `data/artifacts/application_63_classes_splt_train_eval/`

Ignored:

- generated predictions
- generated `model.joblib`
- packet captures
- local cache files

This keeps the project cloneable while still letting users reproduce the model
from the bundled 63-class SPLT dataset.
