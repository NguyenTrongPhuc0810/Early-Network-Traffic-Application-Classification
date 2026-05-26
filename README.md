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

Run these commands from the repository root:

```powershell
cd D:\CCNA\ml-flow-class-tutorial\splt63-github-export
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

```powershell
python -m ml_pipeline.train `
  --data data/raw/final_dataset_63_classes_splt.parquet `
  --out-dir data/artifacts/application_63_classes_splt_train_eval `
  --target-column application_name `
  --feature-width 25 `
  --n-estimators 100 `
  --class-weight balanced
```

Equivalent installed CLI:

```powershell
traffic-clf train `
  --data data/raw/final_dataset_63_classes_splt.parquet `
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

```powershell
python -m ml_pipeline.predict `
  --model data/artifacts/application_63_classes_splt_train_eval/model.joblib `
  --data data/raw/final_dataset_63_classes_splt.parquet `
  --out data/artifacts/predictions.parquet
```

## Test a Local PCAP with the Trained Model

Use this one-step command when you only want to run a local `.pcap` or
`.pcapng` file through the trained model and see model predictions. It extracts
NFStream SPLT flows in memory, applies the same inference filter used by the
model pipeline (`bidirectional_packets >= 10`), then writes compact model
predictions only.

Example for:

```text
D:\CCNA\raw_pcap\ytb_full_hd.pcapng
```

```powershell
python -m ml_pipeline.pcap_predict `
  --pcap D:\CCNA\raw_pcap\ytb_full_hd.pcapng `
  --model data/artifacts/application_63_classes_splt_train_eval/model.joblib `
  --out-dir data/artifacts/pcap_ytb_full_hd `
  --min-packets 10
```

Equivalent installed CLI after `python -m pip install -e .`:

```powershell
traffic-clf predict-pcap `
  --pcap D:\CCNA\raw_pcap\ytb_full_hd.pcapng `
  --model data/artifacts/application_63_classes_splt_train_eval/model.joblib `
  --out-dir data/artifacts/pcap_ytb_full_hd
```

The command prints `prediction_counts` and writes:

```text
data/artifacts/pcap_ytb_full_hd/predictions.parquet
data/artifacts/pcap_ytb_full_hd/pcap_prediction_metadata.json
```

`predictions.parquet` is intentionally compact and follows the same row schema
as `data/raw/final_dataset_63_classes_splt.parquet`, but without
`application_category_name`:

```text
splt_direction
splt_ps
splt_piat_ms
application_name
```

Here `application_name` is the model prediction for the filtered PCAP flow.

For another PCAP, replace only `--pcap` and optionally `--out-dir`.

## Evaluate Existing Model

```powershell
python -m ml_pipeline.evaluate `
  --model data/artifacts/application_63_classes_splt_train_eval/model.joblib `
  --data data/raw/final_dataset_63_classes_splt.parquet `
  --out-dir data/artifacts/application_63_classes_splt_eval
```

## Start from a PCAP

Install NFStream first:

```bash
python -m pip install -e .[pcap]
```

Extract flows:

```powershell
python -m ml_pipeline.nfstream_ingest `
  --pcap samples/input.pcap `
  --out data/interim/flows.parquet `
  --n-dissections 20 `
  --splt-analysis 25
```

Prepare a curated SPLT dataset:

```powershell
python -m ml_pipeline.prepare `
  --input data/interim/flows.parquet `
  --out data/processed/splt_dataset.parquet `
  --target-column application_name
```

Then train or predict using the prepared parquet.

## PowerShell Shortcuts

```powershell
.\scripts\01_extract_pcap.ps1 -Pcap samples\input.pcap
.\scripts\02_prepare_dataset.ps1
.\scripts\03_train_eval.ps1
.\scripts\04_predict.ps1
.\scripts\05_predict_pcap.ps1 -Pcap D:\CCNA\raw_pcap\ytb_full_hd.pcapng
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
