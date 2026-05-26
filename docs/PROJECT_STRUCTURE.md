# Project Structure

The repository follows a simple ML project layout with one package and a clear
runtime path from packet capture to classification output.

```text
.
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   │   └── final_dataset_63_classes_splt.parquet
│   ├── interim/
│   ├── processed/
│   └── artifacts/
├── scripts/
│   ├── 01_extract_pcap.ps1
│   ├── 02_prepare_dataset.ps1
│   ├── 03_train_eval.ps1
│   └── 04_predict.ps1
├── src/
│   ├── ml_pipeline/
│   │   ├── nfstream_ingest.py
│   │   ├── prepare.py
│   │   ├── data_loader.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   └── cli.py
│   └── ebpf_export/
│       └── __init__.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Source Extraction

The clean modules were extracted from these original workspace files:

- `act2-project/src/act2_project/pipeline/pcap_ingest.py`
- `act2-project/src/act2_project/pipeline/act2_prepare.py`
- `act2-project/src/act2_project/pipeline/splt_features.py`
- `act2-project/src/act2_project/pipeline/train.py`
- `act2-project/src/act2_project/pipeline/evaluate.py`
- `act2-project/tools/splt_width_experiments.py`
- `act2-project/data/artifacts/application_63_classes_splt_train_eval/run_metadata.json`

The reference run uses:

- target column: `application_name`
- class count: 63
- SPLT width: 25 packets
- feature count: 75 flattened values
- model: Random Forest, 100 estimators, balanced class weights

## Git Policy

The curated 63-class SPLT parquet is tracked so users can train immediately
after cloning. Generated models and prediction outputs remain ignored.
