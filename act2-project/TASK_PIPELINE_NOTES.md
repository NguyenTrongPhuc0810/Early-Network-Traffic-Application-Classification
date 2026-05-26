# Task Pipeline Notes

This file records the task-level experiments built on top of `act2-project`.

## Confirmed Task Labels

- `ytb*` -> `Media`
- `pubg*`, `lienquan*`, `lol*` -> `Game`
- `zoom*`, `MS_team*` -> `Collaborative`
- `download*`, `chplay_download*`, `download_EA*` -> `Download`

PCAP source folder:

- `D:\CCNA\raw_pcap`

## Implemented Commands

- `build-task-dataset`
- `build-aux-task-data`
- `task-train-eval`
- `task-pcap-predict`

## Main Datasets Created

- Task-labeled PCAP SPLT dataset:
  - [task_labeled_splt_real.parquet](/D:/CCNA/ml-flow-class-tutorial/ml-flow-class-tutorial/act2-project/data/processed/task_labeled_splt_real.parquet)
- Auxiliary SPLT dataset from `df_final_model_data`:
  - [my_df_act3_splt.parquet](/D:/CCNA/ml-flow-class-tutorial/ml-flow-class-tutorial/act2-project/data/processed/my_df_act3_splt.parquet)
- Tutorial-style SPLT dataset from `df_final_model_data`:
  - [df_final_model_data_splt_real.parquet](/D:/CCNA/ml-flow-class-tutorial/ml-flow-class-tutorial/act2-project/data/processed/df_final_model_data_splt_real.parquet)

## Task Dataset Summary

- `task_labeled_splt_real.parquet`
  - `3525` flows
  - `26` captures
  - classes:
    - `Collaborative: 563`
    - `Download: 455`
    - `Game: 1991`
    - `Media: 516`

## Best Task-Level Result So Far

Current best capture-level result is the task-only model:

- artifacts:
  - [run_metadata.json](/D:/CCNA/ml-flow-class-tutorial/ml-flow-class-tutorial/act2-project/data/artifacts/task_train_eval_task_only/run_metadata.json)
  - [capture_classification_report.txt](/D:/CCNA/ml-flow-class-tutorial/ml-flow-class-tutorial/act2-project/data/artifacts/task_train_eval_task_only/capture_classification_report.txt)
  - [flow_classification_report.txt](/D:/CCNA/ml-flow-class-tutorial/ml-flow-class-tutorial/act2-project/data/artifacts/task_train_eval_task_only/flow_classification_report.txt)
  - [capture_predictions.parquet](/D:/CCNA/ml-flow-class-tutorial/ml-flow-class-tutorial/act2-project/data/artifacts/task_train_eval_task_only/capture_predictions.parquet)
  - [oof_flow_predictions.parquet](/D:/CCNA/ml-flow-class-tutorial/ml-flow-class-tutorial/act2-project/data/artifacts/task_train_eval_task_only/oof_flow_predictions.parquet)
- metrics:
  - capture accuracy: `0.6538`
  - capture macro F1: `0.5485`
  - flow accuracy: `0.6085`
  - flow macro F1: `0.4100`

Capture-level report:

- `Collaborative`: precision `0.75`, recall `0.86`, f1 `0.80`
- `Download`: precision `0.00`, recall `0.00`, f1 `0.00`
- `Game`: precision `0.54`, recall `0.88`, f1 `0.67`
- `Media`: precision `1.00`, recall `0.57`, f1 `0.73`

## Auxiliary Result

Task model with auxiliary SPLT data did not improve capture-level accuracy:

- artifacts:
  - [run_metadata.json](/D:/CCNA/ml-flow-class-tutorial/ml-flow-class-tutorial/act2-project/data/artifacts/task_train_eval_with_aux/run_metadata.json)
  - [capture_classification_report.txt](/D:/CCNA/ml-flow-class-tutorial/ml-flow-class-tutorial/act2-project/data/artifacts/task_train_eval_with_aux/capture_classification_report.txt)
- metrics:
  - capture accuracy: `0.6154`
  - capture macro F1: `0.5095`

## Extra Experiments Tried

- `support_weight = 1.0`, `min_packets = 5`
  - [run_metadata.json](/D:/CCNA/ml-flow-class-tutorial/ml-flow-class-tutorial/act2-project/data/artifacts/task_train_eval_support1/run_metadata.json)
  - capture accuracy stayed at `0.6538`
- `support_weight = 1.0`, `min_packets = 3`
  - [run_metadata.json](/D:/CCNA/ml-flow-class-tutorial/ml-flow-class-tutorial/act2-project/data/artifacts/task_train_eval_support1_min3/run_metadata.json)
  - capture accuracy dropped to `0.6154`

## Observations

- Game captures often contain many generic `TLS`, `DNS`, `Microsoft`, and other background flows.
- Several game captures do not expose clear game-specific DPI app names after early filtering.
- Download captures are the hardest class under the current SPLT-only setup.
- The current flow-level task model is not sufficient on its own for strong `Download` recall.

## Suggested Next Step

The next promising step is a two-stage model:

1. Use a SPLT flow model as a feature extractor.
2. Build a capture-level meta-classifier from aggregated flow probabilities and packet statistics.

That would stay aligned with early classification while using the capture label as the real supervision target.
