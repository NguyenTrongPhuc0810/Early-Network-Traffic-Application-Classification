# Act 2 Project

This project extracts the Act 2 SPLT idea from the tutorial into a CLI-first codebase.

There are now two practical paths:

1. `df_final_model_data` path
   - build a SPLT parquet from `02-app-classification/data/data.parquet`
   - train a category model from the filtered tutorial data

2. task-level PCAP path
   - build a labeled SPLT dataset from a folder of `.pcap` / `.pcapng`
   - infer labels from filename patterns
   - train a flow model with capture-level voting

## Core SPLT Commands

Prepare the tutorial-style SPLT parquet from `df_final_model_data`:

```powershell
python -m act2_project.cli prepare-train-data `
  --data-path ..\02-app-classification\data\data.parquet `
  --out data\processed\df_final_model_data_splt.parquet
```

Train and evaluate the tutorial-style category model:

```powershell
python -m act2_project.cli train-eval `
  --data-path data\processed\df_final_model_data_splt.parquet `
  --out-dir data\artifacts\df_final_model_train_eval
```

Predict categories on one capture:

```powershell
python -m act2_project.cli pcap-predict `
  --pcap D:\path\to\capture.pcapng `
  --train-data data\processed\df_final_model_data_splt.parquet `
  --out-dir data\artifacts\pcap_predict_run
```

## Task-Level Commands

Build a task-labeled SPLT dataset from a folder of labeled captures:

```powershell
python -m act2_project.cli build-task-dataset `
  --pcap-dir D:\CCNA\raw_pcap `
  --out data\processed\task_labeled_splt.parquet
```

Build the auxiliary SPLT dataset from `df_final_model_data` for the final task classes:

```powershell
python -m act2_project.cli build-aux-task-data `
  --data-path ..\02-app-classification\data\data.parquet `
  --out data\processed\my_df_act3_splt.parquet
```

Train and evaluate the task-level model:

```powershell
python -m act2_project.cli task-train-eval `
  --dataset data\processed\task_labeled_splt.parquet `
  --out-dir data\artifacts\task_train_eval
```

Train and evaluate with auxiliary SPLT data:

```powershell
python -m act2_project.cli task-train-eval `
  --dataset data\processed\task_labeled_splt.parquet `
  --aux-data data\processed\my_df_act3_splt.parquet `
  --out-dir data\artifacts\task_train_eval_with_aux
```

Predict the task for one capture:

```powershell
python -m act2_project.cli task-pcap-predict `
  --pcap D:\path\to\capture.pcapng `
  --dataset data\processed\task_labeled_splt.parquet `
  --out-dir data\artifacts\task_predict_run
```

## Config Files

- [default.yaml](/D:/CCNA/ml-flow-class-tutorial/ml-flow-class-tutorial/act2-project/configs/default.yaml)
- [class_subset.yaml](/D:/CCNA/ml-flow-class-tutorial/ml-flow-class-tutorial/act2-project/configs/class_subset.yaml)
- [task_labels.yaml](/D:/CCNA/ml-flow-class-tutorial/ml-flow-class-tutorial/act2-project/configs/task_labels.yaml)

## Notes

- The task-level pipeline uses only SPLT features for the model:
  - `splt_direction`
  - `splt_ps`
  - `splt_piat_ms`
- DPI is used only to extract optional labels and to score training relevance heuristics.
- Capture-level evaluation is done by splitting on capture files, not by random flow split.
