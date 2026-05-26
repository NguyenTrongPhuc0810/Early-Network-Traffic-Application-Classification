param(
    [string]$Data = "data/raw/final_dataset_63_classes_splt.parquet",
    [string]$OutDir = "data/artifacts/application_63_classes_splt_train_eval"
)

python -m ml_pipeline.train `
    --data $Data `
    --out-dir $OutDir `
    --target-column application_name `
    --feature-width 25 `
    --n-estimators 100 `
    --class-weight balanced
