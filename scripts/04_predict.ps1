param(
    [string]$Model = "data/artifacts/application_63_classes_splt_train_eval/model.joblib",
    [string]$Data = "data/raw/final_dataset_63_classes_splt.parquet",
    [string]$Out = "data/artifacts/predictions.parquet"
)

python -m ml_pipeline.predict `
    --model $Model `
    --data $Data `
    --out $Out `
    --min-packets 10
