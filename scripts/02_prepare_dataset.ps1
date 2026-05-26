param(
    [string]$Input = "data/interim/flows.parquet",
    [string]$Out = "data/processed/splt_dataset.parquet"
)

python -m ml_pipeline.prepare `
    --input $Input `
    --out $Out `
    --target-column application_name `
    --feature-width 25
