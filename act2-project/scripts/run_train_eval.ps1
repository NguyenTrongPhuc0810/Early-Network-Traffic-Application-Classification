param(
    [string]$DataPath = "..\02-app-classification\data\data.parquet",
    [string]$OutDir = "data\artifacts\df_final_model_train_eval"
)

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = Join-Path $projectRoot "src"

if ([System.IO.Path]::IsPathRooted($DataPath)) {
    $resolvedDataPath = Resolve-Path $DataPath
} else {
    $resolvedDataPath = Resolve-Path (Join-Path $projectRoot $DataPath)
}

if ([System.IO.Path]::IsPathRooted($OutDir)) {
    $resolvedOutDir = $OutDir
} else {
    $resolvedOutDir = Join-Path $projectRoot $OutDir
}

python -m act2_project.cli train-eval `
    --data-path $resolvedDataPath `
    --out-dir $resolvedOutDir
