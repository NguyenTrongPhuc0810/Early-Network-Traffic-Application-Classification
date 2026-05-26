param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [Parameter(Mandatory = $true)]
    [ValidateSet("quic", "tls")]
    [string]$Dataset,

    [int]$MaxRows = 100000,
    [int]$MaxPerClass = 0,
    [int]$ChunkSize = 50000
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $ProjectRoot

$ModelPath = Join-Path $ProjectRoot "data\artifacts\application_63_classes_splt_train_eval\model.joblib"
if (-not (Test-Path $ModelPath)) {
    throw "Model not found: $ModelPath"
}

$InputName = [System.IO.Path]::GetFileNameWithoutExtension($InputPath)
$ConvertedPath = Join-Path $ProjectRoot "data\processed\${Dataset}_${InputName}_act2.parquet"
$EvalDir = Join-Path $ProjectRoot "data\artifacts\${Dataset}_${InputName}_eval"

$ConvertArgs = @(
    "-m", "act2_project.cli", "convert-cesnet",
    "--input", $InputPath,
    "--dataset", $Dataset,
    "--out", $ConvertedPath,
    "--chunksize", "$ChunkSize"
)

if ($MaxRows -gt 0) {
    $ConvertArgs += @("--max-rows", "$MaxRows")
}
if ($MaxPerClass -gt 0) {
    $ConvertArgs += @("--max-per-class", "$MaxPerClass")
}

python @ConvertArgs

python -m act2_project.cli eval-parquet-pretrained `
    --data-path $ConvertedPath `
    --model-path $ModelPath `
    --out-dir $EvalDir

Write-Host "Converted parquet: $ConvertedPath"
Write-Host "Evaluation output: $EvalDir"
