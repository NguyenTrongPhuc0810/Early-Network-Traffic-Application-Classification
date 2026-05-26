param(
    [string]$PcapPath = "..\01-data-collection\pcap\traffic_trace.pcap",
    [string]$TrainData = "data\processed\df_final_model_data_splt.parquet",
    [string]$OutDir = "data\artifacts\pcap_predict",
    [switch]$NoDpi
)

$projectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$env:PYTHONPATH = Join-Path $projectRoot "src"

if ([System.IO.Path]::IsPathRooted($PcapPath)) {
    $resolvedPcapPath = Resolve-Path $PcapPath
} else {
    $resolvedPcapPath = Resolve-Path (Join-Path $projectRoot $PcapPath)
}

if ([System.IO.Path]::IsPathRooted($TrainData)) {
    $resolvedTrainData = Resolve-Path $TrainData
} else {
    $resolvedTrainData = Resolve-Path (Join-Path $projectRoot $TrainData)
}

if ([System.IO.Path]::IsPathRooted($OutDir)) {
    $resolvedOutDir = $OutDir
} else {
    $resolvedOutDir = Join-Path $projectRoot $OutDir
}

$args = @(
    "-m", "act2_project.cli",
    "pcap-predict",
    "--pcap", $resolvedPcapPath,
    "--train-data", $resolvedTrainData,
    "--out-dir", $resolvedOutDir
)

if ($NoDpi) {
    $args += "--no-dpi"
}

python @args
