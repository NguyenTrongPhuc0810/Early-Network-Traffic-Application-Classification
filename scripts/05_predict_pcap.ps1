param(
    [Parameter(Mandatory=$true)][string]$Pcap,
    [string]$Model = "data/artifacts/application_63_classes_splt_train_eval/model.joblib",
    [string]$OutDir = ""
)

$argsList = @(
    "-m", "ml_pipeline.pcap_predict",
    "--pcap", $Pcap,
    "--model", $Model,
    "--min-packets", "10"
)

if ($OutDir -ne "") {
    $argsList += @("--out-dir", $OutDir)
}

python @argsList
