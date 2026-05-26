param(
    [Parameter(Mandatory=$true)][string]$Pcap,
    [string]$Out = "data/interim/flows.parquet"
)

python -m ml_pipeline.nfstream_ingest `
    --pcap $Pcap `
    --out $Out `
    --n-dissections 20 `
    --splt-analysis 25 `
    --no-statistical-analysis
