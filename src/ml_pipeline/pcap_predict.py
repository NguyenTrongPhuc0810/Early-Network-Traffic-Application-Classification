"""One-command PCAP prediction workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml_pipeline.evaluate import to_jsonable, write_json
from ml_pipeline.nfstream_ingest import NFStreamConfig, extract_pcap_to_parquet
from ml_pipeline.predict import predict_parquet


DEFAULT_MODEL_PATH = Path("data/artifacts/application_63_classes_splt_train_eval/model.joblib")


def run_pcap_prediction(
    *,
    pcap_path: Path,
    model_path: Path = DEFAULT_MODEL_PATH,
    out_dir: Path | None = None,
    min_packets: int = 10,
    n_dissections: int = 20,
    splt_analysis: int = 25,
    statistical_analysis: bool = False,
) -> dict[str, Any]:
    """Extract SPLT flows from a PCAP and run model prediction in one call."""

    resolved_out_dir = out_dir or Path("data/artifacts") / f"pcap_{pcap_path.stem}"
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    flows_path = resolved_out_dir / "flows.parquet"
    predictions_path = resolved_out_dir / "predictions.parquet"

    ingest_metadata = extract_pcap_to_parquet(
        pcap_path=pcap_path,
        out_path=flows_path,
        config=NFStreamConfig(
            n_dissections=n_dissections,
            splt_analysis=splt_analysis,
            statistical_analysis=statistical_analysis,
        ),
    )
    prediction_metadata = predict_parquet(
        model_path=model_path,
        data_path=flows_path,
        out_path=predictions_path,
        min_packets=min_packets,
    )

    metadata = {
        "pcap_path": str(pcap_path),
        "model_path": str(model_path),
        "out_dir": str(resolved_out_dir),
        "flows_path": str(flows_path),
        "predictions_path": str(predictions_path),
        "input_flows": int(ingest_metadata["rows"]),
        "predicted_flows": int(prediction_metadata["rows"]),
        "min_packets": int(min_packets),
        "prediction_counts": prediction_metadata["prediction_counts"],
    }
    write_json(resolved_out_dir / "pcap_prediction_metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a PCAP through NFStream and the trained SPLT model.")
    parser.add_argument("--pcap", required=True, type=Path, help="Input .pcap or .pcapng file.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to model.joblib.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory.")
    parser.add_argument("--min-packets", type=int, default=10)
    parser.add_argument("--n-dissections", type=int, default=20)
    parser.add_argument("--splt-analysis", type=int, default=25)
    parser.add_argument("--statistical-analysis", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = run_pcap_prediction(
        pcap_path=args.pcap,
        model_path=args.model,
        out_dir=args.out_dir,
        min_packets=args.min_packets,
        n_dissections=args.n_dissections,
        splt_analysis=args.splt_analysis,
        statistical_analysis=args.statistical_analysis,
    )
    print(json.dumps(to_jsonable(metadata), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
