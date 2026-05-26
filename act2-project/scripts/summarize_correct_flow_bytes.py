from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np


def _parse_sequence(value: Any) -> list[int]:
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, list):
            raise TypeError(f"Expected list, got {type(parsed)!r}")
        return parsed
    raise TypeError(f"Unsupported sequence type: {type(value)!r}")


def _positive_sum(value: Any) -> int:
    return int(sum(item for item in _parse_sequence(value) if int(item) > 0))


def _format_bytes(value: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{value} B"


def summarize(
    *,
    predictions_path: Path,
    source_file: Path,
    out_dir: Path,
    target_column: str = "application_name",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(predictions_path)
    predicted_column = f"predicted_{target_column}"
    if predicted_column not in df.columns:
        raise ValueError(f"Missing prediction column: {predicted_column}")
    if target_column not in df.columns:
        raise ValueError(f"Missing target column: {target_column}")

    df = df.copy()
    df["first25_packet_bytes"] = df["splt_ps"].apply(_positive_sum)
    correct_mask = df[target_column] == df[predicted_column]
    correct_df = df[correct_mask].copy()

    total_rows = int(len(df))
    correct_rows = int(len(correct_df))
    total_first25_bytes = int(df["first25_packet_bytes"].sum())
    correct_first25_bytes = int(correct_df["first25_packet_bytes"].sum())
    source_file_bytes = int(source_file.stat().st_size)

    correct_by_class = (
        correct_df.groupby(target_column)["first25_packet_bytes"]
        .agg(["count", "sum", "mean", "median"])
        .reset_index()
        .sort_values(target_column)
    )
    correct_by_class_path = out_dir / "correct_flow_bytes_by_class.csv"
    correct_flows_path = out_dir / "correct_predictions.parquet"
    summary_json_path = out_dir / "correct_flow_bytes_summary.json"
    summary_text_path = out_dir / "summary_and_notes.txt"

    correct_by_class.to_csv(correct_by_class_path, index=False)
    correct_df.to_parquet(correct_flows_path, index=False)

    summary = {
        "predictions_path": str(predictions_path),
        "source_file": str(source_file),
        "source_file_bytes": source_file_bytes,
        "source_file_size": _format_bytes(source_file_bytes),
        "total_test_flows": total_rows,
        "correct_test_flows": correct_rows,
        "correct_flow_rate": correct_rows / total_rows if total_rows else 0.0,
        "total_first25_packet_bytes": total_first25_bytes,
        "total_first25_packet_size": _format_bytes(total_first25_bytes),
        "correct_first25_packet_bytes": correct_first25_bytes,
        "correct_first25_packet_size": _format_bytes(correct_first25_bytes),
        "correct_first25_packet_percent_of_test_first25_bytes": (
            correct_first25_bytes / total_first25_bytes * 100 if total_first25_bytes else 0.0
        ),
        "correct_first25_packet_percent_of_source_file": (
            correct_first25_bytes / source_file_bytes * 100 if source_file_bytes else 0.0
        ),
        "correct_predictions_path": str(correct_flows_path),
        "correct_by_class_csv": str(correct_by_class_path),
    }
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    text = "\n".join(
        [
            "CESNET-QUIC22 XS strict-mapping external test summary",
            "",
            f"Source file: {source_file}",
            f"Predictions: {predictions_path}",
            f"Total test flows: {total_rows:,}",
            f"Correct test flows: {correct_rows:,} ({summary['correct_flow_rate'] * 100:.2f}%)",
            "",
            "First-25-packet byte volume",
            f"All tested flows: {_format_bytes(total_first25_bytes)}",
            f"Correct flows only: {_format_bytes(correct_first25_bytes)}",
            (
                "Correct-flow first-25 bytes / all tested first-25 bytes: "
                f"{summary['correct_first25_packet_percent_of_test_first25_bytes']:.2f}%"
            ),
            (
                "Correct-flow first-25 bytes / source H5 file size: "
                f"{summary['correct_first25_packet_percent_of_source_file']:.4f}%"
            ),
            "",
            "Notes",
            "- The byte volume is computed from the positive packet sizes in splt_ps for the first 25 packets only.",
            "- This is payload/packet-size feature volume used by the model, not the compressed H5 storage size.",
            "- The source-file percentage is only a rough storage-size comparison because H5 is compressed and includes metadata/other features.",
            "- The test uses only labels explicitly listed in configs/cesnet_quic22_strict_mapping.json.",
            "",
            f"Correct predictions parquet: {correct_flows_path}",
            f"Correct-flow byte stats by class: {correct_by_class_path}",
        ]
    )
    summary_text_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize first-25-packet bytes for correct predictions.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--target-column", default="application_name")
    args = parser.parse_args()
    summarize(
        predictions_path=args.predictions,
        source_file=args.source_file,
        out_dir=args.out_dir,
        target_column=args.target_column,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
