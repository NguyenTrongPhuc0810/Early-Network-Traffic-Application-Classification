"""Prediction helpers for trained SPLT Random Forest bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml_pipeline.data_loader import (
    DEFAULT_SPLT_COLUMNS,
    build_feature_matrix,
    load_parquet,
    parse_splt_columns,
)
from ml_pipeline.evaluate import to_jsonable, write_json


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    """Load and validate a joblib model bundle."""

    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_path}")
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Expected a model bundle dict containing a 'model' key.")
    return bundle


def _format_sequence_for_output(sequence: list[float]) -> list[int | float]:
    formatted: list[int | float] = []
    for value in sequence:
        if float(value).is_integer():
            formatted.append(int(value))
        else:
            formatted.append(float(value))
    return formatted


def predict_parquet(
    *,
    model_path: Path,
    data_path: Path,
    out_path: Path,
    target_column: str | None = None,
    min_packets: int = 10,
) -> dict[str, Any]:
    """Run a trained model bundle on an SPLT parquet and save predictions."""

    raw_dataframe = load_parquet(data_path)
    return predict_dataframe(
        model_path=model_path,
        dataframe=raw_dataframe,
        out_path=out_path,
        source_name=str(data_path),
        target_column=target_column,
        min_packets=min_packets,
    )


def predict_dataframe(
    *,
    model_path: Path,
    dataframe: pd.DataFrame,
    out_path: Path,
    source_name: str,
    target_column: str | None = None,
    min_packets: int = 10,
    compact_output: bool = True,
) -> dict[str, Any]:
    """Run a trained model bundle on an in-memory SPLT dataframe."""

    bundle = load_model_bundle(model_path)
    resolved_target = target_column or str(bundle.get("target_column", "application_name"))
    feature_width = int(bundle["feature_width"]) if "feature_width" in bundle else None

    raw_dataframe = dataframe.copy()
    if "bidirectional_packets" in raw_dataframe.columns:
        raw_dataframe = raw_dataframe[raw_dataframe["bidirectional_packets"] >= min_packets].copy()
    dataframe = parse_splt_columns(raw_dataframe, DEFAULT_SPLT_COLUMNS)
    features, resolved_width = build_feature_matrix(
        dataframe,
        splt_columns=DEFAULT_SPLT_COLUMNS,
        width=feature_width,
    )
    model = bundle["model"]
    predictions = model.predict(features)

    if compact_output:
        output = dataframe.loc[:, list(DEFAULT_SPLT_COLUMNS)].copy()
        for column in DEFAULT_SPLT_COLUMNS:
            output[column] = output[column].apply(_format_sequence_for_output)
        output[resolved_target] = predictions
    else:
        output = dataframe.copy()
        output[f"predicted_{resolved_target}"] = predictions

    if not compact_output and hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        output["prediction_confidence"] = probabilities.max(axis=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(out_path, index=False)

    summary = pd.Series(predictions).value_counts().sort_index().to_dict()
    metadata = {
        "model_path": str(model_path),
        "data_path": source_name,
        "predictions_path": str(out_path),
        "rows": int(len(output)),
        "target_column": resolved_target,
        "min_packets": int(min_packets),
        "feature_width": int(resolved_width),
        "feature_count": int(features.shape[1]),
        "output_columns": list(output.columns),
        "prediction_counts": {str(key): int(value) for key, value in summary.items()},
    }
    write_json(out_path.with_suffix(".metadata.json"), metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict labels for an SPLT parquet dataset.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--target-column", default=None)
    parser.add_argument("--min-packets", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = predict_parquet(
        model_path=args.model,
        data_path=args.data,
        out_path=args.out,
        target_column=args.target_column,
        min_packets=args.min_packets,
    )
    print(json.dumps(to_jsonable(metadata), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
