"""Prediction helpers for trained SPLT Random Forest bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml_pipeline.data_loader import DatasetConfig, load_prepared_dataset
from ml_pipeline.evaluate import to_jsonable, write_json


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    """Load and validate a joblib model bundle."""

    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_path}")
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Expected a model bundle dict containing a 'model' key.")
    return bundle


def predict_parquet(
    *,
    model_path: Path,
    data_path: Path,
    out_path: Path,
    target_column: str | None = None,
) -> dict[str, Any]:
    """Run a trained model bundle on an SPLT parquet and save predictions."""

    bundle = load_model_bundle(model_path)
    resolved_target = target_column or str(bundle.get("target_column", "application_name"))
    feature_width = int(bundle["feature_width"]) if "feature_width" in bundle else None

    dataset = load_prepared_dataset(
        data_path,
        DatasetConfig(
            target_column=resolved_target,
            feature_width=feature_width,
            min_samples_per_application=1,
        ),
    )
    model = bundle["model"]
    predictions = model.predict(dataset.features)

    output = dataset.dataframe.copy()
    output[f"predicted_{resolved_target}"] = predictions
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(dataset.features)
        output["prediction_confidence"] = probabilities.max(axis=1)
        for index, label in enumerate(model.classes_):
            output[f"prob_{label}"] = probabilities[:, index]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(out_path, index=False)

    summary = pd.Series(predictions).value_counts().sort_index().to_dict()
    metadata = {
        "model_path": str(model_path),
        "data_path": str(data_path),
        "predictions_path": str(out_path),
        "rows": int(len(output)),
        "target_column": resolved_target,
        "feature_width": int(dataset.feature_width),
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = predict_parquet(
        model_path=args.model,
        data_path=args.data,
        out_path=args.out,
        target_column=args.target_column,
    )
    print(json.dumps(to_jsonable(metadata), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
