"""Train a Random Forest classifier on 63-class SPLT network-flow features."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml_pipeline.data_loader import DatasetConfig, PreparedDataset, load_prepared_dataset
from ml_pipeline.evaluate import evaluate_predictions, to_jsonable, write_json


@dataclass(frozen=True)
class TrainingConfig:
    """Training parameters for the SPLT Random Forest model."""

    n_estimators: int = 100
    test_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1
    class_weight: str | dict[str, float] | None = "balanced"
    target_column: str = "application_name"
    min_packets: int = 10
    min_samples_per_application: int = 1000
    feature_width: int | None = 25


def build_random_forest(config: TrainingConfig) -> RandomForestClassifier:
    """Create the Random Forest configuration used by the SPLT 63-class run."""

    return RandomForestClassifier(
        n_estimators=config.n_estimators,
        random_state=config.random_state,
        n_jobs=config.n_jobs,
        class_weight=config.class_weight,
    )


def build_model_bundle(
    *,
    model: RandomForestClassifier,
    dataset: PreparedDataset,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Package the fitted estimator with feature metadata needed for export."""

    return {
        "model": model,
        "model_type": "RandomForestClassifier",
        "feature_columns": dataset.feature_columns,
        "feature_width": int(dataset.feature_width),
        "splt_columns": ["splt_direction", "splt_ps", "splt_piat_ms"],
        "target_column": dataset.target_column,
        "classes": [str(label) for label in model.classes_.tolist()],
        "n_estimators": int(config.n_estimators),
        "class_weight": config.class_weight,
        "random_state": int(config.random_state),
        "preparation": dataset.metadata,
    }


def save_model_bundle(bundle: dict[str, Any], model_path: Path) -> None:
    """Persist a model bundle with joblib."""

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)


def train_random_forest(
    *,
    data_path: Path,
    out_dir: Path,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Run the complete SPLT training/evaluation pipeline.

    The default target is ``application_name`` because the 63-class artifact in
    the workspace was trained as application-level classification, not coarse
    category classification.
    """

    dataset = load_prepared_dataset(
        data_path,
        DatasetConfig(
            target_column=config.target_column,
            min_packets=config.min_packets,
            min_samples_per_application=config.min_samples_per_application,
            feature_width=config.feature_width,
        ),
    )

    x_train, x_test, y_train, y_test = train_test_split(
        dataset.features,
        dataset.labels,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=dataset.labels,
    )

    model = build_random_forest(config)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    out_dir.mkdir(parents=True, exist_ok=True)
    eval_metadata = evaluate_predictions(
        y_test,
        y_pred,
        out_dir=out_dir,
        labels=[str(label) for label in model.classes_.tolist()],
    )

    bundle = build_model_bundle(model=model, dataset=dataset, config=config)
    model_path = out_dir / "model.joblib"
    save_model_bundle(bundle, model_path)

    metadata = {
        "data_path": str(data_path),
        "out_dir": str(out_dir),
        "model_path": str(model_path),
        "target_column": config.target_column,
        "input_rows": int(len(dataset.labels)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "feature_width": int(dataset.feature_width),
        "feature_count": int(dataset.features.shape[1]),
        "feature_columns": dataset.feature_columns,
        "classes": [str(label) for label in model.classes_.tolist()],
        "class_count": int(len(model.classes_)),
        "n_estimators": int(config.n_estimators),
        "class_weight": config.class_weight,
        "random_state": int(config.random_state),
        "test_size": float(config.test_size),
        "evaluation": eval_metadata,
        "preparation": dataset.metadata,
    }
    write_json(out_dir / "run_metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SPLT 63-class Random Forest model.")
    parser.add_argument("--data", required=True, type=Path, help="Path to SPLT parquet dataset.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Artifact output directory.")
    parser.add_argument("--target-column", default="application_name")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--class-weight", default="balanced")
    parser.add_argument("--min-packets", type=int, default=10)
    parser.add_argument("--min-samples-per-application", type=int, default=1000)
    parser.add_argument("--feature-width", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    class_weight = None if args.class_weight.lower() == "none" else args.class_weight
    metadata = train_random_forest(
        data_path=args.data,
        out_dir=args.out_dir,
        config=TrainingConfig(
            n_estimators=args.n_estimators,
            test_size=args.test_size,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
            class_weight=class_weight,
            target_column=args.target_column,
            min_packets=args.min_packets,
            min_samples_per_application=args.min_samples_per_application,
            feature_width=args.feature_width,
        ),
    )
    print(json.dumps(to_jsonable(metadata), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
