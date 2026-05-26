"""Evaluation helpers for SPLT traffic classification models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

from ml_pipeline.data_loader import DatasetConfig, load_prepared_dataset


def ensure_dir(path: Path) -> Path:
    """Create a directory and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def to_jsonable(value: Any) -> Any:
    """Convert common numpy/pandas scalar objects into JSON-safe values."""

    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write stable, readable JSON."""

    ensure_dir(path.parent)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    """Write UTF-8 text and create parent directories."""

    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def generate_classification_reports(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    digits: int = 4,
) -> tuple[str, dict[str, Any]]:
    """Return both text and dictionary classification reports."""

    text_report = classification_report(
        y_true,
        y_pred,
        digits=digits,
        zero_division=0,
    )
    json_report = classification_report(
        y_true,
        y_pred,
        digits=digits,
        output_dict=True,
        zero_division=0,
    )
    return text_report, json_report


def save_classification_reports(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    text_path: Path,
    json_path: Path,
    digits: int = 4,
) -> tuple[str, dict[str, Any]]:
    """Persist text and JSON classification reports."""

    text_report, json_report = generate_classification_reports(
        y_true,
        y_pred,
        digits=digits,
    )
    write_text(text_path, text_report)
    write_json(json_path, json_report)
    return text_report, json_report


def save_confusion_matrix(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    out_path: Path,
    title: str = "Random Forest SPLT Confusion Matrix",
    labels: Sequence[str] | None = None,
    normalize: str | None = None,
) -> None:
    """Save a confusion matrix image for model diagnostics."""

    ensure_dir(out_path.parent)
    class_count = len(labels) if labels is not None else len(set(y_true) | set(y_pred))
    figure_size = max(10.0, min(28.0, class_count * 0.38))

    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=list(labels) if labels is not None else None,
        normalize=normalize,
        ax=ax,
        xticks_rotation="vertical",
        colorbar=True,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_predictions(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    out_dir: Path,
    labels: Sequence[str] | None = None,
    digits: int = 4,
    confusion_matrix_name: str = "confusion_matrix.png",
) -> dict[str, Any]:
    """Write all standard evaluation artifacts and return summary metadata."""

    ensure_dir(out_dir)
    report_text_path = out_dir / "classification_report.txt"
    report_json_path = out_dir / "classification_report.json"
    confusion_matrix_path = out_dir / confusion_matrix_name

    _, report = save_classification_reports(
        y_true,
        y_pred,
        text_path=report_text_path,
        json_path=report_json_path,
        digits=digits,
    )
    save_confusion_matrix(
        y_true,
        y_pred,
        out_path=confusion_matrix_path,
        labels=labels,
    )

    return {
        "classification_report_text": str(report_text_path),
        "classification_report_json": str(report_json_path),
        "confusion_matrix_path": str(confusion_matrix_path),
        "accuracy": float(report.get("accuracy", 0.0)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "weighted_f1": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
    }


def evaluate_model_bundle(
    *,
    model_path: Path,
    data_path: Path,
    out_dir: Path,
    target_column: str = "application_name",
    feature_width: int | None = None,
) -> dict[str, Any]:
    """Load a trained joblib bundle and evaluate it on a parquet dataset."""

    bundle = joblib.load(model_path)
    model = bundle["model"] if isinstance(bundle, dict) and "model" in bundle else bundle
    width = feature_width or bundle.get("feature_width") if isinstance(bundle, dict) else None
    dataset = load_prepared_dataset(
        data_path,
        DatasetConfig(
            target_column=target_column,
            feature_width=width,
            min_samples_per_application=1,
        ),
    )
    y_pred = model.predict(dataset.features)
    labels = bundle.get("classes") if isinstance(bundle, dict) else None

    metadata = evaluate_predictions(
        dataset.labels,
        y_pred,
        out_dir=out_dir,
        labels=labels,
    )
    metadata.update(
        {
            "model_path": str(model_path),
            "data_path": str(data_path),
            "rows": int(len(dataset.labels)),
            "target_column": target_column,
            "feature_width": int(dataset.feature_width),
        }
    )
    write_json(out_dir / "run_metadata.json", metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SPLT Random Forest bundle.")
    parser.add_argument("--model", required=True, type=Path, help="Path to model.joblib.")
    parser.add_argument("--data", required=True, type=Path, help="Path to SPLT parquet dataset.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for reports.")
    parser.add_argument("--target-column", default="application_name")
    parser.add_argument("--feature-width", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = evaluate_model_bundle(
        model_path=args.model,
        data_path=args.data,
        out_dir=args.out_dir,
        target_column=args.target_column,
        feature_width=args.feature_width,
    )
    print(json.dumps(to_jsonable(metadata), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
