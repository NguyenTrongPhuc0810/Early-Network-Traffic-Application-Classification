from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from act2_project.config import load_app_config
from act2_project.pipeline.evaluate import save_classification_reports, save_confusion_matrix
from act2_project.pipeline.splt_features import build_feature_matrix, parse_splt_sequence
from act2_project.utils.io import ensure_dir


SEQUENCE_COLUMNS = ("splt_direction", "splt_ps", "splt_piat_ms")


def _parse_sequence(value: Any) -> list[int]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)
    raise TypeError(f"Unsupported sequence type: {type(value)!r}")


def _truncate_sequence(value: Any, width: int) -> str:
    parsed = _parse_sequence(value)
    return str(parsed[:width])


def create_truncated_dataset(
    source_path: Path,
    out_path: Path,
    *,
    width: int,
) -> dict[str, Any]:
    df = pd.read_parquet(source_path)
    work = df.copy()
    for column in SEQUENCE_COLUMNS:
        work[column] = work[column].apply(lambda value: _truncate_sequence(value, width))

    ensure_dir(out_path.parent)
    work.to_parquet(out_path, index=False)
    return {
        "source_path": str(source_path),
        "out_path": str(out_path),
        "rows": int(len(work)),
        "width": width,
        "columns": list(work.columns),
    }


def run_train_eval_for_width(
    data_path: Path,
    out_dir: Path,
    *,
    target_column: str,
) -> dict[str, Any]:
    app_config = load_app_config()
    df = pd.read_parquet(data_path).copy()
    for column in app_config.dataset.splt_feature_columns:
        df[column] = df[column].apply(parse_splt_sequence)
    X, feature_width = build_feature_matrix(
        df,
        splt_column=app_config.dataset.splt_column,
        sequence_columns=app_config.dataset.splt_feature_columns,
    )
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=app_config.split.test_size,
        random_state=app_config.split.random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=app_config.model.n_estimators,
        random_state=app_config.split.random_state,
        n_jobs=app_config.model.n_jobs,
        class_weight=app_config.model.class_weight,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    ensure_dir(out_dir)
    report_text_path = out_dir / "classification_report.txt"
    report_json_path = out_dir / "classification_report.json"
    confusion_matrix_path = out_dir / "confusion_matrix.png"
    metadata_path = out_dir / "run_metadata.json"

    save_classification_reports(
        y_true=y_test,
        y_pred=y_pred,
        text_path=report_text_path,
        json_path=report_json_path,
    )
    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        out_path=confusion_matrix_path,
        title=f"Confusion Matrix for RF SPLT width {feature_width}",
    )

    with report_json_path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    metadata = {
        "data_path": str(data_path),
        "out_dir": str(out_dir),
        "target_column": target_column,
        "feature_width": int(feature_width),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "n_estimators": app_config.model.n_estimators,
        "class_weight": app_config.model.class_weight,
        "random_state": app_config.split.random_state,
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "classification_report_text": str(report_text_path),
        "classification_report_json": str(report_json_path),
        "confusion_matrix_path": str(confusion_matrix_path),
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def main() -> int:
    source_path = Path(
        r"D:\CCNA\ml-flow-class-tutorial\ml-flow-class-tutorial\02-app-classification\data\final_dataset_63_classes_splt.parquet"
    )
    data_out_dir = Path(
        r"D:\CCNA\ml-flow-class-tutorial\ml-flow-class-tutorial\02-app-classification\data\splt_width_variants"
    )
    artifacts_root = Path(
        r"D:\CCNA\ml-flow-class-tutorial\ml-flow-class-tutorial\act2-project\data\artifacts"
    )
    widths = (10, 12, 15, 17, 20)
    summaries: list[dict[str, Any]] = []

    for width in widths:
        dataset_path = data_out_dir / f"final_dataset_63_classes_splt_{width}pkt.parquet"
        dataset_meta = create_truncated_dataset(source_path, dataset_path, width=width)
        out_dir = artifacts_root / f"application_63_classes_splt_{width}pkt_train_eval"
        train_meta = run_train_eval_for_width(
            dataset_path,
            out_dir,
            target_column="application_name",
        )
        summaries.append(
            {
                "width": width,
                "dataset": dataset_meta,
                "train_eval": train_meta,
            }
        )

    summary_path = artifacts_root / "application_63_classes_splt_width_experiments.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, indent=2)
    print(json.dumps(summaries, indent=2))
    print(f"SUMMARY_PATH {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
