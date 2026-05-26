from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from act2_project.config import AppConfig
from act2_project.pipeline.evaluate import save_classification_reports, save_confusion_matrix
from act2_project.pipeline.predict import load_model_bundle
from act2_project.pipeline.splt_features import build_feature_matrix, load_act2_dataframe
from act2_project.utils.io import ensure_dir, write_json, write_text


def _format_eval_summary(
    *,
    data_path: Path,
    model_path: Path,
    rows: int,
    evaluable_rows: int,
    predicted_counts: dict[str, int],
    report_text_path: Path | None,
) -> str:
    lines = [
        f"Data: {data_path}",
        f"Model: {model_path}",
        f"Rows loaded: {rows}",
        f"Rows evaluable with known labels: {evaluable_rows}",
    ]
    if predicted_counts:
        lines.append(
            "Predicted labels: "
            + ", ".join(f"{label} = {count}" for label, count in predicted_counts.items())
        )
    if report_text_path is not None:
        lines.append(f"Classification report: {report_text_path}")
    return "\n".join(lines) + "\n"


def run_parquet_eval_pretrained(
    *,
    data_path: Path,
    model_path: Path,
    out_dir: Path,
    app_config: AppConfig,
    target_column: str | None = None,
) -> dict[str, Any]:
    out_dir = ensure_dir(out_dir)
    predictions_path = out_dir / app_config.artifacts.predictions_file
    report_text_path = out_dir / app_config.artifacts.classification_report_text
    report_json_path = out_dir / app_config.artifacts.classification_report_json
    confusion_matrix_path = out_dir / app_config.artifacts.confusion_matrix_png
    summary_path = out_dir / app_config.artifacts.prediction_summary_text
    metadata_path = out_dir / app_config.artifacts.run_metadata_file

    model_bundle = load_model_bundle(model_path)
    resolved_target = target_column or model_bundle["target_column"]

    df_eval = load_act2_dataframe(
        data_path,
        splt_column=app_config.dataset.splt_column,
        target_column=resolved_target if resolved_target else None,
        sequence_columns=app_config.dataset.splt_feature_columns,
    )
    sequence_columns = [
        column for column in app_config.dataset.splt_feature_columns if column in df_eval.columns
    ] or [app_config.dataset.splt_column]

    predicted_column = f"predicted_{resolved_target}"
    report_paths: tuple[Path | None, Path | None] = (None, None)

    if df_eval.empty:
        predictions_df = df_eval.copy()
        predictions_df[predicted_column] = pd.Series(dtype="object")
        predictions_df["prediction_confidence"] = pd.Series(dtype="float64")
        predicted_counts: dict[str, int] = {}
        evaluable_rows = 0
    else:
        X_eval, _ = build_feature_matrix(
            df_eval,
            splt_column=app_config.dataset.splt_column,
            width=int(model_bundle["feature_width"]),
            sequence_columns=sequence_columns,
        )
        model = model_bundle["model"]
        predictions_df = df_eval.copy()
        predictions_df[predicted_column] = model.predict(X_eval)
        predicted_counts = (
            predictions_df[predicted_column].value_counts().sort_index().astype(int).to_dict()
        )
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_eval)
            predictions_df["prediction_confidence"] = probabilities.max(axis=1)

        eval_mask = predictions_df[resolved_target].isin(model_bundle["classes"])
        evaluable_rows = int(eval_mask.sum())
        if evaluable_rows > 0:
            save_classification_reports(
                y_true=predictions_df.loc[eval_mask, resolved_target],
                y_pred=predictions_df.loc[eval_mask, predicted_column],
                text_path=report_text_path,
                json_path=report_json_path,
                digits=4,
            )
            save_confusion_matrix(
                y_true=predictions_df.loc[eval_mask, resolved_target],
                y_pred=predictions_df.loc[eval_mask, predicted_column],
                out_path=confusion_matrix_path,
                title=f"Pretrained Model Evaluation on {data_path.name}",
            )
            report_paths = (report_text_path, confusion_matrix_path)

    predictions_df.to_parquet(predictions_path, index=False)
    summary_text = _format_eval_summary(
        data_path=data_path,
        model_path=model_path,
        rows=int(len(df_eval)),
        evaluable_rows=evaluable_rows,
        predicted_counts=predicted_counts,
        report_text_path=report_paths[0],
    )
    write_text(summary_path, summary_text)
    print(summary_text, end="")

    metadata = {
        "data_path": str(data_path),
        "model_path": str(model_path),
        "out_dir": str(out_dir),
        "target_column": resolved_target,
        "feature_width": int(model_bundle["feature_width"]),
        "rows": int(len(df_eval)),
        "evaluable_rows": evaluable_rows,
        "predicted_counts": predicted_counts,
        "predictions_path": str(predictions_path),
        "classification_report_text": str(report_paths[0]) if report_paths[0] else None,
        "classification_report_json": str(report_json_path) if report_paths[0] else None,
        "confusion_matrix_path": str(report_paths[1]) if report_paths[1] else None,
        "summary_path": str(summary_path),
        "trained_classes": model_bundle["classes"],
    }
    write_json(metadata_path, metadata)
    return metadata
