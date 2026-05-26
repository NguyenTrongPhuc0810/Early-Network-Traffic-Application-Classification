from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from act2_project.config import AppConfig
from act2_project.pipeline.evaluate import save_classification_reports, save_confusion_matrix
from act2_project.pipeline.predict import load_model_bundle
from act2_project.pipeline.splt_features import build_feature_matrix, load_act2_dataframe
from act2_project.utils.io import ensure_dir, write_json, write_text


def _safe_dirname(value: str, *, max_len: int = 120) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        cleaned = "label"
    return cleaned[:max_len]


def _format_eval_summary(
    *,
    data_path: Path,
    model_path: Path,
    label: str,
    rows: int,
    evaluable_rows: int,
    correct_rows: int,
    predicted_counts: dict[str, int],
    report_text_path: Path | None,
) -> str:
    lines = [
        f"Data: {data_path}",
        f"Model: {model_path}",
        f"Label: {label}",
        f"Rows loaded: {rows}",
        f"Rows evaluable with known labels: {evaluable_rows}",
        f"Correct: {correct_rows} / {rows}" if rows else "Correct: 0 / 0",
    ]
    if predicted_counts:
        lines.append(
            "Predicted labels: "
            + ", ".join(f"{name} = {count}" for name, count in predicted_counts.items())
        )
    if report_text_path is not None:
        lines.append(f"Classification report: {report_text_path}")
    return "\n".join(lines) + "\n"


def _resolve_labels(df: pd.DataFrame, *, target_column: str, labels: Iterable[str] | None) -> list[str]:
    if labels:
        return [str(label) for label in labels]
    if df.empty or target_column not in df.columns:
        return []
    uniques = df[target_column].dropna().astype(str).unique().tolist()
    return sorted(uniques)


def run_parquet_eval_pretrained_per_label(
    *,
    data_path: Path,
    model_path: Path,
    out_dir: Path,
    app_config: AppConfig,
    target_column: str | None = None,
    labels: Iterable[str] | None = None,
    max_per_label: int | None = None,
    random_seed: int = 42,
    save_subset: bool = False,
) -> dict[str, Any]:
    """Evaluate a labeled parquet one label at a time with a pretrained model.

    This is useful when you want per-application confusion and metrics without mixing labels.
    Each label gets its own output directory under out_dir.
    """

    base_out_dir = ensure_dir(out_dir)

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

    resolved_labels = _resolve_labels(df_eval, target_column=resolved_target, labels=labels)
    label_dir_map: dict[str, str] = {}
    summary_rows: list[dict[str, Any]] = []

    for label in resolved_labels:
        df_label = df_eval[df_eval[resolved_target].astype(str) == str(label)].copy()
        if max_per_label is not None and len(df_label) > max_per_label:
            df_label = df_label.sample(n=int(max_per_label), random_state=int(random_seed))

        safe_label = _safe_dirname(label)
        label_dir = ensure_dir(base_out_dir / safe_label)
        label_dir_map[safe_label] = label

        subset_path = label_dir / "eval_subset.parquet"
        if save_subset:
            df_label.to_parquet(subset_path, index=False)

        predictions_path = label_dir / app_config.artifacts.predictions_file
        report_text_path = label_dir / app_config.artifacts.classification_report_text
        report_json_path = label_dir / app_config.artifacts.classification_report_json
        confusion_matrix_path = label_dir / app_config.artifacts.confusion_matrix_png
        summary_path = label_dir / app_config.artifacts.prediction_summary_text
        metadata_path = label_dir / app_config.artifacts.run_metadata_file

        predicted_column = f"predicted_{resolved_target}"
        report_paths: tuple[Path | None, Path | None] = (None, None)

        if df_label.empty:
            predictions_df = df_label.copy()
            predictions_df[predicted_column] = pd.Series(dtype="object")
            predictions_df["prediction_confidence"] = pd.Series(dtype="float64")
            predicted_counts: dict[str, int] = {}
            evaluable_rows = 0
            correct_rows = 0
        else:
            X_eval, _ = build_feature_matrix(
                df_label,
                splt_column=app_config.dataset.splt_column,
                width=int(model_bundle["feature_width"]),
                sequence_columns=sequence_columns,
            )
            model = model_bundle["model"]
            predictions_df = df_label.copy()
            predictions_df[predicted_column] = model.predict(X_eval)
            predicted_counts = (
                predictions_df[predicted_column].value_counts().sort_index().astype(int).to_dict()
            )
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X_eval)
                predictions_df["prediction_confidence"] = probabilities.max(axis=1)

            eval_mask = predictions_df[resolved_target].isin(model_bundle["classes"])
            evaluable_rows = int(eval_mask.sum())
            correct_rows = int(
                (predictions_df.loc[eval_mask, resolved_target]
                == predictions_df.loc[eval_mask, predicted_column]).sum()
            )

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
                    title=f"Pretrained Model Evaluation on {data_path.name} ({label})",
                )
                report_paths = (report_text_path, confusion_matrix_path)

        predictions_df.to_parquet(predictions_path, index=False)

        summary_text = _format_eval_summary(
            data_path=data_path,
            model_path=model_path,
            label=label,
            rows=int(len(df_label)),
            evaluable_rows=evaluable_rows,
            correct_rows=correct_rows,
            predicted_counts=predicted_counts,
            report_text_path=report_paths[0],
        )
        write_text(summary_path, summary_text)
        print(summary_text, end="")

        most_predicted_label = None
        most_predicted_count = 0
        if predicted_counts:
            most_predicted_label = max(predicted_counts, key=predicted_counts.get)
            most_predicted_count = int(predicted_counts[most_predicted_label])

        summary_rows.append(
            {
                "label": label,
                "safe_label": safe_label,
                "rows": int(len(df_label)),
                "evaluable_rows": evaluable_rows,
                "correct": correct_rows,
                "accuracy": (correct_rows / len(df_label)) if len(df_label) else 0.0,
                "most_predicted_label": most_predicted_label,
                "most_predicted_count": most_predicted_count,
                "most_predicted_pct": (
                    most_predicted_count / len(df_label) if len(df_label) else 0.0
                ),
                "predictions_path": str(predictions_path),
                "classification_report_text": str(report_paths[0]) if report_paths[0] else None,
                "classification_report_json": str(report_json_path) if report_paths[0] else None,
                "confusion_matrix_path": str(report_paths[1]) if report_paths[1] else None,
                "summary_path": str(summary_path),
                "subset_path": str(subset_path) if save_subset else None,
            }
        )

        metadata = {
            "data_path": str(data_path),
            "model_path": str(model_path),
            "out_dir": str(label_dir),
            "target_column": resolved_target,
            "feature_width": int(model_bundle["feature_width"]),
            "label": label,
            "safe_label": safe_label,
            "rows": int(len(df_label)),
            "evaluable_rows": evaluable_rows,
            "correct": correct_rows,
            "accuracy": (correct_rows / len(df_label)) if len(df_label) else 0.0,
            "predicted_counts": predicted_counts,
            "subset_path": str(subset_path) if save_subset else None,
            "predictions_path": str(predictions_path),
            "classification_report_text": str(report_paths[0]) if report_paths[0] else None,
            "classification_report_json": str(report_json_path) if report_paths[0] else None,
            "confusion_matrix_path": str(report_paths[1]) if report_paths[1] else None,
            "summary_path": str(summary_path),
            "trained_classes": model_bundle["classes"],
        }
        write_json(metadata_path, metadata)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = base_out_dir / "per_label_summary.csv"
    summary_json_path = base_out_dir / "per_label_summary.json"
    labels_json_path = base_out_dir / "labels.json"
    summary_df.to_csv(summary_csv_path, index=False)
    write_json(summary_json_path, {"rows": summary_rows})
    write_json(labels_json_path, label_dir_map)

    return {
        "data_path": str(data_path),
        "model_path": str(model_path),
        "out_dir": str(base_out_dir),
        "target_column": resolved_target,
        "labels": resolved_labels,
        "labels_count": int(len(resolved_labels)),
        "summary_csv": str(summary_csv_path),
        "summary_json": str(summary_json_path),
        "labels_json": str(labels_json_path),
    }
