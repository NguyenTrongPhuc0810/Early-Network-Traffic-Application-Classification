from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from act2_project.config import AppConfig
from act2_project.domain.constants import DEFAULT_TASK_LABEL_COLUMN
from act2_project.pipeline.evaluate import (
    generate_classification_reports,
    save_classification_reports,
    save_confusion_matrix,
)
from act2_project.pipeline.pcap_ingest import extract_pcap_dataframe
from act2_project.pipeline.splt_features import build_feature_matrix, parse_splt_sequence
from act2_project.pipeline.task_dataset import infer_task_label
from act2_project.pipeline.train import build_random_forest, save_model_bundle
from act2_project.task_config import TaskConfig
from act2_project.utils.io import ensure_dir, write_json, write_text
from act2_project.utils.logging import get_logger

LOGGER = get_logger(__name__)


def _parse_sequence_columns(df: pd.DataFrame, sequence_columns: tuple[str, ...]) -> pd.DataFrame:
    normalized = df.copy()
    for column in sequence_columns:
        if column in normalized.columns:
            normalized[column] = normalized[column].apply(parse_splt_sequence)
    return normalized


def load_task_dataframe(data_path: Path, app_config: AppConfig) -> pd.DataFrame:
    df = pd.read_parquet(data_path)
    required_columns = {
        "capture_name",
        "capture_path",
        DEFAULT_TASK_LABEL_COLUMN,
        "bidirectional_packets",
        "capture_normalized_weight",
    }
    required_columns.update(app_config.dataset.splt_feature_columns)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Task dataset is missing required columns: {missing}")
    return _parse_sequence_columns(df, app_config.dataset.splt_feature_columns)


def _build_task_feature_matrix(
    df: pd.DataFrame,
    app_config: AppConfig,
    *,
    width: int | None = None,
) -> tuple[pd.DataFrame, int]:
    return build_feature_matrix(
        df,
        splt_column=app_config.dataset.splt_column,
        width=width,
        sequence_columns=app_config.dataset.splt_feature_columns,
    )


def _prepare_auxiliary_dataset(
    auxiliary_data_path: Path | None,
    app_config: AppConfig,
    *,
    width: int,
) -> tuple[pd.DataFrame | None, pd.Series | None]:
    if auxiliary_data_path is None:
        return None, None
    df_aux = load_task_dataframe(auxiliary_data_path, app_config)
    X_aux, _ = _build_task_feature_matrix(df_aux, app_config, width=width)
    y_aux = df_aux[DEFAULT_TASK_LABEL_COLUMN]
    return X_aux, y_aux


def _combined_training_frame(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: np.ndarray,
    X_aux: pd.DataFrame | None,
    y_aux: pd.Series | None,
    auxiliary_weight_ratio: float,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    if X_aux is None or y_aux is None or X_aux.empty:
        return X_train, y_train, sample_weight

    aux_total_weight = float(sample_weight.sum()) * auxiliary_weight_ratio
    per_row_aux_weight = aux_total_weight / len(X_aux)
    auxiliary_weights = np.full(len(X_aux), per_row_aux_weight, dtype="float64")

    combined_X = pd.concat([X_train, X_aux], axis=0, ignore_index=True)
    combined_y = pd.concat([y_train.reset_index(drop=True), y_aux.reset_index(drop=True)], axis=0)
    combined_weight = np.concatenate([sample_weight, auxiliary_weights])
    return combined_X, combined_y, combined_weight


def aggregate_capture_votes(
    prediction_df: pd.DataFrame,
    *,
    classes: list[str],
    task_config: TaskConfig,
) -> pd.DataFrame:
    capture_rows: list[dict[str, Any]] = []
    probability_columns = [f"prob_{label}" for label in classes]

    for capture_name, group in prediction_df.groupby("capture_name", sort=True):
        work = group.copy()
        work["max_prob"] = work[probability_columns].max(axis=1)
        packet_weight = np.power(
            np.maximum(work["bidirectional_packets"].astype("float64"), 1.0),
            task_config.dataset.packet_weight_power,
        )
        work["vote_weight"] = np.where(
            work["max_prob"] >= task_config.dataset.vote_min_confidence,
            work["max_prob"] * packet_weight,
            0.0,
        )

        selected = work[work["vote_weight"] > 0].copy()
        if selected.empty:
            selected = work.copy()
            selected["vote_weight"] = selected["max_prob"] * packet_weight

        selected = selected.sort_values("vote_weight", ascending=False).head(
            task_config.dataset.vote_top_flows
        )

        class_scores = {
            label: float((selected[f"prob_{label}"] * selected["vote_weight"]).sum())
            for label in classes
        }
        predicted_task = max(class_scores, key=class_scores.get)
        sorted_scores = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)
        second_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0

        capture_rows.append(
            {
                "capture_name": capture_name,
                "capture_path": group["capture_path"].iloc[0],
                DEFAULT_TASK_LABEL_COLUMN: group[DEFAULT_TASK_LABEL_COLUMN].iloc[0],
                f"predicted_{DEFAULT_TASK_LABEL_COLUMN}": predicted_task,
                "selected_flow_count": int(len(selected)),
                "total_flow_count": int(len(group)),
                "winning_score": float(class_scores[predicted_task]),
                "score_margin": float(class_scores[predicted_task] - second_score),
                **{f"capture_score_{label}": score for label, score in class_scores.items()},
            }
        )

    return pd.DataFrame(capture_rows)


def _task_train_eval_paths(out_dir: Path) -> dict[str, Path]:
    base = ensure_dir(out_dir)
    return {
        "base": base,
        "model": base / "model.joblib",
        "flow_report_text": base / "flow_classification_report.txt",
        "flow_report_json": base / "flow_classification_report.json",
        "flow_confusion": base / "flow_confusion_matrix.png",
        "capture_report_text": base / "capture_classification_report.txt",
        "capture_report_json": base / "capture_classification_report.json",
        "capture_confusion": base / "capture_confusion_matrix.png",
        "flow_predictions": base / "oof_flow_predictions.parquet",
        "capture_predictions": base / "capture_predictions.parquet",
        "metadata": base / "run_metadata.json",
    }


def run_task_train_eval(
    task_data_path: Path,
    out_dir: Path,
    app_config: AppConfig,
    task_config: TaskConfig,
    *,
    auxiliary_data_path: Path | None = None,
) -> dict[str, Any]:
    paths = _task_train_eval_paths(out_dir)
    task_df = load_task_dataframe(task_data_path, app_config)
    X_task, feature_width = _build_task_feature_matrix(task_df, app_config)
    y_task = task_df[DEFAULT_TASK_LABEL_COLUMN]

    capture_labels = task_df[["capture_name", DEFAULT_TASK_LABEL_COLUMN]].drop_duplicates()
    class_capture_counts = capture_labels[DEFAULT_TASK_LABEL_COLUMN].value_counts()
    n_splits = min(4, int(class_capture_counts.min()))
    if n_splits < 2:
        raise ValueError("Need at least 2 captures per class for task-level evaluation.")

    X_aux, y_aux = _prepare_auxiliary_dataset(
        auxiliary_data_path,
        app_config,
        width=feature_width,
    )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=app_config.split.random_state,
    )

    fold_predictions: list[pd.DataFrame] = []

    for fold_index, (train_idx, test_idx) in enumerate(
        splitter.split(X_task, y_task, groups=task_df["capture_name"]),
        start=1,
    ):
        X_train = X_task.iloc[train_idx].reset_index(drop=True)
        y_train = y_task.iloc[train_idx].reset_index(drop=True)
        train_weights = (
            task_df.iloc[train_idx]["capture_normalized_weight"].astype("float64").to_numpy()
        )

        combined_X, combined_y, combined_weight = _combined_training_frame(
            X_train=X_train,
            y_train=y_train,
            sample_weight=train_weights,
            X_aux=X_aux,
            y_aux=y_aux,
            auxiliary_weight_ratio=task_config.dataset.auxiliary_weight_ratio,
        )

        model = build_random_forest(
            n_estimators=app_config.model.n_estimators,
            random_state=app_config.split.random_state,
            n_jobs=app_config.model.n_jobs,
            class_weight=app_config.model.class_weight,
        )
        model.fit(combined_X, combined_y, sample_weight=combined_weight)

        X_test = X_task.iloc[test_idx].reset_index(drop=True)
        probabilities = model.predict_proba(X_test)
        predicted = model.predict(X_test)

        fold_df = task_df.iloc[test_idx].reset_index(drop=True).copy()
        fold_df["fold"] = fold_index
        fold_df[f"predicted_{DEFAULT_TASK_LABEL_COLUMN}"] = predicted
        fold_df["prediction_confidence"] = probabilities.max(axis=1)
        for class_index, label in enumerate(model.classes_):
            fold_df[f"prob_{label}"] = probabilities[:, class_index]
        fold_predictions.append(fold_df)

    oof_predictions = pd.concat(fold_predictions, ignore_index=True)
    oof_predictions.to_parquet(paths["flow_predictions"], index=False)

    save_classification_reports(
        y_true=oof_predictions[DEFAULT_TASK_LABEL_COLUMN],
        y_pred=oof_predictions[f"predicted_{DEFAULT_TASK_LABEL_COLUMN}"],
        text_path=paths["flow_report_text"],
        json_path=paths["flow_report_json"],
    )
    save_confusion_matrix(
        y_true=oof_predictions[DEFAULT_TASK_LABEL_COLUMN],
        y_pred=oof_predictions[f"predicted_{DEFAULT_TASK_LABEL_COLUMN}"],
        out_path=paths["flow_confusion"],
        title="Flow-Level Task Classification",
    )

    capture_predictions = aggregate_capture_votes(
        oof_predictions,
        classes=sorted(task_df[DEFAULT_TASK_LABEL_COLUMN].unique().tolist()),
        task_config=task_config,
    )
    capture_predictions.to_parquet(paths["capture_predictions"], index=False)

    save_classification_reports(
        y_true=capture_predictions[DEFAULT_TASK_LABEL_COLUMN],
        y_pred=capture_predictions[f"predicted_{DEFAULT_TASK_LABEL_COLUMN}"],
        text_path=paths["capture_report_text"],
        json_path=paths["capture_report_json"],
    )
    save_confusion_matrix(
        y_true=capture_predictions[DEFAULT_TASK_LABEL_COLUMN],
        y_pred=capture_predictions[f"predicted_{DEFAULT_TASK_LABEL_COLUMN}"],
        out_path=paths["capture_confusion"],
        title="Capture-Level Task Classification",
    )

    final_weights = task_df["capture_normalized_weight"].astype("float64").to_numpy()
    combined_X, combined_y, combined_weight = _combined_training_frame(
        X_train=X_task.reset_index(drop=True),
        y_train=y_task.reset_index(drop=True),
        sample_weight=final_weights,
        X_aux=X_aux,
        y_aux=y_aux,
        auxiliary_weight_ratio=task_config.dataset.auxiliary_weight_ratio,
    )
    final_model = build_random_forest(
        n_estimators=app_config.model.n_estimators,
        random_state=app_config.split.random_state,
        n_jobs=app_config.model.n_jobs,
        class_weight=app_config.model.class_weight,
    )
    final_model.fit(combined_X, combined_y, sample_weight=combined_weight)

    model_bundle = {
        "model": final_model,
        "feature_width": feature_width,
        "feature_columns": list(X_task.columns),
        "target_column": DEFAULT_TASK_LABEL_COLUMN,
        "classes": final_model.classes_.tolist(),
        "task_classes": list(task_config.final_classes),
        "vote_min_confidence": task_config.dataset.vote_min_confidence,
        "vote_top_flows": task_config.dataset.vote_top_flows,
        "packet_weight_power": task_config.dataset.packet_weight_power,
    }
    save_model_bundle(model_bundle, paths["model"])

    with paths["capture_report_json"].open("r", encoding="utf-8") as handle:
        capture_report = json.load(handle)
    with paths["flow_report_json"].open("r", encoding="utf-8") as handle:
        flow_report = json.load(handle)

    metadata = {
        "task_data_path": str(task_data_path),
        "auxiliary_data_path": str(auxiliary_data_path) if auxiliary_data_path else None,
        "out_dir": str(paths["base"]),
        "rows": int(len(task_df)),
        "captures": int(task_df["capture_name"].nunique()),
        "class_counts": task_df[DEFAULT_TASK_LABEL_COLUMN].value_counts().sort_index().to_dict(),
        "n_splits": n_splits,
        "feature_width": feature_width,
        "flow_accuracy": float(flow_report.get("accuracy", 0.0)),
        "capture_accuracy": float(capture_report.get("accuracy", 0.0)),
        "flow_macro_f1": float(flow_report.get("macro avg", {}).get("f1-score", 0.0)),
        "capture_macro_f1": float(capture_report.get("macro avg", {}).get("f1-score", 0.0)),
        "flow_report_text": str(paths["flow_report_text"]),
        "capture_report_text": str(paths["capture_report_text"]),
        "flow_predictions": str(paths["flow_predictions"]),
        "capture_predictions": str(paths["capture_predictions"]),
        "model_path": str(paths["model"]),
    }
    write_json(paths["metadata"], metadata)
    LOGGER.info("Saved task train/eval artifacts to %s", paths["base"])
    return metadata


def _fit_full_task_model(
    task_data_path: Path,
    app_config: AppConfig,
    task_config: TaskConfig,
    *,
    auxiliary_data_path: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    task_df = load_task_dataframe(task_data_path, app_config)
    X_task, feature_width = _build_task_feature_matrix(task_df, app_config)
    y_task = task_df[DEFAULT_TASK_LABEL_COLUMN]
    final_weights = task_df["capture_normalized_weight"].astype("float64").to_numpy()
    X_aux, y_aux = _prepare_auxiliary_dataset(auxiliary_data_path, app_config, width=feature_width)

    combined_X, combined_y, combined_weight = _combined_training_frame(
        X_train=X_task.reset_index(drop=True),
        y_train=y_task.reset_index(drop=True),
        sample_weight=final_weights,
        X_aux=X_aux,
        y_aux=y_aux,
        auxiliary_weight_ratio=task_config.dataset.auxiliary_weight_ratio,
    )
    model = build_random_forest(
        n_estimators=app_config.model.n_estimators,
        random_state=app_config.split.random_state,
        n_jobs=app_config.model.n_jobs,
        class_weight=app_config.model.class_weight,
    )
    model.fit(combined_X, combined_y, sample_weight=combined_weight)

    model_bundle = {
        "model": model,
        "feature_width": feature_width,
        "feature_columns": list(X_task.columns),
        "target_column": DEFAULT_TASK_LABEL_COLUMN,
        "classes": model.classes_.tolist(),
        "task_classes": list(task_config.final_classes),
        "vote_min_confidence": task_config.dataset.vote_min_confidence,
        "vote_top_flows": task_config.dataset.vote_top_flows,
        "packet_weight_power": task_config.dataset.packet_weight_power,
    }
    return model_bundle, task_df


def run_task_pcap_predict(
    pcap_path: Path,
    task_data_path: Path,
    out_dir: Path,
    app_config: AppConfig,
    task_config: TaskConfig,
    *,
    auxiliary_data_path: Path | None = None,
    n_dissections_override: int | None = None,
) -> dict[str, Any]:
    base = ensure_dir(out_dir)
    model_bundle, _ = _fit_full_task_model(
        task_data_path,
        app_config,
        task_config,
        auxiliary_data_path=auxiliary_data_path,
    )
    save_model_bundle(model_bundle, base / "model.joblib")

    resolved_n_dissections = (
        app_config.nfstream.n_dissections
        if n_dissections_override is None
        else n_dissections_override
    )
    df = extract_pcap_dataframe(
        pcap_path=pcap_path,
        n_meters=app_config.nfstream.n_meters,
        n_dissections=resolved_n_dissections,
        statistical_analysis=app_config.nfstream.statistical_analysis,
        splt_analysis=app_config.nfstream.splt_analysis,
        accounting_mode=app_config.nfstream.accounting_mode,
        interim_required_columns=app_config.dataset.interim_required_columns,
        interim_optional_dpi_columns=app_config.dataset.interim_optional_dpi_columns,
    )
    extracted_flows = int(len(df))
    df = df[df["bidirectional_packets"] >= task_config.dataset.min_packets].copy()
    df["capture_name"] = pcap_path.stem
    df["capture_path"] = str(pcap_path)
    inferred_label = infer_task_label(pcap_path.name, task_config)
    df[DEFAULT_TASK_LABEL_COLUMN] = inferred_label or "Unknown"
    df = _parse_sequence_columns(df, app_config.dataset.splt_feature_columns)

    if df.empty:
        predictions_df = df.copy()
        predictions_df[f"predicted_{DEFAULT_TASK_LABEL_COLUMN}"] = pd.Series(dtype="object")
        capture_prediction = {
            "capture_name": pcap_path.stem,
            f"predicted_{DEFAULT_TASK_LABEL_COLUMN}": "Unknown",
            "selected_flow_count": 0,
            "total_flow_count": 0,
        }
    else:
        X_predict, _ = _build_task_feature_matrix(
            df,
            app_config,
            width=int(model_bundle["feature_width"]),
        )
        model = model_bundle["model"]
        probabilities = model.predict_proba(X_predict)
        predicted = model.predict(X_predict)

        predictions_df = df.copy()
        predictions_df[f"predicted_{DEFAULT_TASK_LABEL_COLUMN}"] = predicted
        predictions_df["prediction_confidence"] = probabilities.max(axis=1)
        for class_index, label in enumerate(model.classes_):
            predictions_df[f"prob_{label}"] = probabilities[:, class_index]

        capture_prediction_df = aggregate_capture_votes(
            predictions_df,
            classes=model.classes_.tolist(),
            task_config=task_config,
        )
        capture_prediction = capture_prediction_df.iloc[0].to_dict()

    predictions_path = base / "predictions.parquet"
    metadata_path = base / "run_metadata.json"
    summary_path = base / "prediction_summary.txt"
    predictions_df.to_parquet(predictions_path, index=False)

    predicted_task = capture_prediction.get(f"predicted_{DEFAULT_TASK_LABEL_COLUMN}", "Unknown")
    expected_text = inferred_label if inferred_label is not None else "Unknown"
    summary_lines = [
        f"PCAP: {pcap_path}",
        f"Expected task from filename: {expected_text}",
        f"Extracted flows: {extracted_flows}",
        f"Flows after min_packets>={task_config.dataset.min_packets}: {len(predictions_df)}",
        f"Predicted task: {predicted_task}",
    ]
    if "selected_flow_count" in capture_prediction:
        summary_lines.append(
            f"Voting used {capture_prediction['selected_flow_count']} / {capture_prediction['total_flow_count']} flows."
        )
    summary_text = "\n".join(summary_lines) + "\n"
    write_text(summary_path, summary_text)
    print(summary_text, end="")

    metadata = {
        "pcap_path": str(pcap_path),
        "task_data_path": str(task_data_path),
        "auxiliary_data_path": str(auxiliary_data_path) if auxiliary_data_path else None,
        "predictions_path": str(predictions_path),
        "model_path": str(base / "model.joblib"),
        "summary_path": str(summary_path),
        "n_dissections": resolved_n_dissections,
        "extracted_flows": extracted_flows,
        "post_filter_flows": int(len(predictions_df)),
        "expected_task_from_filename": inferred_label,
        "predicted_task": predicted_task,
        "capture_prediction": capture_prediction,
    }
    write_json(metadata_path, metadata)
    LOGGER.info("Saved task PCAP prediction artifacts to %s", base)
    return metadata
