from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from act2_project.config import AppConfig
from act2_project.paths import build_pcap_predict_artifacts
from act2_project.pipeline.act2_prepare import build_pcap_inference_dataset
from act2_project.pipeline.evaluate import save_classification_reports, save_confusion_matrix
from act2_project.pipeline.pcap_ingest import run_pcap_ingest
from act2_project.pipeline.splt_features import build_feature_matrix, load_act2_dataframe
from act2_project.pipeline.train import fit_model_bundle, save_model_bundle
from act2_project.utils.io import write_json, write_text
from act2_project.utils.logging import get_logger

LOGGER = get_logger(__name__)


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found: {model_path}")
    payload = joblib.load(model_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected model bundle dict in {model_path}, got {type(payload)!r}")
    return payload


def _format_prediction_summary(
    *,
    pcap_path: Path,
    extracted_flows: int,
    model_input_flows: int,
    predicted_counts: dict[str, int],
    proxy_eval_rows: int,
    report_text_path: Path | None,
    confusion_matrix_path: Path | None,
) -> str:
    lines = [
        f"PCAP: {pcap_path}",
        (
            f"So lieu: {extracted_flows} flows trich ra, "
            f"{model_input_flows} flows vao model sau filter bidirectional_packets>=10."
        ),
    ]

    if predicted_counts:
        label_summary = ", ".join(f"{label} = {count}" for label, count in predicted_counts.items())
    else:
        label_summary = "Khong co flow nao du dieu kien de dua vao model"
    lines.append(f"Nhan du doan ra: {label_summary}.")

    if proxy_eval_rows > 0 and report_text_path is not None and confusion_matrix_path is not None:
        lines.append(
            f"Proxy evaluation rows (DPI label thuoc tap category train): {proxy_eval_rows}."
        )
        lines.append(f"Proxy classification report: {report_text_path}")
        lines.append(f"Proxy confusion matrix: {confusion_matrix_path}")

    return "\n".join(lines) + "\n"


def run_pcap_predict(
    pcap_path: Path,
    train_data_path: Path,
    out_dir: Path,
    app_config: AppConfig,
    *,
    target_column: str | None = None,
    n_dissections_override: int | None = None,
    ingest_extra_columns: tuple[str, ...] = (),
) -> dict[str, Any]:
    artifacts = build_pcap_predict_artifacts(out_dir, app_config.artifacts)

    ingest_metadata = run_pcap_ingest(
        pcap_path=pcap_path,
        out_path=artifacts.flows_parquet_path,
        app_config=app_config,
        n_dissections_override=n_dissections_override,
        extra_columns=ingest_extra_columns,
    )
    prepare_metadata = build_pcap_inference_dataset(
        flows_path=artifacts.flows_parquet_path,
        out_path=artifacts.act2_parquet_path,
        app_config=app_config,
    )

    model_bundle = fit_model_bundle(
        data_path=train_data_path,
        app_config=app_config,
        target_column=target_column,
    )
    save_model_bundle(model_bundle, artifacts.model_path)

    resolved_target = model_bundle["target_column"]
    df_predict = load_act2_dataframe(
        artifacts.act2_parquet_path,
        splt_column=app_config.dataset.splt_column,
        sequence_columns=app_config.dataset.splt_feature_columns,
    )
    sequence_columns = [
        column for column in app_config.dataset.splt_feature_columns if column in df_predict.columns
    ] or [app_config.dataset.splt_column]

    predicted_column = f"predicted_{resolved_target}"
    proxy_eval_rows = 0
    report_paths: tuple[Path | None, Path | None] = (None, None)

    if df_predict.empty:
        predictions_df = df_predict.copy()
        predictions_df[predicted_column] = pd.Series(dtype="object")
        predictions_df["prediction_confidence"] = pd.Series(dtype="float64")
        predicted_counts: dict[str, int] = {}
    else:
        X_predict, _ = build_feature_matrix(
            df_predict,
            splt_column=app_config.dataset.splt_column,
            width=int(model_bundle["feature_width"]),
            sequence_columns=sequence_columns,
        )
        model = model_bundle["model"]
        predictions_df = df_predict.copy()
        predictions_df[predicted_column] = model.predict(X_predict)
        predicted_counts = (
            predictions_df[predicted_column].value_counts().sort_index().astype(int).to_dict()
        )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_predict)
            predictions_df["prediction_confidence"] = probabilities.max(axis=1)

        if resolved_target in predictions_df.columns:
            proxy_eval_mask = predictions_df[resolved_target].isin(model_bundle["classes"])
            proxy_eval_rows = int(proxy_eval_mask.sum())

            if proxy_eval_rows > 0:
                y_true = predictions_df.loc[proxy_eval_mask, resolved_target]
                y_pred = predictions_df.loc[proxy_eval_mask, predicted_column]

                save_classification_reports(
                    y_true=y_true,
                    y_pred=y_pred,
                    text_path=artifacts.report_text_path,
                    json_path=artifacts.report_json_path,
                )
                save_confusion_matrix(
                    y_true=y_true,
                    y_pred=y_pred,
                    out_path=artifacts.confusion_matrix_path,
                    title=f"Proxy Confusion Matrix for {pcap_path.name}",
                )
                report_paths = (artifacts.report_text_path, artifacts.confusion_matrix_path)

    predictions_df.to_parquet(artifacts.predictions_path, index=False)

    summary_text = _format_prediction_summary(
        pcap_path=pcap_path,
        extracted_flows=int(ingest_metadata["rows"]),
        model_input_flows=int(prepare_metadata["output_rows"]),
        predicted_counts=predicted_counts,
        proxy_eval_rows=proxy_eval_rows,
        report_text_path=report_paths[0],
        confusion_matrix_path=report_paths[1],
    )
    write_text(artifacts.summary_path, summary_text)
    print(summary_text, end="")

    metadata: dict[str, Any] = {
        "pcap_path": str(pcap_path),
        "train_data_path": str(train_data_path),
        "target_column": resolved_target,
        "feature_width": int(model_bundle["feature_width"]),
        "ingest": ingest_metadata,
        "prepare": prepare_metadata,
        "flows_path": str(artifacts.flows_parquet_path),
        "prepared_inference_path": str(artifacts.act2_parquet_path),
        "model_path": str(artifacts.model_path),
        "predictions_path": str(artifacts.predictions_path),
        "prediction_rows": int(len(predictions_df)),
        "predicted_counts": predicted_counts,
        "proxy_eval_rows": proxy_eval_rows,
        "classification_report_text": str(artifacts.report_text_path) if report_paths[0] else None,
        "classification_report_json": str(artifacts.report_json_path) if report_paths[0] else None,
        "confusion_matrix_path": str(artifacts.confusion_matrix_path) if report_paths[1] else None,
        "summary_path": str(artifacts.summary_path),
        "trained_categories": model_bundle["classes"],
    }
    write_json(artifacts.metadata_path, metadata)

    LOGGER.info("Saved PCAP prediction artifacts to %s", artifacts.base_out_dir)
    return metadata


def run_pcap_predict_pretrained(
    pcap_path: Path,
    model_path: Path,
    out_dir: Path,
    app_config: AppConfig,
    *,
    n_dissections_override: int | None = None,
    ingest_extra_columns: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Chạy dự đoán PCAP bằng model đã train sẵn, không train lại.

    Đây là đường chạy phù hợp để benchmark gần runtime thực tế hơn
    so với run_pcap_predict(), vì nó bỏ hẳn chi phí fit model.
    """

    artifacts = build_pcap_predict_artifacts(out_dir, app_config.artifacts)

    ingest_metadata = run_pcap_ingest(
        pcap_path=pcap_path,
        out_path=artifacts.flows_parquet_path,
        app_config=app_config,
        n_dissections_override=n_dissections_override,
        extra_columns=ingest_extra_columns,
    )
    prepare_metadata = build_pcap_inference_dataset(
        flows_path=artifacts.flows_parquet_path,
        out_path=artifacts.act2_parquet_path,
        app_config=app_config,
    )

    model_bundle = load_model_bundle(model_path)
    resolved_target = model_bundle["target_column"]

    df_predict = load_act2_dataframe(
        artifacts.act2_parquet_path,
        splt_column=app_config.dataset.splt_column,
        sequence_columns=app_config.dataset.splt_feature_columns,
    )
    sequence_columns = [
        column for column in app_config.dataset.splt_feature_columns if column in df_predict.columns
    ] or [app_config.dataset.splt_column]

    predicted_column = f"predicted_{resolved_target}"
    proxy_eval_rows = 0
    report_paths: tuple[Path | None, Path | None] = (None, None)

    if df_predict.empty:
        predictions_df = df_predict.copy()
        predictions_df[predicted_column] = pd.Series(dtype="object")
        predictions_df["prediction_confidence"] = pd.Series(dtype="float64")
        predicted_counts: dict[str, int] = {}
    else:
        X_predict, _ = build_feature_matrix(
            df_predict,
            splt_column=app_config.dataset.splt_column,
            width=int(model_bundle["feature_width"]),
            sequence_columns=sequence_columns,
        )
        model = model_bundle["model"]
        predictions_df = df_predict.copy()
        predictions_df[predicted_column] = model.predict(X_predict)
        predicted_counts = (
            predictions_df[predicted_column].value_counts().sort_index().astype(int).to_dict()
        )

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_predict)
            predictions_df["prediction_confidence"] = probabilities.max(axis=1)

        if resolved_target in predictions_df.columns:
            proxy_eval_mask = predictions_df[resolved_target].isin(model_bundle["classes"])
            proxy_eval_rows = int(proxy_eval_mask.sum())
            if proxy_eval_rows > 0:
                y_true = predictions_df.loc[proxy_eval_mask, resolved_target]
                y_pred = predictions_df.loc[proxy_eval_mask, predicted_column]
                save_classification_reports(
                    y_true=y_true,
                    y_pred=y_pred,
                    text_path=artifacts.report_text_path,
                    json_path=artifacts.report_json_path,
                )
                save_confusion_matrix(
                    y_true=y_true,
                    y_pred=y_pred,
                    out_path=artifacts.confusion_matrix_path,
                    title=f"Proxy Confusion Matrix for {pcap_path.name}",
                )
                report_paths = (artifacts.report_text_path, artifacts.confusion_matrix_path)

    predictions_df.to_parquet(artifacts.predictions_path, index=False)

    summary_text = _format_prediction_summary(
        pcap_path=pcap_path,
        extracted_flows=int(ingest_metadata["rows"]),
        model_input_flows=int(prepare_metadata["output_rows"]),
        predicted_counts=predicted_counts,
        proxy_eval_rows=proxy_eval_rows,
        report_text_path=report_paths[0],
        confusion_matrix_path=report_paths[1],
    )
    write_text(artifacts.summary_path, summary_text)
    print(summary_text, end="")

    metadata: dict[str, Any] = {
        "pcap_path": str(pcap_path),
        "target_column": resolved_target,
        "feature_width": int(model_bundle["feature_width"]),
        "ingest": ingest_metadata,
        "prepare": prepare_metadata,
        "flows_path": str(artifacts.flows_parquet_path),
        "prepared_inference_path": str(artifacts.act2_parquet_path),
        "model_path": str(model_path),
        "predictions_path": str(artifacts.predictions_path),
        "prediction_rows": int(len(predictions_df)),
        "predicted_counts": predicted_counts,
        "proxy_eval_rows": proxy_eval_rows,
        "classification_report_text": str(artifacts.report_text_path) if report_paths[0] else None,
        "classification_report_json": str(artifacts.report_json_path) if report_paths[0] else None,
        "confusion_matrix_path": str(artifacts.confusion_matrix_path) if report_paths[1] else None,
        "summary_path": str(artifacts.summary_path),
        "trained_categories": model_bundle["classes"],
    }
    write_json(artifacts.metadata_path, metadata)

    LOGGER.info("Saved pretrained PCAP prediction artifacts to %s", artifacts.base_out_dir)
    return metadata
