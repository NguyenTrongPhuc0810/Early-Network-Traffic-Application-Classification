from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from act2_project.config import AppConfig
from act2_project.utils.io import ensure_dir
from act2_project.utils.logging import get_logger

LOGGER = get_logger(__name__)


def require_columns(df: pd.DataFrame, cols: Sequence[str], context: str) -> None:
    missing = [column for column in cols if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {missing}")


def _dedupe_preserve_order(columns: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        ordered.append(column)
    return ordered


def _filter_clean_rows(df: pd.DataFrame, *, min_packets: int) -> pd.DataFrame:
    df_clean = df[df["application_is_guessed"] == 0].copy()
    df_clean = df_clean[df_clean["application_confidence"] == 6].copy()
    df_clean = df_clean[df_clean["application_name"] != "Unknown"].copy()
    df_clean = df_clean[df_clean["bidirectional_packets"] >= min_packets].copy()
    return df_clean


def _build_df_final_model_data(
    df_clean: pd.DataFrame,
    *,
    application_column: str,
    min_samples_per_application: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    app_counts = df_clean[application_column].value_counts()
    apps_to_keep = app_counts[app_counts >= min_samples_per_application].index.tolist()
    df_final_model_data = df_clean[df_clean[application_column].isin(apps_to_keep)].copy()

    app_to_category_counts = (
        df_final_model_data.groupby(application_column)["application_category_name"].nunique()
    )
    ambiguous_applications = app_to_category_counts[app_to_category_counts > 1].sort_values(
        ascending=False
    )

    metadata = {
        "min_samples_per_application": int(min_samples_per_application),
        "applications_retained": int(len(apps_to_keep)),
        "after_min_samples_rows": int(len(df_final_model_data)),
        "ambiguous_application_count": int(len(ambiguous_applications)),
        "top_ambiguous_applications": ambiguous_applications.head(10).to_dict(),
    }
    return df_final_model_data, metadata


def prepare_training_dataframe(
    df: pd.DataFrame,
    *,
    application_column: str,
    training_filter_columns: Sequence[str],
    splt_feature_columns: Sequence[str],
    label_columns: Sequence[str],
    target_column: str,
    category_subset: Sequence[str],
    min_packets: int,
    min_samples_per_application: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    require_columns(df, training_filter_columns, context="training filters")
    require_columns(
        df,
        list(splt_feature_columns) + [application_column, target_column],
        context="SPLT training columns",
    )

    df_clean = _filter_clean_rows(df, min_packets=min_packets)
    df_final_model_data, threshold_metadata = _build_df_final_model_data(
        df_clean,
        application_column=application_column,
        min_samples_per_application=min_samples_per_application,
    )

    if category_subset:
        df_selected = df_final_model_data[
            df_final_model_data[target_column].isin(category_subset)
        ].copy()
    else:
        df_selected = df_final_model_data.copy()

    output_columns = _dedupe_preserve_order(
        list(splt_feature_columns)
        + [target_column]
        + [column for column in label_columns if column in df_selected.columns]
    )
    df_prepared = df_selected[output_columns].copy()

    present_categories = df_prepared[target_column].value_counts().sort_index().to_dict()
    metadata = {
        "input_rows": int(len(df)),
        "after_cleaning_rows": int(len(df_clean)),
        "after_min_samples_rows": int(threshold_metadata["after_min_samples_rows"]),
        "output_rows": int(len(df_prepared)),
        "output_columns": output_columns,
        "present_categories": present_categories,
        "missing_categories": [
            category for category in category_subset if category not in present_categories
        ],
        "application_column": application_column,
        "category_subset_applied": bool(category_subset),
        "category_subset": list(category_subset),
        "unique_applications_in_output": int(df_selected[application_column].nunique()),
        "unique_categories_in_output": int(df_selected[target_column].nunique()),
        **threshold_metadata,
    }
    return df_prepared, metadata


def prepare_core_training_dataframe(
    df: pd.DataFrame,
    *,
    application_column: str,
    training_filter_columns: Sequence[str],
    splt_feature_columns: Sequence[str],
    label_columns: Sequence[str],
    target_column: str,
    core_category_subset: Sequence[str],
    min_packets: int,
    min_samples_per_application: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return prepare_training_dataframe(
        df,
        application_column=application_column,
        training_filter_columns=training_filter_columns,
        splt_feature_columns=splt_feature_columns,
        label_columns=label_columns,
        target_column=target_column,
        category_subset=core_category_subset,
        min_packets=min_packets,
        min_samples_per_application=min_samples_per_application,
    )


def prepare_pcap_inference_dataframe(
    df: pd.DataFrame,
    *,
    inference_required_columns: Sequence[str],
    splt_feature_columns: Sequence[str],
    label_columns: Sequence[str],
    min_packets: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    require_columns(df, inference_required_columns, context="pcap SPLT inference columns")

    df_filtered = df[df["bidirectional_packets"] >= min_packets].copy()
    output_columns = _dedupe_preserve_order(
        list(splt_feature_columns) + [column for column in label_columns if column in df_filtered.columns]
    )
    df_prepared = df_filtered[output_columns].copy()

    metadata = {
        "input_rows": int(len(df)),
        "after_min_packets_rows": int(len(df_filtered)),
        "output_rows": int(len(df_prepared)),
        "output_columns": output_columns,
    }
    return df_prepared, metadata


def build_training_dataset(
    data_path: Path,
    out_path: Path,
    app_config: AppConfig,
    *,
    target_column: str | None = None,
) -> dict[str, Any]:
    if not data_path.exists():
        raise FileNotFoundError(f"Training parquet not found: {data_path}")

    resolved_target = target_column or app_config.dataset.target_column
    df = pd.read_parquet(data_path)
    df_prepared, metadata = prepare_training_dataframe(
        df,
        application_column=app_config.dataset.application_column,
        training_filter_columns=app_config.dataset.training_filter_columns,
        splt_feature_columns=app_config.dataset.splt_feature_columns,
        label_columns=app_config.dataset.label_columns,
        target_column=resolved_target,
        category_subset=app_config.category_subset,
        min_packets=app_config.dataset.min_packets,
        min_samples_per_application=app_config.dataset.min_samples_per_application,
    )

    ensure_dir(out_path.parent)
    df_prepared.to_parquet(out_path, index=False)

    metadata["input_path"] = str(data_path)
    metadata["output_path"] = str(out_path)
    metadata["target_column"] = resolved_target

    LOGGER.info("Prepared SPLT training dataset with %s rows at %s", len(df_prepared), out_path)
    return metadata


def build_core_training_dataset(
    data_path: Path,
    out_path: Path,
    app_config: AppConfig,
    *,
    target_column: str | None = None,
) -> dict[str, Any]:
    return build_training_dataset(
        data_path=data_path,
        out_path=out_path,
        app_config=app_config,
        target_column=target_column,
    )


def build_pcap_inference_dataset(
    flows_path: Path,
    out_path: Path,
    app_config: AppConfig,
) -> dict[str, Any]:
    if not flows_path.exists():
        raise FileNotFoundError(f"Flows parquet not found: {flows_path}")

    df = pd.read_parquet(flows_path)
    df_prepared, metadata = prepare_pcap_inference_dataframe(
        df,
        inference_required_columns=app_config.dataset.inference_required_columns,
        splt_feature_columns=app_config.dataset.splt_feature_columns,
        label_columns=app_config.dataset.label_columns,
        min_packets=app_config.dataset.min_packets,
    )

    ensure_dir(out_path.parent)
    df_prepared.to_parquet(out_path, index=False)

    metadata["input_path"] = str(flows_path)
    metadata["output_path"] = str(out_path)

    LOGGER.info("Prepared PCAP inference dataset with %s rows at %s", len(df_prepared), out_path)
    return metadata
