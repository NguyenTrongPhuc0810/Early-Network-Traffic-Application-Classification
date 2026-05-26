from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from act2_project.config import AppConfig
from act2_project.domain.constants import DEFAULT_TASK_LABEL_COLUMN
from act2_project.pipeline.act2_prepare import _build_df_final_model_data, _filter_clean_rows
from act2_project.pipeline.pcap_ingest import extract_pcap_dataframe
from act2_project.task_config import TaskConfig
from act2_project.utils.io import ensure_dir
from act2_project.utils.logging import get_logger

LOGGER = get_logger(__name__)


def infer_task_label(filename: str, task_config: TaskConfig) -> str | None:
    normalized = filename.lower()
    for label in task_config.label_priority:
        patterns = task_config.label_patterns.get(label, ())
        if any(pattern.lower() in normalized for pattern in patterns):
            return label
    return None


def _is_background_flow(application_name: str, application_category_name: str, task_config: TaskConfig) -> bool:
    app_name = application_name.lower()
    category_name = application_category_name
    if any(app_name.startswith(prefix.lower()) for prefix in task_config.background.application_prefixes):
        return True
    return category_name in set(task_config.background.category_names)


def _matches_foreground_rule(
    *,
    task_label: str,
    application_name: str,
    application_category_name: str,
    task_config: TaskConfig,
) -> bool:
    rule = task_config.foreground.get(task_label)
    if rule is None:
        return False
    app_name = application_name.lower()
    if application_category_name in set(rule.category_names):
        return True
    return any(keyword.lower() in app_name for keyword in rule.application_keywords)


def score_task_flow(
    *,
    task_label: str,
    application_name: str,
    application_category_name: str,
    task_config: TaskConfig,
) -> tuple[float, str]:
    if _is_background_flow(application_name, application_category_name, task_config):
        return task_config.dataset.background_weight, "background"

    if _matches_foreground_rule(
        task_label=task_label,
        application_name=application_name,
        application_category_name=application_category_name,
        task_config=task_config,
    ):
        return task_config.dataset.foreground_weight, "foreground"

    return task_config.dataset.support_weight, "support"


def _normalize_capture_weights(
    df: pd.DataFrame,
    *,
    capture_total_weight: float,
) -> pd.DataFrame:
    normalized = df.copy()
    if normalized.empty:
        normalized["capture_normalized_weight"] = pd.Series(dtype="float64")
        return normalized

    totals = normalized.groupby("capture_name")["flow_relevance_weight"].transform("sum")
    normalized["capture_normalized_weight"] = 0.0
    positive_mask = totals > 0
    normalized.loc[positive_mask, "capture_normalized_weight"] = (
        normalized.loc[positive_mask, "flow_relevance_weight"] / totals.loc[positive_mask]
    ) * capture_total_weight
    return normalized


def _task_dataset_output_columns(app_config: AppConfig) -> list[str]:
    columns = [
        "capture_name",
        "capture_path",
        DEFAULT_TASK_LABEL_COLUMN,
        "bidirectional_packets",
        "flow_relevance_weight",
        "flow_role",
        "capture_normalized_weight",
    ]
    for column in app_config.dataset.splt_feature_columns:
        if column not in columns:
            columns.append(column)
    for column in ("application_name", "application_category_name"):
        if column not in columns:
            columns.append(column)
    return columns


def build_task_dataset(
    pcap_dir: Path,
    out_path: Path,
    app_config: AppConfig,
    task_config: TaskConfig,
    *,
    n_dissections_override: int | None = None,
) -> dict[str, Any]:
    if not pcap_dir.exists():
        raise FileNotFoundError(f"PCAP directory not found: {pcap_dir}")

    pcap_files = sorted(
        [
            path
            for path in pcap_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".pcap", ".pcapng"}
        ]
    )
    rows: list[pd.DataFrame] = []
    skipped_files: list[str] = []
    errors: dict[str, str] = {}

    resolved_n_dissections = (
        app_config.nfstream.n_dissections
        if n_dissections_override is None
        else n_dissections_override
    )

    for pcap_path in pcap_files:
        task_label = infer_task_label(pcap_path.name, task_config)
        if task_label is None:
            skipped_files.append(pcap_path.name)
            continue

        try:
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
        except Exception as exc:
            errors[pcap_path.name] = str(exc)
            LOGGER.exception("Failed to extract PCAP %s", pcap_path)
            continue

        df = df[df["bidirectional_packets"] >= task_config.dataset.min_packets].copy()
        if df.empty:
            skipped_files.append(f"{pcap_path.name}:no_rows_after_min_packets")
            continue

        df["capture_name"] = pcap_path.stem
        df["capture_path"] = str(pcap_path)
        df[DEFAULT_TASK_LABEL_COLUMN] = task_label

        scored = df.apply(
            lambda row: score_task_flow(
                task_label=task_label,
                application_name=str(row.get("application_name", "")),
                application_category_name=str(row.get("application_category_name", "")),
                task_config=task_config,
            ),
            axis=1,
            result_type="expand",
        )
        scored.columns = ["flow_relevance_weight", "flow_role"]
        df = pd.concat([df, scored], axis=1)
        rows.append(df)

    if not rows:
        raise ValueError(f"No labeled task rows were produced from {pcap_dir}")

    dataset = pd.concat(rows, ignore_index=True)
    dataset = _normalize_capture_weights(
        dataset,
        capture_total_weight=task_config.dataset.capture_total_weight,
    )

    output_columns = _task_dataset_output_columns(app_config)
    dataset = dataset.loc[:, [column for column in output_columns if column in dataset.columns]].copy()

    ensure_dir(out_path.parent)
    dataset.to_parquet(out_path, index=False)

    metadata = {
        "pcap_dir": str(pcap_dir),
        "out_path": str(out_path),
        "rows": int(len(dataset)),
        "captures": int(dataset["capture_name"].nunique()),
        "task_counts": dataset[DEFAULT_TASK_LABEL_COLUMN].value_counts().sort_index().to_dict(),
        "flow_role_counts": dataset["flow_role"].value_counts().sort_index().to_dict(),
        "skipped_files": skipped_files,
        "errors": errors,
        "n_dissections": resolved_n_dissections,
        "min_packets": task_config.dataset.min_packets,
    }
    LOGGER.info(
        "Prepared task-labeled SPLT dataset with %s rows from %s captures at %s",
        len(dataset),
        dataset["capture_name"].nunique(),
        out_path,
    )
    return metadata


def build_auxiliary_task_dataset(
    raw_data_path: Path,
    out_path: Path,
    app_config: AppConfig,
    task_config: TaskConfig,
) -> dict[str, Any]:
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Raw parquet not found: {raw_data_path}")

    df_raw = pd.read_parquet(raw_data_path)
    df_clean = _filter_clean_rows(df_raw, min_packets=app_config.dataset.min_packets)
    df_final_model_data, threshold_metadata = _build_df_final_model_data(
        df_clean,
        application_column=app_config.dataset.application_column,
        min_samples_per_application=app_config.dataset.min_samples_per_application,
    )

    df_task = df_final_model_data[
        df_final_model_data[app_config.dataset.target_column].isin(task_config.final_classes)
    ].copy()
    if df_task.empty:
        raise ValueError("No auxiliary rows matched the configured final task classes.")

    app_category_counts = (
        df_task.groupby(
            [app_config.dataset.application_column, app_config.dataset.target_column]
        )
        .size()
        .rename("count")
        .reset_index()
    )
    total_counts = (
        app_category_counts.groupby(app_config.dataset.application_column)["count"]
        .sum()
        .rename("total")
        .reset_index()
    )
    dominant_counts = app_category_counts.sort_values("count", ascending=False).drop_duplicates(
        subset=[app_config.dataset.application_column]
    )
    dominant_counts = dominant_counts.merge(total_counts, on=app_config.dataset.application_column)
    dominant_counts["purity"] = dominant_counts["count"] / dominant_counts["total"]

    keep_apps = dominant_counts[
        dominant_counts["purity"] >= task_config.dataset.auxiliary_purity_threshold
    ][app_config.dataset.application_column].tolist()
    df_task = df_task[df_task[app_config.dataset.application_column].isin(keep_apps)].copy()
    if df_task.empty:
        raise ValueError("Auxiliary task dataset is empty after purity filtering.")

    df_task["capture_name"] = "__aux__" + df_task[app_config.dataset.application_column].astype(str)
    df_task["capture_path"] = "__aux__"
    df_task[DEFAULT_TASK_LABEL_COLUMN] = df_task[app_config.dataset.target_column]
    df_task["flow_relevance_weight"] = 1.0
    df_task["flow_role"] = "auxiliary"
    df_task["capture_normalized_weight"] = 1.0

    output_columns = _task_dataset_output_columns(app_config)
    df_task = df_task.loc[:, [column for column in output_columns if column in df_task.columns]].copy()

    ensure_dir(out_path.parent)
    df_task.to_parquet(out_path, index=False)

    metadata = {
        "raw_data_path": str(raw_data_path),
        "out_path": str(out_path),
        "rows": int(len(df_task)),
        "task_counts": df_task[DEFAULT_TASK_LABEL_COLUMN].value_counts().sort_index().to_dict(),
        "unique_applications": int(df_task["application_name"].nunique()),
        "purity_threshold": task_config.dataset.auxiliary_purity_threshold,
        "threshold_metadata": threshold_metadata,
    }
    LOGGER.info("Prepared auxiliary task SPLT dataset with %s rows at %s", len(df_task), out_path)
    return metadata
