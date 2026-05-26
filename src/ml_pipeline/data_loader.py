"""Data loading and SPLT feature engineering for 63-class traffic classification.

The original notebooks and ``act2-project`` scripts used three SPLT sequence
columns from NFStream:

``splt_direction``, ``splt_ps`` and ``splt_piat_ms``.

This module keeps only the reusable production logic: validate parquet inputs,
filter clean DPI-labelled rows when those columns are available, parse sequence
values, and expand fixed-width SPLT vectors into a flat numeric matrix that can
later be exported to C/eBPF.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_SPLT_COLUMNS: tuple[str, ...] = ("splt_direction", "splt_ps", "splt_piat_ms")
DEFAULT_TARGET_COLUMN = "application_name"
DEFAULT_APPLICATION_COLUMN = "application_name"
DEFAULT_CATEGORY_COLUMN = "application_category_name"
DEFAULT_TRAINING_FILTER_COLUMNS: tuple[str, ...] = (
    "application_is_guessed",
    "application_confidence",
    "application_name",
    "application_category_name",
    "bidirectional_packets",
)


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for preparing the SPLT parquet dataset."""

    target_column: str = DEFAULT_TARGET_COLUMN
    splt_columns: tuple[str, ...] = DEFAULT_SPLT_COLUMNS
    application_column: str = DEFAULT_APPLICATION_COLUMN
    label_columns: tuple[str, ...] = (DEFAULT_APPLICATION_COLUMN, DEFAULT_CATEGORY_COLUMN)
    min_packets: int = 10
    min_samples_per_application: int = 1000
    category_subset: tuple[str, ...] = ()
    feature_width: int | None = None
    pad_value: float = -1.0


@dataclass(frozen=True)
class PreparedDataset:
    """Prepared labels, flattened SPLT features and preparation metadata."""

    dataframe: pd.DataFrame
    features: pd.DataFrame
    labels: pd.Series
    feature_width: int
    feature_columns: list[str]
    target_column: str
    metadata: dict[str, Any]


def require_columns(df: pd.DataFrame, columns: Sequence[str], *, context: str) -> None:
    """Raise a clear error when required columns are missing."""

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for {context}: {missing}")


def parse_splt_sequence(value: Any) -> list[float]:
    """Parse one SPLT sequence cell into a list of numbers.

    Parquet files in the workspace store SPLT columns either as real arrays or
    as stringified Python lists. This parser accepts both forms and rejects
    scalar/object values early so the model never trains on malformed rows.
    """

    if isinstance(value, list):
        return [float(item) for item in value]
    if isinstance(value, tuple):
        return [float(item) for item in value]
    if isinstance(value, np.ndarray):
        return [float(item) for item in value.tolist()]
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (list, tuple)):
            return [float(item) for item in parsed]
        raise TypeError(f"Parsed SPLT value is not a sequence: {type(parsed)!r}")
    raise TypeError(f"Unsupported SPLT value type: {type(value)!r}")


def load_parquet(path: Path | str) -> pd.DataFrame:
    """Load a parquet file and validate that it exists."""

    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Parquet dataset not found: {resolved}")
    return pd.read_parquet(resolved)


def parse_splt_columns(df: pd.DataFrame, splt_columns: Sequence[str]) -> pd.DataFrame:
    """Return a copy of ``df`` with SPLT columns parsed into Python lists."""

    require_columns(df, splt_columns, context="SPLT feature parsing")
    parsed = df.copy()
    for column in splt_columns:
        parsed[column] = parsed[column].apply(parse_splt_sequence)
    return parsed


def _dedupe_preserve_order(columns: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for column in columns:
        if column in seen:
            continue
        seen.add(column)
        ordered.append(column)
    return ordered


def _filter_clean_dpi_rows(df: pd.DataFrame, *, min_packets: int) -> pd.DataFrame:
    """Apply NFStream DPI quality filters when raw DPI columns are present."""

    clean = df[df["application_is_guessed"] == 0].copy()
    clean = clean[clean["application_confidence"] == 6].copy()
    clean = clean[clean["application_name"] != "Unknown"].copy()
    clean = clean[clean["bidirectional_packets"] >= min_packets].copy()
    return clean


def _apply_min_samples(
    df: pd.DataFrame,
    *,
    application_column: str,
    min_samples_per_application: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    counts = df[application_column].value_counts()
    labels_to_keep = counts[counts >= min_samples_per_application].index.tolist()
    filtered = df[df[application_column].isin(labels_to_keep)].copy()
    return filtered, {
        "min_samples_per_application": int(min_samples_per_application),
        "labels_retained": int(len(labels_to_keep)),
        "after_min_samples_rows": int(len(filtered)),
    }


def prepare_dataframe(df: pd.DataFrame, config: DatasetConfig) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean a raw or already-curated SPLT dataframe for training/evaluation."""

    require_columns(
        df,
        [*config.splt_columns, config.target_column],
        context="SPLT training dataset",
    )

    if set(DEFAULT_TRAINING_FILTER_COLUMNS).issubset(df.columns):
        work = _filter_clean_dpi_rows(df, min_packets=config.min_packets)
    else:
        work = df.copy()
        if "bidirectional_packets" in work.columns:
            work = work[work["bidirectional_packets"] >= config.min_packets].copy()

    if config.application_column in work.columns:
        work, threshold_metadata = _apply_min_samples(
            work,
            application_column=config.application_column,
            min_samples_per_application=config.min_samples_per_application,
        )
    else:
        threshold_metadata = {
            "min_samples_per_application": int(config.min_samples_per_application),
            "labels_retained": None,
            "after_min_samples_rows": int(len(work)),
        }

    if config.category_subset:
        work = work[work[config.target_column].isin(config.category_subset)].copy()

    output_columns = _dedupe_preserve_order(
        [*config.splt_columns, config.target_column]
        + [column for column in config.label_columns if column in work.columns]
    )
    prepared = work[output_columns].copy()
    prepared = parse_splt_columns(prepared, config.splt_columns)

    label_counts = prepared[config.target_column].value_counts().sort_index().to_dict()
    metadata = {
        "input_rows": int(len(df)),
        "after_cleaning_rows": int(len(work)),
        "output_rows": int(len(prepared)),
        "output_columns": output_columns,
        "target_column": config.target_column,
        "present_labels": {str(key): int(value) for key, value in label_counts.items()},
        "missing_subset_labels": [
            label for label in config.category_subset if label not in label_counts
        ],
        "category_subset_applied": bool(config.category_subset),
        "category_subset": list(config.category_subset),
        **threshold_metadata,
    }
    return prepared, metadata


def _feature_prefix(column: str) -> str:
    mapping = {
        "splt_direction": "dir",
        "splt_ps": "ps",
        "splt_piat_ms": "piat",
    }
    return mapping.get(column, column.replace("splt_", ""))


def feature_column_names(width: int, *, prefix: str) -> list[str]:
    """Create stable feature names such as ``ps_1`` through ``ps_25``."""

    return [f"{prefix}_{index + 1}" for index in range(width)]


def _pad_or_truncate(sequence: Sequence[float], width: int, pad_value: float) -> list[float]:
    if len(sequence) >= width:
        return list(sequence[:width])
    return [*sequence, *([pad_value] * (width - len(sequence)))]


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    splt_columns: Sequence[str] = DEFAULT_SPLT_COLUMNS,
    width: int | None = None,
    pad_value: float = -1.0,
) -> tuple[pd.DataFrame, int]:
    """Flatten SPLT sequence columns into a fixed-width feature matrix."""

    require_columns(df, splt_columns, context="SPLT feature matrix")
    if df.empty and width is None:
        raise ValueError("Cannot infer SPLT width from an empty dataframe.")

    resolved_width = (
        int(width)
        if width is not None
        else int(max(df[column].apply(len).max() for column in splt_columns))
    )

    frames: list[pd.DataFrame] = []
    for column in splt_columns:
        prefix = _feature_prefix(column)
        rows = [
            _pad_or_truncate(sequence, resolved_width, pad_value)
            for sequence in df[column].tolist()
        ]
        frames.append(
            pd.DataFrame(
                rows,
                columns=feature_column_names(resolved_width, prefix=prefix),
                index=df.index,
                dtype="float32",
            )
        )

    return pd.concat(frames, axis=1), resolved_width


def load_prepared_dataset(path: Path | str, config: DatasetConfig | None = None) -> PreparedDataset:
    """Load parquet data and return model-ready SPLT features and labels."""

    resolved_config = config or DatasetConfig()
    raw_df = load_parquet(path)
    prepared_df, metadata = prepare_dataframe(raw_df, resolved_config)
    if prepared_df.empty:
        raise ValueError(f"Prepared SPLT dataset is empty: {path}")

    features, feature_width = build_feature_matrix(
        prepared_df,
        splt_columns=resolved_config.splt_columns,
        width=resolved_config.feature_width,
        pad_value=resolved_config.pad_value,
    )
    labels = prepared_df[resolved_config.target_column].astype("string")

    metadata.update(
        {
            "feature_width": int(feature_width),
            "feature_columns": list(features.columns),
            "feature_count": int(features.shape[1]),
        }
    )
    return PreparedDataset(
        dataframe=prepared_df,
        features=features,
        labels=labels,
        feature_width=feature_width,
        feature_columns=list(features.columns),
        target_column=resolved_config.target_column,
        metadata=metadata,
    )
