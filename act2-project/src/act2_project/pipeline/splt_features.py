from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import numpy as np


def parse_splt_sequence(value: Any) -> list[int]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, list):
            raise TypeError(f"Parsed SPLT value is not a list: {type(parsed)!r}")
        return parsed
    raise TypeError(f"Unsupported SPLT value type: {type(value)!r}")


def load_act2_dataframe(
    data_path: Path,
    *,
    splt_column: str,
    target_column: str | None = None,
    sequence_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    df = pd.read_parquet(data_path)

    columns_to_parse = list(sequence_columns) if sequence_columns else [splt_column]
    for column in columns_to_parse:
        if column not in df.columns:
            raise ValueError(f"Dataset is missing required SPLT column: {column}")
    if target_column is not None and target_column not in df.columns:
        raise ValueError(f"Dataset is missing required target column: {target_column}")

    df = df.copy()
    for column in columns_to_parse:
        df[column] = df[column].apply(parse_splt_sequence)
    return df


def _normalize_column_prefix(column: str) -> str:
    mapping = {
        "splt_direction": "dir",
        "splt_ps": "ps",
        "splt_piat_ms": "piat",
    }
    if column in mapping:
        return mapping[column]
    return column.replace("splt_", "")


def feature_column_names(width: int, *, prefix: str) -> list[str]:
    return [f"{prefix}_{index + 1}" for index in range(width)]


def _normalize_sequence(sequence: list[int], width: int) -> list[int]:
    if len(sequence) >= width:
        return sequence[:width]
    return sequence + ([-1] * (width - len(sequence)))


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    splt_column: str,
    width: int | None = None,
    sequence_columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, int]:
    columns_to_use = list(sequence_columns) if sequence_columns else [splt_column]
    if df.empty:
        if width is None:
            raise ValueError("Cannot infer SPLT width from an empty dataframe.")
        frames = [
            pd.DataFrame(
                columns=feature_column_names(width, prefix=_normalize_column_prefix(column)),
                index=df.index,
            )
            for column in columns_to_use
        ]
        return pd.concat(frames, axis=1), width

    if width is None:
        resolved_width = int(max(df[column].apply(len).max() for column in columns_to_use))
    else:
        resolved_width = width

    frames: list[pd.DataFrame] = []
    for column in columns_to_use:
        rows = [_normalize_sequence(sequence, resolved_width) for sequence in df[column].tolist()]
        frame = pd.DataFrame(
            rows,
            columns=feature_column_names(resolved_width, prefix=_normalize_column_prefix(column)),
            index=df.index,
        )
        frames.append(frame)

    matrix = pd.concat(frames, axis=1)
    return matrix, resolved_width
