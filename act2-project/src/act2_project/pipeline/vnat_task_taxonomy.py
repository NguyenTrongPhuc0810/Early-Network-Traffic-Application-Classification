from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from act2_project.paths import project_root, resolve_project_path
from act2_project.utils.io import ensure_dir, read_yaml, write_json


@dataclass(frozen=True)
class VnatTaxonomyConfig:
    target_column: str
    source_dataset: str
    task_patterns: dict[str, tuple[str, ...]]


def load_vnat_taxonomy_config(config_path: Path | None = None) -> VnatTaxonomyConfig:
    root = project_root()
    resolved_path = resolve_project_path(config_path or "configs/vnat_task_taxonomy.yaml", root)
    raw = read_yaml(resolved_path)
    taxonomy = raw.get("taxonomy", {})
    return VnatTaxonomyConfig(
        target_column=str(taxonomy.get("target_column", "task_label")),
        source_dataset=str(taxonomy.get("source_dataset", "vnat")),
        task_patterns={
            str(label): tuple(str(value) for value in values or ())
            for label, values in (taxonomy.get("task_patterns", {}) or {}).items()
        },
    )


def _infer_task_label_from_filename(file_name: str, config: VnatTaxonomyConfig) -> str | None:
    normalized = file_name.lower()
    for task_label, patterns in config.task_patterns.items():
        if any(pattern.lower() in normalized for pattern in patterns):
            return task_label
    return None


def _safe_literal_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Unsupported sequence type: {type(value)!r}")


def _serialize_sequence(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return str(value)
    if isinstance(value, tuple):
        return str(list(value))
    if hasattr(value, "tolist"):
        return str(value.tolist())
    raise TypeError(f"Unsupported sequence type for serialization: {type(value)!r}")


def _to_splt_lists(
    timestamps: list[float],
    sizes: list[int],
    directions: list[int],
    *,
    max_packets: int = 25,
) -> tuple[list[int], list[int], list[float]]:
    trimmed_timestamps = timestamps[:max_packets]
    trimmed_sizes = sizes[:max_packets]
    trimmed_directions = directions[:max_packets]

    splt_piat_ms: list[float] = []
    previous_ts: float | None = None
    for ts in trimmed_timestamps:
        if previous_ts is None:
            splt_piat_ms.append(0.0)
        else:
            splt_piat_ms.append(max((ts - previous_ts) * 1000.0, 0.0))
        previous_ts = ts

    return trimmed_directions, trimmed_sizes, splt_piat_ms


def build_vnat_task_taxonomy_dataset(
    data_path: Path,
    out_path: Path,
    *,
    taxonomy_config_path: Path | None = None,
    max_packets: int = 25,
) -> dict[str, Any]:
    """Chuyển VNAT dataframe release sang schema SPLT-like cho act2-project.

    Chỉ giữ những flow map được chắc chắn vào taxonomy nhiệm vụ hiện tại.
    """

    if not data_path.exists():
        raise FileNotFoundError(f"VNAT H5 not found: {data_path}")

    config = load_vnat_taxonomy_config(taxonomy_config_path)
    df = pd.read_hdf(data_path, key="data")
    required_columns = {"timestamps", "sizes", "directions", "file_names"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"VNAT H5 is missing required columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        file_name = str(row["file_names"])
        task_label = _infer_task_label_from_filename(file_name, config)
        if task_label is None:
            continue

        timestamps = [float(x) for x in _safe_literal_list(row["timestamps"])]
        sizes = [int(x) for x in _safe_literal_list(row["sizes"])]
        directions = [int(x) for x in _safe_literal_list(row["directions"])]
        if not timestamps or not (len(timestamps) == len(sizes) == len(directions)):
            continue

        splt_direction, splt_ps, splt_piat_ms = _to_splt_lists(
            timestamps,
            sizes,
            directions,
            max_packets=max_packets,
        )

        rows.append(
            {
                "splt_direction": splt_direction,
                "splt_ps": splt_ps,
                "splt_piat_ms": splt_piat_ms,
                config.target_column: task_label,
                "application_name": Path(file_name).stem,
                "application_category_name": task_label,
                "source_dataset": config.source_dataset,
                "capture_name": Path(file_name).stem,
                "original_file_name": file_name,
            }
        )

    prepared = pd.DataFrame(rows)
    ensure_dir(out_path.parent)
    prepared.to_parquet(out_path, index=False)

    metadata = {
        "input_path": str(data_path),
        "output_path": str(out_path),
        "output_rows": int(len(prepared)),
        "target_column": config.target_column,
        "source_dataset": config.source_dataset,
        "task_counts": prepared[config.target_column].value_counts().sort_index().to_dict()
        if not prepared.empty
        else {},
        "task_patterns": {label: list(patterns) for label, patterns in config.task_patterns.items()},
        "max_packets": max_packets,
    }
    write_json(out_path.with_suffix(".metadata.json"), metadata)
    return metadata


def build_combined_task_dataset(
    act2_path: Path,
    vnat_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    """Hợp nhất Act2 task dataset và VNAT task dataset theo schema chung."""

    if not act2_path.exists():
        raise FileNotFoundError(f"Act2 task dataset not found: {act2_path}")
    if not vnat_path.exists():
        raise FileNotFoundError(f"VNAT task dataset not found: {vnat_path}")

    act2_df = pd.read_parquet(act2_path)
    vnat_df = pd.read_parquet(vnat_path)

    common_columns = sorted(set(act2_df.columns).intersection(vnat_df.columns))
    if "task_label" not in common_columns:
        raise ValueError("Expected task_label in both input datasets.")

    for frame in (act2_df, vnat_df):
        for column in ("splt_direction", "splt_ps", "splt_piat_ms"):
            if column in frame.columns:
                frame[column] = frame[column].apply(_serialize_sequence)

    combined = pd.concat(
        [act2_df.loc[:, common_columns], vnat_df.loc[:, common_columns]],
        ignore_index=True,
    )
    ensure_dir(out_path.parent)
    combined.to_parquet(out_path, index=False)

    metadata = {
        "act2_path": str(act2_path),
        "vnat_path": str(vnat_path),
        "output_path": str(out_path),
        "rows": int(len(combined)),
        "columns": common_columns,
        "task_counts": combined["task_label"].value_counts().sort_index().to_dict(),
        "source_counts": combined["source_dataset"].value_counts().sort_index().to_dict(),
        "task_source_counts": {
            f"{source_dataset}::{task_label}": int(count)
            for (source_dataset, task_label), count in (
                combined.groupby(["source_dataset", "task_label"]).size().sort_index().to_dict().items()
            )
        },
    }
    write_json(out_path.with_suffix(".metadata.json"), metadata)
    return metadata
