from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from act2_project.paths import project_root, resolve_project_path
from act2_project.utils.io import ensure_dir, read_yaml, write_json


@dataclass(frozen=True)
class TaxonomyConfig:
    target_column: str
    source_dataset: str
    task_labels: dict[str, tuple[str, ...]]


def load_taxonomy_config(config_path: Path | None = None) -> TaxonomyConfig:
    root = project_root()
    resolved_path = resolve_project_path(config_path or "configs/task_taxonomy.yaml", root)
    raw = read_yaml(resolved_path)
    taxonomy = raw.get("taxonomy", {})
    return TaxonomyConfig(
        target_column=str(taxonomy.get("target_column", "task_label")),
        source_dataset=str(taxonomy.get("source_dataset", "act2")),
        task_labels={
            str(label): tuple(str(value) for value in values or ())
            for label, values in (taxonomy.get("task_labels", {}) or {}).items()
        },
    )


def build_task_taxonomy_dataset(
    data_path: Path,
    out_path: Path,
    *,
    taxonomy_config_path: Path | None = None,
) -> dict[str, Any]:
    """Tạo dataset task-level sạch từ parquet SPLT đã chuẩn bị."""

    if not data_path.exists():
        raise FileNotFoundError(f"Training parquet not found: {data_path}")

    taxonomy = load_taxonomy_config(taxonomy_config_path)
    df = pd.read_parquet(data_path)
    if "application_name" not in df.columns:
        raise ValueError("Expected application_name column in source parquet.")

    app_to_label: dict[str, str] = {}
    for task_label, app_names in taxonomy.task_labels.items():
        for app_name in app_names:
            if app_name in app_to_label:
                raise ValueError(f"Application {app_name!r} is assigned to multiple task labels.")
            app_to_label[app_name] = task_label

    filtered = df[df["application_name"].isin(app_to_label)].copy()
    filtered[taxonomy.target_column] = filtered["application_name"].map(app_to_label)
    filtered["source_dataset"] = taxonomy.source_dataset

    output_columns = [
        "splt_direction",
        "splt_ps",
        "splt_piat_ms",
        taxonomy.target_column,
        "application_name",
        "application_category_name",
        "source_dataset",
    ]
    prepared = filtered.loc[:, [column for column in output_columns if column in filtered.columns]].copy()

    ensure_dir(out_path.parent)
    prepared.to_parquet(out_path, index=False)

    metadata = {
        "input_path": str(data_path),
        "output_path": str(out_path),
        "input_rows": int(len(df)),
        "output_rows": int(len(prepared)),
        "target_column": taxonomy.target_column,
        "source_dataset": taxonomy.source_dataset,
        "task_counts": prepared[taxonomy.target_column].value_counts().sort_index().to_dict(),
        "application_counts": prepared["application_name"].value_counts().sort_index().to_dict(),
        "task_labels": {label: list(apps) for label, apps in taxonomy.task_labels.items()},
    }
    write_json(out_path.with_suffix(".metadata.json"), metadata)
    return metadata

