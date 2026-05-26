from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from act2_project.domain.constants import (
    DEFAULT_APPLICATION_COLUMN,
    DEFAULT_CATEGORY_SUBSET,
    DEFAULT_INFERENCE_REQUIRED_COLUMNS,
    DEFAULT_INTERIM_OPTIONAL_DPI_COLUMNS,
    DEFAULT_INTERIM_REQUIRED_COLUMNS,
    DEFAULT_LABEL_COLUMNS,
    DEFAULT_MIN_SAMPLES_PER_APPLICATION,
    DEFAULT_SPLT_COLUMN,
    DEFAULT_SPLT_FEATURE_COLUMNS,
    DEFAULT_TARGET_COLUMN,
    DEFAULT_TRAINING_FILTER_COLUMNS,
)
from act2_project.paths import project_root, resolve_project_path
from act2_project.utils.io import read_yaml


@dataclass(frozen=True)
class PathsConfig:
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    artifacts_dir: Path


@dataclass(frozen=True)
class DatasetConfig:
    train_data_path: Path
    application_column: str
    target_column: str
    splt_column: str
    splt_feature_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    training_filter_columns: tuple[str, ...]
    inference_required_columns: tuple[str, ...]
    interim_required_columns: tuple[str, ...]
    interim_optional_dpi_columns: tuple[str, ...]
    min_packets: int
    min_samples_per_application: int


@dataclass(frozen=True)
class SplitConfig:
    test_size: float
    random_state: int


@dataclass(frozen=True)
class ModelConfig:
    kind: str
    n_estimators: int
    n_jobs: int
    class_weight: Any | None


@dataclass(frozen=True)
class NFStreamConfig:
    n_meters: int
    n_dissections: int
    statistical_analysis: bool
    splt_analysis: int
    accounting_mode: int


@dataclass(frozen=True)
class ArtifactConfig:
    model_file: str
    classification_report_text: str
    classification_report_json: str
    confusion_matrix_png: str
    predictions_file: str
    prediction_summary_text: str
    run_metadata_file: str


@dataclass(frozen=True)
class AppConfig:
    project_name: str
    project_root: Path
    paths: PathsConfig
    dataset: DatasetConfig
    split: SplitConfig
    model: ModelConfig
    nfstream: NFStreamConfig
    artifacts: ArtifactConfig
    category_subset: tuple[str, ...]


def load_app_config(
    config_path: Path | None = None,
    class_subset_path: Path | None = None,
) -> AppConfig:
    root = project_root()
    default_config_path = resolve_project_path(config_path or "configs/default.yaml", root)
    class_config_path = resolve_project_path(class_subset_path or "configs/class_subset.yaml", root)

    raw_config = read_yaml(default_config_path)
    class_config = read_yaml(class_config_path)

    project_cfg = raw_config.get("project", {})
    paths_cfg = raw_config.get("paths", {})
    dataset_cfg = raw_config.get("dataset", {})
    split_cfg = raw_config.get("split", {})
    model_cfg = raw_config.get("model", {})
    nfstream_cfg = raw_config.get("nfstream", {})
    artifacts_cfg = raw_config.get("artifacts", {})
    raw_category_subset = class_config.get("category_subset")
    if raw_category_subset is None:
        raw_category_subset = class_config.get("core_category_subset", DEFAULT_CATEGORY_SUBSET)

    return AppConfig(
        project_name=str(project_cfg.get("name", "act2-project")),
        project_root=root,
        paths=PathsConfig(
            raw_dir=resolve_project_path(paths_cfg.get("raw_dir", "data/raw"), root),
            interim_dir=resolve_project_path(paths_cfg.get("interim_dir", "data/interim"), root),
            processed_dir=resolve_project_path(paths_cfg.get("processed_dir", "data/processed"), root),
            artifacts_dir=resolve_project_path(paths_cfg.get("artifacts_dir", "data/artifacts"), root),
        ),
        dataset=DatasetConfig(
            train_data_path=resolve_project_path(
                dataset_cfg.get(
                    "train_data_path",
                    "../02-app-classification/data/data.parquet",
                ),
                root,
            ),
            application_column=str(
                dataset_cfg.get("application_column", DEFAULT_APPLICATION_COLUMN)
            ),
            target_column=str(dataset_cfg.get("target_column", DEFAULT_TARGET_COLUMN)),
            splt_column=str(dataset_cfg.get("splt_column", DEFAULT_SPLT_COLUMN)),
            splt_feature_columns=tuple(
                dataset_cfg.get("splt_feature_columns", DEFAULT_SPLT_FEATURE_COLUMNS)
            ),
            label_columns=tuple(dataset_cfg.get("label_columns", DEFAULT_LABEL_COLUMNS)),
            training_filter_columns=tuple(
                dataset_cfg.get("training_filter_columns", DEFAULT_TRAINING_FILTER_COLUMNS)
            ),
            inference_required_columns=tuple(
                dataset_cfg.get("inference_required_columns", DEFAULT_INFERENCE_REQUIRED_COLUMNS)
            ),
            interim_required_columns=tuple(
                dataset_cfg.get("interim_required_columns", DEFAULT_INTERIM_REQUIRED_COLUMNS)
            ),
            interim_optional_dpi_columns=tuple(
                dataset_cfg.get(
                    "interim_optional_dpi_columns",
                    DEFAULT_INTERIM_OPTIONAL_DPI_COLUMNS,
                )
            ),
            min_packets=int(dataset_cfg.get("min_packets", 10)),
            min_samples_per_application=int(
                dataset_cfg.get(
                    "min_samples_per_application",
                    DEFAULT_MIN_SAMPLES_PER_APPLICATION,
                )
            ),
        ),
        split=SplitConfig(
            test_size=float(split_cfg.get("test_size", 0.2)),
            random_state=int(split_cfg.get("random_state", 42)),
        ),
        model=ModelConfig(
            kind=str(model_cfg.get("kind", "random_forest")),
            n_estimators=int(model_cfg.get("n_estimators", 100)),
            n_jobs=int(model_cfg.get("n_jobs", -1)),
            class_weight=model_cfg.get("class_weight"),
        ),
        nfstream=NFStreamConfig(
            n_meters=int(nfstream_cfg.get("n_meters", 1)),
            n_dissections=int(nfstream_cfg.get("n_dissections", 20)),
            statistical_analysis=bool(nfstream_cfg.get("statistical_analysis", False)),
            splt_analysis=int(nfstream_cfg.get("splt_analysis", 25)),
            accounting_mode=int(nfstream_cfg.get("accounting_mode", 2)),
        ),
        artifacts=ArtifactConfig(
            model_file=str(artifacts_cfg.get("model_file", "model.joblib")),
            classification_report_text=str(
                artifacts_cfg.get("classification_report_text", "classification_report.txt")
            ),
            classification_report_json=str(
                artifacts_cfg.get("classification_report_json", "classification_report.json")
            ),
            confusion_matrix_png=str(
                artifacts_cfg.get("confusion_matrix_png", "confusion_matrix.png")
            ),
            predictions_file=str(artifacts_cfg.get("predictions_file", "predictions.parquet")),
            prediction_summary_text=str(
                artifacts_cfg.get("prediction_summary_text", "prediction_summary.txt")
            ),
            run_metadata_file=str(artifacts_cfg.get("run_metadata_file", "run_metadata.json")),
        ),
        category_subset=tuple(raw_category_subset or ()),
    )
