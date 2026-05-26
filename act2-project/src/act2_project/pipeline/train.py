from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from act2_project.config import AppConfig
from act2_project.paths import build_train_eval_artifacts
from act2_project.pipeline.act2_prepare import prepare_training_dataframe
from act2_project.pipeline.evaluate import save_classification_reports, save_confusion_matrix
from act2_project.pipeline.splt_features import (
    build_feature_matrix,
    load_act2_dataframe,
    parse_splt_sequence,
)
from act2_project.utils.io import write_json
from act2_project.utils.logging import get_logger

LOGGER = get_logger(__name__)


def build_random_forest(
    *,
    n_estimators: int,
    random_state: int,
    n_jobs: int,
    class_weight: object | None,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight=class_weight,
    )


def _resolve_target_column(app_config: AppConfig, target_column: str | None) -> str:
    return target_column or app_config.dataset.target_column


def _normalize_prepared_dataframe(
    df: pd.DataFrame,
    *,
    sequence_columns: Sequence[str],
) -> pd.DataFrame:
    normalized = df.copy()
    for column in sequence_columns:
        if column in normalized.columns:
            normalized[column] = normalized[column].apply(parse_splt_sequence)
    return normalized


def prepare_training_dataframe_from_path(
    data_path: Path,
    app_config: AppConfig,
    *,
    target_column: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved_target = _resolve_target_column(app_config, target_column)
    df_raw = pd.read_parquet(data_path)

    if set(app_config.dataset.training_filter_columns).issubset(df_raw.columns):
        prepared_df, metadata = prepare_training_dataframe(
            df_raw,
            application_column=app_config.dataset.application_column,
            training_filter_columns=app_config.dataset.training_filter_columns,
            splt_feature_columns=app_config.dataset.splt_feature_columns,
            label_columns=app_config.dataset.label_columns,
            target_column=resolved_target,
            category_subset=app_config.category_subset,
            min_packets=app_config.dataset.min_packets,
            min_samples_per_application=app_config.dataset.min_samples_per_application,
        )
        return _normalize_prepared_dataframe(
            prepared_df,
            sequence_columns=app_config.dataset.splt_feature_columns,
        ), metadata

    available_sequence_columns = [
        column for column in app_config.dataset.splt_feature_columns if column in df_raw.columns
    ]
    df_prepared = load_act2_dataframe(
        data_path,
        splt_column=app_config.dataset.splt_column,
        target_column=resolved_target,
        sequence_columns=available_sequence_columns,
    )
    if app_config.category_subset:
        df_prepared = df_prepared[df_prepared[resolved_target].isin(app_config.category_subset)].copy()

    metadata = {
        "input_rows": int(len(df_raw)),
        "after_cleaning_rows": int(len(df_prepared)),
        "after_min_samples_rows": int(len(df_prepared)),
        "output_rows": int(len(df_prepared)),
        "output_columns": list(df_prepared.columns),
        "present_categories": df_prepared[resolved_target].value_counts().sort_index().to_dict(),
        "missing_categories": [
            category
            for category in app_config.category_subset
            if category not in set(df_prepared[resolved_target].unique().tolist())
        ],
        "category_subset_applied": bool(app_config.category_subset),
        "category_subset": list(app_config.category_subset),
    }
    return df_prepared, metadata


def fit_model_bundle(
    data_path: Path,
    app_config: AppConfig,
    *,
    target_column: str | None = None,
) -> dict[str, Any]:
    resolved_target = _resolve_target_column(app_config, target_column)
    df_prepared, prepare_metadata = prepare_training_dataframe_from_path(
        data_path,
        app_config,
        target_column=resolved_target,
    )
    if df_prepared.empty:
        raise ValueError(f"Training dataset is empty after SPLT preparation: {data_path}")

    sequence_columns = [
        column for column in app_config.dataset.splt_feature_columns if column in df_prepared.columns
    ] or [app_config.dataset.splt_column]
    X, feature_width = build_feature_matrix(
        df_prepared,
        splt_column=app_config.dataset.splt_column,
        sequence_columns=sequence_columns,
    )
    y = df_prepared[resolved_target]

    model = build_random_forest(
        n_estimators=app_config.model.n_estimators,
        random_state=app_config.split.random_state,
        n_jobs=app_config.model.n_jobs,
        class_weight=app_config.model.class_weight,
    )
    model.fit(X, y)

    return {
        "model": model,
        "feature_columns": list(X.columns),
        "feature_width": feature_width,
        "target_column": resolved_target,
        "classes": model.classes_.tolist(),
        "prepare_metadata": prepare_metadata,
    }


def save_model_bundle(model_bundle: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_bundle, out_path)


def run_train_eval(
    data_path: Path,
    out_dir: Path,
    app_config: AppConfig,
    *,
    target_column: str | None = None,
) -> dict[str, Any]:
    artifacts = build_train_eval_artifacts(out_dir, app_config.artifacts)
    resolved_target = _resolve_target_column(app_config, target_column)

    df_prepared, prepare_metadata = prepare_training_dataframe_from_path(
        data_path,
        app_config,
        target_column=resolved_target,
    )
    if df_prepared.empty:
        raise ValueError(f"Prepared SPLT training dataset is empty: {data_path}")

    sequence_columns = [
        column for column in app_config.dataset.splt_feature_columns if column in df_prepared.columns
    ] or [app_config.dataset.splt_column]
    X, feature_width = build_feature_matrix(
        df_prepared,
        splt_column=app_config.dataset.splt_column,
        sequence_columns=sequence_columns,
    )
    y = df_prepared[resolved_target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=app_config.split.test_size,
        random_state=app_config.split.random_state,
        stratify=y,
    )

    model = build_random_forest(
        n_estimators=app_config.model.n_estimators,
        random_state=app_config.split.random_state,
        n_jobs=app_config.model.n_jobs,
        class_weight=app_config.model.class_weight,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    save_classification_reports(
        y_true=y_test,
        y_pred=y_pred,
        text_path=artifacts.report_text_path,
        json_path=artifacts.report_json_path,
    )
    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        out_path=artifacts.confusion_matrix_path,
        title="Confusion Matrix for Random Forest (SPLT Category Model)",
    )

    model_bundle = {
        "model": model,
        "feature_columns": list(X.columns),
        "feature_width": feature_width,
        "target_column": resolved_target,
        "classes": model.classes_.tolist(),
        "category_subset": list(app_config.category_subset),
        "prepare_metadata": prepare_metadata,
    }
    save_model_bundle(model_bundle, artifacts.model_path)

    metadata = {
        "data_path": str(data_path),
        "out_dir": str(artifacts.out_dir),
        "target_column": resolved_target,
        "input_rows": int(len(df_prepared)),
        "feature_width": feature_width,
        "feature_columns": list(X.columns),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "model_path": str(artifacts.model_path),
        "classification_report_text": str(artifacts.report_text_path),
        "classification_report_json": str(artifacts.report_json_path),
        "confusion_matrix_path": str(artifacts.confusion_matrix_path),
        "random_state": app_config.split.random_state,
        "n_estimators": app_config.model.n_estimators,
        "class_weight": app_config.model.class_weight,
        "prepare": prepare_metadata,
        "category_subset": list(app_config.category_subset),
    }
    write_json(artifacts.metadata_path, metadata)

    LOGGER.info("Saved train/eval artifacts to %s", artifacts.out_dir)
    return metadata
