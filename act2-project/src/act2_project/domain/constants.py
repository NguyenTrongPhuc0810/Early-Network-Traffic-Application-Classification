from __future__ import annotations

DEFAULT_TARGET_COLUMN = "application_category_name"
DEFAULT_APPLICATION_COLUMN = "application_name"
DEFAULT_SPLT_COLUMN = "splt_ps"
DEFAULT_TASK_LABEL_COLUMN = "task_label"

DEFAULT_LABEL_COLUMNS = (
    "application_name",
    "application_category_name",
)

DEFAULT_SPLT_FEATURE_COLUMNS = (
    "splt_direction",
    "splt_ps",
    "splt_piat_ms",
)

DEFAULT_TRAINING_FILTER_COLUMNS = (
    "application_is_guessed",
    "application_confidence",
    "application_name",
    "application_category_name",
    "bidirectional_packets",
)

DEFAULT_MIN_SAMPLES_PER_APPLICATION = 1000

DEFAULT_INFERENCE_REQUIRED_COLUMNS = (
    "bidirectional_packets",
    "splt_direction",
    "splt_ps",
    "splt_piat_ms",
)

DEFAULT_INTERIM_REQUIRED_COLUMNS = DEFAULT_INFERENCE_REQUIRED_COLUMNS

DEFAULT_INTERIM_OPTIONAL_DPI_COLUMNS = (
    "application_is_guessed",
    "application_confidence",
    "application_name",
    "application_category_name",
)

DEFAULT_CATEGORY_SUBSET: tuple[str, ...] = ()

DEFAULT_DIAGNOSTIC_LABEL_COLUMNS = (
    "application_name",
    "application_category_name",
)

DEFAULT_FLOWS_FILENAME = "flows.parquet"
DEFAULT_ACT2_DATASET_FILENAME = "pcap_inference_dataset.parquet"
DEFAULT_TRAINING_DATASET_FILENAME = "df_final_model_data_splt.parquet"
