from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainEvalArtifacts:
    out_dir: Path
    model_path: Path
    report_text_path: Path
    report_json_path: Path
    confusion_matrix_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class PcapPredictArtifacts:
    base_out_dir: Path
    interim_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    flows_parquet_path: Path
    act2_parquet_path: Path
    model_path: Path
    report_text_path: Path
    report_json_path: Path
    confusion_matrix_path: Path
    predictions_path: Path
    summary_path: Path
    metadata_path: Path
