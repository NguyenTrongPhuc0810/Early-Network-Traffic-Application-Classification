from __future__ import annotations

from pathlib import Path
from typing import Any

from act2_project.domain.constants import DEFAULT_ACT2_DATASET_FILENAME, DEFAULT_FLOWS_FILENAME
from act2_project.domain.schemas import PcapPredictArtifacts, TrainEvalArtifacts
from act2_project.utils.io import ensure_dir


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(pathlike: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(pathlike)
    if path.is_absolute():
        return path
    root = base_dir or project_root()
    return (root / path).resolve()


def build_train_eval_artifacts(out_dir: Path, artifact_config: Any) -> TrainEvalArtifacts:
    out_dir = ensure_dir(out_dir)
    return TrainEvalArtifacts(
        out_dir=out_dir,
        model_path=out_dir / artifact_config.model_file,
        report_text_path=out_dir / artifact_config.classification_report_text,
        report_json_path=out_dir / artifact_config.classification_report_json,
        confusion_matrix_path=out_dir / artifact_config.confusion_matrix_png,
        metadata_path=out_dir / artifact_config.run_metadata_file,
    )


def build_pcap_predict_artifacts(
    base_out_dir: Path,
    artifact_config: Any,
) -> PcapPredictArtifacts:
    base_out_dir = ensure_dir(base_out_dir)
    interim_dir = ensure_dir(base_out_dir / "interim")
    processed_dir = ensure_dir(base_out_dir / "processed")
    artifacts_dir = ensure_dir(base_out_dir / "artifacts")

    return PcapPredictArtifacts(
        base_out_dir=base_out_dir,
        interim_dir=interim_dir,
        processed_dir=processed_dir,
        artifacts_dir=artifacts_dir,
        flows_parquet_path=interim_dir / DEFAULT_FLOWS_FILENAME,
        act2_parquet_path=processed_dir / DEFAULT_ACT2_DATASET_FILENAME,
        model_path=artifacts_dir / artifact_config.model_file,
        report_text_path=artifacts_dir / artifact_config.classification_report_text,
        report_json_path=artifacts_dir / artifact_config.classification_report_json,
        confusion_matrix_path=artifacts_dir / artifact_config.confusion_matrix_png,
        predictions_path=artifacts_dir / artifact_config.predictions_file,
        summary_path=artifacts_dir / artifact_config.prediction_summary_text,
        metadata_path=artifacts_dir / artifact_config.run_metadata_file,
    )
