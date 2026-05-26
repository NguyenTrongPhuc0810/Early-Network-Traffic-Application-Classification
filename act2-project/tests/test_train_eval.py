from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from act2_project.config import load_app_config
from act2_project.pipeline.train import run_train_eval


def _make_raw_training_dataset(path: Path) -> None:
    rows: list[dict[str, object]] = []
    classes = [
        ("WebBrowser", "Web", [10, 12, 14, -1]),
        ("YouTube", "Media", [90, 95, 92, -1]),
        ("Zoom", "Collaborative", [50, 52, 53, -1]),
    ]
    for app_name, category_name, base_sequence in classes:
        for index in range(20):
            rows.append(
                {
                    "application_is_guessed": 0,
                    "application_confidence": 6,
                    "application_name": app_name,
                    "application_category_name": category_name,
                    "bidirectional_packets": 12 + (index % 3),
                    "splt_direction": "[0, 1, 0, 1]",
                    "splt_ps": str([base_sequence[0], base_sequence[1] + (index % 2), base_sequence[2], -1]),
                    "splt_piat_ms": "[1, 1, 1, -1]",
                }
            )

    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_test_config(tmp_path: Path) -> tuple[Path, Path]:
    config_path = tmp_path / "config.yaml"
    class_subset_path = tmp_path / "class_subset.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "train_data_path": str(tmp_path / "unused.parquet"),
                    "min_samples_per_application": 1,
                }
            }
        ),
        encoding="utf-8",
    )
    class_subset_path.write_text(yaml.safe_dump({"category_subset": []}), encoding="utf-8")
    return config_path, class_subset_path


def test_run_train_eval_writes_expected_artifacts(tmp_path: Path) -> None:
    data_path = tmp_path / "raw_training.parquet"
    out_dir = tmp_path / "artifacts"
    _make_raw_training_dataset(data_path)
    config_path, class_subset_path = _write_test_config(tmp_path)

    metadata = run_train_eval(
        data_path=data_path,
        out_dir=out_dir,
        app_config=load_app_config(config_path=config_path, class_subset_path=class_subset_path),
    )

    assert metadata["target_column"] == "application_category_name"
    assert metadata["feature_width"] == 4
    assert metadata["prepare"]["output_rows"] == 60
    assert (out_dir / "model.joblib").exists()
    assert (out_dir / "classification_report.txt").exists()
    assert (out_dir / "classification_report.json").exists()
    assert (out_dir / "confusion_matrix.png").exists()

    saved_metadata = json.loads((out_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert saved_metadata["input_rows"] == 60
    assert saved_metadata["test_rows"] > 0
