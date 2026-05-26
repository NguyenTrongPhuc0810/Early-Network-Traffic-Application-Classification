from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from act2_project.cli import main


def _make_raw_training_dataset(path: Path) -> None:
    rows: list[dict[str, object]] = []
    classes = [
        ("WebBrowser", "Web", [10, 12, 14, -1]),
        ("YouTube", "Media", [90, 95, 92, -1]),
        ("Zoom", "Collaborative", [50, 52, 53, -1]),
    ]
    for app_name, category_name, base_sequence in classes:
        for index in range(12):
            rows.append(
                {
                    "application_is_guessed": 0,
                    "application_confidence": 6,
                    "application_name": app_name,
                    "application_category_name": category_name,
                    "bidirectional_packets": 12 + (index % 2),
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


def test_cli_train_eval_runs_end_to_end(tmp_path: Path) -> None:
    data_path = tmp_path / "dataset.parquet"
    out_dir = tmp_path / "train_eval"
    _make_raw_training_dataset(data_path)
    config_path, class_subset_path = _write_test_config(tmp_path)

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--category-config",
            str(class_subset_path),
            "train-eval",
            "--data-path",
            str(data_path),
            "--out-dir",
            str(out_dir),
        ]
    )

    assert exit_code == 0
    assert (out_dir / "model.joblib").exists()


def test_cli_pcap_predict_dispatches(monkeypatch, tmp_path: Path) -> None:
    pcap_path = tmp_path / "capture.pcap"
    train_data_path = tmp_path / "train.parquet"
    out_dir = tmp_path / "predict_run"
    pcap_path.write_bytes(b"pcap")
    train_data_path.write_bytes(b"parquet")
    config_path, class_subset_path = _write_test_config(tmp_path)

    called: dict[str, object] = {}

    def fake_run_pcap_predict(**kwargs):
        called.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr("act2_project.cli.run_pcap_predict", fake_run_pcap_predict)

    exit_code = main(
        [
            "--config",
            str(config_path),
            "--category-config",
            str(class_subset_path),
            "pcap-predict",
            "--pcap",
            str(pcap_path),
            "--train-data",
            str(train_data_path),
            "--out-dir",
            str(out_dir),
            "--no-dpi",
        ]
    )

    assert exit_code == 0
    assert called["pcap_path"] == pcap_path.resolve()
    assert called["train_data_path"] == train_data_path.resolve()
    assert called["out_dir"] == out_dir.resolve()
    assert called["n_dissections_override"] == 0
