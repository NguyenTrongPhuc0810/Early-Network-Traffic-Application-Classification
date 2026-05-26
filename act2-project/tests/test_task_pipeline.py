from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from act2_project.pipeline.task_dataset import infer_task_label, score_task_flow
from act2_project.pipeline.task_model import aggregate_capture_votes
from act2_project.task_config import load_task_config


def test_infer_task_label_uses_expected_priority() -> None:
    task_config = load_task_config()

    assert infer_task_label("ytb_iphone.pcap", task_config) == "Media"
    assert infer_task_label("zoom_galaxy.pcap", task_config) == "Collaborative"
    assert infer_task_label("pubg_iphone.pcap", task_config) == "Game"
    assert infer_task_label("download_EA.pcapng", task_config) == "Download"
    assert infer_task_label("lienquan_galaxy_download.pcap", task_config) == "Download"


def test_score_task_flow_prefers_foreground_over_background() -> None:
    task_config = load_task_config()

    weight, role = score_task_flow(
        task_label="Media",
        application_name="QUIC.YouTube",
        application_category_name="Media",
        task_config=task_config,
    )
    assert (weight, role) == (1.0, "foreground")

    weight, role = score_task_flow(
        task_label="Media",
        application_name="DNS.YouTube",
        application_category_name="Network",
        task_config=task_config,
    )
    assert (weight, role) == (0.0, "background")

    weight, role = score_task_flow(
        task_label="Game",
        application_name="TLS",
        application_category_name="Web",
        task_config=task_config,
    )
    assert (weight, role) == (0.25, "support")


def test_aggregate_capture_votes_prefers_high_confidence_relevant_flows() -> None:
    task_config = load_task_config()
    prediction_df = pd.DataFrame(
        [
            {
                "capture_name": "cap1",
                "capture_path": "cap1.pcap",
                "task_label": "Media",
                "bidirectional_packets": 30,
                "prob_Media": 0.91,
                "prob_Game": 0.05,
                "prob_Collaborative": 0.02,
                "prob_Download": 0.02,
            },
            {
                "capture_name": "cap1",
                "capture_path": "cap1.pcap",
                "task_label": "Media",
                "bidirectional_packets": 40,
                "prob_Media": 0.82,
                "prob_Game": 0.08,
                "prob_Collaborative": 0.05,
                "prob_Download": 0.05,
            },
            {
                "capture_name": "cap1",
                "capture_path": "cap1.pcap",
                "task_label": "Media",
                "bidirectional_packets": 12,
                "prob_Media": 0.20,
                "prob_Game": 0.50,
                "prob_Collaborative": 0.15,
                "prob_Download": 0.15,
            },
        ]
    )

    capture_df = aggregate_capture_votes(
        prediction_df,
        classes=["Collaborative", "Download", "Game", "Media"],
        task_config=task_config,
    )

    assert capture_df.iloc[0]["predicted_task_label"] == "Media"
