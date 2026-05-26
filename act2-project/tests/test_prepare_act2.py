from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from act2_project.pipeline.act2_prepare import (
    prepare_training_dataframe,
    prepare_pcap_inference_dataframe,
)


def _row(
    app_name: str,
    category_name: str,
    *,
    guessed: int = 0,
    confidence: int = 6,
    packets: int = 12,
) -> dict[str, object]:
    return {
        "application_is_guessed": guessed,
        "application_confidence": confidence,
        "application_name": app_name,
        "application_category_name": category_name,
        "bidirectional_packets": packets,
        "splt_direction": "[0, 1, 0]",
        "splt_ps": "[10, 20, -1]",
        "splt_piat_ms": "[1, 2, -1]",
    }


def test_prepare_training_matches_df_final_model_data_logic() -> None:
    df = pd.DataFrame(
        [
            _row("YouTube", "Media"),
            _row("YouTube", "Media"),
            _row("WebBrowser", "Web"),
            _row("WebBrowser", "Web"),
            _row("OneOff", "Media"),
            _row("BadGuess", "Media", guessed=1),
            _row("Unknown", "Unspecified"),
            _row("TooShort", "Game", packets=9),
            _row("OtherApp", "OtherCategory"),
        ]
    )

    prepared, metadata = prepare_training_dataframe(
        df,
        application_column="application_name",
        training_filter_columns=(
            "application_is_guessed",
            "application_confidence",
            "application_name",
            "application_category_name",
            "bidirectional_packets",
        ),
        splt_feature_columns=("splt_direction", "splt_ps", "splt_piat_ms"),
        label_columns=("application_name", "application_category_name"),
        target_column="application_category_name",
        category_subset=("Media", "Web", "Music"),
        min_packets=10,
        min_samples_per_application=2,
    )

    assert list(prepared.columns) == [
        "splt_direction",
        "splt_ps",
        "splt_piat_ms",
        "application_category_name",
        "application_name",
    ]
    assert prepared["application_category_name"].tolist() == ["Media", "Media", "Web", "Web"]
    assert metadata["after_min_samples_rows"] == 4
    assert metadata["output_rows"] == 4
    assert metadata["missing_categories"] == ["Music"]


def test_prepare_pcap_inference_keeps_all_apps_above_min_packets() -> None:
    df = pd.DataFrame(
        [
            _row("YouTube", "Media"),
            _row("Zoom", "Collaborative"),
            _row("PUBG", "Game", packets=9),
        ]
    )

    prepared, metadata = prepare_pcap_inference_dataframe(
        df,
        inference_required_columns=(
            "bidirectional_packets",
            "splt_direction",
            "splt_ps",
            "splt_piat_ms",
        ),
        splt_feature_columns=("splt_direction", "splt_ps", "splt_piat_ms"),
        label_columns=("application_name", "application_category_name"),
        min_packets=10,
    )

    assert prepared["application_name"].tolist() == ["YouTube", "Zoom"]
    assert metadata["after_min_packets_rows"] == 2
