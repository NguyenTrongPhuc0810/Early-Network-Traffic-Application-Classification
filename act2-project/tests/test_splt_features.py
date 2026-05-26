from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from act2_project.pipeline.splt_features import build_feature_matrix, parse_splt_sequence


def test_parse_splt_sequence_supports_string_and_list() -> None:
    assert parse_splt_sequence("[1, 2, -1]") == [1, 2, -1]
    assert parse_splt_sequence([5, 6]) == [5, 6]


def test_build_feature_matrix_pads_short_sequences() -> None:
    df = pd.DataFrame({"splt_ps": [[1, 2, -1], [9]]})

    matrix, width = build_feature_matrix(df, splt_column="splt_ps", width=3)

    assert width == 3
    assert matrix.to_dict("records") == [
        {"ps_1": 1, "ps_2": 2, "ps_3": -1},
        {"ps_1": 9, "ps_2": -1, "ps_3": -1},
    ]


def test_build_feature_matrix_supports_multiple_sequences() -> None:
    df = pd.DataFrame(
        {
            "splt_direction": [[0, 1], [1]],
            "splt_ps": [[10, 20], [30]],
            "splt_piat_ms": [[5, 6], [7]],
        }
    )

    matrix, width = build_feature_matrix(
        df,
        splt_column="splt_ps",
        sequence_columns=("splt_direction", "splt_ps", "splt_piat_ms"),
        width=2,
    )

    assert width == 2
    assert matrix.to_dict("records") == [
        {
            "dir_1": 0,
            "dir_2": 1,
            "ps_1": 10,
            "ps_2": 20,
            "piat_1": 5,
            "piat_2": 6,
        },
        {
            "dir_1": 1,
            "dir_2": -1,
            "ps_1": 30,
            "ps_2": -1,
            "piat_1": 7,
            "piat_2": -1,
        },
    ]
