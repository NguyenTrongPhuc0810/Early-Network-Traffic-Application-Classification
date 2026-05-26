from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from act2_project.pipeline.cesnet_convert import (
    infer_application_category,
    map_cesnet_label,
    normalize_ppi,
)


def test_normalize_ppi_keeps_first_width_and_converts_padding() -> None:
    ppi = "[[0, 10, 20, 0], [1, -1, 0, 1], [0, 1200, 55, 0]]"

    direction, size, ipt = normalize_ppi(ppi, width=3)

    assert direction == [0, 1, -1]
    assert size == [-1, 1200, 55]
    assert ipt == [-1, 10, 20]


def test_normalize_ppi_pads_short_sequences() -> None:
    ppi = [[5], [1], [100]]

    direction, size, ipt = normalize_ppi(ppi, width=3)

    assert direction == [0, -1, -1]
    assert size == [100, -1, -1]
    assert ipt == [5, -1, -1]


def test_map_cesnet_label_uses_dataset_protocol() -> None:
    assert map_cesnet_label("youtube", dataset="quic") == "QUIC.YouTube"
    assert map_cesnet_label("youtube", dataset="tls") == "TLS.YouTube"
    assert map_cesnet_label("steam", dataset="tls") == "TLS.Steam"


def test_infer_application_category_for_mapped_labels() -> None:
    assert infer_application_category("QUIC.YouTube") == "Media"
    assert infer_application_category("TLS.Steam") == "Game"
