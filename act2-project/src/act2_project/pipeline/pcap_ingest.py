from __future__ import annotations

import difflib
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from act2_project.config import AppConfig
from act2_project.pipeline.act2_prepare import require_columns
from act2_project.utils.io import ensure_dir
from act2_project.utils.logging import get_logger

LOGGER = get_logger(__name__)


def extract_pcap_dataframe(
    pcap_path: Path,
    *,
    n_meters: int,
    n_dissections: int,
    statistical_analysis: bool,
    splt_analysis: int,
    accounting_mode: int,
    interim_required_columns: Sequence[str],
    interim_optional_dpi_columns: Sequence[str],
    extra_columns: Sequence[str] = (),
) -> pd.DataFrame:
    if not pcap_path.exists():
        suggestions: list[str] = []
        parent = pcap_path.parent
        if parent.exists():
            try:
                candidates = [p.name for p in parent.iterdir() if p.is_file()]
            except OSError:
                candidates = []

            if candidates:
                # Common case: wrong extension but correct stem.
                same_stem = [name for name in candidates if Path(name).stem.lower() == pcap_path.stem.lower()]
                if same_stem:
                    suggestions.extend(sorted(same_stem)[:10])
                else:
                    suggestions.extend(
                        difflib.get_close_matches(pcap_path.name, candidates, n=10, cutoff=0.55)
                    )

        hint = f" Did you mean: {', '.join(suggestions)}" if suggestions else ""
        raise FileNotFoundError(f"PCAP not found: {pcap_path}.{hint}")

    try:
        from nfstream import NFStreamer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to import nfstream. Install the optional 'pcap' dependencies "
            "or run this step in a Python 3.11/3.12 environment with NFStream."
        ) from exc

    streamer = NFStreamer(
        source=str(pcap_path),
        n_meters=n_meters,
        n_dissections=n_dissections,
        statistical_analysis=statistical_analysis,
        splt_analysis=splt_analysis,
        accounting_mode=accounting_mode,
    )

    df = streamer.to_pandas()
    require_columns(df, interim_required_columns, context="SPLT interim flow selection")

    selected_columns = list(interim_required_columns)
    selected_columns.extend(
        column for column in extra_columns if column in df.columns and column not in selected_columns
    )
    selected_columns.extend(
        column
        for column in interim_optional_dpi_columns
        if column in df.columns and column not in selected_columns
    )
    return df.loc[:, selected_columns].copy()


def extract_pcap_to_flows(
    pcap_path: Path,
    out_path: Path,
    *,
    n_meters: int,
    n_dissections: int,
    statistical_analysis: bool,
    splt_analysis: int,
    accounting_mode: int,
    interim_required_columns: Sequence[str],
    interim_optional_dpi_columns: Sequence[str],
    extra_columns: Sequence[str] = (),
) -> dict[str, Any]:
    ensure_dir(out_path.parent)
    df = extract_pcap_dataframe(
        pcap_path=pcap_path,
        n_meters=n_meters,
        n_dissections=n_dissections,
        statistical_analysis=statistical_analysis,
        splt_analysis=splt_analysis,
        accounting_mode=accounting_mode,
        interim_required_columns=interim_required_columns,
        interim_optional_dpi_columns=interim_optional_dpi_columns,
        extra_columns=extra_columns,
    )
    df.to_parquet(out_path, index=False)

    LOGGER.info("Extracted %s flows to %s", len(df), out_path)

    return {
        "pcap_path": str(pcap_path),
        "flows_path": str(out_path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "n_dissections": n_dissections,
    }


def run_pcap_ingest(
    pcap_path: Path,
    out_path: Path,
    app_config: AppConfig,
    *,
    n_dissections_override: int | None = None,
    extra_columns: Sequence[str] = (),
) -> dict[str, Any]:
    return extract_pcap_to_flows(
        pcap_path=pcap_path,
        out_path=out_path,
        n_meters=app_config.nfstream.n_meters,
        n_dissections=(
            app_config.nfstream.n_dissections if n_dissections_override is None else n_dissections_override
        ),
        statistical_analysis=app_config.nfstream.statistical_analysis,
        splt_analysis=app_config.nfstream.splt_analysis,
        accounting_mode=app_config.nfstream.accounting_mode,
        interim_required_columns=app_config.dataset.interim_required_columns,
        interim_optional_dpi_columns=app_config.dataset.interim_optional_dpi_columns,
        extra_columns=extra_columns,
    )
