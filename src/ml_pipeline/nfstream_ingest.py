"""NFStream PCAP ingestion for the SPLT traffic-classification pipeline."""

from __future__ import annotations

import argparse
import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from ml_pipeline.data_loader import DEFAULT_SPLT_COLUMNS, require_columns
from ml_pipeline.evaluate import to_jsonable, write_json


DEFAULT_INTERIM_REQUIRED_COLUMNS: tuple[str, ...] = (
    "bidirectional_packets",
    *DEFAULT_SPLT_COLUMNS,
)
DEFAULT_DPI_COLUMNS: tuple[str, ...] = (
    "application_is_guessed",
    "application_confidence",
    "application_name",
    "application_category_name",
)


@dataclass(frozen=True)
class NFStreamConfig:
    """NFStream extraction parameters used for early SPLT features."""

    n_meters: int = 1
    n_dissections: int = 20
    statistical_analysis: bool = False
    splt_analysis: int = 25
    accounting_mode: int = 2
    required_columns: tuple[str, ...] = DEFAULT_INTERIM_REQUIRED_COLUMNS
    optional_dpi_columns: tuple[str, ...] = DEFAULT_DPI_COLUMNS


def _pcap_not_found_error(pcap_path: Path) -> FileNotFoundError:
    suggestions: list[str] = []
    if pcap_path.parent.exists():
        candidates = [path.name for path in pcap_path.parent.iterdir() if path.is_file()]
        same_stem = [name for name in candidates if Path(name).stem.lower() == pcap_path.stem.lower()]
        suggestions.extend(sorted(same_stem)[:10])
        if not suggestions:
            suggestions.extend(difflib.get_close_matches(pcap_path.name, candidates, n=10, cutoff=0.55))
    hint = f" Did you mean: {', '.join(suggestions)}" if suggestions else ""
    return FileNotFoundError(f"PCAP not found: {pcap_path}.{hint}")


def extract_pcap_dataframe(
    pcap_path: Path,
    *,
    config: NFStreamConfig | None = None,
    extra_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Extract SPLT-oriented flow rows from a PCAP/PCAPNG file with NFStream."""

    resolved_config = config or NFStreamConfig()
    if not pcap_path.exists():
        raise _pcap_not_found_error(pcap_path)

    try:
        from nfstream import NFStreamer
    except Exception as exc:  # pragma: no cover - depends on native NFStream install.
        raise RuntimeError(
            "NFStream is not installed or failed to import. Install optional PCAP "
            "dependencies with `python -m pip install -e .[pcap]` and prefer "
            "Python 3.11/3.12 if wheels are unavailable for your Python version."
        ) from exc

    streamer = NFStreamer(
        source=str(pcap_path),
        n_meters=resolved_config.n_meters,
        n_dissections=resolved_config.n_dissections,
        statistical_analysis=resolved_config.statistical_analysis,
        splt_analysis=resolved_config.splt_analysis,
        accounting_mode=resolved_config.accounting_mode,
    )
    df = streamer.to_pandas()
    require_columns(df, resolved_config.required_columns, context="NFStream SPLT output")

    selected_columns = list(resolved_config.required_columns)
    selected_columns.extend(
        column for column in extra_columns if column in df.columns and column not in selected_columns
    )
    selected_columns.extend(
        column
        for column in resolved_config.optional_dpi_columns
        if column in df.columns and column not in selected_columns
    )
    return df.loc[:, selected_columns].copy()


def extract_pcap_to_parquet(
    *,
    pcap_path: Path,
    out_path: Path,
    config: NFStreamConfig | None = None,
    extra_columns: Sequence[str] = (),
) -> dict[str, Any]:
    """Extract a PCAP to a parquet file and return run metadata."""

    resolved_config = config or NFStreamConfig()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = extract_pcap_dataframe(
        pcap_path,
        config=resolved_config,
        extra_columns=extra_columns,
    )
    df.to_parquet(out_path, index=False)
    metadata = {
        "pcap_path": str(pcap_path),
        "flows_path": str(out_path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "n_meters": resolved_config.n_meters,
        "n_dissections": resolved_config.n_dissections,
        "statistical_analysis": resolved_config.statistical_analysis,
        "splt_analysis": resolved_config.splt_analysis,
        "accounting_mode": resolved_config.accounting_mode,
    }
    write_json(out_path.with_suffix(".metadata.json"), metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract NFStream SPLT flows from a PCAP.")
    parser.add_argument("--pcap", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n-meters", type=int, default=1)
    parser.add_argument("--n-dissections", type=int, default=20)
    parser.add_argument("--statistical-analysis", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--splt-analysis", type=int, default=25)
    parser.add_argument("--accounting-mode", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--extra-columns", default="", help="Comma-separated optional NFStream columns.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    extra_columns = tuple(column.strip() for column in args.extra_columns.split(",") if column.strip())
    metadata = extract_pcap_to_parquet(
        pcap_path=args.pcap,
        out_path=args.out,
        config=NFStreamConfig(
            n_meters=args.n_meters,
            n_dissections=args.n_dissections,
            statistical_analysis=args.statistical_analysis,
            splt_analysis=args.splt_analysis,
            accounting_mode=args.accounting_mode,
        ),
        extra_columns=extra_columns,
    )
    print(json.dumps(to_jsonable(metadata), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
