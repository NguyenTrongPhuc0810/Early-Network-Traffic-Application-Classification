"""Dataset preparation commands for SPLT 63-class training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml_pipeline.data_loader import DatasetConfig, load_parquet, prepare_dataframe
from ml_pipeline.evaluate import to_jsonable, write_json


def prepare_splt_dataset(
    *,
    input_path: Path,
    output_path: Path,
    config: DatasetConfig | None = None,
) -> dict[str, Any]:
    """Prepare a raw NFStream parquet into a trainable SPLT parquet dataset."""

    resolved_config = config or DatasetConfig()
    raw_df = load_parquet(input_path)
    prepared_df, metadata = prepare_dataframe(raw_df, resolved_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prepared_df.to_parquet(output_path, index=False)

    metadata.update(
        {
            "input_path": str(input_path),
            "output_path": str(output_path),
        }
    )
    write_json(output_path.with_suffix(".metadata.json"), metadata)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare an SPLT parquet dataset for training.")
    parser.add_argument("--input", required=True, type=Path, help="Raw NFStream or curated SPLT parquet.")
    parser.add_argument("--out", required=True, type=Path, help="Output curated SPLT parquet.")
    parser.add_argument("--target-column", default="application_name")
    parser.add_argument("--min-packets", type=int, default=10)
    parser.add_argument("--min-samples-per-application", type=int, default=1000)
    parser.add_argument("--feature-width", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = prepare_splt_dataset(
        input_path=args.input,
        output_path=args.out,
        config=DatasetConfig(
            target_column=args.target_column,
            min_packets=args.min_packets,
            min_samples_per_application=args.min_samples_per_application,
            feature_width=args.feature_width,
        ),
    )
    print(json.dumps(to_jsonable(metadata), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
