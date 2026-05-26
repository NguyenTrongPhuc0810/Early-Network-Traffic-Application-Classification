from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import tables as tb

from act2_project.pipeline.cesnet_convert import infer_application_category, map_cesnet_label


def _to_model_direction(value: Any) -> int:
    numeric = int(value)
    if numeric > 0:
        return 0
    if numeric < 0:
        return 1
    return -1


def _normalize_h5_ppi(ppi, ppi_len: int, *, width: int) -> tuple[list[int], list[int], list[int]]:
    usable = min(int(ppi_len), width)
    ipt = [int(round(float(x))) for x in ppi[0, :usable]]
    direction = [_to_model_direction(x) for x in ppi[1, :usable]]
    size = [int(round(float(x))) for x in ppi[2, :usable]]
    if usable < width:
        pad = width - usable
        ipt.extend([-1] * pad)
        direction.extend([-1] * pad)
        size.extend([-1] * pad)
    return direction, size, ipt


def _enum_inverse(enum) -> dict[int, str]:
    return {int(value): str(name) for name, value in enum._names.items()}


def convert_datazoo_h5(
    *,
    input_path: Path,
    out_path: Path,
    dataset: str,
    mapping: dict[str, str] | None,
    width: int,
    max_per_class: int,
    chunk_size: int,
    random_seed: int,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(random_seed)
    reservoir: dict[str, list[dict[str, Any]]] = {}
    seen_per_class: Counter[str] = Counter()
    raw_counts: Counter[str] = Counter()
    mapped_counts: Counter[str] = Counter()
    unmapped_counts: Counter[str] = Counter()
    rows_seen = 0

    with tb.open_file(input_path, mode="r") as h5:
        table_paths = [node._v_pathname for node in h5.walk_nodes("/flows", classname="Table")]
        for table_index, table_path in enumerate(table_paths, start=1):
            table = h5.get_node(table_path)
            app_inverse = _enum_inverse(table.get_enum("APP"))
            print(f"[{table_index}/{len(table_paths)}] {table_path}: {table.nrows} rows", flush=True)
            for start in range(0, table.nrows, chunk_size):
                stop = min(start + chunk_size, table.nrows)
                apps = table.read(start=start, stop=stop, field="APP")
                ppis = table.read(start=start, stop=stop, field="PPI")
                ppi_lens = table.read(start=start, stop=stop, field="PPI_LEN")

                for app_id, ppi, ppi_len in zip(apps, ppis, ppi_lens):
                    rows_seen += 1
                    raw_label = app_inverse[int(app_id)]
                    raw_counts[raw_label] += 1
                    application_name = (
                        mapping.get(raw_label)
                        if mapping is not None
                        else map_cesnet_label(raw_label, dataset=dataset)
                    )
                    if application_name is None:
                        unmapped_counts[raw_label] += 1
                        continue

                    splt_direction, splt_ps, splt_piat_ms = _normalize_h5_ppi(
                        ppi,
                        int(ppi_len),
                        width=width,
                    )
                    mapped_counts[application_name] += 1
                    row = {
                        "splt_direction": str(splt_direction),
                        "splt_ps": str(splt_ps),
                        "splt_piat_ms": str(splt_piat_ms),
                        "application_name": application_name,
                        "application_category_name": infer_application_category(application_name),
                        "cesnet_app": raw_label,
                    }
                    seen_per_class[application_name] += 1
                    class_rows = reservoir.setdefault(application_name, [])
                    if len(class_rows) < max_per_class:
                        class_rows.append(row)
                    else:
                        replacement_index = rng.randrange(seen_per_class[application_name])
                        if replacement_index < max_per_class:
                            class_rows[replacement_index] = row

    sampled_rows = [row for label in sorted(reservoir) for row in reservoir[label]]
    df = pd.DataFrame(sampled_rows)
    df.to_parquet(out_path, index=False)

    metadata = {
        "input_path": str(input_path),
        "out_path": str(out_path),
        "dataset": dataset,
        "mapping": mapping,
        "width": width,
        "max_per_class": max_per_class,
        "chunk_size": chunk_size,
        "random_seed": random_seed,
        "rows_seen": rows_seen,
        "rows_written": int(len(df)),
        "mapped_counts_total": dict(mapped_counts.most_common()),
        "sampled_counts": df["application_name"].value_counts().sort_index().astype(int).to_dict()
        if not df.empty
        else {},
        "top_raw_labels": dict(raw_counts.most_common(50)),
        "top_unmapped_labels": dict(unmapped_counts.most_common(50)),
    }
    out_path.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(df)} rows to {out_path}")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert CESNET DataZoo H5 to Act2 SPLT parquet.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", choices=("quic", "tls"), required=True)
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=None,
        help="Optional raw APP label to model class mapping JSON. If set, only these labels are mapped.",
    )
    parser.add_argument("--width", type=int, default=25)
    parser.add_argument("--max-per-class", type=int, default=5000)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()
    mapping = None
    if args.mapping_json is not None:
        mapping = json.loads(args.mapping_json.read_text(encoding="utf-8"))
    convert_datazoo_h5(
        input_path=args.input,
        out_path=args.out,
        dataset=args.dataset,
        mapping=mapping,
        width=args.width,
        max_per_class=args.max_per_class,
        chunk_size=args.chunk_size,
        random_seed=args.random_seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
