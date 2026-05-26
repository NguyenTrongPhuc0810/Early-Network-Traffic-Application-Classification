from __future__ import annotations

import ast
import csv
import json
import random
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_WIDTH = 25

CESNET_TO_APPLICATION_NAME: dict[str, str] = {
    "apple-icloud": "TLS.AppleiCloud",
    "apple-itunes": "TLS.AppleiTunes",
    "apple-privaterelay": "QUIC.iCloudPrivateRelay",
    "discord": "Discord",
    "dns-doh": "QUIC.DoH_DoT",
    "doh": "TLS.DoH_DoT",
    "facebook-messenger": "TLS.FacebookMessenger",
    "google-ads": "TLS.ADS_Analytic_Track",
    "google-play": "TLS.PlayStore",
    "google-services": "TLS.GoogleServices",
    "instagram": "TLS.Instagram",
    "microsoft-onedrive": "TLS.MS_OneDrive",
    "microsoft-update": "TLS.WindowsUpdate",
    "office365": "TLS.Microsoft365",
    "riot-games": "TLS.RiotGames",
    "spotify": "TLS.Spotify",
    "steam": "TLS.Steam",
    "teams": "TLS.Teams",
    "tiktok": "TLS.TikTok",
    "twitch": "TLS.Twitch",
    "whatsapp": "WhatsApp",
    "youtube": "TLS.YouTube",
}

QUIC_TAG_OVERRIDES: dict[str, str] = {
    "apple-privaterelay": "QUIC.iCloudPrivateRelay",
    "cloudflare-cdnjs": "QUIC.Cloudflare",
    "discord": "QUIC.Discord",
    "dns-doh": "QUIC.DoH_DoT",
    "facebook-connect": "QUIC.Facebook",
    "facebook-gamesgraph": "QUIC.Facebook",
    "facebook-graph": "QUIC.Facebook",
    "facebook-media": "QUIC.Facebook",
    "facebook-messenger": "QUIC.Facebook",
    "facebook-rupload": "QUIC.Facebook",
    "facebook-web": "QUIC.Facebook",
    "gmail": "QUIC.Google",
    "google-authentication": "QUIC.GoogleServices",
    "google-autofill": "QUIC.GoogleServices",
    "google-background": "QUIC.GoogleServices",
    "google-calendar": "QUIC.GoogleServices",
    "google-docs": "QUIC.GoogleServices",
    "google-drive": "QUIC.GoogleServices",
    "google-fonts": "QUIC.GoogleServices",
    "google-gstatic": "QUIC.GoogleServices",
    "google-pay": "QUIC.GoogleServices",
    "google-photos": "QUIC.GoogleServices",
    "google-play": "QUIC.PlayStore",
    "google-safebrowsing": "QUIC.GoogleServices",
    "google-services": "QUIC.GoogleServices",
    "google-translate": "QUIC.GoogleServices",
    "google-usercontent": "QUIC.GoogleServices",
    "google-www": "QUIC.Google",
    "instagram": "QUIC.Instagram",
    "microsoft-outlook": "QUIC.Microsoft365",
    "microsoft-substrate": "QUIC.Microsoft365",
    "spotify": "QUIC.Spotify",
    "tiktok": "QUIC.TikTok",
    "twitch": "QUIC.Twitch",
    "whatsapp": "WhatsApp",
    "youtube": "QUIC.YouTube",
}

TLS_TAG_OVERRIDES: dict[str, str] = {
    "apple-icloud": "TLS.AppleiCloud",
    "apple-itunes": "TLS.AppleiTunes",
    "facebook-media": "TLS.Facebook",
    "facebook-web": "TLS.Facebook",
    "google-play": "TLS.PlayStore",
    "google-services": "TLS.GoogleServices",
    "google-www": "TLS.Google",
    "microsoft-onedrive": "TLS.MS_OneDrive",
    "microsoft-update": "TLS.WindowsUpdate",
    "office365": "TLS.Microsoft365",
    "youtube": "TLS.YouTube",
}

APPLICATION_CATEGORY_BY_PREFIX: dict[str, str] = {
    "ADS_Analytic_Track": "Advertisement",
    "Apple": "Web",
    "AppleiCloud": "Cloud",
    "AppleiTunes": "Streaming",
    "Discord": "Chat",
    "DoH_DoT": "Web",
    "Facebook": "SocialNetwork",
    "FacebookMessenger": "Chat",
    "Google": "Web",
    "GoogleServices": "Web",
    "Instagram": "SocialNetwork",
    "MS_OneDrive": "Collaborative",
    "Microsoft365": "Collaborative",
    "PlayStore": "Download",
    "RiotGames": "Game",
    "Spotify": "Music",
    "Steam": "Game",
    "Teams": "Collaborative",
    "TikTok": "SocialNetwork",
    "Twitch": "Video",
    "WindowsUpdate": "SoftwareUpdate",
    "YouTube": "Media",
    "iCloudPrivateRelay": "VPN",
}


def _normalize_tag(value: Any) -> str:
    return str(value).strip().lower().replace("_", "-")


def map_cesnet_label(tag: Any, *, dataset: str) -> str | None:
    normalized = _normalize_tag(tag)
    if dataset == "quic":
        return QUIC_TAG_OVERRIDES.get(normalized)
    if dataset == "tls":
        return TLS_TAG_OVERRIDES.get(normalized) or CESNET_TO_APPLICATION_NAME.get(normalized)
    raise ValueError(f"Unsupported CESNET dataset kind: {dataset!r}")


def infer_application_category(application_name: str) -> str:
    name = application_name.split(".", 1)[-1]
    return APPLICATION_CATEGORY_BY_PREFIX.get(name, "Web")


def _parse_json_or_python(value: Any) -> Any:
    if isinstance(value, str):
        value = value.strip()
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return ast.literal_eval(value)
    return value


def parse_ppi(value: Any) -> tuple[list[int], list[int], list[int]]:
    parsed = _parse_json_or_python(value)
    if not isinstance(parsed, (list, tuple)) or len(parsed) < 3:
        raise ValueError("PPI must contain IPT, DIR, and SIZE sequences.")

    ipt = list(parsed[0])
    direction = list(parsed[1])
    size = list(parsed[2])
    return direction, size, ipt


def _to_model_direction(value: Any) -> int:
    numeric = int(value)
    if numeric == 0:
        return -1
    if numeric > 0:
        return 0
    return 1


def _to_int_or_pad(value: Any) -> int:
    numeric = int(float(value))
    return -1 if numeric == 0 else numeric


def normalize_ppi(value: Any, *, width: int) -> tuple[list[int], list[int], list[int]]:
    direction, size, ipt = parse_ppi(value)
    direction_out = [_to_model_direction(item) for item in direction[:width]]
    size_out = [_to_int_or_pad(item) for item in size[:width]]
    ipt_out = [_to_int_or_pad(item) for item in ipt[:width]]

    if len(direction_out) < width:
        direction_out.extend([-1] * (width - len(direction_out)))
    if len(size_out) < width:
        size_out.extend([-1] * (width - len(size_out)))
    if len(ipt_out) < width:
        ipt_out.extend([-1] * (width - len(ipt_out)))
    return direction_out, size_out, ipt_out


def _iter_csv_sources(input_path: Path) -> Iterable[Any]:
    if input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path) as archive:
            names = [
                name
                for name in archive.namelist()
                if name.lower().endswith((".csv", ".csv.gz")) and "servicemap" not in name.lower()
            ]
            if not names:
                raise ValueError(f"No data CSV found in zip: {input_path}")
            for name in names:
                with archive.open(name) as handle:
                    yield handle
    else:
        yield input_path


def _resolve_label_column(columns: list[str]) -> str:
    for candidate in ("APP", "Tag", "tag", "app"):
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not find APP/Tag label column. Columns: {columns}")


def convert_cesnet_to_act2_parquet(
    *,
    input_path: Path,
    out_path: Path,
    dataset: str,
    width: int = DEFAULT_WIDTH,
    chunksize: int = 100_000,
    max_rows: int | None = None,
    max_per_class: int | None = None,
    random_seed: int = 42,
    keep_unmapped: bool = False,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    rows_seen = 0
    rows_unmapped = 0
    mapped_counts: Counter[str] = Counter()
    unmapped_counts: Counter[str] = Counter()
    seen_per_class: Counter[str] = Counter()
    reservoir: dict[str, list[dict[str, Any]]] = {}
    rng = random.Random(random_seed)

    try:
        for source in _iter_csv_sources(input_path):
            for chunk in pd.read_csv(
                source,
                chunksize=chunksize,
                usecols=lambda column: column in {"PPI", "APP", "Tag", "tag", "app"},
                quoting=csv.QUOTE_MINIMAL,
            ):
                if "PPI" not in chunk.columns:
                    raise ValueError("CESNET input is missing required PPI column.")
                label_column = _resolve_label_column(list(chunk.columns))
                output_rows: list[dict[str, Any]] = []

                for record in chunk[["PPI", label_column]].itertuples(index=False, name=None):
                    ppi_value, raw_label = record
                    rows_seen += 1
                    application_name = map_cesnet_label(raw_label, dataset=dataset)
                    if application_name is None:
                        rows_unmapped += 1
                        unmapped_counts[_normalize_tag(raw_label)] += 1
                        if not keep_unmapped:
                            continue
                        application_name = f"UNMAPPED.{_normalize_tag(raw_label)}"

                    try:
                        splt_direction, splt_ps, splt_piat_ms = normalize_ppi(
                            ppi_value,
                            width=width,
                        )
                    except Exception:
                        continue

                    mapped_counts[application_name] += 1
                    converted_row = {
                        "splt_direction": str(splt_direction),
                        "splt_ps": str(splt_ps),
                        "splt_piat_ms": str(splt_piat_ms),
                        "application_name": application_name,
                        "application_category_name": infer_application_category(application_name),
                    }
                    if max_per_class is None:
                        output_rows.append(converted_row)
                    else:
                        seen_per_class[application_name] += 1
                        class_rows = reservoir.setdefault(application_name, [])
                        if len(class_rows) < max_per_class:
                            class_rows.append(converted_row)
                        else:
                            replacement_index = rng.randrange(seen_per_class[application_name])
                            if replacement_index < max_per_class:
                                class_rows[replacement_index] = converted_row

                    if (
                        max_per_class is None
                        and max_rows is not None
                        and rows_written + len(output_rows) >= max_rows
                    ):
                        break

                if max_per_class is None and output_rows:
                    table = pa.Table.from_pandas(pd.DataFrame(output_rows), preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(out_path, table.schema)
                    writer.write_table(table)
                    rows_written += len(output_rows)

                if max_per_class is None and max_rows is not None and rows_written >= max_rows:
                    break
            if max_per_class is None and max_rows is not None and rows_written >= max_rows:
                break
    finally:
        if max_per_class is not None:
            sampled_rows = [row for label in sorted(reservoir) for row in reservoir[label]]
            if max_rows is not None:
                sampled_rows = sampled_rows[:max_rows]
            if sampled_rows:
                table = pa.Table.from_pandas(pd.DataFrame(sampled_rows), preserve_index=False)
                writer = pq.ParquetWriter(out_path, table.schema)
                writer.write_table(table)
                rows_written = len(sampled_rows)
        if writer is not None:
            writer.close()

    metadata = {
        "input_path": str(input_path),
        "out_path": str(out_path),
        "dataset": dataset,
        "width": width,
        "chunksize": chunksize,
        "max_rows": max_rows,
        "max_per_class": max_per_class,
        "random_seed": random_seed,
        "rows_seen": rows_seen,
        "rows_written": rows_written,
        "rows_unmapped": rows_unmapped,
        "mapped_counts": dict(mapped_counts.most_common()),
        "top_unmapped_labels": dict(unmapped_counts.most_common(50)),
    }
    metadata_path = out_path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata
