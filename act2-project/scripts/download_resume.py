from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests


def _format_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{value} B"


def download_with_resume(url: str, out_path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    existing_size = out_path.stat().st_size if out_path.exists() else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size > 0 else {}

    with requests.get(url, headers=headers, stream=True, timeout=(30, 120), allow_redirects=True) as response:
        if existing_size > 0 and response.status_code == 200:
            print("Server did not accept resume; restarting download.")
            existing_size = 0
            mode = "wb"
        elif response.status_code in (200, 206):
            mode = "ab" if existing_size > 0 else "wb"
        else:
            response.raise_for_status()
            mode = "ab" if existing_size > 0 else "wb"

        total_header = response.headers.get("Content-Length")
        total_size = int(total_header) + existing_size if total_header else None
        downloaded = existing_size
        started = time.monotonic()
        last_print = started

        print(f"URL: {url}")
        print(f"Output: {out_path}")
        if existing_size:
            print(f"Resuming from {_format_bytes(existing_size)}")
        if total_size:
            print(f"Expected total: {_format_bytes(total_size)}")

        with out_path.open(mode + "") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_print >= 5:
                    elapsed = max(now - started, 0.001)
                    speed = (downloaded - existing_size) / elapsed
                    if total_size:
                        pct = downloaded / total_size * 100
                        print(
                            f"{_format_bytes(downloaded)} / {_format_bytes(total_size)} "
                            f"({pct:.2f}%) at {_format_bytes(int(speed))}/s",
                            flush=True,
                        )
                    else:
                        print(
                            f"{_format_bytes(downloaded)} at {_format_bytes(int(speed))}/s",
                            flush=True,
                        )
                    last_print = now

    print(f"Done: {_format_bytes(out_path.stat().st_size)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download a URL with HTTP Range resume support.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    download_with_resume(args.url, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
