r"""Restore local-only datasets and model artifacts for this project.

The repository intentionally does not track parquet/csv/pcap files or trained
model binaries. Use this script after cloning to restore a local data bundle
from an external archive.

Examples:
    python setup_data.py --archive D:\datasets\ml-flow-class-data.zip
    python setup_data.py --url https://example.com/ml-flow-class-data.zip
    python setup_data.py --url-env DATA_ARCHIVE_URL
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT

EXPECTED_PATHS = (
    Path("data/raw/final_dataset_63_classes_splt.parquet"),
    Path("data/artifacts/application_63_classes_splt_train_eval"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore datasets and trained artifacts from an external zip archive."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="Path to a local .zip data bundle.")
    source.add_argument("--url", help="HTTP(S) URL to a .zip data bundle.")
    source.add_argument(
        "--url-env",
        default=None,
        help="Environment variable containing the HTTP(S) archive URL.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Destination repository root. Defaults to this script directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow extraction over existing files.",
    )
    return parser.parse_args()


def resolve_archive(args: argparse.Namespace, tmp_dir: Path) -> Path:
    if args.archive:
        archive = args.archive.expanduser().resolve()
        if not archive.exists():
            raise FileNotFoundError(f"Archive not found: {archive}")
        return archive

    url = args.url
    if args.url_env:
        url = os.environ.get(args.url_env)
        if not url:
            raise RuntimeError(f"Environment variable is not set: {args.url_env}")

    if not url:
        raise RuntimeError("No archive URL provided.")

    archive = tmp_dir / "data_bundle.zip"
    print(f"Downloading data bundle from {url}")
    urllib.request.urlretrieve(url, archive)
    return archive


def validate_zip_members(archive: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = Path(member.filename)
            if target.is_absolute() or ".." in target.parts:
                raise RuntimeError(f"Unsafe zip member path: {member.filename}")


def extract_archive(archive: Path, dest: Path, force: bool) -> None:
    dest = dest.expanduser().resolve()
    if not dest.exists():
        raise FileNotFoundError(f"Destination does not exist: {dest}")

    validate_zip_members(archive)

    with zipfile.ZipFile(archive) as zf:
        if not force:
            collisions = [
                name
                for name in zf.namelist()
                if name and not name.endswith("/") and (dest / name).exists()
            ]
            if collisions:
                preview = "\n".join(f"  - {item}" for item in collisions[:10])
                raise RuntimeError(
                    "Extraction would overwrite existing files. "
                    "Re-run with --force if this is intended.\n"
                    f"{preview}"
                )
        zf.extractall(dest)


def print_expected_paths(dest: Path) -> None:
    print("\nExpected local files/directories:")
    missing = False
    for rel_path in EXPECTED_PATHS:
        path = dest / rel_path
        status = "OK" if path.exists() else "MISSING"
        missing = missing or not path.exists()
        print(f"  [{status}] {rel_path}")

    if missing:
        print(
            "\nSome expected paths are missing. Check that the archive preserves "
            "paths relative to the repository root."
        )


def main() -> int:
    args = parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        archive = resolve_archive(args, Path(tmp))
        extract_archive(archive, args.dest, args.force)
        print_expected_paths(args.dest.expanduser().resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"setup_data.py failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
