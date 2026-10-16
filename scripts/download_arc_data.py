#!/usr/bin/env python3
"""Download the public ARC-AGI task data into ``data/arc_eval/``.

Fetches the individual per-task JSON files from the official ARC-AGI GitHub
repository (``fchollet/ARC-AGI``) and stores them locally so the agent can run
against real training/evaluation tasks.  The script uses only the Python
standard library (``urllib``) and is idempotent: already-downloaded files are
skipped unless ``--force`` is passed.

Usage::

    py scripts/download_arc_data.py                    # training + evaluation
    py scripts/download_arc_data.py --subset training  # training only
    py scripts/download_arc_data.py --dest data/arc_eval --force
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Official ARC-AGI repository and its per-task data directories.
_REPO = "fchollet/ARC-AGI"
_BRANCH = "master"
_SUBSETS = ("training", "evaluation")

# Repo root is scripts/.. ; the default destination is data/arc_eval/.
_DEFAULT_DEST = Path(__file__).resolve().parents[1] / "data" / "arc_eval"

# GitHub requires a User-Agent header for API requests.
_HEADERS = {"User-Agent": "paper-track-agent"}


def _http_get_json(url: str):
    """GET ``url`` and return the decoded JSON body."""
    request = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _download_file(url: str, dest: Path) -> None:
    """Download ``url`` and write its raw bytes to ``dest``."""
    request = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    dest.write_bytes(data)


def _list_task_files(subset: str) -> list[dict]:
    """List task-file metadata for ``subset`` via the GitHub contents API."""
    url = f"https://api.github.com/repos/{_REPO}/contents/data/{subset}"
    entries = _http_get_json(url)
    if not isinstance(entries, list):
        raise RuntimeError(f"Unexpected GitHub API response for '{subset}': {entries!r}")
    return [e for e in entries if e.get("type") == "file" and e["name"].endswith(".json")]


def _download_subset(subset: str, dest: Path, force: bool) -> int:
    """Download every task in ``subset`` into ``dest / subset``; return count."""
    files = _list_task_files(subset)
    subset_dir = dest / subset
    subset_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for index, entry in enumerate(files, start=1):
        name = entry["name"]
        target = subset_dir / name
        if target.exists() and not force:
            print(f"[{index}/{len(files)}] skip (exists) {subset}/{name}")
            continue
        _download_file(entry["download_url"], target)
        downloaded += 1
        print(f"[{index}/{len(files)}] downloaded {subset}/{name}")

    return downloaded


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset",
        choices=(*_SUBSETS, "all"),
        default="all",
        help="which ARC-AGI subset(s) to download (default: all)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=_DEFAULT_DEST,
        help="destination directory (default: data/arc_eval)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download files that already exist locally",
    )
    args = parser.parse_args()

    subsets = _SUBSETS if args.subset == "all" else (args.subset,)
    args.dest.mkdir(parents=True, exist_ok=True)

    total = 0
    for subset in subsets:
        try:
            count = _download_subset(subset, args.dest, args.force)
        except urllib.error.URLError as exc:
            print(f"ERROR: failed to download '{subset}': {exc.reason}", file=sys.stderr)
            return 1
        total += count
        print(f"Done: {count} file(s) written for '{subset}'.")

    print(f"ARC-AGI data ready under {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
