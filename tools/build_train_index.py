#!/usr/bin/env python3
"""Build the JSON shard index consumed by Echo-SR's streaming dataset."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("patterns", nargs="+", help="Tar paths or glob patterns.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--relative",
        action="store_true",
        help="Write paths relative to the current directory instead of absolute paths.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths: set[Path] = set()
    for pattern in args.patterns:
        matches = glob.glob(pattern, recursive=True)
        if not matches and Path(pattern).is_file():
            matches = [pattern]
        paths.update(Path(match) for match in matches if Path(match).is_file())

    tar_paths = sorted(path for path in paths if path.suffix.lower() == ".tar")
    if not tar_paths:
        raise SystemExit("No .tar shards matched the supplied paths or patterns.")

    cwd = Path.cwd().resolve()
    resolved_files = []
    for path in tar_paths:
        absolute = path.expanduser().resolve()
        resolved_files.append(str(absolute.relative_to(cwd) if args.relative else absolute))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"resolved_files": resolved_files}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(resolved_files)} shards to {args.output}")


if __name__ == "__main__":
    main()
