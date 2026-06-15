#!/usr/bin/env python
"""Preprocess a custom audio folder into training chunks.

Default uses the dependency-light fallback (resample + slice + manifest).
Pass --use-rave to invoke the official `rave preprocess` CLI instead.

Examples:
    python scripts/preprocess_dataset.py --input ./raw --output ./dataset
    python scripts/preprocess_dataset.py --input ./raw --output ./db --use-rave
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from a source checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timbre_transfer.training.preprocess import preprocess_dataset, rave_preprocess  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Preprocess audio for fine-tuning.")
    p.add_argument("--input", required=True, help="Folder of source audio files.")
    p.add_argument("--output", required=True, help="Output dataset directory.")
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--chunk-seconds", type=float, default=2.0)
    p.add_argument("--use-rave", action="store_true", help="Use the `rave preprocess` CLI.")
    args = p.parse_args(argv)

    if args.use_rave:
        return rave_preprocess(args.input, args.output, sample_rate=args.sample_rate)
    preprocess_dataset(
        args.input,
        args.output,
        sample_rate=args.sample_rate,
        chunk_seconds=args.chunk_seconds,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
