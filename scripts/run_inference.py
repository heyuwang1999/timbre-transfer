#!/usr/bin/env python
"""Run RAVE timbre transfer on an audio file.

Example:
    python scripts/run_inference.py --input in.wav --model guitar --output out.wav
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a source checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timbre_transfer.cli import infer_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(infer_main())
