#!/usr/bin/env python
"""List or download pre-trained RAVE models.

Examples:
    python scripts/download_models.py --list
    python scripts/download_models.py --model guitar
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a source checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timbre_transfer.cli import download_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(download_main())
