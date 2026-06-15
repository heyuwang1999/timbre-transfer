"""Configuration loading helpers.

The default config ships at ``configs/default.yaml`` in the repo root. We locate
it relative to the installed package so it works both from a source checkout and
from an editable install.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repo root = three levels up from this file: src/timbre_transfer/config.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config file, falling back to the packaged default.

    Args:
        path: Optional explicit path to a YAML config. When ``None`` the bundled
            ``configs/default.yaml`` is used.

    Returns:
        Parsed configuration dictionary.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data


def resolve_device(requested: str = "auto") -> str:
    """Resolve a device string ("auto" -> cuda if available else cpu)."""
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested
