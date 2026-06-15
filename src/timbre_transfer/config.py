"""Configuration loading helpers.

``load_config()`` resolves config in this order:

1. An explicit ``path`` argument (must exist, else error).
2. The bundled ``configs/default.yaml`` next to the package (source checkout /
   editable install).
3. Built-in :data:`EMBEDDED_DEFAULTS` -- so the package works even when
   installed non-editable (e.g. ``pip install .`` in Colab), where the repo's
   ``configs/`` directory is not shipped alongside the package.

This means importing and using the package never crashes just because the YAML
file isn't where the source layout expects it.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import yaml

# Repo root = three levels up from this file: src/timbre_transfer/config.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "default.yaml"

# Built-in fallback, kept in sync with configs/default.yaml. Used when the YAML
# file cannot be found (e.g. a non-editable install that did not ship configs/).
EMBEDDED_DEFAULTS: dict[str, Any] = {
    "models": {
        "repos": ["Intelligent-Instruments-Lab/rave-models"],
        "fallback": [
            {
                "repo": "Intelligent-Instruments-Lab/rave-models",
                "filename": "guitar_iil_b2048_r48000_z16.ts",
                "label": "guitar",
            },
            {
                "repo": "Intelligent-Instruments-Lab/rave-models",
                "filename": "organ_archive_b2048_r48000_z16.ts",
                "label": "organ",
            },
            {
                "repo": "Intelligent-Instruments-Lab/rave-models",
                "filename": "sax_soprano_franziskaschroeder_b2048_r48000_z20.ts",
                "label": "sax",
            },
            {
                "repo": "Intelligent-Instruments-Lab/rave-models",
                "filename": "voice_jvs_b2048_r44100_z16.ts",
                "label": "voice",
            },
        ],
    },
    "cache_dir": "models",
    "audio": {
        "default_sample_rate": 48000,
        "chunk_seconds": 0,
        "chunk_overlap_seconds": 0.0,
    },
    "inference": {
        "device": "auto",
        "latent_bias": 0.0,
        "latent_scale": 1.0,
        "latent_temperature": 0.0,
    },
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config, falling back to bundled YAML then embedded defaults.

    Args:
        path: Optional explicit path to a YAML config. When provided it must
            exist. When ``None``, the bundled ``configs/default.yaml`` is used
            if present, otherwise :data:`EMBEDDED_DEFAULTS`.

    Returns:
        Parsed configuration dictionary.
    """
    if path is not None:
        cfg_path = Path(path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    if DEFAULT_CONFIG_PATH.exists():
        with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    warnings.warn(
        f"Bundled config not found at {DEFAULT_CONFIG_PATH}; using embedded defaults. "
        "(Expected when installed non-editable.)",
        stacklevel=2,
    )
    # Return a deep-ish copy so callers can mutate without touching the constant.
    import copy

    return copy.deepcopy(EMBEDDED_DEFAULTS)


def resolve_device(requested: str = "auto") -> str:
    """Resolve a device string ("auto" -> cuda if available else cpu)."""
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested
