"""timbre_transfer: audio timbre transfer with pre-trained RAVE models.

Primary path: RAVE (ACIDS-IRCAM) TorchScript models for encode -> decode
timbre transfer. DDSP is provided as an optional/experimental module.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import load_config

__all__ = ["load_config", "__version__"]
