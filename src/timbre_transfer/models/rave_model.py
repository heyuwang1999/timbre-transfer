"""Load and wrap pre-trained RAVE TorchScript (.ts) models.

A RAVE ``.ts`` export is a *frozen, inference-only* TorchScript module. It
typically exposes:
    * ``encode(x)``  -> latent tensor ``z`` of shape (B, latent, T')
    * ``decode(z)``  -> waveform of shape (B, 1, T)
    * ``forward(x)`` -> reconstruct-in-timbre (encode then decode)

Some exports only script ``forward``; we handle both. This module does NOT
support fine-tuning -- see ``timbre_transfer.training`` for that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class RaveModel:
    """A loaded RAVE TorchScript model plus introspected metadata."""

    module: torch.jit.ScriptModule
    sample_rate: int
    device: str
    has_encode_decode: bool

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a waveform (B, 1, T) into a latent (B, latent, T')."""
        if not self.has_encode_decode:
            raise RuntimeError(
                "This model does not expose separate encode/decode. "
                "Use timbre_transfer() which falls back to forward()."
            )
        return self.module.encode(x.to(self.device))

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent (B, latent, T') back into a waveform (B, 1, T)."""
        if not self.has_encode_decode:
            raise RuntimeError("This model does not expose separate encode/decode.")
        return self.module.decode(z.to(self.device))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full encode->decode reconstruction (timbre re-synthesis)."""
        return self.module(x.to(self.device))


def _infer_sample_rate(module: torch.jit.ScriptModule, default: int) -> int:
    """Best-effort extraction of a model's native sample rate.

    RAVE exports register the sample rate under various names across versions;
    we probe the common ones and fall back to ``default``.
    """
    for attr in ("sr", "sampling_rate", "sample_rate", "fs"):
        if hasattr(module, attr):
            try:
                value = getattr(module, attr)
                value = value.item() if isinstance(value, torch.Tensor) else value
                if value:
                    return int(value)
            except Exception:
                pass
    # Probe registered buffers (e.g. a buffer literally named "sampling_rate").
    try:
        for name, buf in module.named_buffers():
            if any(k in name.lower() for k in ("sr", "sampling_rate", "sample_rate")):
                if buf.numel() == 1:
                    return int(buf.item())
    except Exception:
        pass
    return int(default)


def _has_encode_decode(module: torch.jit.ScriptModule) -> bool:
    """Return True if the scripted module exposes callable encode AND decode."""
    return all(
        callable(getattr(module, name, None)) and hasattr(module, name)
        for name in ("encode", "decode")
    )


def load_rave(
    path: str | Path,
    device: str = "cpu",
    default_sample_rate: int = 48000,
) -> RaveModel:
    """Load a RAVE TorchScript model from disk.

    Args:
        path: Path to a ``.ts`` file.
        device: Torch device string ("cpu" / "cuda").
        default_sample_rate: Fallback SR if the model does not advertise one.

    Returns:
        A :class:`RaveModel` ready for inference.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    module = torch.jit.load(str(path), map_location=device)
    module.eval()
    sr = _infer_sample_rate(module, default_sample_rate)
    return RaveModel(
        module=module,
        sample_rate=sr,
        device=device,
        has_encode_decode=_has_encode_decode(module),
    )
