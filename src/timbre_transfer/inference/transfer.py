"""Core RAVE timbre-transfer pipeline.

The transfer works by encoding the source audio into RAVE's latent space and
decoding it through a model trained on the *target* timbre. Optional latent
controls (bias / scale / temperature) expose creative manipulation. When a model
only scripts ``forward`` (no separate encode/decode), we fall back to running the
full reconstruction, which still re-synthesises the input in the model's timbre.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..audio_io import chunk_waveform, load_audio, save_audio
from ..models.rave_model import RaveModel


@dataclass
class LatentControls:
    """Creative controls applied to the RAVE latent before decoding."""

    bias: float = 0.0
    scale: float = 1.0
    temperature: float = 0.0  # std of gaussian noise added to the latent

    @property
    def is_identity(self) -> bool:
        return self.bias == 0.0 and self.scale == 1.0 and self.temperature == 0.0

    def apply(self, z: torch.Tensor) -> torch.Tensor:
        z = z * self.scale + self.bias
        if self.temperature > 0.0:
            z = z + torch.randn_like(z) * self.temperature
        return z


@torch.no_grad()
def _transfer_chunk(model: RaveModel, x: torch.Tensor, controls: LatentControls) -> torch.Tensor:
    """Run timbre transfer on a single (1, 1, T) chunk."""
    if model.has_encode_decode:
        z = model.encode(x)
        if not controls.is_identity:
            z = controls.apply(z)
        return model.decode(z)
    # Fallback: model only exposes forward(). Latent controls are unavailable.
    return model.forward(x)


@torch.no_grad()
def timbre_transfer(
    model: RaveModel,
    wav: torch.Tensor,
    controls: LatentControls | None = None,
    chunk_samples: int = 0,
    overlap_samples: int = 0,
) -> torch.Tensor:
    """Apply timbre transfer to a mono batch waveform ``(1, 1, T)``.

    Args:
        model: A loaded :class:`RaveModel`.
        wav: Input waveform at the model's sample rate, shape ``(1, 1, T)``.
        controls: Optional latent manipulation (ignored for forward-only models).
        chunk_samples: If > 0, process in chunks of this length (samples).
        overlap_samples: Overlap between chunks; trimmed on concatenation.

    Returns:
        Output waveform tensor ``(1, 1, T')`` on the model's device.
    """
    controls = controls or LatentControls()
    chunks = chunk_waveform(wav, chunk_samples, overlap_samples)
    if len(chunks) == 1:
        return _transfer_chunk(model, chunks[0], controls)

    outputs = [_transfer_chunk(model, c, controls) for c in chunks]
    if overlap_samples <= 0:
        return torch.cat(outputs, dim=-1)
    # Trim the trailing overlap from every chunk except the last to avoid dupes.
    trimmed = [o[..., :-overlap_samples] for o in outputs[:-1]] + [outputs[-1]]
    return torch.cat(trimmed, dim=-1)


def transfer_file(
    model: RaveModel,
    input_path: str | Path,
    output_path: str | Path,
    controls: LatentControls | None = None,
    chunk_seconds: float = 0.0,
    overlap_seconds: float = 0.0,
) -> Path:
    """Load -> transfer -> save convenience wrapper.

    Returns the output path written.
    """
    wav, sr = load_audio(input_path, target_sr=model.sample_rate)
    chunk_samples = int(chunk_seconds * sr)
    overlap_samples = int(overlap_seconds * sr)
    out = timbre_transfer(
        model,
        wav,
        controls=controls,
        chunk_samples=chunk_samples,
        overlap_samples=overlap_samples,
    )
    save_audio(output_path, out, model.sample_rate)
    return Path(output_path)
