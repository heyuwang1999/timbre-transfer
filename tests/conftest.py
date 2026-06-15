"""Shared fixtures.

The key fixture builds a *tiny* TorchScript module that mimics a RAVE export
(exposing ``encode``/``decode``/``forward``). This lets the full load->transfer
pipeline be exercised offline with no model downloads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn


class _DummyRave(nn.Module):
    """Minimal encode/decode autoencoder standing in for a RAVE export."""

    def __init__(self, sampling_rate: int = 44100, latent: int = 4, hop: int = 8):
        super().__init__()
        self.sampling_rate = sampling_rate  # introspected by load_rave()
        self.enc = nn.Conv1d(1, latent, kernel_size=hop, stride=hop)
        self.dec = nn.ConvTranspose1d(latent, 1, kernel_size=hop, stride=hop)

    @torch.jit.export
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x)

    @torch.jit.export
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.dec(z))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


@pytest.fixture
def dummy_rave_path(tmp_path: Path) -> Path:
    """Path to a scripted dummy RAVE .ts file on disk."""
    model = _DummyRave()
    scripted = torch.jit.script(model)
    out = tmp_path / "dummy_rave.ts"
    scripted.save(str(out))
    return out


@pytest.fixture
def sine_wave() -> torch.Tensor:
    """A 0.5s mono sine as a (1, 1, T) tensor at 44100 Hz."""
    sr = 44100
    t = torch.arange(int(0.5 * sr)) / sr
    wav = 0.3 * torch.sin(2 * torch.pi * 220.0 * t)
    return wav.view(1, 1, -1)
