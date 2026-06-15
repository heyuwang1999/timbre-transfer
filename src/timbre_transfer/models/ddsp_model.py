"""Minimal DDSP decoder interface (EXPERIMENTAL / OPTIONAL).

This is a *scaffold*, not a pre-trained model. The canonical DDSP decoders (with
violin/flute/etc. weights) ship as TensorFlow checkpoints; porting or training a
PyTorch equivalent is left as a fine-tuning exercise. The class below defines the
control->audio contract so the rest of the pipeline (feature extraction in
``timbre_transfer.features.ddsp_features``) has a target to plug into.
"""

from __future__ import annotations

import torch
from torch import nn


class DdspDecoder(nn.Module):
    """A tiny harmonic-plus-noise style DDSP decoder skeleton.

    Maps frame-rate controls (F0, loudness) to additive-synth parameters. The
    forward pass here is intentionally minimal -- replace the synth with a full
    harmonic + filtered-noise synthesiser when training for real timbre transfer.
    """

    def __init__(self, n_harmonics: int = 64, hidden: int = 256):
        super().__init__()
        self.n_harmonics = n_harmonics
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.LayerNorm(hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, n_harmonics + 1),  # harmonic amps + overall gain
        )

    def forward(self, f0_hz: torch.Tensor, loudness_db: torch.Tensor) -> torch.Tensor:
        """Predict per-frame harmonic distribution and gain.

        Args:
            f0_hz: ``(B, frames)`` fundamental frequency.
            loudness_db: ``(B, frames)`` loudness.

        Returns:
            ``(B, frames, n_harmonics + 1)`` synth controls (un-synthesised).
            Hook a harmonic/noise synthesiser here to produce audio.
        """
        x = torch.stack([f0_hz, loudness_db], dim=-1)  # (B, frames, 2)
        return self.net(x)


def load_ddsp(path: str | None = None) -> DdspDecoder:
    """Instantiate a DDSP decoder, optionally loading a state_dict.

    Raises a clear message if asked to load weights that don't exist, since no
    pre-trained PyTorch DDSP checkpoint ships with this project.
    """
    model = DdspDecoder()
    if path:
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
    model.eval()
    return model
