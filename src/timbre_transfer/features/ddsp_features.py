"""Feature extraction for DDSP timbre transfer (EXPERIMENTAL / OPTIONAL).

DDSP synthesises audio from two control signals extracted from the source:
    * fundamental frequency (F0) -- here via ``torchcrepe``
    * loudness -- A-weighted log-power of the STFT

These features condition a decoder trained on a target instrument. NOTE: robust
pre-trained *PyTorch* DDSP decoders are scarce (the canonical checkpoints are
TensorFlow); this module supplies the front-end so you can train or port one.

Requires the optional dependencies: ``pip install -e .[ddsp]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def _require_torchcrepe():
    try:
        import torchcrepe  # noqa: F401

        return torchcrepe
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise ImportError(
            "DDSP feature extraction needs torchcrepe. Install the optional extra: "
            "pip install -e '.[ddsp]'"
        ) from exc


@dataclass
class DdspFeatures:
    """Frame-rate control signals for a DDSP decoder."""

    f0_hz: torch.Tensor  # (1, frames)
    loudness_db: torch.Tensor  # (1, frames)
    sample_rate: int
    hop_length: int


def extract_f0(
    wav: torch.Tensor,
    sample_rate: int,
    hop_length: int = 256,
    fmin: float = 50.0,
    fmax: float = 2000.0,
    model: str = "full",
) -> torch.Tensor:
    """Estimate F0 (Hz) per frame with CREPE. Input ``wav`` is ``(1, samples)``."""
    torchcrepe = _require_torchcrepe()
    f0 = torchcrepe.predict(
        wav,
        sample_rate,
        hop_length=hop_length,
        fmin=fmin,
        fmax=fmax,
        model=model,
        batch_size=512,
        device=str(wav.device),
    )
    return f0  # (1, frames)


def compute_loudness(
    wav: torch.Tensor,
    sample_rate: int,
    hop_length: int = 256,
    n_fft: int = 2048,
    ref_db: float = 20.0,
) -> torch.Tensor:
    """A-weighted log-power loudness per frame. Input ``wav`` is ``(1, samples)``."""
    window = torch.hann_window(n_fft, device=wav.device)
    stft = torch.stft(
        wav.squeeze(0),
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    power = stft.abs() ** 2  # (freq, frames)

    freqs = np.linspace(0, sample_rate / 2, power.size(0))
    a_weight = _a_weighting(freqs)
    a_weight = torch.from_numpy(a_weight).to(wav.device).float().unsqueeze(-1)

    weighted = power * (10.0 ** (a_weight / 10.0))
    loudness = 10.0 * torch.log10(weighted.sum(dim=0) + 1e-10)
    loudness = loudness - ref_db
    return loudness.unsqueeze(0)  # (1, frames)


def _a_weighting(freqs: np.ndarray) -> np.ndarray:
    """A-weighting curve (dB) for an array of frequencies (Hz)."""
    f2 = np.square(freqs) + 1e-9
    num = (12194.0**2) * (f2**2)
    den = (f2 + 20.6**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * (f2 + 12194.0**2)
    ra = num / den
    return 2.0 + 20.0 * np.log10(ra + 1e-9)


def extract_features(
    wav: torch.Tensor,
    sample_rate: int,
    hop_length: int = 256,
) -> DdspFeatures:
    """Extract F0 + loudness from a mono ``(1, samples)`` waveform."""
    if wav.dim() == 3:
        wav = wav.squeeze(0)
    f0 = extract_f0(wav, sample_rate, hop_length=hop_length)
    loudness = compute_loudness(wav, sample_rate, hop_length=hop_length)
    # Align frame counts (CREPE and STFT can differ by one frame).
    frames = min(f0.size(-1), loudness.size(-1))
    return DdspFeatures(
        f0_hz=f0[..., :frames],
        loudness_db=loudness[..., :frames],
        sample_rate=sample_rate,
        hop_length=hop_length,
    )
