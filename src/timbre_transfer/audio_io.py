"""Audio loading, resampling, mono conversion, and saving.

Tensors flow through the pipeline as ``(batch, channels, samples)`` with a
single channel (mono), matching what RAVE TorchScript models expect.

File I/O goes through ``soundfile`` (libsndfile) rather than ``torchaudio.load``
/ ``torchaudio.save``: recent torchaudio routes I/O through ``torchcodec`` +
FFmpeg, which is often unavailable. ``torchaudio`` is still used for its
pure-torch resampler. soundfile reads/writes WAV/FLAC/OGG without FFmpeg.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def to_mono(wav: torch.Tensor) -> torch.Tensor:
    """Collapse a (channels, samples) tensor to mono (1, samples)."""
    if wav.dim() != 2:
        raise ValueError(f"Expected (channels, samples), got shape {tuple(wav.shape)}")
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav


def resample(wav: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample a (channels, samples) tensor if sample rates differ."""
    if orig_sr == target_sr:
        return wav
    return torchaudio.functional.resample(wav, orig_freq=orig_sr, new_freq=target_sr)


def load_audio(path: str | Path, target_sr: int | None = None) -> tuple[torch.Tensor, int]:
    """Load an audio file as a mono batch tensor.

    Args:
        path: Path to an audio file (wav/flac/mp3/ogg, anything torchaudio reads).
        target_sr: If given, resample to this rate.

    Returns:
        ``(wav, sr)`` where ``wav`` has shape ``(1, 1, samples)`` (batch, mono)
        and ``sr`` is the (possibly resampled) sample rate.
    """
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (samples, channels)
    wav = torch.from_numpy(data.T.copy())  # (channels, samples)
    wav = to_mono(wav)
    if target_sr is not None:
        wav = resample(wav, sr, target_sr)
        sr = target_sr
    return wav.unsqueeze(0), sr  # (1, 1, samples)


def save_audio(path: str | Path, wav: torch.Tensor, sr: int) -> None:
    """Save a waveform tensor to disk.

    Accepts ``(samples,)``, ``(channels, samples)`` or ``(batch, channels,
    samples)`` (batch is squeezed). Always written on CPU.
    """
    wav = wav.detach().cpu()
    if wav.dim() == 3:
        wav = wav.squeeze(0)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.dim() != 2:
        raise ValueError(f"Cannot save tensor of shape {tuple(wav.shape)}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data: np.ndarray = wav.transpose(0, 1).numpy()  # (samples, channels)
    sf.write(str(path), data, sr)


def chunk_waveform(
    wav: torch.Tensor, chunk_samples: int, overlap_samples: int = 0
) -> list[torch.Tensor]:
    """Split a (1, 1, T) waveform into overlapping chunks for streamed inference.

    Returns a single-element list when ``chunk_samples <= 0`` or the signal is
    shorter than one chunk.
    """
    if chunk_samples <= 0:
        return [wav]
    total = wav.size(-1)
    if total <= chunk_samples:
        return [wav]
    step = max(1, chunk_samples - overlap_samples)
    chunks = []
    start = 0
    while start < total:
        chunks.append(wav[..., start : start + chunk_samples])
        start += step
    return chunks
