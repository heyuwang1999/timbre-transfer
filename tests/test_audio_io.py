"""Tests for audio loading, resampling, mono conversion, chunking."""

from __future__ import annotations

import torch

from timbre_transfer.audio_io import (
    chunk_waveform,
    load_audio,
    resample,
    save_audio,
    to_mono,
)


def test_to_mono_collapses_channels():
    stereo = torch.randn(2, 1000)
    mono = to_mono(stereo)
    assert mono.shape == (1, 1000)


def test_resample_changes_length():
    wav = torch.randn(1, 1000)
    out = resample(wav, 1000, 2000)
    assert out.shape[-1] == 2000


def test_resample_noop_when_equal():
    wav = torch.randn(1, 1000)
    out = resample(wav, 1000, 1000)
    assert torch.equal(out, wav)


def test_save_and_load_roundtrip(tmp_path):
    sr = 16000
    t = torch.arange(sr) / sr
    wav = (0.2 * torch.sin(2 * torch.pi * 440 * t)).view(1, 1, -1)
    path = tmp_path / "tone.wav"
    save_audio(path, wav, sr)

    loaded, loaded_sr = load_audio(path)
    assert loaded_sr == sr
    assert loaded.dim() == 3 and loaded.shape[0] == 1 and loaded.shape[1] == 1
    assert abs(loaded.shape[-1] - wav.shape[-1]) <= 1


def test_load_resamples_to_target(tmp_path):
    sr = 16000
    wav = torch.randn(1, 1, sr)
    path = tmp_path / "noise.wav"
    save_audio(path, wav, sr)
    loaded, loaded_sr = load_audio(path, target_sr=8000)
    assert loaded_sr == 8000
    assert loaded.shape[-1] == 8000


def test_chunk_waveform_no_chunking():
    wav = torch.randn(1, 1, 1000)
    chunks = chunk_waveform(wav, chunk_samples=0)
    assert len(chunks) == 1


def test_chunk_waveform_splits():
    wav = torch.randn(1, 1, 1000)
    chunks = chunk_waveform(wav, chunk_samples=300, overlap_samples=0)
    assert len(chunks) == 4  # 300,300,300,100
    assert chunks[0].shape[-1] == 300
