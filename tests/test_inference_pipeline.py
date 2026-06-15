"""End-to-end pipeline tests using the offline dummy RAVE TorchScript model."""

from __future__ import annotations

import torch

from timbre_transfer.inference.transfer import (
    LatentControls,
    timbre_transfer,
    transfer_file,
)
from timbre_transfer.models.rave_model import load_rave


def test_load_rave_introspects_sample_rate(dummy_rave_path):
    model = load_rave(dummy_rave_path, device="cpu")
    assert model.sample_rate == 44100
    assert model.has_encode_decode is True


def test_timbre_transfer_runs(dummy_rave_path, sine_wave):
    model = load_rave(dummy_rave_path, device="cpu")
    out = timbre_transfer(model, sine_wave)
    assert out.dim() == 3
    assert out.shape[0] == 1 and out.shape[1] == 1
    assert torch.isfinite(out).all()


def test_latent_controls_change_output(dummy_rave_path, sine_wave):
    model = load_rave(dummy_rave_path, device="cpu")
    baseline = timbre_transfer(model, sine_wave)
    scaled = timbre_transfer(model, sine_wave, controls=LatentControls(scale=0.5))
    assert not torch.allclose(baseline, scaled)


def test_chunked_matches_unchunked_length(dummy_rave_path, sine_wave):
    model = load_rave(dummy_rave_path, device="cpu")
    whole = timbre_transfer(model, sine_wave)
    chunked = timbre_transfer(model, sine_wave, chunk_samples=4096, overlap_samples=0)
    # No overlap -> concatenated chunks reconstruct the same total length.
    assert chunked.shape[-1] == whole.shape[-1]


def test_transfer_file_roundtrip(dummy_rave_path, sine_wave, tmp_path):
    from timbre_transfer.audio_io import load_audio, save_audio

    model = load_rave(dummy_rave_path, device="cpu")
    in_path = tmp_path / "in.wav"
    out_path = tmp_path / "out.wav"
    save_audio(in_path, sine_wave, model.sample_rate)

    result = transfer_file(model, in_path, out_path)
    assert result.exists()
    loaded, sr = load_audio(out_path)
    assert sr == model.sample_rate
    assert loaded.shape[0] == 1 and loaded.shape[1] == 1
