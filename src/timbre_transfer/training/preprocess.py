"""Preprocess custom audio datasets for fine-tuning.

Two paths are provided:

1. :func:`rave_preprocess` -- thin wrapper around the official ``rave preprocess``
   CLI (from the ``acids-rave`` package). This builds the on-disk dataset format
   that ``rave train`` expects and is the **recommended** path.

2. :func:`preprocess_dataset` -- a dependency-light fallback that resamples a
   folder of audio files to mono at a target rate, slices them into fixed-length
   chunks, and writes a JSON manifest. Useful for the illustrative training loop
   or for inspecting your data.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import torch

from ..audio_io import load_audio, save_audio


def rave_preprocess(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 48000,
    channels: int = 1,
    extra_args: list[str] | None = None,
) -> int:
    """Run ``rave preprocess`` (requires ``pip install acids-rave``).

    Returns the subprocess exit code.
    """
    if shutil.which("rave") is None:
        raise RuntimeError("The 'rave' CLI was not found. Install it with: pip install acids-rave")
    cmd = [
        "rave",
        "preprocess",
        "--input_path",
        str(input_path),
        "--output_path",
        str(output_path),
        "--sampling_rate",
        str(sample_rate),
        "--channels",
        str(channels),
    ]
    if extra_args:
        cmd.extend(extra_args)
    print("[preprocess] running:", " ".join(cmd))
    return subprocess.call(cmd)


def preprocess_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    sample_rate: int = 48000,
    chunk_seconds: float = 2.0,
    extensions: tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg", ".aif", ".aiff"),
) -> Path:
    """Resample + slice a folder of audio into fixed-length training chunks.

    Writes chunk ``.wav`` files plus a ``manifest.json`` into ``output_dir`` and
    returns the manifest path.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_samples = int(chunk_seconds * sample_rate)

    files = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in extensions)
    if not files:
        raise FileNotFoundError(f"No audio files found under {input_dir}")

    manifest: list[dict] = []
    idx = 0
    for src in files:
        wav, _ = load_audio(src, target_sr=sample_rate)  # (1, 1, T)
        wav = wav.squeeze(0)  # (1, T)
        total = wav.size(-1)
        for start in range(0, total, chunk_samples):
            chunk = wav[..., start : start + chunk_samples]
            if chunk.size(-1) < chunk_samples:
                # Pad the final short chunk so all chunks are equal length.
                pad = chunk_samples - chunk.size(-1)
                chunk = torch.nn.functional.pad(chunk, (0, pad))
            out_name = f"chunk_{idx:06d}.wav"
            save_audio(output_dir / out_name, chunk, sample_rate)
            manifest.append({"file": out_name, "source": str(src), "sample_rate": sample_rate})
            idx += 1

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[preprocess] wrote {idx} chunks + manifest to {output_dir}")
    return manifest_path
