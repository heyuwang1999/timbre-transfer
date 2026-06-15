"""Fine-tuning entry points for RAVE.

Two paths, mirroring ``preprocess.py``:

1. :func:`rave_train` -- the **real** path. Wraps the ``rave train`` CLI from the
   ``acids-rave`` package, resuming from a pre-trained Lightning ``.ckpt`` so you
   continue training rather than starting from scratch.

2. :func:`finetune_illustrative` -- a minimal, self-contained PyTorch loop that
   demonstrates the *shape* of a training step (dataloader -> forward ->
   multi-scale STFT reconstruction loss -> optimiser step -> checkpoint). It
   operates on a trainable ``nn.Module`` you provide. It is a TEACHING SCAFFOLD
   and explicitly does NOT work on a frozen ``.ts`` export.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..audio_io import load_audio


def rave_train(
    db_path: str | Path,
    name: str,
    config: str = "v2",
    ckpt: str | Path | None = None,
    out_dir: str | Path = "runs",
    extra_args: list[str] | None = None,
) -> int:
    """Run ``rave train`` (requires ``pip install acids-rave``).

    Args:
        db_path: Path to a dataset produced by ``rave preprocess``.
        name: Run name.
        config: RAVE config (e.g. ``"v2"``, ``"v2_small"``, ``"discrete"``).
        ckpt: Optional pre-trained ``.ckpt`` to resume/fine-tune from.
        out_dir: Where to write run artifacts.
        extra_args: Additional CLI flags passed through verbatim.

    Returns:
        The subprocess exit code.
    """
    if shutil.which("rave") is None:
        raise RuntimeError("The 'rave' CLI was not found. Install it with: pip install acids-rave")
    cmd = [
        "rave",
        "train",
        "--config",
        config,
        "--db_path",
        str(db_path),
        "--name",
        name,
        "--out_path",
        str(out_dir),
    ]
    if ckpt:
        # Resume from a pre-trained checkpoint == fine-tuning.
        cmd += ["--ckpt", str(ckpt)]
    if extra_args:
        cmd.extend(extra_args)
    print("[finetune] running:", " ".join(cmd))
    return subprocess.call(cmd)


# --------------------------------------------------------------------------- #
# Illustrative pure-PyTorch loop (teaching scaffold -- NOT for .ts files)
# --------------------------------------------------------------------------- #
class ChunkDataset(Dataset):
    """Loads fixed-length wav chunks from a manifest produced by preprocess."""

    def __init__(self, manifest_path: str | Path):
        manifest_path = Path(manifest_path)
        self.root = manifest_path.parent
        self.items = json.loads(manifest_path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> torch.Tensor:
        item = self.items[idx]
        wav, _ = load_audio(self.root / item["file"])  # (1, 1, T)
        return wav.squeeze(0)  # (1, T)


def multiscale_stft_loss(
    y: torch.Tensor,
    target: torch.Tensor,
    scales: tuple[int, ...] = (2048, 1024, 512, 256),
) -> torch.Tensor:
    """Sum of L1 spectral-magnitude losses across several STFT window sizes."""
    loss = y.new_zeros(())
    for n_fft in scales:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=y.device)
        sy = torch.stft(y.squeeze(1), n_fft, hop, window=window, return_complex=True).abs()
        st = torch.stft(target.squeeze(1), n_fft, hop, window=window, return_complex=True).abs()
        loss = loss + (sy - st).abs().mean()
    return loss


def finetune_illustrative(
    model: nn.Module,
    manifest_path: str | Path,
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: str = "cpu",
    out_path: str | Path = "checkpoints/finetuned.pt",
) -> Path:
    """A minimal reconstruction-based fine-tuning loop.

    ``model`` must be a *trainable* module whose ``forward(x)`` returns audio of
    the same shape as ``x`` (e.g. a non-scripted RAVE built via ``acids-rave``,
    or your own autoencoder). Do NOT pass a ``torch.jit`` ScriptModule loaded
    from a ``.ts`` file -- those are frozen and have no gradients.

    Returns the path of the saved checkpoint.
    """
    if isinstance(model, torch.jit.ScriptModule):
        raise TypeError(
            "Cannot fine-tune a TorchScript (.ts) model -- it is inference-only. "
            "Use rave_train() with a .ckpt, or pass a trainable nn.Module."
        )

    dataset = ChunkDataset(manifest_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = model.to(device).train()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        running = 0.0
        for batch in loader:
            batch = batch.to(device)  # (B, 1, T)
            optim.zero_grad()
            recon = model(batch)
            loss = multiscale_stft_loss(recon, batch)
            loss.backward()
            optim.step()
            running += loss.item()
        avg = running / max(1, len(loader))
        print(f"[finetune] epoch {epoch + 1}/{epochs}  loss={avg:.4f}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_path)
    print(f"[finetune] saved checkpoint -> {out_path}")
    return out_path
