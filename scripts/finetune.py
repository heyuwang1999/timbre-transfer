#!/usr/bin/env python
"""Fine-tune a RAVE model.

Recommended (real) path -- continue training from a pre-trained .ckpt via the
`rave` CLI (requires `pip install acids-rave`):

    python scripts/finetune.py rave --db ./db --name myrun --ckpt pretrained.ckpt

Illustrative path -- a minimal PyTorch reconstruction loop over preprocessed
chunks, for teaching / a trainable nn.Module (NOT a .ts file):

    python scripts/finetune.py demo --manifest ./dataset/manifest.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from a source checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from timbre_transfer.training.finetune import finetune_illustrative, rave_train  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fine-tune RAVE.")
    sub = p.add_subparsers(dest="mode", required=True)

    pr = sub.add_parser("rave", help="Wrap the official `rave train` CLI (recommended).")
    pr.add_argument("--db", required=True, help="Dataset path from `rave preprocess`.")
    pr.add_argument("--name", required=True, help="Run name.")
    pr.add_argument("--config", default="v2", help="RAVE config (default: v2).")
    pr.add_argument("--ckpt", default=None, help="Pre-trained .ckpt to resume/fine-tune.")
    pr.add_argument("--out", default="runs", help="Output directory.")

    pd = sub.add_parser("demo", help="Illustrative PyTorch loop (teaching scaffold).")
    pd.add_argument("--manifest", required=True, help="manifest.json from preprocess_dataset.")
    pd.add_argument("--epochs", type=int, default=5)
    pd.add_argument("--batch-size", type=int, default=4)
    pd.add_argument("--lr", type=float, default=1e-4)
    pd.add_argument("--device", default="cpu")
    pd.add_argument("--out", default="checkpoints/finetuned.pt")

    args = p.parse_args(argv)

    if args.mode == "rave":
        return rave_train(
            db_path=args.db, name=args.name, config=args.config, ckpt=args.ckpt, out_dir=args.out
        )

    # demo: build a tiny trainable autoencoder so the scaffold runs out of the box.
    import torch
    from torch import nn

    class TinyAutoencoder(nn.Module):
        """Toy 1-D conv autoencoder -- placeholder for a real trainable RAVE."""

        def __init__(self):
            super().__init__()
            self.enc = nn.Conv1d(1, 16, 9, stride=4, padding=4)
            self.dec = nn.ConvTranspose1d(16, 1, 9, stride=4, padding=4, output_padding=3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tanh(self.dec(torch.relu(self.enc(x))))

    finetune_illustrative(
        TinyAutoencoder(),
        manifest_path=args.manifest,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        out_path=args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
