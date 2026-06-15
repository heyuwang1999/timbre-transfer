"""Argparse entry points shared by the console scripts and ``scripts/*.py``.

Keeping the logic here (rather than in ``scripts/``) means it is importable,
testable, and exposed as ``timbre-download`` / ``timbre-infer`` console scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, resolve_device
from .inference.transfer import LatentControls, transfer_file
from .models.rave_model import load_rave
from .models.registry import list_available_models, resolve_entry
from .utils.download import fetch_model


def download_main(argv: list[str] | None = None) -> int:
    """Entry point: list or download pre-trained models."""
    parser = argparse.ArgumentParser(description="Download / list pre-trained RAVE models.")
    parser.add_argument("--list", action="store_true", help="List available models and exit.")
    parser.add_argument("--model", help="Model selector (label, filename, or key).")
    parser.add_argument("--config", help="Path to a YAML config.")
    parser.add_argument("--offline", action="store_true", help="Skip live HF discovery.")
    parser.add_argument("--token", help="Hugging Face token for private/gated repos.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    allow_network = not args.offline

    if args.list or not args.model:
        models = list_available_models(cfg, token=args.token, allow_network=allow_network)
        print(f"Available models ({len(models)}):")
        for m in models:
            print(f"  - {m.label:12s} {m.filename}   [{m.repo}]")
        return 0

    entry = resolve_entry(args.model, cfg, token=args.token, allow_network=allow_network)
    path = fetch_model(
        entry.repo, entry.filename, cache_dir=cfg.get("cache_dir", "models"), token=args.token
    )
    print(f"Downloaded {entry.label} -> {path}")
    return 0


def infer_main(argv: list[str] | None = None) -> int:
    """Entry point: run timbre transfer on an audio file."""
    parser = argparse.ArgumentParser(description="RAVE timbre transfer on an audio file.")
    parser.add_argument("--input", required=True, help="Input mono audio file.")
    parser.add_argument("--output", required=True, help="Output audio path.")
    parser.add_argument("--model", required=True, help="Model selector (label/filename/key).")
    parser.add_argument("--config", help="Path to a YAML config.")
    parser.add_argument("--device", default=None, help="cpu | cuda | auto (overrides config).")
    parser.add_argument("--latent-bias", type=float, default=None)
    parser.add_argument("--latent-scale", type=float, default=None)
    parser.add_argument("--latent-temperature", type=float, default=None)
    parser.add_argument("--token", help="Hugging Face token for private/gated repos.")
    parser.add_argument("--offline", action="store_true", help="Skip live HF discovery.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    inf_cfg = cfg.get("inference", {})
    audio_cfg = cfg.get("audio", {})

    device = resolve_device(args.device or inf_cfg.get("device", "auto"))
    # --model may be a local .ts path (use directly) or a registry selector (download).
    local = Path(args.model)
    if local.is_file() and local.suffix == ".ts":
        model_path = local
    else:
        entry = resolve_entry(args.model, cfg, token=args.token, allow_network=not args.offline)
        model_path = fetch_model(
            entry.repo, entry.filename, cache_dir=cfg.get("cache_dir", "models"), token=args.token
        )
    model = load_rave(
        model_path,
        device=device,
        default_sample_rate=audio_cfg.get("default_sample_rate", 48000),
    )

    controls = LatentControls(
        bias=args.latent_bias if args.latent_bias is not None else inf_cfg.get("latent_bias", 0.0),
        scale=(
            args.latent_scale if args.latent_scale is not None else inf_cfg.get("latent_scale", 1.0)
        ),
        temperature=(
            args.latent_temperature
            if args.latent_temperature is not None
            else inf_cfg.get("latent_temperature", 0.0)
        ),
    )

    out = transfer_file(
        model,
        args.input,
        args.output,
        controls=controls,
        chunk_seconds=audio_cfg.get("chunk_seconds", 0),
        overlap_seconds=audio_cfg.get("chunk_overlap_seconds", 0.0),
    )
    print(f"Wrote transferred audio -> {out}  (sr={model.sample_rate}, device={device})")
    return 0
