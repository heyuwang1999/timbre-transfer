"""Gradio interface for RAVE timbre transfer.

Upload a (preferably monophonic) source clip, pick a pre-trained timbre, tweak
optional latent controls, and play back the result. Loaded models are cached
in-process so switching timbres is fast.

Run:
    python app/gradio_app.py            # local
    python app/gradio_app.py --share    # public link (e.g. from Colab)
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Allow running directly from a source checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gradio as gr  # noqa: E402

from timbre_transfer.config import load_config, resolve_device  # noqa: E402
from timbre_transfer.inference.transfer import LatentControls, transfer_file  # noqa: E402
from timbre_transfer.models.rave_model import load_rave  # noqa: E402
from timbre_transfer.models.registry import list_available_models, resolve_entry  # noqa: E402
from timbre_transfer.utils.download import fetch_model  # noqa: E402

CONFIG = load_config()
DEVICE = resolve_device(CONFIG.get("inference", {}).get("device", "auto"))
CACHE_DIR = CONFIG.get("cache_dir", "models")
_MODEL_CACHE: dict[str, object] = {}


def _get_model(selector: str):
    """Download (if needed) and load a model, caching the loaded module."""
    entry = resolve_entry(selector, CONFIG)
    if entry.filename in _MODEL_CACHE:
        return _MODEL_CACHE[entry.filename]
    path = fetch_model(entry.repo, entry.filename, cache_dir=CACHE_DIR)
    model = load_rave(
        path,
        device=DEVICE,
        default_sample_rate=CONFIG.get("audio", {}).get("default_sample_rate", 48000),
    )
    _MODEL_CACHE[entry.filename] = model
    return model


def run_transfer(audio_path, model_selector, bias, scale, temperature):
    """Gradio callback: returns the output audio file path for playback."""
    if not audio_path:
        raise gr.Error("Please upload an audio file first.")
    model = _get_model(model_selector)
    controls = LatentControls(bias=float(bias), scale=float(scale), temperature=float(temperature))
    out_path = Path(tempfile.gettempdir()) / "timbre_transfer_output.wav"
    transfer_file(model, audio_path, out_path, controls=controls)
    return str(out_path)


def build_demo() -> gr.Blocks:
    models = list_available_models(CONFIG)
    choices = [m.key for m in models]
    default = choices[0] if choices else None

    with gr.Blocks(title="Timbre Transfer (RAVE)") as demo:
        gr.Markdown(
            "# 🎛️ Timbre Transfer (RAVE)\n"
            "Upload a **monophonic** clip, choose a target timbre, and play the result. "
            "Latent controls only apply to models that expose `encode`/`decode`."
        )
        with gr.Row():
            with gr.Column():
                src = gr.Audio(type="filepath", label="Source audio")
                model_dd = gr.Dropdown(choices=choices, value=default, label="Target model")
                with gr.Accordion("Latent controls", open=False):
                    bias = gr.Slider(-3.0, 3.0, value=0.0, step=0.05, label="bias")
                    scale = gr.Slider(0.0, 3.0, value=1.0, step=0.05, label="scale")
                    temp = gr.Slider(0.0, 2.0, value=0.0, step=0.05, label="temperature")
                go = gr.Button("Transfer timbre", variant="primary")
            with gr.Column():
                out = gr.Audio(label="Output", type="filepath")

        go.click(run_transfer, inputs=[src, model_dd, bias, scale, temp], outputs=out)
    return demo


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch the timbre-transfer Gradio app.")
    parser.add_argument("--share", action="store_true", help="Create a public share link.")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args(argv)

    build_demo().launch(
        share=args.share, server_name=args.server_name, server_port=args.server_port
    )


if __name__ == "__main__":
    main()
