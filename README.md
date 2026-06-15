# 🎛️ Timbre Transfer

Audio **timbre transfer** built around SOTA non-LLM architectures, prioritising
**RAVE** (ACIDS-IRCAM) with **pre-trained TorchScript models** from the Hugging
Face Hub so inference works out of the box — with a clear path to fine-tuning on
your own data. A **DDSP** front-end is included as an optional/experimental
module.

> **How RAVE timbre transfer works:** the source audio is *encoded* into RAVE's
> latent space and *decoded* through a model trained on the **target** timbre.
> Pick a different pre-trained model → get a different instrument/voice.

---

## Features

- ⬇️ **Pre-trained model integration** — discover & download `.ts` checkpoints
  from Hugging Face (defaults to `Intelligent-Instruments-Lab/rave-models`).
- 🔁 **Inference pipeline** — load a `.ts`, `encode → (latent controls) → decode`,
  with chunked processing for long inputs.
- 🖥️ **Gradio app** — upload audio, pick a timbre, tweak latent controls, play back.
- 🎚️ **Fine-tuning** — preprocess custom datasets + continue training from a
  pre-trained checkpoint.
- ☁️ **Colab notebook** — `pip install`, HF download, Drive mount, inference + training.
- ✅ **CI** — ruff + black + pytest (fully offline tests, no downloads).

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# editable install exposes the `timbre-download` / `timbre-infer` console scripts:
pip install -e .
```

> Install the torch/torchaudio build that matches your CUDA setup. For
> fine-tuning RAVE you also need: `pip install acids-rave`.

---

## Quickstart (inference)

```bash
# 1) See which pre-trained models are available
python scripts/download_models.py --list

# 2) Run timbre transfer on a mono clip
python scripts/run_inference.py --input data/my_voice.wav --model guitar --output out.wav

# 3) Or launch the interactive UI
python app/gradio_app.py            # local  -> http://127.0.0.1:7860
python app/gradio_app.py --share    # public share link (useful from Colab)
```

Latent controls (only for models exposing `encode`/`decode`):

```bash
python scripts/run_inference.py --input in.wav --model organ --output out.wav \
    --latent-scale 1.2 --latent-temperature 0.1
```

---

## Fine-tuning

> ⚠️ **A RAVE `.ts` file is inference-only and cannot be fine-tuned** (it is a
> frozen TorchScript graph). Real RAVE fine-tuning uses the `acids-rave` package
> on a Lightning `.ckpt`. This repo wraps that workflow.

**Recommended (real) path:**

```bash
pip install acids-rave

# 1) Preprocess your audio into RAVE's dataset format
python scripts/preprocess_dataset.py --input ./data/my_instrument --output ./db --use-rave

# 2) Continue training from a pre-trained checkpoint (.ckpt, not .ts)
python scripts/finetune.py rave --db ./db --name my_run --config v2 --ckpt pretrained.ckpt
```

**Illustrative path** (teaching scaffold — a tiny trainable autoencoder, *not*
RAVE-quality, for understanding the loop):

```bash
python scripts/preprocess_dataset.py --input ./data/my_instrument --output ./dataset
python scripts/finetune.py demo --manifest ./dataset/manifest.json --epochs 5
```

---

## DDSP (optional / experimental)

```bash
pip install -e ".[ddsp]"   # adds torchcrepe + librosa
```

`timbre_transfer.features.ddsp_features` extracts **F0 (CREPE)** + **loudness**
control signals, and `timbre_transfer.models.ddsp_model` defines a decoder
skeleton. Note: robust pre-trained *PyTorch* DDSP decoders are scarce (canonical
checkpoints are TensorFlow), so this is scaffolding for training/porting rather
than a turnkey pre-trained path.

---

## Project layout

```
src/timbre_transfer/   core package (models, audio_io, inference, training, utils)
app/gradio_app.py      interactive UI
scripts/               CLI entrypoints (download / inference / preprocess / finetune)
configs/default.yaml   model registry + sample rates + inference defaults
notebooks/             Google Colab notebook
tests/                 offline pytest suite (dummy TorchScript model)
```

---

## Development

```bash
pip install -r requirements-dev.txt
ruff check . && black --check . && pytest
```

CI runs the same checks on Python 3.10 & 3.11 (see `.github/workflows/ci.yml`).
Tests are fully offline — they script a tiny dummy RAVE model rather than
downloading weights.

---

## Notes & caveats

- Best results come from **monophonic** source material matching the target
  model's training domain.
- Exact Hub filenames drift over time; the registry discovers `.ts` files live
  and only falls back to the curated list in `configs/default.yaml`.
- For gated/private repos pass `--token` (or set `HF_TOKEN`).

## License

MIT.
