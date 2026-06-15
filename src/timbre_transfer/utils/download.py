"""Download pre-trained model checkpoints from the Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import hf_hub_download


def fetch_model(
    repo_id: str,
    filename: str,
    cache_dir: str | Path = "models",
    revision: str | None = None,
    token: str | None = None,
) -> Path:
    """Download a single model file from the Hub and return its local path.

    The file is materialised inside ``cache_dir`` (``local_dir``) so it lives in
    the project tree (gitignored) rather than the global HF cache -- handy for
    Colab/Drive workflows where you want artifacts in a known location.

    Args:
        repo_id: Hugging Face repo id, e.g. ``"Intelligent-Instruments-Lab/rave-models"``.
        filename: File within the repo, e.g. ``"guitar_iil_b2048_r48000_z16.ts"``.
        cache_dir: Local directory to download into.
        revision: Optional git revision / branch / tag.
        token: Optional HF auth token (for gated/private repos).

    Returns:
        Path to the downloaded file on local disk.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(cache_dir),
        revision=revision,
        token=token,
    )
    return Path(local_path)
