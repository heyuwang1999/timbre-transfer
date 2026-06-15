"""Discover and resolve pre-trained RAVE models hosted on the Hugging Face Hub.

Filenames on the Hub drift over time, so we prefer *live discovery* of ``.ts``
files via the HF API and only fall back to a curated list (from the config) when
the network is unavailable. This keeps the UI dropdown populated offline while
staying correct online.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import load_config


@dataclass(frozen=True)
class ModelEntry:
    """A selectable pre-trained model."""

    repo: str
    filename: str
    label: str

    @property
    def key(self) -> str:
        """Stable identifier used by CLIs and the UI dropdown."""
        return f"{self.label} ({self.filename})"


def _label_from_filename(filename: str) -> str:
    """Derive a friendly label from a RAVE filename (best effort)."""
    stem = filename.rsplit("/", 1)[-1]
    stem = stem[:-3] if stem.endswith(".ts") else stem
    # RAVE filenames look like "<name>_<tags>_b2048_r48000_z16"; keep the head.
    return stem.split("_")[0] or stem


def _discover_repo(repo_id: str, token: str | None) -> list[ModelEntry]:
    """List ``.ts`` files in a repo via the HF API. Returns [] on any failure."""
    try:
        from huggingface_hub import HfApi

        files = HfApi().list_repo_files(repo_id=repo_id, token=token)
    except Exception:
        return []
    entries: list[ModelEntry] = []
    for f in files:
        if f.endswith(".ts"):
            entries.append(ModelEntry(repo=repo_id, filename=f, label=_label_from_filename(f)))
    return entries


def list_available_models(
    config: dict | None = None,
    token: str | None = None,
    allow_network: bool = True,
) -> list[ModelEntry]:
    """Return the list of selectable models (discovered + curated fallback).

    Args:
        config: Parsed config dict (defaults to the bundled config).
        token: Optional HF token for private/gated repos.
        allow_network: If ``False``, skip live discovery and return only the
            curated fallback (useful for tests / offline runs).
    """
    cfg = config or load_config()
    models_cfg = cfg.get("models", {})
    repos: list[str] = list(models_cfg.get("repos", []))
    fallback_cfg: list[dict] = list(models_cfg.get("fallback", []))

    discovered: list[ModelEntry] = []
    if allow_network:
        for repo in repos:
            discovered.extend(_discover_repo(repo, token))

    fallback = [
        ModelEntry(
            repo=item["repo"],
            filename=item["filename"],
            label=item.get("label", _label_from_filename(item["filename"])),
        )
        for item in fallback_cfg
    ]

    # Merge, de-duplicating on (repo, filename); discovered entries win.
    seen: set[tuple[str, str]] = set()
    merged: list[ModelEntry] = []
    for entry in [*discovered, *fallback]:
        sig = (entry.repo, entry.filename)
        if sig in seen:
            continue
        seen.add(sig)
        merged.append(entry)
    return merged


def resolve_entry(
    selector: str,
    config: dict | None = None,
    token: str | None = None,
    allow_network: bool = True,
) -> ModelEntry:
    """Resolve a user selector to a :class:`ModelEntry`.

    ``selector`` may be an exact filename, a label (e.g. ``"guitar"``), or the
    composite key from :pyattr:`ModelEntry.key`.
    """
    models = list_available_models(config=config, token=token, allow_network=allow_network)
    # Exact filename match first, then key, then label (first hit wins).
    for entry in models:
        if selector == entry.filename or selector == entry.key:
            return entry
    for entry in models:
        if selector == entry.label:
            return entry
    available = ", ".join(sorted({m.label for m in models})) or "<none>"
    raise ValueError(f"Unknown model selector {selector!r}. Available labels: {available}")
