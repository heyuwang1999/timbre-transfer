"""Tests for the model registry (offline -- no network)."""

from __future__ import annotations

import pytest

from timbre_transfer.models.registry import (
    ModelEntry,
    list_available_models,
    resolve_entry,
)

CONFIG = {
    "models": {
        "repos": ["example/repo"],
        "fallback": [
            {"repo": "example/repo", "filename": "guitar_b2048.ts", "label": "guitar"},
            {"repo": "example/repo", "filename": "organ_b2048.ts", "label": "organ"},
        ],
    }
}


def test_fallback_listing_offline():
    models = list_available_models(CONFIG, allow_network=False)
    labels = {m.label for m in models}
    assert labels == {"guitar", "organ"}
    assert all(isinstance(m, ModelEntry) for m in models)


def test_model_entry_key_format():
    entry = ModelEntry(repo="r", filename="guitar_b2048.ts", label="guitar")
    assert entry.key == "guitar (guitar_b2048.ts)"


def test_resolve_by_label():
    entry = resolve_entry("organ", CONFIG, allow_network=False)
    assert entry.filename == "organ_b2048.ts"


def test_resolve_by_filename():
    entry = resolve_entry("guitar_b2048.ts", CONFIG, allow_network=False)
    assert entry.label == "guitar"


def test_resolve_by_key():
    entry = resolve_entry("guitar (guitar_b2048.ts)", CONFIG, allow_network=False)
    assert entry.filename == "guitar_b2048.ts"


def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        resolve_entry("tuba", CONFIG, allow_network=False)


def test_dedup_prefers_discovered(monkeypatch):
    import timbre_transfer.models.registry as reg

    discovered = [ModelEntry(repo="example/repo", filename="guitar_b2048.ts", label="GUITAR")]
    monkeypatch.setattr(reg, "_discover_repo", lambda repo, token: discovered)
    models = list_available_models(CONFIG, allow_network=True)
    # Discovered entry wins the dedup on (repo, filename).
    guitars = [m for m in models if m.filename == "guitar_b2048.ts"]
    assert len(guitars) == 1
    assert guitars[0].label == "GUITAR"
