"""
tests/conftest.py -- Shared pytest fixtures for Advanced Watermarking Studio.

Fixtures provide deterministic synthetic images so every test is
reproducible and independent of real photos.
"""
from __future__ import annotations

import numpy as np
import pytest


# ── Image helpers ─────────────────────────────────────────────────────────────

def _make_bgr(h: int = 256, w: int = 256, seed: int = 0) -> np.ndarray:
    """Random-noise BGR uint8 image (worst-case for watermarking)."""
    rng = np.random.default_rng(seed)
    return rng.integers(30, 220, (h, w, 3), dtype=np.uint8)


def _make_wm(h: int = 64, w: int = 64) -> np.ndarray:
    """Binary watermark: white square on black background."""
    wm = np.zeros((h, w), dtype=np.uint8)
    wm[h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 255
    return wm


def _png_roundtrip(bgr: np.ndarray) -> np.ndarray:
    """Simulate saving to PNG and reloading (uint8 quantisation round-trip)."""
    import cv2, tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    cv2.imwrite(path, bgr)
    reloaded = cv2.imread(path)
    os.unlink(path)
    return reloaded


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def cover_256():
    return _make_bgr(256, 256, seed=1)


@pytest.fixture(scope="session")
def cover_512():
    return _make_bgr(512, 512, seed=2)


@pytest.fixture(scope="session")
def wm_64():
    return _make_wm(64, 64)


@pytest.fixture(scope="session")
def wm_32():
    return _make_wm(32, 32)


@pytest.fixture
def png_roundtrip():
    return _png_roundtrip
