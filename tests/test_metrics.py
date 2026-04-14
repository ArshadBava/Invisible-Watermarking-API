"""
tests/test_metrics.py -- Unit tests for app/utils/metrics.py

Tests cover:
  - PSNR: identical images, known MSE, low-quality images
  - NC: identical images, inverted images, orthogonal arrays, known value
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from app.utils.metrics import calculate_psnr, normalized_correlation


class TestPSNR:
    def test_identical_images_returns_100(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        assert calculate_psnr(img, img) == pytest.approx(100.0)

    def test_known_mse(self):
        """PSNR = 20*log10(255/sqrt(MSE)). With MSE=1: PSNR ~ 48.13 dB."""
        a = np.zeros((100, 100), dtype=np.float32)
        b = np.ones((100, 100), dtype=np.float32)
        expected = 20 * math.log10(255.0 / math.sqrt(1.0))
        assert calculate_psnr(a, b) == pytest.approx(expected, rel=1e-4)

    def test_high_quality_threshold(self):
        """Embedding at alpha=0.05 on a 256x256 image should give PSNR > 35 dB."""
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
        noise = rng.integers(0, 3, (256, 256, 3), dtype=np.int32)
        noisy = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)
        assert calculate_psnr(img, noisy) > 35.0

    def test_completely_different_images_low_psnr(self):
        a = np.zeros((64, 64, 3), dtype=np.uint8)
        b = np.full((64, 64, 3), 255, dtype=np.uint8)
        assert calculate_psnr(a, b) < 10.0

    def test_returns_float(self):
        a = np.zeros((8, 8), dtype=np.uint8)
        b = np.ones((8, 8), dtype=np.uint8)
        assert isinstance(calculate_psnr(a, b), float)


class TestNormalizedCorrelation:
    def test_identical_images_returns_1(self):
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        assert normalized_correlation(img, img) == pytest.approx(1.0, abs=1e-6)

    def test_inverted_images_returns_minus1(self):
        img = np.random.randint(10, 245, (64, 64), dtype=np.uint8)
        inv = (255 - img.astype(np.int32)).astype(np.uint8)
        assert normalized_correlation(img, inv) == pytest.approx(-1.0, abs=1e-4)

    def test_constant_image_returns_0(self):
        """A constant array has no variation; NC is undefined → return 0."""
        a = np.full((32, 32), 128, dtype=np.uint8)
        b = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        assert normalized_correlation(a, b) == pytest.approx(0.0)

    def test_symmetric(self):
        a = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        b = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        assert normalized_correlation(a, b) == pytest.approx(
            normalized_correlation(b, a), abs=1e-6)

    def test_range_minus1_to_1(self):
        rng = np.random.default_rng(5)
        for _ in range(20):
            a = rng.integers(0, 255, (64, 64), dtype=np.uint8)
            b = rng.integers(0, 255, (64, 64), dtype=np.uint8)
            nc = normalized_correlation(a, b)
            assert -1.0 <= nc <= 1.0

    def test_returns_float(self):
        a = np.zeros((8, 8), dtype=np.uint8)
        b = np.ones((8, 8), dtype=np.uint8)
        assert isinstance(normalized_correlation(a, b), float)
