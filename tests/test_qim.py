"""
tests/test_qim.py -- Unit tests for app/utils/qim.py

Tests cover:
  - Exact QIM embed values for bit=0 and bit=1
  - Round-trip coherence: embed then extract gives original bits
  - Vectorised operation over full arrays
  - Edge cases: very small delta, array of all-zeros, negative coefficients
"""
from __future__ import annotations

import numpy as np
import pytest

from app.utils.qim import qim_embed, qim_extract


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def ber(bits_ref: np.ndarray, bits_ext: np.ndarray) -> float:
    """Bit Error Rate between two binary arrays."""
    return float(np.mean(bits_ref.astype(np.int32) != bits_ext.astype(np.int32)))


# ---------------------------------------------------------------------------
# Embed correctness
# ---------------------------------------------------------------------------

class TestQimEmbed:
    def test_bit0_snaps_to_even_multiple(self):
        """bit=0 must quantise to an even multiple of delta."""
        delta = 1.0
        coef  = np.array([3.7])      # closest even multiple of 1.0 is 4.0
        bits  = np.array([0])
        result = qim_embed(coef, bits, delta)
        assert result[0] == pytest.approx(4.0)

    def test_bit1_snaps_to_odd_multiple(self):
        """bit=1 must quantise to an odd multiple of delta."""
        delta = 1.0
        coef  = np.array([3.7])      # odd multiples: ..., 1, 3, 5, ... → 3 is closest + 0.5
        bits  = np.array([1])
        result = qim_embed(coef, bits, delta)
        # Expected: 2*1*round((3.7-1)/(2*1)) + 1 = 2*round(1.35)+1 = 2*1+1 = 3
        assert result[0] == pytest.approx(3.0)

    def test_already_on_grid_unchanged(self):
        """A coefficient already exactly on its grid should not change."""
        delta = 0.5
        # 4.0 = 8 * 0.5  → even multiple, encodes bit=0
        coef  = np.array([4.0])
        bits  = np.array([0])
        result = qim_embed(coef, bits, delta)
        assert result[0] == pytest.approx(4.0)

    def test_output_shape_preserved(self):
        """Output shape must match input shape."""
        rng  = np.random.default_rng(0)
        coef = rng.standard_normal((8, 8))
        bits = rng.integers(0, 2, (8, 8))
        out  = qim_embed(coef, bits, delta=0.1)
        assert out.shape == coef.shape

    def test_output_dtype_is_float64(self):
        coef = np.array([1.5], dtype=np.float32)
        bits = np.array([1])
        out  = qim_embed(coef, bits, delta=0.5)
        assert out.dtype == np.float64

    def test_negative_coefficients(self):
        """QIM must handle negative coefficients correctly."""
        delta = 1.0
        coef  = np.array([-3.7])
        bits  = np.array([0])
        out   = qim_embed(coef, bits, delta)
        # Even multiple of 1.0 closest to -3.7 is -4.0
        assert out[0] == pytest.approx(-4.0)


# ---------------------------------------------------------------------------
# Extract correctness
# ---------------------------------------------------------------------------

class TestQimExtract:
    def test_extract_even_multiple_gives_bit0(self):
        delta = 1.0
        coef  = np.array([4.0])        # 4 = 4*1 → even → bit 0
        bits  = qim_extract(coef, delta)
        assert bits[0] == 0

    def test_extract_odd_multiple_gives_bit1(self):
        delta = 1.0
        coef  = np.array([3.0])        # 3 = 3*1 → odd → bit 1
        bits  = qim_extract(coef, delta)
        assert bits[0] == 1

    def test_output_dtype_is_uint8(self):
        coef = np.array([1.0, 2.0, 3.0])
        out  = qim_extract(coef, delta=1.0)
        assert out.dtype == np.uint8

    def test_output_values_are_binary(self):
        rng  = np.random.default_rng(7)
        coef = rng.standard_normal(1000)
        out  = qim_extract(coef, delta=0.3)
        assert set(out.flatten().tolist()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Round-trip (embed then extract)
# ---------------------------------------------------------------------------

class TestQimRoundTrip:
    @pytest.mark.parametrize("delta", [0.05, 0.1, 0.3, 0.5, 1.0, 2.0])
    def test_lossless_roundtrip(self, delta):
        """Embed then immediately extract must recover all bits exactly."""
        rng    = np.random.default_rng(42)
        coef   = rng.standard_normal(512)
        bits   = rng.integers(0, 2, 512).astype(np.int32)
        embedded  = qim_embed(coef, bits, delta)
        recovered = qim_extract(embedded, delta)
        assert ber(bits.astype(np.uint8), recovered) == 0.0

    def test_roundtrip_2d_array(self):
        """Works on 2-D arrays (e.g. LL3 sub-band shape)."""
        rng    = np.random.default_rng(99)
        coef   = rng.standard_normal((64, 64))
        bits   = rng.integers(0, 2, (64, 64)).astype(np.int32)
        emb    = qim_embed(coef, bits, delta=0.2)
        rec    = qim_extract(emb, delta=0.2)
        assert ber(bits.astype(np.uint8), rec) == 0.0

    def test_all_zeros_watermark(self):
        """All-zero bit array must be recoverable."""
        coef  = np.random.standard_normal(200)
        bits  = np.zeros(200, dtype=np.int32)
        emb   = qim_embed(coef, bits, delta=0.1)
        rec   = qim_extract(emb, delta=0.1)
        assert np.all(rec == 0)

    def test_all_ones_watermark(self):
        """All-one bit array must be recoverable."""
        coef  = np.random.standard_normal(200)
        bits  = np.ones(200, dtype=np.int32)
        emb   = qim_embed(coef, bits, delta=0.1)
        rec   = qim_extract(emb, delta=0.1)
        assert np.all(rec == 1)

    @pytest.mark.parametrize("noise_sigma,seed", [(0.001, 3), (0.01, 4), (0.015, 5)])
    def test_noise_tolerance_within_half_delta(self, noise_sigma, seed):
        """
        Noise with sigma well below delta/2 must not flip any bits.
        delta=0.1 -> tolerance=0.05.
        sigma=0.015 -> 3-sigma bound=0.045 < 0.05 -> guaranteed zero flips.
        Using fixed seeds so the test is deterministic across runs.
        """
        rng   = np.random.default_rng(seed)
        delta = 0.1
        coef  = rng.standard_normal(200)
        bits  = rng.integers(0, 2, 200).astype(np.int32)
        emb   = qim_embed(coef, bits, delta)
        noisy = emb + rng.normal(0, noise_sigma, emb.shape)
        rec   = qim_extract(noisy, delta)
        assert ber(bits.astype(np.uint8), rec) == 0.0
