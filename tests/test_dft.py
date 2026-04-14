"""
tests/test_dft.py -- Unit tests for DFT watermarking (non-blind + blind).

Blind DFT uses Block-DFT (8x8) + log-domain QIM — tested extensively
because it required the most engineering effort to get right.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import tempfile, os

from app.services.dft_watermark import (
    embed_dft, extract_dft,
    embed_dft_blind, extract_dft_blind,
)
from app.utils.metrics import normalized_correlation


def _make_cover(h=256, w=256, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(30, 220, (h, w, 3), dtype=np.uint8)


def _make_wm(h=64, w=64):
    wm = np.zeros((h, w), dtype=np.uint8)
    wm[h//4:3*h//4, w//4:3*w//4] = 255
    return wm


def _png_roundtrip(bgr):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    cv2.imwrite(path, bgr)
    loaded = cv2.imread(path)
    os.unlink(path)
    return loaded


# ── Non-Blind ─────────────────────────────────────────────────────────────────

class TestEmbedDFT:
    def test_output_shape(self):
        cover = _make_cover()
        out, _ = embed_dft(cover, _make_wm(), alpha=0.1)
        assert out.shape == cover.shape

    def test_output_uint8(self):
        out, _ = embed_dft(_make_cover(), _make_wm(), alpha=0.1)
        assert out.dtype == np.uint8

    def test_psnr_in_range(self):
        _, psnr = embed_dft(_make_cover(), _make_wm(), alpha=0.1)
        assert 25.0 < psnr < 65.0

    def test_higher_alpha_lower_psnr(self):
        cover = _make_cover(seed=40)
        wm    = _make_wm()
        _, p_low  = embed_dft(cover, wm, alpha=0.05)
        _, p_high = embed_dft(cover, wm, alpha=0.3)
        assert p_low > p_high

    def test_watermarked_differs_from_original(self):
        cover = _make_cover()
        out, _ = embed_dft(cover, _make_wm(), alpha=0.1)
        assert not np.array_equal(cover, out)

    def test_none_cover_raises(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            embed_dft(None, _make_wm(), alpha=0.1)

    def test_alpha_out_of_range_raises(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            embed_dft(_make_cover(), _make_wm(), alpha=2.0)

    def test_too_small_raises(self):
        from fastapi import HTTPException
        tiny = np.zeros((16, 16, 3), dtype=np.uint8)
        with pytest.raises(HTTPException):
            embed_dft(tiny, _make_wm(), alpha=0.1)


class TestExtractDFT:
    def test_nc_above_threshold(self):
        cover = _make_cover(seed=50)
        wm    = _make_wm()
        wm_img, _ = embed_dft(cover, wm, alpha=0.1)
        ext = extract_dft(cover, wm_img, alpha=0.1, watermark_size=(64, 64))
        assert normalized_correlation(wm, ext) > 0.70

    def test_output_is_2d(self):
        cover = _make_cover()
        wm    = _make_wm()
        wm_img, _ = embed_dft(cover, wm, alpha=0.1)
        ext = extract_dft(cover, wm_img, alpha=0.1)
        assert ext.ndim == 2

    def test_none_raises(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            extract_dft(None, _make_cover(), alpha=0.1)


# ── Blind (Block-DFT) ─────────────────────────────────────────────────────────

class TestEmbedDFTBlind:
    def test_returns_3_tuple(self):
        result = embed_dft_blind(_make_cover(), _make_wm(), alpha=0.1)
        assert len(result) == 3

    def test_output_shape_matches_cover(self):
        cover = _make_cover()
        out, _, _ = embed_dft_blind(cover, _make_wm(), alpha=0.1)
        assert out.shape == cover.shape

    def test_output_dtype_uint8(self):
        out, _, _ = embed_dft_blind(_make_cover(), _make_wm(), alpha=0.1)
        assert out.dtype == np.uint8

    def test_psnr_reasonable(self):
        _, psnr, _ = embed_dft_blind(_make_cover(), _make_wm(), alpha=0.1)
        assert 20.0 < psnr < 65.0

    def test_grid_dimensions_match_image(self):
        """For a 256×256 cover: grid should be 32×32 (256//8)."""
        cover = _make_cover(256, 256)
        _, _, (bh, bw) = embed_dft_blind(cover, _make_wm(), alpha=0.1)
        assert bh == 256 // 8
        assert bw == 256 // 8

    def test_none_cover_raises(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            embed_dft_blind(None, _make_wm(), alpha=0.1)

    def test_too_small_raises(self):
        from fastapi import HTTPException
        tiny = np.zeros((16, 16, 3), dtype=np.uint8)
        with pytest.raises(HTTPException):
            embed_dft_blind(tiny, _make_wm(), alpha=0.1)


class TestExtractDFTBlind:
    @pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
    def test_zero_ber_through_png(self, alpha):
        """
        Core correctness test: block-DFT blind extraction after PNG round-trip
        must have 0% BER.  This was the bug (global FFT gave pure noise).
        """
        cover = _make_cover(256, 256, seed=60)
        wm    = _make_wm(32, 32)
        watermarked, _, (bh, bw) = embed_dft_blind(cover, wm, alpha=alpha)
        reloaded  = _png_roundtrip(watermarked)
        extracted = extract_dft_blind(reloaded, alpha=alpha,
                                      watermark_size=(bh, bw))
        ref      = cv2.resize(wm, (bw, bh), interpolation=cv2.INTER_AREA)
        bits_ref = (ref >= 128).astype(np.uint8)
        bits_ext = (extracted >= 128).astype(np.uint8)
        ber = 1.0 - float(np.mean(bits_ref == bits_ext))
        assert ber == pytest.approx(0.0), f"BER={ber:.4f} at alpha={alpha}"

    @pytest.mark.parametrize("size", [(128, 128), (256, 256), (512, 512)])
    def test_zero_ber_various_cover_sizes(self, size):
        """Grid must be image-size–independent as long as divisible by 8."""
        cover = _make_cover(*size, seed=61)
        wm    = _make_wm(16, 16)
        watermarked, _, (bh, bw) = embed_dft_blind(cover, wm, alpha=0.1)
        reloaded  = _png_roundtrip(watermarked)
        extracted = extract_dft_blind(reloaded, alpha=0.1,
                                      watermark_size=(bh, bw))
        ref = cv2.resize(wm, (bw, bh), interpolation=cv2.INTER_AREA)
        bits_ref = (ref >= 128).astype(np.uint8)
        bits_ext = (extracted >= 128).astype(np.uint8)
        ber = 1.0 - float(np.mean(bits_ref == bits_ext))
        assert ber == pytest.approx(0.0), f"BER={ber:.4f} for size={size}"

    def test_output_is_binary(self):
        cover = _make_cover()
        wm    = _make_wm()
        wm_img, _, (bh, bw) = embed_dft_blind(cover, wm, alpha=0.1)
        ext = extract_dft_blind(wm_img, alpha=0.1, watermark_size=(bh, bw))
        assert set(ext.flatten().tolist()).issubset({0, 255})

    def test_output_shape_matches_watermark_size(self):
        cover = _make_cover(256, 256)
        wm    = _make_wm(100, 80)
        wm_img, _, (bh, bw) = embed_dft_blind(cover, wm, alpha=0.1)
        # Request original watermark dimensions back
        ext = extract_dft_blind(wm_img, alpha=0.1, watermark_size=(100, 80))
        assert ext.shape == (100, 80)

    def test_wrong_alpha_increases_ber(self):
        """Secret key (alpha) must match — wrong key degrades extraction."""
        cover = _make_cover(256, 256, seed=62)
        wm    = _make_wm(32, 32)
        wm_img, _, (bh, bw) = embed_dft_blind(cover, wm, alpha=0.1)
        reloaded = _png_roundtrip(wm_img)

        ext_ok  = extract_dft_blind(reloaded, alpha=0.1, watermark_size=(bh, bw))
        ext_bad = extract_dft_blind(reloaded, alpha=0.7, watermark_size=(bh, bw))

        ref = cv2.resize(wm, (bw, bh), interpolation=cv2.INTER_AREA)
        bits_ref = (ref >= 128).astype(np.uint8)
        ber_ok  = 1.0 - float(np.mean(bits_ref == (ext_ok  >= 128)))
        ber_bad = 1.0 - float(np.mean(bits_ref == (ext_bad >= 128)))
        assert ber_bad > ber_ok

    def test_no_original_needed(self):
        """extract_dft_blind must not require an original image argument."""
        import inspect
        sig = inspect.signature(extract_dft_blind)
        params = list(sig.parameters.keys())
        assert "original_bgr" not in params

    def test_none_raises(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            extract_dft_blind(None, alpha=0.1)
