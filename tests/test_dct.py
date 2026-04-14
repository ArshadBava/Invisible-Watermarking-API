"""
tests/test_dct.py -- Unit tests for DCT watermarking (non-blind + blind).
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import tempfile, os

from app.services.dct_watermark import (
    embed_dct, extract_dct,
    embed_dct_blind, extract_dct_blind,
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

class TestEmbedDCT:
    def test_output_shape(self):
        cover = _make_cover()
        out, _ = embed_dct(cover, _make_wm(), alpha=0.1)
        assert out.shape == cover.shape

    def test_output_uint8(self):
        out, _ = embed_dct(_make_cover(), _make_wm(), alpha=0.1)
        assert out.dtype == np.uint8

    def test_psnr_in_range(self):
        _, psnr = embed_dct(_make_cover(), _make_wm(), alpha=0.1)
        assert 25.0 < psnr < 60.0

    def test_higher_alpha_lower_psnr(self):
        cover = _make_cover(seed=11)
        wm    = _make_wm()
        _, p_low  = embed_dct(cover, wm, alpha=0.05)
        _, p_high = embed_dct(cover, wm, alpha=0.3)
        assert p_low > p_high

    def test_watermarked_differs_from_original(self):
        cover = _make_cover()
        out, _ = embed_dct(cover, _make_wm(), alpha=0.1)
        assert not np.array_equal(cover, out)

    def test_none_cover_raises(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            embed_dct(None, _make_wm(), alpha=0.1)

    def test_too_small_raises(self):
        from fastapi import HTTPException
        tiny = np.zeros((4, 4, 3), dtype=np.uint8)
        with pytest.raises(HTTPException):
            embed_dct(tiny, _make_wm(), alpha=0.1)


class TestExtractDCT:
    def test_nc_above_threshold(self):
        cover = _make_cover(seed=20)
        wm    = _make_wm()
        wm_img, _ = embed_dct(cover, wm, alpha=0.1)
        ext = extract_dct(cover, wm_img, alpha=0.1, watermark_size=(64, 64))
        assert normalized_correlation(wm, ext) > 0.70

    def test_output_grayscale_2d(self):
        cover = _make_cover()
        wm    = _make_wm()
        wm_img, _ = embed_dct(cover, wm, alpha=0.1)
        ext = extract_dct(cover, wm_img, alpha=0.1)
        assert ext.ndim == 2

    def test_wrong_alpha_degrades_nc(self):
        cover = _make_cover(seed=21)
        wm    = _make_wm()
        wm_img, _ = embed_dct(cover, wm, alpha=0.1)
        ext_right = extract_dct(cover, wm_img, alpha=0.1, watermark_size=(64, 64))
        ext_wrong = extract_dct(cover, wm_img, alpha=0.5, watermark_size=(64, 64))
        assert normalized_correlation(wm, ext_right) > normalized_correlation(wm, ext_wrong)

    def test_none_raises(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            extract_dct(None, _make_cover(), alpha=0.1)


# ── Blind ─────────────────────────────────────────────────────────────────────

class TestEmbedDCTBlind:
    def test_returns_3_tuple(self):
        result = embed_dct_blind(_make_cover(), _make_wm(), alpha=0.1)
        assert len(result) == 3

    def test_output_shape(self):
        cover = _make_cover()
        out, _, _ = embed_dct_blind(cover, _make_wm(), alpha=0.1)
        assert out.shape == cover.shape

    def test_psnr_reasonable(self):
        _, psnr, _ = embed_dct_blind(_make_cover(), _make_wm(), alpha=0.1)
        assert 20.0 < psnr < 60.0

    def test_grid_size_positive(self):
        _, _, (bh, bw) = embed_dct_blind(_make_cover(256, 256), _make_wm(), alpha=0.1)
        assert bh > 0 and bw > 0

    def test_none_cover_raises(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            embed_dct_blind(None, _make_wm(), alpha=0.1)


class TestExtractDCTBlind:
    @pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
    def test_zero_ber_through_png(self, alpha):
        cover = _make_cover(256, 256, seed=30)
        wm    = _make_wm(32, 32)
        watermarked, _, (bh, bw) = embed_dct_blind(cover, wm, alpha=alpha)
        reloaded  = _png_roundtrip(watermarked)
        extracted = extract_dct_blind(reloaded, alpha=alpha,
                                      watermark_size=(bh, bw))
        ref      = cv2.resize(wm, (bw, bh), interpolation=cv2.INTER_AREA)
        bits_ref = (ref >= 128).astype(np.uint8)
        bits_ext = (extracted >= 128).astype(np.uint8)
        ber = 1.0 - float(np.mean(bits_ref == bits_ext))
        assert ber == pytest.approx(0.0), f"BER={ber:.4f} at alpha={alpha}"

    def test_output_binary(self):
        cover = _make_cover()
        wm    = _make_wm()
        wm_img, _, (bh, bw) = embed_dct_blind(cover, wm, alpha=0.1)
        ext = extract_dct_blind(wm_img, alpha=0.1, watermark_size=(bh, bw))
        assert set(ext.flatten().tolist()).issubset({0, 255})

    def test_wrong_alpha_increases_ber(self):
        cover = _make_cover(256, 256, seed=31)
        wm    = _make_wm(32, 32)
        wm_img, _, (bh, bw) = embed_dct_blind(cover, wm, alpha=0.1)
        reloaded = _png_roundtrip(wm_img)
        ext_ok  = extract_dct_blind(reloaded, alpha=0.1, watermark_size=(bh, bw))
        ext_bad = extract_dct_blind(reloaded, alpha=0.9, watermark_size=(bh, bw))
        ref = cv2.resize(wm, (bw, bh), interpolation=cv2.INTER_AREA)
        bits_ref = (ref >= 128).astype(np.uint8)
        ber_ok  = 1.0 - float(np.mean(bits_ref == (ext_ok  >= 128)))
        ber_bad = 1.0 - float(np.mean(bits_ref == (ext_bad >= 128)))
        assert ber_bad > ber_ok

    def test_none_raises(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            extract_dct_blind(None, alpha=0.1)
