"""
tests/test_dwt.py -- Unit tests for DWT watermarking (non-blind + blind).

Covers:
  - embed_dwt: output shape, PSNR range, watermark is present
  - extract_dwt: reconstructs correct watermark (NC > 0.7)
  - embed_dwt_blind: output shape, PSNR, grid size returned
  - extract_dwt_blind: 0% BER through PNG round-trip
  - Input validation: None images, too-small images
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
import tempfile, os

from app.services.dwt_watermark import (
    embed_dwt, extract_dwt,
    embed_dwt_blind, extract_dwt_blind,
)
from app.utils.metrics import normalized_correlation


# ── Helpers ───────────────────────────────────────────────────────────────────

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


# ── Non-Blind Tests ───────────────────────────────────────────────────────────

class TestEmbedDWT:
    def test_output_same_shape_as_cover(self):
        cover = _make_cover()
        wm    = _make_wm()
        out, psnr = embed_dwt(cover, wm, alpha=0.1)
        assert out.shape == cover.shape

    def test_output_dtype_uint8(self):
        cover = _make_cover()
        wm    = _make_wm()
        out, _ = embed_dwt(cover, wm, alpha=0.1)
        assert out.dtype == np.uint8

    def test_psnr_in_acceptable_range(self):
        cover = _make_cover()
        wm    = _make_wm()
        _, psnr = embed_dwt(cover, wm, alpha=0.1)
        assert 25.0 < psnr < 60.0

    def test_higher_alpha_lowers_psnr(self):
        cover = _make_cover(seed=5)
        wm    = _make_wm()
        _, psnr_low  = embed_dwt(cover, wm, alpha=0.05)
        _, psnr_high = embed_dwt(cover, wm, alpha=0.3)
        assert psnr_low > psnr_high

    def test_watermarked_differs_from_original(self):
        cover = _make_cover()
        wm    = _make_wm()
        out, _ = embed_dwt(cover, wm, alpha=0.1)
        assert not np.array_equal(cover, out)

    def test_none_cover_raises(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            embed_dwt(None, _make_wm(), alpha=0.1)

    def test_none_watermark_raises(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            embed_dwt(_make_cover(), None, alpha=0.1)

    def test_too_small_image_raises(self):
        from fastapi import HTTPException
        tiny = np.zeros((16, 16, 3), dtype=np.uint8)
        with pytest.raises(HTTPException):
            embed_dwt(tiny, _make_wm(), alpha=0.1)


class TestExtractDWT:
    def test_nc_above_threshold(self):
        """Extracted watermark should correlate strongly with original."""
        cover = _make_cover()
        wm    = _make_wm()
        watermarked, _ = embed_dwt(cover, wm, alpha=0.1)
        extracted = extract_dwt(cover, watermarked, alpha=0.1)
        # Resize to wm size for NC comparison
        extracted_rs = cv2.resize(extracted, (64, 64), interpolation=cv2.INTER_NEAREST)
        nc = normalized_correlation(wm, extracted_rs)
        assert nc > 0.70

    def test_output_grayscale(self):
        cover = _make_cover()
        wm    = _make_wm()
        watermarked, _ = embed_dwt(cover, wm, alpha=0.1)
        extracted = extract_dwt(cover, watermarked, alpha=0.1)
        assert extracted.ndim == 2

    def test_wrong_alpha_gives_low_nc(self):
        """Using a different alpha at extraction should degrade quality."""
        cover = _make_cover(seed=10)
        wm    = _make_wm()
        watermarked, _ = embed_dwt(cover, wm, alpha=0.1)
        extracted_correct = extract_dwt(cover, watermarked, alpha=0.1)
        extracted_wrong   = extract_dwt(cover, watermarked, alpha=0.4)
        # Resize both to wm dims for fair comparison
        rs = lambda x: cv2.resize(x, (64, 64), interpolation=cv2.INTER_NEAREST)
        nc_correct = normalized_correlation(wm, rs(extracted_correct))
        nc_wrong   = normalized_correlation(wm, rs(extracted_wrong))
        assert nc_correct > nc_wrong


# ── Blind Tests ───────────────────────────────────────────────────────────────

class TestEmbedDWTBlind:
    def test_returns_tuple_of_3(self):
        cover = _make_cover()
        wm    = _make_wm()
        result = embed_dwt_blind(cover, wm, alpha=0.1)
        assert len(result) == 3

    def test_output_shape_matches_cover(self):
        cover = _make_cover()
        wm    = _make_wm()
        out, _, _ = embed_dwt_blind(cover, wm, alpha=0.1)
        assert out.shape == cover.shape

    def test_psnr_reasonable(self):
        cover = _make_cover()
        wm    = _make_wm()
        _, psnr, _ = embed_dwt_blind(cover, wm, alpha=0.1)
        assert 20.0 < psnr < 60.0

    def test_grid_size_returned(self):
        cover = _make_cover(256, 256)
        wm    = _make_wm()
        _, _, (bh, bw) = embed_dwt_blind(cover, wm, alpha=0.1)
        assert bh > 0 and bw > 0

    def test_none_cover_raises(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            embed_dwt_blind(None, _make_wm(), alpha=0.1)


class TestExtractDWTBlind:
    @pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
    def test_zero_ber_through_png(self, alpha):
        """
        Blind extraction after PNG save/reload must have 0% BER.
        Uses spatial patch-average QIM (delta = max(alpha*0.2, 0.01)).
        """
        cover = _make_cover(256, 256, seed=7)
        wm    = _make_wm(32, 32)
        watermarked, psnr, (bh, bw) = embed_dwt_blind(cover, wm, alpha=alpha)
        assert psnr > 25.0, f"PSNR={psnr:.1f} too low at alpha={alpha}"
        reloaded  = _png_roundtrip(watermarked)
        extracted = extract_dwt_blind(reloaded, alpha=alpha,
                                      watermark_size=(bh, bw))
        ref      = cv2.resize(wm, (bw, bh), interpolation=cv2.INTER_AREA)
        bits_ref = (ref >= 128).astype(np.uint8)
        bits_ext = (extracted >= 128).astype(np.uint8)
        ber = 1.0 - float(np.mean(bits_ref == bits_ext))
        assert ber == pytest.approx(0.0), f"BER={ber:.4f} at alpha={alpha}"

    def test_output_is_binary(self):
        cover = _make_cover()
        wm    = _make_wm()
        watermarked, _, (bh, bw) = embed_dwt_blind(cover, wm, alpha=0.1)
        extracted = extract_dwt_blind(watermarked, alpha=0.1,
                                      watermark_size=(bh, bw))
        unique = set(extracted.flatten().tolist())
        assert unique.issubset({0, 255})

    def test_wrong_alpha_increases_ber(self):
        """Using wrong alpha at extraction must degrade BER significantly."""
        cover = _make_cover(256, 256, seed=3)
        wm    = _make_wm(32, 32)
        watermarked, _, (bh, bw) = embed_dwt_blind(cover, wm, alpha=0.1)
        reloaded = _png_roundtrip(watermarked)

        ext_correct = extract_dwt_blind(reloaded, alpha=0.1,
                                        watermark_size=(bh, bw))
        ext_wrong   = extract_dwt_blind(reloaded, alpha=0.8,
                                        watermark_size=(bh, bw))
        ref = cv2.resize(wm, (bw, bh), interpolation=cv2.INTER_AREA)
        bits_ref = (ref >= 128).astype(np.uint8)

        ber_correct = 1.0 - float(np.mean(bits_ref == (ext_correct >= 128)))
        ber_wrong   = 1.0 - float(np.mean(bits_ref == (ext_wrong   >= 128)))
        assert ber_wrong > ber_correct

    def test_none_image_raises(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            extract_dwt_blind(None, alpha=0.1)
