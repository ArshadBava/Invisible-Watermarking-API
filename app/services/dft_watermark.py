from __future__ import annotations

import cv2
import numpy as np
from fastapi import HTTPException
import logging

from app.utils.metrics import calculate_psnr

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_luma01(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert BGR to YCrCb and return normalised Y [0,1] plus Cr, Cb."""
    if bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image files")
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise HTTPException(status_code=400, detail="Expected BGR image")
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    return y.astype(np.float32) / 255.0, cr, cb


def _merge_luma01(y01: np.ndarray, cr: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """Merge normalised Y [0,1] with Cr, Cb back to BGR uint8."""
    y_uint8 = np.uint8(np.clip(y01, 0.0, 1.0) * 255.0)
    return cv2.cvtColor(cv2.merge([y_uint8, cr, cb]), cv2.COLOR_YCrCb2BGR)


def _dual_block_regions(
    h: int, w: int, offset_ratio: float = 0.25, block_ratio: float = 0.125
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    """
    Returns two conjugate-symmetric rectangle regions in the shifted DFT spectrum
    for dual-symmetric embedding (non-blind).
    """
    cy, cx = h // 2, w // 2
    dy = max(1, int(h * offset_ratio))
    dx = max(1, int(w * offset_ratio))
    bh = max(16, int(h * block_ratio))
    bw = max(16, int(w * block_ratio))
    bh = bh if bh % 2 == 0 else bh + 1
    bw = bw if bw % 2 == 0 else bw + 1
    hh, hw = bh // 2, bw // 2
    y1, x1 = cy - dy, cx + dx
    y2, x2 = cy + dy, cx - dx
    r1 = (slice(y1 - hh, y1 + hh), slice(x1 - hw, x1 + hw))
    r2 = (slice(y2 - hh, y2 + hh), slice(x2 - hw, x2 + hw))
    return r1, r2


# ─────────────────────────────────────────────────────────────────────────────
# Non-blind DFT (requires original image at extraction)
# ─────────────────────────────────────────────────────────────────────────────

def embed_dft(
    cover_bgr: np.ndarray,
    watermark_gray: np.ndarray,
    alpha: float,
    band_ratio: float = 0.125,
) -> tuple[np.ndarray, float]:
    """
    Dual-symmetric DFT embedding — additive magnitude modification.
    Embeds watermark symmetrically into two conjugate FFT magnitude regions.
    """
    if cover_bgr is None or watermark_gray is None:
        raise HTTPException(status_code=400, detail="Invalid image processing")
    if cover_bgr.shape[0] < 64 or cover_bgr.shape[1] < 64:
        raise HTTPException(status_code=400, detail="Cover image too small (min 64×64)")
    if not 0.01 <= alpha <= 0.5:
        raise HTTPException(status_code=400, detail=f"Alpha {alpha} out of range [0.01, 0.5]")

    y01, cr, cb = _to_luma01(cover_bgr)
    h, w = y01.shape
    r1, r2 = _dual_block_regions(h, w, offset_ratio=0.25, block_ratio=band_ratio)

    wm = watermark_gray
    if wm.ndim == 3:
        wm = cv2.cvtColor(wm, cv2.COLOR_BGR2GRAY)
    bh = r1[0].stop - r1[0].start
    bw = r1[1].stop - r1[1].start
    wm_block = cv2.resize(wm, (bw, bh), interpolation=cv2.INTER_AREA)
    wm_signal = (wm_block.astype(np.float32) / 255.0) * 2.0 - 1.0
    wm_signal_flipped = np.flip(wm_signal, axis=(0, 1))

    f_shift = np.fft.fftshift(np.fft.fft2(y01))
    mag = np.abs(f_shift)
    phase = np.angle(f_shift)

    band_avg = np.mean(mag[r1])
    if band_avg < 1e-6:
        band_avg = 1.0
    scale_factor = band_avg * 0.5

    mag[r1] = np.maximum(mag[r1] + alpha * wm_signal         * scale_factor, 0.0)
    mag[r2] = np.maximum(mag[r2] + alpha * wm_signal_flipped * scale_factor, 0.0)

    y_new = np.fft.ifft2(np.fft.ifftshift(mag * np.exp(1j * phase))).real
    y_new = np.clip(y_new, 0.0, 1.0)

    watermarked_bgr = _merge_luma01(y_new, cr, cb)
    return watermarked_bgr, calculate_psnr(cover_bgr, watermarked_bgr)


def extract_dft(
    original_bgr: np.ndarray,
    watermarked_bgr: np.ndarray,
    alpha: float,
    watermark_size: tuple[int, int] | None = None,
    band_ratio: float = 0.125,
) -> np.ndarray:
    """
    Dual-symmetric DFT extraction — requires original image (non-blind).
    Averages signals extracted from both symmetric quadrants.
    """
    if original_bgr is None or watermarked_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image files")
    if original_bgr.shape[0] < 64 or original_bgr.shape[1] < 64:
        raise HTTPException(status_code=400, detail="Image too small (min 64×64)")
    if not 0.01 <= alpha <= 0.5:
        raise HTTPException(status_code=400, detail=f"Alpha {alpha} out of range")

    y_o, _, _ = _to_luma01(original_bgr)
    y_w, _, _ = _to_luma01(watermarked_bgr)
    h = min(y_o.shape[0], y_w.shape[0])
    w = min(y_o.shape[1], y_w.shape[1])
    y_o, y_w = y_o[:h, :w], y_w[:h, :w]

    r1, r2 = _dual_block_regions(h, w, offset_ratio=0.25, block_ratio=band_ratio)
    mag_o = np.abs(np.fft.fftshift(np.fft.fft2(y_o)))
    mag_w = np.abs(np.fft.fftshift(np.fft.fft2(y_w)))

    band_avg = np.mean(mag_o[r1])
    if band_avg < 1e-6:
        band_avg = 1.0
    scale_factor = band_avg * 0.5

    ext1 = (mag_w[r1] - mag_o[r1]) / (alpha * scale_factor)
    ext2 = np.flip((mag_w[r2] - mag_o[r2]) / (alpha * scale_factor), axis=(0, 1))
    wm_signal = np.clip((ext1 + ext2) / 2.0, -1.0, 1.0)

    extracted = np.uint8((wm_signal + 1.0) / 2.0 * 255.0)
    if watermark_size is not None:
        wm_h, wm_w = watermark_size
        return cv2.resize(extracted, (wm_w, wm_h), interpolation=cv2.INTER_CUBIC)
    return extracted


# ─────────────────────────────────────────────────────────────────────────────
# BLIND DFT  —  Block-DFT + Log-domain QIM
#
# WHY BLOCK-DFT (not global FFT)?
#   Global FFT causes quantization noise amplification:
#     error_per_bin ≈ sqrt(N×M) × 0.5/255 ≈ 512 × 0.002 ≈ 0.58 (linear)
#   In log domain at typical mid-freq magnitudes: ≈ 0.05–0.5 per bin.
#   With alpha=0.1, QIM tolerance = 0.05 → virtually every bit decodes wrong.
#
#   Block-DFT (8×8 tiles) limits spreading per block:
#     error_per_bin ≈ sqrt(64) × 0.5/255 ≈ 8 × 0.002 ≈ 0.016 (linear)
#   Log domain: ≈ 0.002–0.005 → well within alpha=0.1 tolerance (0.05).
#   This is why blind DCT works too: same block-wise principle.
#
#   Fixed delta = alpha in log domain means embed-time delta = extract-time
#   delta exactly — no adaptive mismatch possible.
# ─────────────────────────────────────────────────────────────────────────────

from app.utils.qim import qim_embed, qim_extract  # noqa: E402

_DFT_BLOCK = 8
# Four mid-frequency positions in an 8×8 DFT block.
# None are self-conjugate (conjugate of (r,c) is ((8-r)%8, (8-c)%8)).
_DFT_MID_COEFS = [(2, 3), (3, 2), (4, 3), (3, 4)]
_DFT_N_COEFS   = len(_DFT_MID_COEFS)


def embed_dft_blind(
    cover_bgr: np.ndarray,
    watermark_gray: np.ndarray,
    alpha: float,
    band_ratio: float = 0.125,   # unused in block mode, kept for API compat
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Blind DFT embedding using 8×8 Block-DFT + Log-domain QIM.

    Divides the Y channel into 8×8 tiles, applies np.fft.fft2 per tile, and
    embeds one watermark bit per tile into 4 mid-frequency magnitude positions
    (log1p domain, fixed delta = alpha, Hermitian symmetry enforced).

    Returns (watermarked_bgr, psnr_dB, (bh, bw)).
    """
    if cover_bgr is None or watermark_gray is None:
        raise HTTPException(status_code=400, detail="Invalid image processing")
    if cover_bgr.shape[0] < 64 or cover_bgr.shape[1] < 64:
        raise HTTPException(status_code=400, detail="Cover image too small (min 64×64)")

    y01, cr, cb = _to_luma01(cover_bgr)
    h, w = y01.shape
    bh, bw = h // _DFT_BLOCK, w // _DFT_BLOCK
    if bh == 0 or bw == 0:
        raise HTTPException(status_code=400, detail="Cover image too small for DFT blocks")

    wm = watermark_gray
    if wm is None:
        raise HTTPException(status_code=400, detail="Invalid watermark image")
    if wm.ndim == 3:
        wm = cv2.cvtColor(wm, cv2.COLOR_BGR2GRAY)
    wm_small  = cv2.resize(wm, (bw, bh), interpolation=cv2.INTER_AREA)
    wm_binary = (wm_small >= 128).astype(np.int32)

    y_work = y01.copy()
    B = _DFT_BLOCK

    for i in range(bh):
        for j in range(bw):
            block     = y_work[i*B:(i+1)*B, j*B:(j+1)*B].copy()
            dft_block = np.fft.fft2(block)
            bit       = int(wm_binary[i, j])

            for (r, c) in _DFT_MID_COEFS:
                log_m     = np.log1p(abs(dft_block[r, c]))
                log_m_new = float(
                    qim_embed(np.array([log_m]), np.array([bit]), delta=alpha)[0])
                new_mag  = np.expm1(max(0.0, log_m_new))
                phase_rc = np.angle(dft_block[r, c])
                val      = new_mag * np.exp(1j * phase_rc)
                # Set conjugate pair to preserve Hermitian symmetry → real IFFT
                dft_block[r, c]                     = val
                dft_block[(B - r) % B, (B - c) % B] = np.conj(val)

            y_work[i*B:(i+1)*B, j*B:(j+1)*B] = np.fft.ifft2(dft_block).real

    watermarked_bgr = _merge_luma01(y_work, cr, cb)
    return watermarked_bgr, calculate_psnr(cover_bgr, watermarked_bgr), (bh, bw)


def extract_dft_blind(
    watermarked_bgr: np.ndarray,
    alpha: float,
    watermark_size: tuple[int, int] | None = None,
    band_ratio: float = 0.125,   # unused in block mode, kept for API compat
) -> np.ndarray:
    """
    Blind DFT extraction using 8×8 Block-DFT + Log-domain QIM + majority vote.

    Reads log1p(|DFT[r,c]|) from the same 4 positions per tile and takes
    majority vote. Fixed delta = alpha — no original image needed.
    Returns binary grayscale image (0 or 255 per pixel).
    """
    if watermarked_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image files")
    if watermarked_bgr.shape[0] < 64 or watermarked_bgr.shape[1] < 64:
        raise HTTPException(status_code=400, detail="Image too small (min 64×64)")

    y_wm, _, _ = _to_luma01(watermarked_bgr)
    h = (y_wm.shape[0] // _DFT_BLOCK) * _DFT_BLOCK
    w = (y_wm.shape[1] // _DFT_BLOCK) * _DFT_BLOCK
    if h == 0 or w == 0:
        raise HTTPException(status_code=400, detail="Image too small for DFT blocks")

    bh, bw = h // _DFT_BLOCK, w // _DFT_BLOCK
    B       = _DFT_BLOCK
    wm_bits = np.zeros((bh, bw), dtype=np.uint8)

    for i in range(bh):
        for j in range(bw):
            block     = y_wm[i*B:(i+1)*B, j*B:(j+1)*B]
            dft_block = np.fft.fft2(block)
            votes = [
                int(qim_extract(
                    np.array([np.log1p(abs(dft_block[r, c]))]),
                    delta=alpha)[0])
                for (r, c) in _DFT_MID_COEFS
            ]
            wm_bits[i, j] = 255 if sum(votes) > _DFT_N_COEFS // 2 else 0

    if watermark_size is not None:
        wm_h, wm_w = watermark_size
        return cv2.resize(wm_bits, (wm_w, wm_h), interpolation=cv2.INTER_NEAREST)
    return wm_bits
