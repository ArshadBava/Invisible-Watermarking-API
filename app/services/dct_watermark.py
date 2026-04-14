from __future__ import annotations

import cv2
import numpy as np
from fastapi import HTTPException

from app.utils.metrics import calculate_psnr

# ── DCT configuration ─────────────────────────────────────────────────────────
# Mid-frequency 8×8 DCT coefficient positions (row, col) selected from the
# zigzag scan order (roughly indices 10–25).
#
# These positions are:
#   - Above the DC / very-low-freq zone  → perceptually invisible changes
#   - Below the high-freq zone           → survive JPEG compression (quality ≥ 70)
#   - Symmetric layout                   → balanced spectral modification
#
# 6 positions instead of the previous 2 gives 3× redundancy:
#   • Extraction averages 6 independent readings → √3 better SNR
#   • If JPEG destroys 1–2 coefficients, the remaining still reconstruct the signal
#
#   Zigzag layout of chosen positions inside a single 8×8 block:
#
#     col→  0   1   2   3   4   5   6   7
#   row↓
#     0    DC  .   .   .   .   .   .   .
#     1    .   .   .   .   .   .   .   .
#     2    .   .   .  [A]  [B] .   .   .       A=(2,3)  B=(2,4)
#     3    .   .  [C]  .  [D]  .   .   .       C=(3,2)  D=(3,4)
#     4    .   .   . [E] [F]   .   .   .       E=(4,3)  F=(4,4)
#     5    .   .   .   .   .   .   .   .
#
_COEF_POSITIONS: list[tuple[int, int]] = [
    (2, 3), (2, 4),   # inner-upper ring
    (3, 2), (3, 4),   # inner-middle ring
    (4, 3), (4, 4),   # inner-lower ring
]
_N_COEFS = len(_COEF_POSITIONS)

# ── HVS (Human Visual System) masking ────────────────────────────────────────
# Perceptual sensitivity varies across blocks:
#   • Flat blocks  (low std) → even tiny changes are visible   → embed lightly
#   • Textured blocks (high std) → changes are masked by texture → embed strongly
#
# Per-block scale:  s = clip( std(block) / _HVS_REF_STD, _HVS_MIN, _HVS_MAX )
#
# Effect at embed  : delta = alpha * watermark_signal * s
# Effect at extract: recovered = (dct_w - dct_o) / (alpha * s)
#
# Since this is non-blind watermarking (original is always available at
# extraction), the identical original block gives the same `s`, making
# the scaling perfectly invertible — no information is lost.
_HVS_REF_STD: float = 0.08   # std that maps to scale = 1.0 (empirical, [0,1] range)
_HVS_MIN:     float = 0.4    # floor: always embed something in very flat blocks
_HVS_MAX:     float = 2.5    # ceiling: cap to avoid large coefficient distortion


def _to_luma01(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert BGR → YCrCb and return Y in [0,1] plus Cr, Cb channels."""
    if bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image files")
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise HTTPException(status_code=400, detail="Invalid image format")
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    return y.astype(np.float32) / 255.0, cr, cb


def _merge_luma01(y01: np.ndarray, cr: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """Merge normalised Y [0,1] with Cr, Cb back to BGR uint8."""
    y_uint8 = np.uint8(np.clip(y01, 0.0, 1.0) * 255.0)
    ycrcb_new = cv2.merge([y_uint8, cr, cb])
    return cv2.cvtColor(ycrcb_new, cv2.COLOR_YCrCb2BGR)


def _hvs_scale(block: np.ndarray) -> float:
    """
    Compute the HVS-masking scale factor for one 8×8 block.

    Uses local standard deviation as a texture-activity proxy:
      - High std (busy texture) → large scale → embed more aggressively
      - Low  std (flat region)  → small scale → embed lightly to stay invisible

    Returns a value clamped to [_HVS_MIN, _HVS_MAX].
    """
    std = float(np.std(block))
    return float(np.clip(std / _HVS_REF_STD, _HVS_MIN, _HVS_MAX))


def embed_dct(
    cover_bgr: np.ndarray,
    watermark_gray: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    """
    Improved non-blind block-DCT (8×8) watermarking on the luminance channel.

    Improvements over the previous version
    ----------------------------------------
    1. **6 mid-frequency coefficient positions** (was 2).
       Redundancy: each watermark pixel is recorded in 6 independent DCT bins.
       Extraction averages all 6 readings → ~√3 better signal-to-noise ratio.

    2. **HVS-adaptive per-block embedding strength**.
       Textured blocks carry a stronger mark (masked by local texture).
       Flat / smooth blocks carry a lighter mark (avoids visible banding).
       Both effects improve PSNR while maintaining or increasing robustness.

    Parameters
    ----------
    cover_bgr      : Original cover image (BGR, uint8).
    watermark_gray : Grayscale watermark (uint8). Will be resized to (bh, bw).
    alpha          : Base embedding strength [0.01, 0.5].

    Returns
    -------
    (watermarked_bgr, psnr_dB)
    """
    if cover_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image processing")
    if cover_bgr.shape[0] < 64 or cover_bgr.shape[1] < 64:
        raise HTTPException(status_code=400, detail="Cover image is too small (min 64×64 required)")

    y01, cr, cb = _to_luma01(cover_bgr)
    h, w = y01.shape
    bh, bw = h // 8, w // 8
    if bh == 0 or bw == 0:
        raise HTTPException(status_code=400, detail="Cover image is too small for DCT blocks")

    # Prepare watermark signal in [-1, 1]
    wm = watermark_gray
    if wm is None:
        raise HTTPException(status_code=400, detail="Invalid watermark image")
    if wm.ndim == 3:
        wm = cv2.cvtColor(wm, cv2.COLOR_BGR2GRAY)
    wm_small = (
        cv2.resize(wm, (bw, bh), interpolation=cv2.INTER_AREA)
        .astype(np.float32) / 255.0
    )
    wm_signal = wm_small * 2.0 - 1.0   # remap [0,1] → [-1,1]

    y_work = y01.copy()
    region = y_work[: bh * 8, : bw * 8]

    for i in range(bh):
        for j in range(bw):
            block = region[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8].copy()
            dct_block = cv2.dct(block)

            s = float(wm_signal[i, j])
            # HVS masking: same scale used consistently at extraction
            scale = _hvs_scale(block)
            delta = alpha * s * scale

            # Embed into all 6 mid-frequency positions
            for (r, c) in _COEF_POSITIONS:
                dct_block[r, c] += delta

            region[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8] = cv2.idct(dct_block)

    y_work[: bh * 8, : bw * 8] = region
    watermarked_bgr = _merge_luma01(y_work, cr, cb)
    psnr_val = calculate_psnr(cover_bgr, watermarked_bgr)
    return watermarked_bgr, psnr_val


def extract_dct(
    original_bgr: np.ndarray,
    watermarked_bgr: np.ndarray,
    alpha: float,
    watermark_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Improved non-blind block-DCT watermark extraction.

    Mirrors the improved embedding exactly:
    - Recomputes the HVS scale from each original block (same value as at embed time).
    - Averages the signal recovered from all 6 coefficient positions.
    - Per-position formula: reading = (dct_w[r,c] - dct_o[r,c]) / (alpha * scale)

    Parameters
    ----------
    original_bgr    : Original (un-watermarked) cover image (BGR, uint8).
    watermarked_bgr : Watermarked (or suspect) image (BGR, uint8).
    alpha           : Embedding strength (must match the value used at embed time).
    watermark_size  : Optional (height, width) of the original watermark.
                      When supplied the tiny block-grid output (H//8 × W//8) is
                      upscaled to these dimensions using bicubic interpolation,
                      producing a visually usable extracted watermark image.
                      When None, the raw block-grid array is returned.

    Returns
    -------
    Grayscale uint8 array — either at block-grid size or resized to watermark_size.
    """
    if original_bgr is None or watermarked_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image files")
    if original_bgr.shape[0] < 64 or original_bgr.shape[1] < 64:
        raise HTTPException(status_code=400, detail="Image is too small (min 64×64 required)")

    y_o, _, _ = _to_luma01(original_bgr)
    y_w, _, _ = _to_luma01(watermarked_bgr)

    # Align to common 8-aligned dimensions
    h = min(y_o.shape[0], y_w.shape[0])
    w = min(y_o.shape[1], y_w.shape[1])
    h = (h // 8) * 8
    w = (w // 8) * 8
    if h == 0 or w == 0:
        raise HTTPException(status_code=400, detail="Image is too small for DCT blocks")

    bh, bw = h // 8, w // 8
    region_o = y_o[:h, :w]
    region_w = y_w[:h, :w]

    wm_est = np.zeros((bh, bw), dtype=np.float32)
    for i in range(bh):
        for j in range(bw):
            bo  = region_o[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
            bwm = region_w[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
            dct_o = cv2.dct(bo)
            dct_w = cv2.dct(bwm)

            # Recover the exact HVS scale used during embedding
            scale = _hvs_scale(bo)
            denom = alpha * scale

            # Average readings from all 6 positions for best SNR
            readings = [
                (float(dct_w[r, c]) - float(dct_o[r, c])) / denom
                for (r, c) in _COEF_POSITIONS
            ]
            wm_est[i, j] = sum(readings) / _N_COEFS

    wm_est = np.clip(wm_est, -1.0, 1.0)
    wm01   = (wm_est + 1.0) / 2.0
    extracted_block = np.uint8(wm01 * 255.0)

    # Upscale from tiny block-grid resolution to original watermark dimensions
    if watermark_size is not None:
        wm_h, wm_w = watermark_size
        return cv2.resize(extracted_block, (wm_w, wm_h), interpolation=cv2.INTER_CUBIC)

    return extracted_block


# ══════════════════════════════════════════════════════════════════════════════
# BLIND DCT  —  QIM on all 6 mid-frequency coefficients + majority vote
# `alpha` = QIM step Δ. No original image required at extraction.
# ══════════════════════════════════════════════════════════════════════════════

from app.utils.qim import qim_embed, qim_extract  # noqa: E402


def embed_dct_blind(
    cover_bgr: np.ndarray,
    watermark_gray: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Blind DCT watermark embedding using QIM across all 6 mid-freq positions.

    Each block carries 1 watermark bit embedded redundantly into 6 DCT
    coefficients.  At extraction, a majority vote over the 6 decoded bits
    recovers the original bit without the cover image.

    Returns
    -------
    (watermarked_bgr, psnr_dB, (bh, bw))
    """
    if cover_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image processing")
    if cover_bgr.shape[0] < 64 or cover_bgr.shape[1] < 64:
        raise HTTPException(status_code=400, detail="Cover image too small (min 64×64)")

    y01, cr, cb = _to_luma01(cover_bgr)
    h, w = y01.shape
    bh, bw = h // 8, w // 8
    if bh == 0 or bw == 0:
        raise HTTPException(status_code=400, detail="Cover image too small for DCT blocks")

    wm = watermark_gray
    if wm is None:
        raise HTTPException(status_code=400, detail="Invalid watermark image")
    if wm.ndim == 3:
        wm = cv2.cvtColor(wm, cv2.COLOR_BGR2GRAY)
    wm_small  = cv2.resize(wm, (bw, bh), interpolation=cv2.INTER_AREA)
    wm_binary = (wm_small >= 128).astype(np.int32)   # 0 or 1 per block

    y_work = y01.copy()
    region = y_work[: bh * 8, : bw * 8]

    for i in range(bh):
        for j in range(bw):
            block     = region[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8].copy()
            dct_block = cv2.dct(block)
            bit       = int(wm_binary[i, j])
            # Embed same bit into all 6 positions using QIM
            for (r, c) in _COEF_POSITIONS:
                coef_arr         = np.array([dct_block[r, c]])
                bit_arr          = np.array([bit])
                dct_block[r, c]  = float(qim_embed(coef_arr, bit_arr, delta=alpha)[0])
            region[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8] = cv2.idct(dct_block)

    y_work[: bh * 8, : bw * 8] = region
    watermarked_bgr = _merge_luma01(y_work, cr, cb)
    psnr_val = calculate_psnr(cover_bgr, watermarked_bgr)
    return watermarked_bgr, psnr_val, (bh, bw)


def extract_dct_blind(
    watermarked_bgr: np.ndarray,
    alpha: float,
    watermark_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Blind DCT watermark extraction using QIM majority vote.

    For each 8×8 block, decodes the bit from all 6 coefficient positions
    and takes the majority vote to recover the embedded bit.
    Original image NOT needed.

    Returns binary grayscale image (0 or 255 per pixel).
    """
    if watermarked_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image files")
    if watermarked_bgr.shape[0] < 64 or watermarked_bgr.shape[1] < 64:
        raise HTTPException(status_code=400, detail="Image too small (min 64×64)")

    y_w, _, _ = _to_luma01(watermarked_bgr)
    h = (y_w.shape[0] // 8) * 8
    w = (y_w.shape[1] // 8) * 8
    if h == 0 or w == 0:
        raise HTTPException(status_code=400, detail="Image too small for DCT blocks")

    bh, bw = h // 8, w // 8
    region_w = y_w[:h, :w]
    wm_bits  = np.zeros((bh, bw), dtype=np.uint8)

    for i in range(bh):
        for j in range(bw):
            bwm      = region_w[i * 8 : (i + 1) * 8, j * 8 : (j + 1) * 8]
            dct_w    = cv2.dct(bwm)
            votes    = [int(qim_extract(np.array([dct_w[r, c]]), delta=alpha)[0])
                        for (r, c) in _COEF_POSITIONS]
            bit      = 1 if sum(votes) > _N_COEFS // 2 else 0
            wm_bits[i, j] = 255 if bit == 1 else 0

    if watermark_size is not None:
        wm_h, wm_w = watermark_size
        return cv2.resize(wm_bits, (wm_w, wm_h), interpolation=cv2.INTER_NEAREST)

    return wm_bits
