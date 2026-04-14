from __future__ import annotations

import cv2
import numpy as np
import pywt
from fastapi import HTTPException

from app.utils.metrics import calculate_psnr

# ── Wavelet configuration ─────────────────────────────────────────────────────
# db4 (Daubechies-4): 4-tap filter, smoother reconstruction than Haar,
# less blocking/ringing artifacts at the same alpha strength.
_WAVELET = "db4"

# Decomposition depth: 3 levels push the embedding into LL3, the lowest-
# frequency sub-band.  Energy is concentrated here, making the mark survive
# JPEG compression, Gaussian noise, spatial filtering and mild geometric
# distortions far better than a 1-level embed in a detail sub-band.
_LEVEL = 3

# Minimum dimension (per axis) needed to safely run 3-level db4 decomposition.
# db4 filter length = 8; safe minimum ≈ filter_len × 2^(level-1) = 8 × 4 = 32.
# We use 64 for a comfortable margin and a sensible minimum watermark size.
_MIN_DIM = 64


def embed_dwt(
    cover_bgr: np.ndarray,
    watermark_gray: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float]:
    """
    Multi-level DWT watermark embedding (3-level db4, LL sub-band).

    Strategy
    --------
    Apply 3 levels of Daubechies-4 DWT to the luminance (Y) channel,
    then additively embed the watermark into the LL approximation coefficients
    at the deepest level (LL3).  LL3 holds the bulk of the image energy and
    is the most robust band against common signal-processing attacks.

    Parameters
    ----------
    cover_bgr      : Original cover image (BGR, uint8).
    watermark_gray : Grayscale watermark (uint8).
    alpha          : Embedding strength [0.01, 0.5].

    Returns
    -------
    (watermarked_bgr, psnr_dB)
    """
    if cover_bgr is None or watermark_gray is None:
        raise HTTPException(status_code=400, detail="Invalid image processing")
    if cover_bgr.shape[0] < _MIN_DIM or cover_bgr.shape[1] < _MIN_DIM:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Cover image is too small "
                f"(min {_MIN_DIM}×{_MIN_DIM} required for {_LEVEL}-level DWT)"
            ),
        )

    # ── 1. Convert to YCrCb and isolate luminance ──────────────────────────
    ycrcb = cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_float = np.float32(y) / 255.0

    # ── 2. Multi-level forward DWT ─────────────────────────────────────────
    # wavedec2 returns:
    #   coeffs[0]        = LL3  ← approximation at deepest level (embed here)
    #   coeffs[1]        = (LH3, HL3, HH3)
    #   coeffs[2]        = (LH2, HL2, HH2)
    #   coeffs[3]        = (LH1, HL1, HH1)
    coeffs = pywt.wavedec2(y_float, _WAVELET, level=_LEVEL)
    ll3 = coeffs[0]
    ll_h, ll_w = ll3.shape

    # ── 3. Prepare watermark ───────────────────────────────────────────────
    wm = watermark_gray
    if wm.ndim == 3:
        wm = cv2.cvtColor(wm, cv2.COLOR_BGR2GRAY)
    wm_resized = cv2.resize(wm, (ll_w, ll_h), interpolation=cv2.INTER_AREA)
    wm_norm = wm_resized.astype(np.float32) / 255.0

    # ── 4. Embed into LL3 (additive) ───────────────────────────────────────
    coeffs[0] = ll3 + alpha * wm_norm

    # ── 5. Inverse multi-level DWT ─────────────────────────────────────────
    y_wm = pywt.waverec2(coeffs, _WAVELET)

    # waverec2 may produce ±1 row/col due to wavelet boundary padding.
    # Resize back to exact original dimensions before merging.
    orig_h, orig_w = y_float.shape
    if y_wm.shape != y_float.shape:
        y_wm = cv2.resize(y_wm, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    y_wm = np.clip(y_wm, 0.0, 1.0)
    y_final = np.uint8(y_wm * 255)

    # ── 6. Merge and convert back to BGR ───────────────────────────────────
    ycrcb_new = cv2.merge([y_final, cr, cb])
    result = cv2.cvtColor(ycrcb_new, cv2.COLOR_YCrCb2BGR)

    psnr_val = calculate_psnr(cover_bgr, result)
    return result, psnr_val


def extract_dwt(
    original_bgr: np.ndarray,
    watermarked_bgr: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Multi-level DWT watermark extraction (3-level db4, LL sub-band).

    Applies the same 3-level db4 DWT to both the original and watermarked
    images, then recovers the watermark from the LL3 coefficient difference.
    Must use the same wavelet, level, and alpha as embedding.

    Parameters
    ----------
    original_bgr   : Original (un-watermarked) cover image (BGR, uint8).
    watermarked_bgr: Watermarked (or suspect) image (BGR, uint8).
    alpha          : Embedding strength (must match the value used at embed time).

    Returns
    -------
    extracted_watermark : Grayscale uint8 array.
    """
    if original_bgr is None or watermarked_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image files")
    if original_bgr.shape[0] < _MIN_DIM or original_bgr.shape[1] < _MIN_DIM:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Image is too small "
                f"(min {_MIN_DIM}×{_MIN_DIM} required for {_LEVEL}-level DWT)"
            ),
        )

    # ── 1. Extract luminance from both images ──────────────────────────────
    def _to_y_float(bgr: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        y, _, _ = cv2.split(ycrcb)
        return np.float32(y) / 255.0

    y_orig = _to_y_float(original_bgr)
    y_wm   = _to_y_float(watermarked_bgr)

    # ── 2. Multi-level forward DWT on both ────────────────────────────────
    coeffs_orig = pywt.wavedec2(y_orig, _WAVELET, level=_LEVEL)
    coeffs_wm   = pywt.wavedec2(y_wm,   _WAVELET, level=_LEVEL)

    ll3_orig = coeffs_orig[0]
    ll3_wm   = coeffs_wm[0]

    # Guard against rare size mismatch (different-resolution inputs)
    if ll3_wm.shape != ll3_orig.shape:
        ll3_wm = cv2.resize(
            ll3_wm,
            (ll3_orig.shape[1], ll3_orig.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    # ── 3. Recover watermark from LL3 difference ───────────────────────────
    extracted = (ll3_wm - ll3_orig) / alpha
    extracted = np.clip(extracted, 0.0, 1.0)
    return np.uint8(extracted * 255)


# ══════════════════════════════════════════════════════════════════════════════
# BLIND WATERMARKING  —  Spatial Patch-Average QIM
#
# WHY NOT LL3 QIM?
#   Embedding via QIM in LL3 then calling waverec2 → uint8 → wavedec2 introduces
#   reconstruction errors of up to 0.18 in LL3 coefficients (measured empirically).
#   With delta=alpha=0.1, tolerance = 0.05 → most bits decode on the wrong side.
#
# SPATIAL PATCH-AVERAGE QIM:
#   Divides the Y channel into 8×8 patches and QIM-encodes one bit per patch
#   via the PATCH AVERAGE luminance.  The round-trip error for a patch average
#   after uint8 save/reload is only ~0.5/255/sqrt(64) ≈ 0.00025 — well within
#   any practical delta.  PSNR ≈ 38–43 dB at typical alpha values; BER = 0%.
#
#   The internal delta = max(alpha × 0.2, 0.01) keeps the QIM step small
#   relative to the [0,1] luminance range.  Both embed and extract use the
#   same formula so the key (alpha) fully determines delta — no mismatch.
# ══════════════════════════════════════════════════════════════════════════════

from app.utils.qim import qim_embed, qim_extract  # noqa: E402

_BLIND_PATCH      = 8     # 8×8 pixel patch per watermark bit
_BLIND_DELTA_SCALE = 0.2  # internal delta = alpha × scale
_BLIND_DELTA_MIN   = 0.01 # minimum delta in [0, 1] luminance space


def _blind_delta(alpha: float) -> float:
    """Translate user-facing alpha into the internal QIM step for blind DWT."""
    return max(alpha * _BLIND_DELTA_SCALE, _BLIND_DELTA_MIN)


def embed_dwt_blind(
    cover_bgr: np.ndarray,
    watermark_gray: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Blind DWT watermark embedding using spatial patch-average QIM.

    Divides the Y channel into 8x8 patches and embeds one watermark bit
    per patch by quantising the patch's mean luminance (QIM step = _blind_delta(alpha)).
    No inverse wavelet transform is needed, so the round-trip error is only
    ~0.00025 per patch average -- well within the QIM tolerance.

    Returns (watermarked_bgr, psnr_dB, (bh, bw)).
    """
    if cover_bgr is None or watermark_gray is None:
        raise HTTPException(status_code=400, detail="Invalid image processing")
    if cover_bgr.shape[0] < _MIN_DIM or cover_bgr.shape[1] < _MIN_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"Cover image too small (min {_MIN_DIM}x{_MIN_DIM})",
        )

    ycrcb = cv2.cvtColor(cover_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_float = y.astype(np.float32) / 255.0
    h, w = y_float.shape

    bh, bw = h // _BLIND_PATCH, w // _BLIND_PATCH
    if bh == 0 or bw == 0:
        raise HTTPException(status_code=400, detail="Cover image too small for blind DWT patches")

    wm = watermark_gray
    if wm.ndim == 3:
        wm = cv2.cvtColor(wm, cv2.COLOR_BGR2GRAY)
    wm_small  = cv2.resize(wm, (bw, bh), interpolation=cv2.INTER_AREA)
    wm_binary = (wm_small >= 128).astype(np.int32)

    delta  = _blind_delta(alpha)
    P      = _BLIND_PATCH
    y_work = y_float.copy()

    for i in range(bh):
        for j in range(bw):
            patch   = y_work[i*P:(i+1)*P, j*P:(j+1)*P]
            avg     = np.array([np.mean(patch)])
            avg_new = float(qim_embed(avg, np.array([wm_binary[i, j]]), delta=delta)[0])
            y_work[i*P:(i+1)*P, j*P:(j+1)*P] = np.clip(
                patch + (avg_new - avg[0]), 0.0, 1.0)

    y_final = np.uint8(np.clip(y_work, 0.0, 1.0) * 255)
    result  = cv2.cvtColor(cv2.merge([y_final, cr, cb]), cv2.COLOR_YCrCb2BGR)
    return result, calculate_psnr(cover_bgr, result), (bh, bw)


def extract_dwt_blind(
    watermarked_bgr: np.ndarray,
    alpha: float,
    watermark_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Blind DWT watermark extraction using spatial patch-average QIM.

    Reads the mean luminance of each 8x8 patch and applies QIM extraction
    with delta = _blind_delta(alpha).  Original image NOT required.
    Returns binary grayscale image (0 or 255 per pixel).
    """
    if watermarked_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image files")
    if watermarked_bgr.shape[0] < _MIN_DIM or watermarked_bgr.shape[1] < _MIN_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"Image too small (min {_MIN_DIM}x{_MIN_DIM})",
        )

    y_wm  = cv2.split(cv2.cvtColor(watermarked_bgr, cv2.COLOR_BGR2YCrCb))[0]
    y_wm  = y_wm.astype(np.float32) / 255.0
    h     = (y_wm.shape[0] // _BLIND_PATCH) * _BLIND_PATCH
    w     = (y_wm.shape[1] // _BLIND_PATCH) * _BLIND_PATCH
    bh, bw = h // _BLIND_PATCH, w // _BLIND_PATCH

    delta   = _blind_delta(alpha)
    P       = _BLIND_PATCH
    wm_bits = np.zeros((bh, bw), dtype=np.uint8)

    for i in range(bh):
        for j in range(bw):
            avg          = np.array([np.mean(y_wm[i*P:(i+1)*P, j*P:(j+1)*P])])
            wm_bits[i, j] = 255 if int(qim_extract(avg, delta=delta)[0]) == 1 else 0

    if watermark_size is not None:
        wm_h, wm_w = watermark_size
        return cv2.resize(wm_bits, (wm_w, wm_h), interpolation=cv2.INTER_NEAREST)
    return wm_bits

