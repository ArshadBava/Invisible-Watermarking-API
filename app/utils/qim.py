"""
app/utils/qim.py  —  Vectorized Quantization Index Modulation (QIM)

Used by all three blind watermarking services.

Binary QIM formulas:
  Embed bit=0 → quantise to even-multiple-of-Δ grid
  Embed bit=1 → quantise to odd-multiple-of-Δ grid
  Extract     → round(coef / Δ) % 2

Works correctly for negative coefficients (Python % always returns ≥ 0).
"""
from __future__ import annotations

import numpy as np


def qim_embed(coeffs: np.ndarray, bits: np.ndarray, delta: float) -> np.ndarray:
    """
    Vectorised QIM embedding.

    Parameters
    ----------
    coeffs : float array of any shape — coefficients to modify
    bits   : int array, same shape — 0 or 1 per coefficient
    delta  : quantisation step (the secret key)

    Returns
    -------
    Modified coefficient array (float64), same shape as `coeffs`.
    """
    c = coeffs.astype(np.float64)
    b = bits.astype(np.int32)
    out = np.empty_like(c)

    m0 = b == 0   # bit-0 mask → even grid
    m1 = b == 1   # bit-1 mask → odd grid

    out[m0] = 2.0 * delta * np.round(c[m0] / (2.0 * delta))
    out[m1] = 2.0 * delta * np.round((c[m1] - delta) / (2.0 * delta)) + delta

    return out


def qim_extract(coeffs: np.ndarray, delta: float) -> np.ndarray:
    """
    Vectorised QIM extraction.

    Parameters
    ----------
    coeffs : float array of any shape — (possibly attacked) coefficients
    delta  : quantisation step used at embedding time

    Returns
    -------
    Recovered bit array (uint8, values 0 or 1), same shape as `coeffs`.
    """
    return (np.round(coeffs.astype(np.float64) / delta).astype(np.int64) % 2).astype(np.uint8)
