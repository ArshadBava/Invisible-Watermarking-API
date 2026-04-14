from __future__ import annotations

import numpy as np


def calculate_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    max_pixel = 255.0
    return float(20 * np.log10(max_pixel / np.sqrt(mse)))


def normalized_correlation(img1: np.ndarray, img2: np.ndarray) -> float:
    a = img1.astype(np.float32)
    b = img2.astype(np.float32)
    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    numerator = float(np.sum((a - mean_a) * (b - mean_b)))
    denominator = float(np.sqrt(np.sum((a - mean_a) ** 2) * np.sum((b - mean_b) ** 2)))
    if denominator == 0:
        return 0.0
    return numerator / denominator

