from __future__ import annotations

from typing import Any

import cv2
import numpy as np


# ── Individual attack functions ───────────────────────────────────────────────

def attack_jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    decoded = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return decoded if decoded is not None else img


def attack_gaussian_noise(img: np.ndarray, std: float) -> np.ndarray:
    noise = np.random.normal(0.0, std, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def attack_salt_pepper(img: np.ndarray, amount: float) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    n = int(h * w * amount)
    out[np.random.randint(0, h, n), np.random.randint(0, w, n)] = 255
    out[np.random.randint(0, h, n), np.random.randint(0, w, n)] = 0
    return out


def attack_gaussian_blur(img: np.ndarray, ksize: int) -> np.ndarray:
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def attack_median_filter(img: np.ndarray, ksize: int) -> np.ndarray:
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.medianBlur(img, k)


def attack_sharpening(img: np.ndarray) -> np.ndarray:
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
    return np.clip(cv2.filter2D(img.astype(np.float32), -1, kernel), 0, 255).astype(np.uint8)


def attack_crop(img: np.ndarray, ratio: float) -> np.ndarray:
    h, w = img.shape[:2]
    dy, dx = max(1, int(h * ratio)), max(1, int(w * ratio))
    out = np.zeros_like(img)
    out[dy: h - dy, dx: w - dx] = img[dy: h - dy, dx: w - dx]
    return out


def attack_rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def attack_scale(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    nh, nw = max(4, int(h * scale)), max(4, int(w * scale))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def attack_histogram_eq(img: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def attack_brightness(img: np.ndarray, delta: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def attack_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


# ── Attack suite (28 attacks across 8 categories) ────────────────────────────

ATTACK_SUITE: list[dict[str, Any]] = [
    # JPEG Compression
    {"category": "JPEG Compression",       "name": "JPEG Q=90",          "fn": lambda i: attack_jpeg(i, 90)},
    {"category": "JPEG Compression",       "name": "JPEG Q=70",          "fn": lambda i: attack_jpeg(i, 70)},
    {"category": "JPEG Compression",       "name": "JPEG Q=50",          "fn": lambda i: attack_jpeg(i, 50)},
    {"category": "JPEG Compression",       "name": "JPEG Q=30",          "fn": lambda i: attack_jpeg(i, 30)},
    # Noise
    {"category": "Noise",                  "name": "Gaussian σ=5",       "fn": lambda i: attack_gaussian_noise(i, 5)},
    {"category": "Noise",                  "name": "Gaussian σ=15",      "fn": lambda i: attack_gaussian_noise(i, 15)},
    {"category": "Noise",                  "name": "Gaussian σ=25",      "fn": lambda i: attack_gaussian_noise(i, 25)},
    {"category": "Noise",                  "name": "Salt & Pepper 1%",   "fn": lambda i: attack_salt_pepper(i, 0.01)},
    {"category": "Noise",                  "name": "Salt & Pepper 3%",   "fn": lambda i: attack_salt_pepper(i, 0.03)},
    # Filtering
    {"category": "Filtering",              "name": "Gaussian Blur 3×3",  "fn": lambda i: attack_gaussian_blur(i, 3)},
    {"category": "Filtering",              "name": "Gaussian Blur 5×5",  "fn": lambda i: attack_gaussian_blur(i, 5)},
    {"category": "Filtering",              "name": "Gaussian Blur 7×7",  "fn": lambda i: attack_gaussian_blur(i, 7)},
    {"category": "Filtering",              "name": "Median Filter 3×3",  "fn": lambda i: attack_median_filter(i, 3)},
    {"category": "Filtering",              "name": "Median Filter 5×5",  "fn": lambda i: attack_median_filter(i, 5)},
    {"category": "Filtering",              "name": "Sharpening",         "fn": attack_sharpening},
    # Geometric
    {"category": "Cropping",               "name": "Crop 5%",            "fn": lambda i: attack_crop(i, 0.05)},
    {"category": "Cropping",               "name": "Crop 10%",           "fn": lambda i: attack_crop(i, 0.10)},
    {"category": "Cropping",               "name": "Crop 20%",           "fn": lambda i: attack_crop(i, 0.20)},
    {"category": "Rotation",               "name": "Rotate 5°",          "fn": lambda i: attack_rotate(i, 5)},
    {"category": "Rotation",               "name": "Rotate 10°",         "fn": lambda i: attack_rotate(i, 10)},
    {"category": "Rotation",               "name": "Rotate 30°",         "fn": lambda i: attack_rotate(i, 30)},
    {"category": "Scaling",                "name": "Scale 75%",          "fn": lambda i: attack_scale(i, 0.75)},
    {"category": "Scaling",                "name": "Scale 50%",          "fn": lambda i: attack_scale(i, 0.50)},
    {"category": "Scaling",                "name": "Scale 25%",          "fn": lambda i: attack_scale(i, 0.25)},
    # Tone / Color
    {"category": "Histogram Equalization", "name": "Hist. Equalization", "fn": attack_histogram_eq},
    {"category": "Brightness / Contrast",  "name": "Brightness +30",     "fn": lambda i: attack_brightness(i, 30)},
    {"category": "Brightness / Contrast",  "name": "Brightness −30",     "fn": lambda i: attack_brightness(i, -30)},
    {"category": "Brightness / Contrast",  "name": "Contrast ×1.3",      "fn": lambda i: attack_contrast(i, 1.3)},
    {"category": "Brightness / Contrast",  "name": "Contrast ×0.7",      "fn": lambda i: attack_contrast(i, 0.7)},
]
