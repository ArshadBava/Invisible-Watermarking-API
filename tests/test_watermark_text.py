"""
tests/test_watermark_text.py -- Unit tests for app/utils/watermark_text.py
"""
from __future__ import annotations

import numpy as np
import pytest
from fastapi import HTTPException

from app.utils.watermark_text import text_to_image
from app.core.config import MAX_TEXT_LENGTH


class TestTextToImage:
    def test_returns_numpy_array(self):
        img = text_to_image("Hello")
        assert isinstance(img, np.ndarray)

    def test_output_is_grayscale(self):
        img = text_to_image("Test")
        assert img.ndim == 2

    def test_output_dtype_uint8(self):
        img = text_to_image("Test")
        assert img.dtype == np.uint8

    def test_output_has_positive_dimensions(self):
        img = text_to_image("Hi")
        assert img.shape[0] > 0
        assert img.shape[1] > 0

    def test_longer_text_is_wider(self):
        short = text_to_image("Hi")
        long  = text_to_image("Hello World")
        assert long.shape[1] > short.shape[1]

    def test_single_character(self):
        img = text_to_image("A")
        assert img.shape[0] > 0 and img.shape[1] > 0

    def test_text_too_long_raises_http_exception(self):
        too_long = "x" * (MAX_TEXT_LENGTH + 1)
        with pytest.raises(HTTPException) as exc_info:
            text_to_image(too_long)
        assert exc_info.value.status_code == 400

    def test_max_length_text_is_accepted(self):
        max_text = "A" * MAX_TEXT_LENGTH
        img = text_to_image(max_text)
        assert img is not None

    def test_output_is_binary_ish(self):
        """Background should be white (~255) and text pixels should be dark."""
        img = text_to_image("W")
        assert img.max() == 255          # white background present
        assert img.min() < 128           # dark text pixels present

    def test_different_texts_produce_different_images(self):
        img_a = text_to_image("AAA")
        img_b = text_to_image("ZZZ")
        # resize to same size before comparing
        import cv2
        h = min(img_a.shape[0], img_b.shape[0])
        w = min(img_a.shape[1], img_b.shape[1])
        a = img_a[:h, :w].astype(np.float32)
        b = img_b[:h, :w].astype(np.float32)
        assert not np.array_equal(a, b)
