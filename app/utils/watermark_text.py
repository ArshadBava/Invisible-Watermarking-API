from __future__ import annotations

import cv2
import numpy as np
from fastapi import HTTPException

from app.core.config import MAX_TEXT_LENGTH


def text_to_image(text: str) -> np.ndarray:
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text exceeds {MAX_TEXT_LENGTH} character limit",
        )

    font_scale = 1.6
    font_thickness = 3
    font = cv2.FONT_HERSHEY_SIMPLEX

    (text_width, text_height), _baseline = cv2.getTextSize(
        text, font, font_scale, font_thickness
    )

    width = text_width + 40
    height = text_height + 40
    img = np.ones((height, width), dtype=np.uint8) * 255

    text_x = 20
    text_y = height - 20
    cv2.putText(
        img,
        text,
        (text_x, text_y),
        font,
        font_scale,
        0,
        font_thickness,
        cv2.LINE_AA,
    )
    return img

