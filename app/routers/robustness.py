from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import DEFAULT_ALPHA
from app.services.robustness import ATTACK_SUITE
from app.services.dwt_watermark import extract_dwt
from app.services.dct_watermark import extract_dct
from app.services.dft_watermark import extract_dft
from app.utils.files import remove_file, validate_file_size, validate_image_mime, validate_magic_bytes
from app.utils.metrics import normalized_correlation
from app.utils.watermark_text import text_to_image


router = APIRouter(prefix="/robustness", tags=["robustness"])


def _status(nc: float) -> str:
    if nc >= 0.75:
        return "robust"
    if nc >= 0.50:
        return "moderate"
    return "degraded"


@router.post("/test")
async def robustness_test(
    original: UploadFile = File(...),
    watermarked: UploadFile = File(...),
    watermark_type: str = Form(...),         # "image" or "text"
    watermark_input: UploadFile = File(None),
    watermark_text: str = Form(None),
    alpha: float = Form(DEFAULT_ALPHA, ge=0.01, le=0.5),
    method: str = Form("dwt"),               # "dwt" | "dct" | "dft"
):
    orig_path = wm_path = ref_path = None
    try:
        if method not in ("dwt", "dct", "dft"):
            raise HTTPException(status_code=400, detail="method must be 'dwt', 'dct', or 'dft'")

        validate_image_mime(original)
        validate_image_mime(watermarked)
        if watermark_type == "image" and watermark_input:
            validate_image_mime(watermark_input)

        tmp = tempfile.gettempdir()
        uid = str(uuid.uuid4())
        orig_path = os.path.join(tmp, f"rb_orig_{uid}.png")
        wm_path   = os.path.join(tmp, f"rb_wm_{uid}.png")
        ref_path  = os.path.join(tmp, f"rb_ref_{uid}.png")

        with open(orig_path, "wb") as f:
            shutil.copyfileobj(original.file, f)
        with open(wm_path, "wb") as f:
            shutil.copyfileobj(watermarked.file, f)

        for p in (orig_path, wm_path):
            validate_file_size(p)
            validate_magic_bytes(p)

        # Build reference watermark
        if watermark_type == "image":
            if not watermark_input:
                raise HTTPException(status_code=400, detail="watermark_input required for image type")
            with open(ref_path, "wb") as f:
                shutil.copyfileobj(watermark_input.file, f)
            validate_file_size(ref_path)
            validate_magic_bytes(ref_path)
        elif watermark_type == "text":
            if not watermark_text:
                raise HTTPException(status_code=400, detail="watermark_text required for text type")
            cv2.imwrite(ref_path, text_to_image(watermark_text))
        else:
            raise HTTPException(status_code=400, detail="watermark_type must be 'image' or 'text'")

        orig_img = cv2.imread(orig_path)
        ref_wm   = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        if orig_img is None:
            raise HTTPException(status_code=400, detail="Cannot read original image")
        if ref_wm is None:
            raise HTTPException(status_code=400, detail="Cannot read reference watermark")

        ref_h, ref_w = ref_wm.shape
        wm_size = (ref_h, ref_w)

        def _extract(attacked: np.ndarray) -> np.ndarray:
            if method == "dwt":
                raw = extract_dwt(orig_img, attacked, alpha)
                if raw.shape[:2] != wm_size:
                    return cv2.resize(raw, (ref_w, ref_h), interpolation=cv2.INTER_CUBIC)
                return raw
            if method == "dct":
                return extract_dct(orig_img, attacked, alpha, watermark_size=wm_size)
            return extract_dft(orig_img, attacked, alpha, watermark_size=wm_size)

        # Run all attacks
        results: list[dict[str, Any]] = []
        for entry in ATTACK_SUITE:
            try:
                wm_img   = cv2.imread(wm_path)
                attacked = entry["fn"](wm_img)
                ext      = _extract(attacked)
                nc       = float(normalized_correlation(ref_wm, ext))
            except Exception:
                nc = 0.0
            results.append({
                "category": entry["category"],
                "name":     entry["name"],
                "nc":       round(nc, 4),
                "status":   _status(nc),
            })

        total    = len(results)
        robust   = sum(1 for r in results if r["status"] == "robust")
        moderate = sum(1 for r in results if r["status"] == "moderate")
        degraded = sum(1 for r in results if r["status"] == "degraded")
        avg_nc   = round(sum(r["nc"] for r in results) / total, 4) if total else 0.0

        return JSONResponse({
            "method":        method.upper(),
            "alpha":         alpha,
            "total_attacks": total,
            "robust":        robust,
            "moderate":      moderate,
            "degraded":      degraded,
            "average_nc":    avg_nc,
            "results":       results,
        })

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for p in (orig_path, wm_path, ref_path):
            if p and os.path.exists(p):
                remove_file(p)
