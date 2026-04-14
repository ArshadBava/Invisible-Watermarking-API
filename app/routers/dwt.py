from __future__ import annotations

import os
import shutil
import tempfile
import uuid

import cv2
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from app.core.config import DEFAULT_ALPHA
from app.services.dwt_watermark import embed_dwt, extract_dwt, embed_dwt_blind, extract_dwt_blind
from app.utils.files import (
    remove_file,
    validate_file_size,
    validate_image_mime,
    validate_magic_bytes,
)
from app.utils.metrics import normalized_correlation
from app.utils.watermark_text import text_to_image


router = APIRouter(tags=["dwt"])


@router.post("/embed")
async def embed_watermark_api(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    watermark_type: str = Form(...),  # "image" or "text"
    watermark: UploadFile = File(None),
    watermark_text: str = Form(None),
    alpha: float = Form(DEFAULT_ALPHA, ge=0.01, le=0.5),
):
    original_path = watermark_path = output_path = None
    try:
        validate_image_mime(image)
        if watermark:
            validate_image_mime(watermark)

        temp_dir = tempfile.gettempdir()
        unique_id = str(uuid.uuid4())

        original_path = os.path.join(temp_dir, f"orig_{unique_id}.png")
        watermark_path = os.path.join(temp_dir, f"wm_{unique_id}.png")
        output_path = os.path.join(temp_dir, f"watermarked_{unique_id}.png")

        with open(original_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        validate_file_size(original_path)
        validate_magic_bytes(original_path)

        if watermark_type == "image":
            if watermark is None:
                raise HTTPException(
                    status_code=400,
                    detail="Watermark file is required when type is 'image'",
                )
            with open(watermark_path, "wb") as buffer:
                shutil.copyfileobj(watermark.file, buffer)
            validate_file_size(watermark_path)
            validate_magic_bytes(watermark_path)
            watermark_img = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        elif watermark_type == "text":
            if not watermark_text:
                raise HTTPException(
                    status_code=400,
                    detail="Watermark text is required when type is 'text'",
                )
            watermark_img = text_to_image(watermark_text)
            cv2.imwrite(watermark_path, watermark_img)
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid watermark_type. Must be 'image' or 'text'",
            )

        original = cv2.imread(original_path)
        if original is None or watermark_img is None:
            raise HTTPException(status_code=400, detail="Invalid image processing")

        result, psnr_val = embed_dwt(original, watermark_img, alpha)
        cv2.imwrite(output_path, result)

        remove_file(original_path)
        if watermark_type == "image":
            remove_file(watermark_path)

        cleanup_task = BackgroundTask(remove_file, output_path)
        return FileResponse(
            output_path,
            media_type="image/png",
            filename="watermarked.png",
            headers={
                "X-PSNR": f"{psnr_val:.2f}",
                "X-Alpha": f"{alpha:.4f}",
                "X-Method": "DWT",
            },
            background=cleanup_task,
        )
    except Exception as e:
        for p in (original_path, watermark_path, output_path):
            if p and os.path.exists(p):
                remove_file(p)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


# ── Blind DWT endpoints ────────────────────────────────────────────────────────

@router.post("/embed-blind")
async def embed_dwt_blind_api(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    watermark_type: str = Form(...),
    watermark: UploadFile = File(None),
    watermark_text: str = Form(None),
    alpha: float = Form(DEFAULT_ALPHA, ge=0.01, le=0.5),
):
    original_path = watermark_path = output_path = None
    try:
        validate_image_mime(image)
        if watermark:
            validate_image_mime(watermark)
        temp_dir = tempfile.gettempdir()
        uid = str(uuid.uuid4())
        original_path  = os.path.join(temp_dir, f"orig_{uid}.png")
        watermark_path = os.path.join(temp_dir, f"wm_{uid}.png")
        output_path    = os.path.join(temp_dir, f"dwt_blind_wm_{uid}.png")
        with open(original_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
        validate_file_size(original_path); validate_magic_bytes(original_path)
        if watermark_type == "image":
            if watermark is None:
                raise HTTPException(status_code=400, detail="Watermark file required")
            with open(watermark_path, "wb") as f:
                shutil.copyfileobj(watermark.file, f)
            validate_file_size(watermark_path); validate_magic_bytes(watermark_path)
            watermark_img = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
        elif watermark_type == "text":
            if not watermark_text:
                raise HTTPException(status_code=400, detail="Watermark text required")
            watermark_img = text_to_image(watermark_text)
            cv2.imwrite(watermark_path, watermark_img)
        else:
            raise HTTPException(status_code=400, detail="Invalid watermark_type")
        cover = cv2.imread(original_path)
        if cover is None or watermark_img is None:
            raise HTTPException(status_code=400, detail="Invalid image processing")
        result, psnr_val, (wm_h, wm_w) = embed_dwt_blind(cover, watermark_img, alpha)
        cv2.imwrite(output_path, result)
        remove_file(original_path)
        if watermark_type == "image":
            remove_file(watermark_path)
        cleanup_task = BackgroundTask(remove_file, output_path)
        return FileResponse(output_path, media_type="image/png",
            filename="watermarked_blind.png",
            headers={"X-PSNR": f"{psnr_val:.2f}", "X-Alpha": f"{alpha:.4f}",
                     "X-Method": "DWT-Blind", "X-WM-Height": str(wm_h), "X-WM-Width": str(wm_w)},
            background=cleanup_task)
    except Exception as e:
        for p in (original_path, watermark_path, output_path):
            if p and os.path.exists(p): remove_file(p)
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-blind")
async def extract_dwt_blind_api(
    background_tasks: BackgroundTasks,
    watermarked: UploadFile = File(...),
    watermark_type: str = Form(None),
    watermark_input: UploadFile = File(None),
    watermark_text: str = Form(None),
    alpha: float = Form(DEFAULT_ALPHA, ge=0.01, le=0.5),
    wm_width: int  = Form(None, ge=1, le=8192),
    wm_height: int = Form(None, ge=1, le=8192),
):
    wm_path = input_wm_path = output_path = None
    try:
        validate_image_mime(watermarked)
        temp_dir = tempfile.gettempdir()
        uid = str(uuid.uuid4())
        wm_path       = os.path.join(temp_dir, f"wm_{uid}.png")
        input_wm_path = os.path.join(temp_dir, f"input_wm_{uid}.png")
        output_path   = os.path.join(temp_dir, f"dwt_blind_ext_{uid}.png")
        with open(wm_path, "wb") as f:
            shutil.copyfileobj(watermarked.file, f)
        validate_file_size(wm_path); validate_magic_bytes(wm_path)
        has_ref = False
        if watermark_type == "image" and watermark_input:
            validate_image_mime(watermark_input)
            with open(input_wm_path, "wb") as f:
                shutil.copyfileobj(watermark_input.file, f)
            has_ref = True
        elif watermark_type == "text" and watermark_text:
            cv2.imwrite(input_wm_path, text_to_image(watermark_text))
            has_ref = True
        wm_img = cv2.imread(wm_path)
        if wm_img is None:
            raise HTTPException(status_code=400, detail="Cannot read watermarked image")
        wm_size   = (wm_height, wm_width) if (wm_height and wm_width) else None
        extracted = extract_dwt_blind(wm_img, alpha, watermark_size=wm_size)
        cv2.imwrite(output_path, extracted)
        nc_val = None
        if has_ref:
            ref = cv2.imread(input_wm_path, cv2.IMREAD_GRAYSCALE)
            if ref is not None:
                ref_r = cv2.resize(ref, (extracted.shape[1], extracted.shape[0]))
                nc_val = normalized_correlation(ref_r, extracted)
        remove_file(wm_path)
        if has_ref: remove_file(input_wm_path)
        headers = {"X-Method": "DWT-Blind"}
        if nc_val is not None: headers["X-NC"] = f"{nc_val:.4f}"
        cleanup_task = BackgroundTask(remove_file, output_path)
        return FileResponse(output_path, media_type="image/png",
            filename="extracted_blind.png", headers=headers, background=cleanup_task)
    except Exception as e:
        for p in (wm_path, input_wm_path, output_path):
            if p and os.path.exists(p): remove_file(p)
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/extract")
async def extract_watermark_api(
    background_tasks: BackgroundTasks,
    original: UploadFile = File(...),
    watermarked: UploadFile = File(...),
    watermark_type: str = Form(None),
    watermark_input: UploadFile = File(None),
    watermark_text: str = Form(None),
    alpha: float = Form(DEFAULT_ALPHA, ge=0.01, le=0.5),
):
    orig_path = wm_path = input_wm_path = output_path = None
    try:
        validate_image_mime(original)
        validate_image_mime(watermarked)
        if watermark_type == "image" and watermark_input:
            validate_image_mime(watermark_input)

        temp_dir = tempfile.gettempdir()
        unique_id = str(uuid.uuid4())

        orig_path = os.path.join(temp_dir, f"orig_{unique_id}.png")
        wm_path = os.path.join(temp_dir, f"wm_{unique_id}.png")
        input_wm_path = os.path.join(temp_dir, f"input_wm_{unique_id}.png")
        output_path = os.path.join(temp_dir, f"extracted_{unique_id}.png")

        with open(orig_path, "wb") as buffer:
            shutil.copyfileobj(original.file, buffer)
        with open(wm_path, "wb") as buffer:
            shutil.copyfileobj(watermarked.file, buffer)
            
        has_reference_wm = False
        
        if watermark_type == "image" and watermark_input:
            with open(input_wm_path, "wb") as buffer:
                shutil.copyfileobj(watermark_input.file, buffer)
            validate_file_size(input_wm_path)
            validate_magic_bytes(input_wm_path)
            has_reference_wm = True
        elif watermark_type == "text" and watermark_text:
            ref_img = text_to_image(watermark_text)
            cv2.imwrite(input_wm_path, ref_img)
            has_reference_wm = True

        validate_file_size(orig_path)
        validate_file_size(wm_path)

        validate_magic_bytes(orig_path)
        validate_magic_bytes(wm_path)

        orig_img = cv2.imread(orig_path)
        wm_img = cv2.imread(wm_path)
        if orig_img is None or wm_img is None:
            raise HTTPException(status_code=400, detail="Invalid image files")

        watermark_extracted = extract_dwt(orig_img, wm_img, alpha)
        cv2.imwrite(output_path, watermark_extracted)

        nc_val = None
        if has_reference_wm:
            original_watermark = cv2.imread(input_wm_path, cv2.IMREAD_GRAYSCALE)
            if original_watermark is None:
                raise HTTPException(status_code=400, detail="Invalid watermark image file")
            extracted_resized = cv2.resize(
                watermark_extracted,
                (original_watermark.shape[1], original_watermark.shape[0]),
            )
            nc_val = normalized_correlation(original_watermark, extracted_resized)

        remove_file(orig_path)
        remove_file(wm_path)
        if has_reference_wm:
            remove_file(input_wm_path)

        cleanup_task = BackgroundTask(remove_file, output_path)
        headers = {"X-Method": "DWT"}
        if nc_val is not None:
            headers["X-NC"] = f"{nc_val:.4f}"

        return FileResponse(
            output_path,
            media_type="image/png",
            filename="extracted.png",
            headers=headers,
            background=cleanup_task,
        )
    except Exception as e:
        for p in (orig_path, wm_path, input_wm_path, output_path):
            if p and os.path.exists(p):
                remove_file(p)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

