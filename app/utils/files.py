from __future__ import annotations

import os
from fastapi import HTTPException, UploadFile

from app.core.config import ALLOWED_MIME_TYPES, MAGIC_BYTES, MAX_FILE_SIZE


def remove_file(path: str) -> None:
    try:
        os.remove(path)
    except Exception:
        # Best-effort cleanup; never fail the request because cleanup failed.
        pass


def validate_image_mime(file: UploadFile) -> None:
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: {ALLOWED_MIME_TYPES}",
        )


def validate_file_size(path: str) -> None:
    if os.path.getsize(path) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File {os.path.basename(path)} exceeds 10MB limit",
        )


def validate_magic_bytes(path: str) -> None:
    with open(path, "rb") as f:
        header = f.read(4)
    if not any(header.startswith(sig) for sig in MAGIC_BYTES):
        raise HTTPException(
            status_code=400,
            detail="File content does not match an allowed image format",
        )

