from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent.parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
MAX_TEXT_LENGTH = 50
DEFAULT_ALPHA = 0.1

ALLOWED_MIME_TYPES = {"image/png", "image/jpeg", "image/bmp"}
MAGIC_BYTES = {
    b"\x89PNG": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"BM": "image/bmp",
}

