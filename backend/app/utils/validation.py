"""
Input validation utilities.
"""
import mimetypes
from typing import List
from fastapi import UploadFile, HTTPException, status

from app.config import get_settings


def detect_image_type(content: bytes) -> str:
    """Detect image type from magic bytes."""
    if len(content) < 12:
        return ""
    
    # Check magic bytes for common image formats
    # JPEG: FF D8 FF
    if content[:3] == b'\xFF\xD8\xFF':
        return "jpeg"
    # PNG: 89 50 4E 47 0D 0A 1A 0A
    if content[:8] == b'\x89PNG\r\n\x1a\n':
        return "png"
    # WebP: RIFF....WEBP
    if content[:4] == b'RIFF' and content[8:12] == b'WEBP':
        return "webp"
    
    # Fallback to mimetypes
    mime_type, _ = mimetypes.guess_type("dummy")
    return mime_type or ""


async def validate_image_file(file: UploadFile) -> bytes:
    """
    Validate a single image file.
    Returns file content as bytes.
    """
    settings = get_settings()
    
    # Check filename
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must have a filename"
        )
    
    # Check extension
    ext = "." + file.filename.split(".")[-1].lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file extension. Allowed: {', '.join(settings.allowed_extensions)}"
        )
    
    # Read content
    content = await file.read()
    await file.seek(0)
    
    # Check file size
    if len(content) > settings.max_file_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds maximum size of {settings.max_file_size / 1024 / 1024:.0f}MB"
        )
    
    # Verify it's actually an image
    detected_type = detect_image_type(content)
    allowed_types = ["jpeg", "png", "webp"]
    if detected_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File is not a valid image. Allowed: JPEG, PNG, WebP"
        )
    
    return content


def validate_uuid(uuid_str: str) -> bool:
    """Validate UUID string format."""
    import uuid
    try:
        uuid.UUID(uuid_str)
        return True
    except ValueError:
        return False


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage."""
    import re
    # Remove path components
    filename = filename.split("/")[-1].split("\\")[-1]
    # Remove special characters
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit(".", 1) if "." in filename else (filename, "")
        filename = name[:255 - len(ext) - 1] + "." + ext if ext else name[:255]
    return filename