"""
Image processing service for validation and preprocessing.
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
        return "image/jpeg"
    # PNG: 89 50 4E 47 0D 0A 1A 0A
    if content[:8] == b'\x89PNG\r\n\x1a\n':
        return "image/png"
    # WebP: RIFF....WEBP
    if content[:4] == b'RIFF' and content[8:12] == b'WEBP':
        return "image/webp"
    
    # Fallback to mimetypes
    mime_type, _ = mimetypes.guess_type("dummy")
    return mime_type or ""


async def validate_upload_files(
    files: list,
    settings
) -> list:
    """
    Validate uploaded files against size and type constraints.
    Returns list of valid files.
    """
    if not files:
        raise ValueError("No files provided")
    
    if len(files) > settings.max_files_per_job:
        raise ValueError(
            f"Too many files. Maximum {settings.max_files_per_job} files per job."
        )
    
    valid_files = []
    total_size = 0
    
    for file in files:
        # Check filename
        if not file.filename:
            continue
            
        # Check extension
        ext = "." + file.filename.split(".")[-1].lower()
        if ext not in settings.allowed_extensions:
            raise ValueError(
                f"File '{file.filename}' has unsupported extension. "
                f"Allowed: {', '.join(settings.allowed_extensions)}"
            )
        
        # Check content type
        content_type = file.content_type or ""
        allowed_types = ["image/jpeg", "image/png", "image/webp"]
        if content_type not in allowed_types:
            # Try to detect from content using magic bytes
            content = await file.read(1024)
            await file.seek(0)
            detected_type = detect_image_type(content)
            if detected_type not in allowed_types:
                raise ValueError(
                    f"File '{file.filename}' is not a valid image. "
                    f"Allowed types: JPEG, PNG, WebP"
                )
        
        # Check file size
        content = await file.read()
        await file.seek(0)
        file_size = len(content)
        
        if file_size > settings.max_file_size:
            raise ValueError(
                f"File '{file.filename}' exceeds maximum size of "
                f"{settings.max_file_size / 1024 / 1024:.0f}MB"
            )
        
        total_size += file_size
        if total_size > 100_000_000:  # 100MB total limit
            raise ValueError("Total upload size exceeds 100MB limit")
        
        valid_files.append(file)
    
    return valid_files


def get_image_dimensions(image_data: bytes) -> tuple:
    """Get image dimensions without fully loading."""
    import cv2
    import numpy as np
    
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is None:
        return (0, 0)
    height, width = img.shape[:2]
    return (width, height)


def resize_image_if_needed(
    image_data: bytes,
    max_dimension: int
) -> bytes:
    """Resize image if it exceeds max dimension."""
    import cv2
    import numpy as np
    
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return image_data
    
    height, width = img.shape[:2]
    
    if max(width, height) <= max_dimension:
        return image_data
    
    # Calculate new dimensions
    if width > height:
        new_width = max_dimension
        new_height = int(height * max_dimension / width)
    else:
        new_height = max_dimension
        new_width = int(width * max_dimension / height)
    
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    # Encode back to bytes
    _, encoded = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return encoded.tobytes()


def extract_face_thumbnail(
    image_data: bytes,
    bbox: list,
    size: tuple = (64, 64)
) -> bytes:
    """Extract and resize face region as thumbnail."""
    import cv2
    import numpy as np
    
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return b""
    
    x, y, w, h = bbox
    
    # Add padding
    padding = int(max(w, h) * 0.2)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(img.shape[1], x + w + padding)
    y2 = min(img.shape[0], y + h + padding)
    
    face_img = img[y1:y2, x1:x2]
    
    if face_img.size == 0:
        return b""
    
    # Resize to thumbnail size
    thumbnail = cv2.resize(face_img, size, interpolation=cv2.INTER_AREA)
    
    # Encode as JPEG
    _, encoded = cv2.imencode('.jpg', thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return encoded.tobytes()