"""
Image processing utilities
"""
import io
import numpy as np
from PIL import Image
from typing import Tuple

from app.config import settings


def load_image_from_bytes(image_bytes: bytes) -> Tuple[Image.Image, np.ndarray]:
    """
    Load image from bytes and convert to both PIL and numpy formats
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        Tuple of (PIL Image, numpy array)
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    # Resize if too large
    if max(image.size) > settings.MAX_IMAGE_SIZE:
        image.thumbnail((settings.MAX_IMAGE_SIZE, settings.MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)
    
    image_np = np.array(image)
    return image, image_np


def save_image(image: Image.Image, path: str, quality: int = None) -> str:
    """
    Save PIL Image to disk
    
    Args:
        image: PIL Image object
        path: Output path
        quality: JPEG quality (default from settings)
        
    Returns:
        Path where image was saved
    """
    if quality is None:
        quality = settings.JPEG_QUALITY
    
    image.save(path, quality=quality, optimize=True)
    return path


def image_to_bytes(image: Image.Image, format: str = "JPEG") -> bytes:
    """
    Convert PIL Image to bytes
    
    Args:
        image: PIL Image object
        format: Output format (JPEG, PNG, etc.)
        
    Returns:
        Image as bytes
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format, quality=settings.JPEG_QUALITY)
    buffer.seek(0)
    return buffer.getvalue()
