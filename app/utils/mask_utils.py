"""
Mask visualization and manipulation utilities
"""
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from typing import Tuple, Optional

from app.utils.logger import logger


def overlay_mask_on_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay mask on image with transparency
    
    Args:
        image: RGB image (H, W, 3)
        mask: Binary mask (H, W) - boolean or uint8
        color: RGB color for mask overlay (default: red)
        alpha: Transparency (0=transparent, 1=opaque)
        
    Returns:
        Image with mask overlay (H, W, 3)
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # Ensure mask is boolean
    if mask.dtype == np.uint8:
        mask = mask > 127
    
    # Create colored overlay
    overlay = image.copy()
    overlay[mask] = color
    
    # Blend with original image
    result = (alpha * overlay + (1 - alpha) * image).astype(np.uint8)
    
    return result


def overlay_mask_with_contour(
    image: np.ndarray,
    mask: np.ndarray,
    fill_color: Tuple[int, int, int] = (255, 0, 0),
    contour_color: Tuple[int, int, int] = (0, 255, 0),
    fill_alpha: float = 0.3,
    contour_width: int = 3
) -> np.ndarray:
    """
    Overlay mask with filled area and contour outline
    
    Args:
        image: RGB image (H, W, 3)
        mask: Binary mask (H, W)
        fill_color: RGB color for filled area
        contour_color: RGB color for contour
        fill_alpha: Transparency for filled area
        contour_width: Width of contour line
        
    Returns:
        Image with mask overlay and contour
    """
    import cv2
    
    # First add filled overlay
    result = overlay_mask_on_image(image, mask, fill_color, fill_alpha)
    
    # Find contours
    mask_uint8 = (mask * 255).astype(np.uint8) if mask.dtype == bool else mask
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours
    cv2.drawContours(result, contours, -1, contour_color, contour_width)
    
    return result


def mask_to_png(
    mask: np.ndarray,
    output_path: Path,
    invert: bool = False
) -> Path:
    """
    Save mask as PNG file
    
    Args:
        mask: Binary mask (H, W) - boolean or uint8
        output_path: Path to save PNG
        invert: If True, invert mask (white<->black)
        
    Returns:
        Path to saved file
    """
    # Convert to uint8
    if mask.dtype == bool:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask
    
    # Invert if requested
    if invert:
        mask_uint8 = 255 - mask_uint8
    
    # Save
    mask_pil = Image.fromarray(mask_uint8)
    mask_pil.save(output_path)
    
    logger.info(f"💾 Saved mask to: {output_path}")
    return output_path


def create_side_by_side_comparison(
    original: np.ndarray,
    masked: np.ndarray,
    output_path: Optional[Path] = None
) -> np.ndarray:
    """
    Create side-by-side comparison of original and masked image
    
    Args:
        original: Original image (H, W, 3)
        masked: Image with mask overlay (H, W, 3)
        output_path: Optional path to save comparison
        
    Returns:
        Side-by-side comparison image
    """
    # Ensure same size
    if original.shape != masked.shape:
        raise ValueError("Images must have same dimensions")
    
    # Concatenate horizontally
    comparison = np.hstack([original, masked])
    
    # Add labels
    comparison_pil = Image.fromarray(comparison)
    draw = ImageDraw.Draw(comparison_pil)
    
    # Add text labels (simple, no font)
    h, w = original.shape[:2]
    draw.text((10, 10), "Original", fill=(255, 255, 255))
    draw.text((w + 10, 10), "Segmented", fill=(255, 255, 255))
    
    comparison = np.array(comparison_pil)
    
    # Save if path provided
    if output_path:
        Image.fromarray(comparison).save(output_path)
        logger.info(f"💾 Saved comparison to: {output_path}")
    
    return comparison


def apply_mask_to_image(
    image: np.ndarray,
    mask: np.ndarray,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> np.ndarray:
    """
    Apply mask to image - keep masked region, replace rest with background
    
    Args:
        image: RGB image (H, W, 3)
        mask: Binary mask (H, W) - True = keep, False = replace
        background_color: RGB color for background
        
    Returns:
        Masked image
    """
    # Ensure mask is boolean
    if mask.dtype == np.uint8:
        mask = mask > 127
    
    # Create result
    result = np.full_like(image, background_color)
    result[mask] = image[mask]
    
    return result


def get_mask_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Get bounding box of mask
    
    Args:
        mask: Binary mask (H, W)
        
    Returns:
        Tuple of (x1, y1, x2, y2)
    """
    # Ensure mask is boolean
    if mask.dtype == np.uint8:
        mask = mask > 127
    
    # Find non-zero pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return (0, 0, 0, 0)
    
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    return (int(x1), int(y1), int(x2), int(y2))


def crop_to_mask(
    image: np.ndarray,
    mask: np.ndarray,
    padding: int = 10
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop image to mask bounding box with padding
    
    Args:
        image: RGB image (H, W, 3)
        mask: Binary mask (H, W)
        padding: Padding around mask in pixels
        
    Returns:
        Tuple of (cropped_image, bbox)
    """
    x1, y1, x2, y2 = get_mask_bbox(mask)
    
    # Add padding
    h, w = image.shape[:2]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    # Crop
    cropped = image[y1:y2, x1:x2]
    
    return cropped, (x1, y1, x2, y2)


def visualize_multiple_masks(
    image: np.ndarray,
    masks: list[np.ndarray],
    colors: Optional[list[Tuple[int, int, int]]] = None,
    alpha: float = 0.4
) -> np.ndarray:
    """
    Visualize multiple masks on same image with different colors
    
    Args:
        image: RGB image (H, W, 3)
        masks: List of binary masks
        colors: List of RGB colors (auto-generated if None)
        alpha: Transparency
        
    Returns:
        Image with all masks overlaid
    """
    if colors is None:
        # Generate distinct colors
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
    
    result = image.copy()
    
    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        result = overlay_mask_on_image(result, mask, color, alpha)
    
    return result
