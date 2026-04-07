"""
Edge Detection Module for ControlNet Generation
================================================
Trich xuat canh (edges) tu anh de su dung voi ControlNet.

Auto Canny mode: Tu dong tinh threshold dua tren median pixel.
Manual Canny mode: Nguoi dung tu chi dinh threshold.

Usage:
    from app.core.edge_detection import auto_canny_edges, detect_canny_edges

    # Auto mode (recommended)
    edges = auto_canny_edges(image_np)

    # Manual mode
    edges = detect_canny_edges(image_np, low_threshold=50, high_threshold=150)
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from app.utils.logger import logger


def auto_canny_edges(
    image: np.ndarray,
    sigma: float = 0.33,
    blur_kernel: int = 5,
) -> np.ndarray:
    """
    Tu dong phat hien canh (edges) bang Canny voi threshold tu dong.

    Thuat toan tinh threshold dua tren median do sang cua anh:
        lower = max(0,   (1.0 - sigma) * median)
        upper = min(255, (1.0 + sigma) * median)

    Gia tri sigma = 0.33 la default tot cho anh noi that (interior):
    - Sigma nho hon (<0.2): Strict, chi bat canh ro rang -> it nhieu, nhung co the bo qua chi tiet
    - Sigma lon hon (>0.5): Rong hon, bat nhieu canh hon -> nhieu noise hon

    Args:
        image: Anh dau vao (numpy array), RGB hoac grayscale, uint8 [0-255]
        sigma: He so dieu chinh do rong cua nguong (default: 0.33)
        blur_kernel: Kich thuoc kernel Gaussian blur truoc khi Canny (default: 5x5)
                     Blur giup giam nhieu, nen dung so le (3, 5, 7)

    Returns:
        edges: Anh canh (grayscale, uint8), pixel=255 la canh, pixel=0 la nen

    Example:
        >>> img = cv2.imread("room.jpg")
        >>> edges = auto_canny_edges(img)
        >>> density = get_edge_density(edges)
        >>> print(f"Edge density: {density:.1f}%")  # Ideal: 5-15%
    """
    # Chuyen sang grayscale neu can
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Gaussian blur de giam nhieu truoc Canny
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # Tinh threshold tu dong dua tren median
    median = float(np.median(blurred))
    lower = max(0.0, (1.0 - sigma) * median)
    upper = min(255.0, (1.0 + sigma) * median)

    # Chay Canny
    edges = cv2.Canny(blurred, lower, upper)

    # Log thong tin
    density = get_edge_density(edges)
    logger.info(
        f"🔍 Auto Canny edges | sigma={sigma:.2f} | "
        f"median={median:.1f} | lower={lower:.1f} | upper={upper:.1f} | "
        f"edge_density={density:.2f}%"
    )

    if density < 3.0:
        logger.warning(
            f"⚠️  Edge density thap ({density:.1f}%). "
            f"Thu giam sigma (hien tai {sigma}) hoac kiem tra anh dau vao."
        )
    elif density > 20.0:
        logger.warning(
            f"⚠️  Edge density cao ({density:.1f}%). "
            f"Thu tang sigma (hien tai {sigma}) de giam nhieu."
        )

    return edges


def detect_canny_edges(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    blur_kernel: int = 5,
) -> np.ndarray:
    """
    Phat hien canh bang Canny voi threshold thu cong (manual mode).

    Su dung khi muon kiem soat chinh xac nguong, hoac khi auto mode
    cho ket qua chua tot voi mot anh cu the.

    Args:
        image: Anh dau vao (numpy array), RGB hoac grayscale, uint8
        low_threshold: Nguong duoi Canny (default: 50)
        high_threshold: Nguong tren Canny (default: 150)
                        Khuyen nghi: high = 2x hoac 3x so voi low
        blur_kernel: Kich thuoc Gaussian blur kernel (default: 5)

    Returns:
        edges: Anh canh (grayscale, uint8), pixel=255 la canh

    Example:
        >>> edges = detect_canny_edges(img, low_threshold=30, high_threshold=100)
    """
    # Chuyen sang grayscale neu can
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()

    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # Canny voi threshold thu cong
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    density = get_edge_density(edges)
    logger.info(
        f"🔍 Manual Canny edges | "
        f"low={low_threshold} | high={high_threshold} | "
        f"edge_density={density:.2f}%"
    )

    return edges


def get_edge_density(edges: np.ndarray) -> float:
    """
    Tinh ty le phan tram so pixel la canh tren tong so pixel.

    Ideal range cho anh noi that: 5-15%
    - < 5%: Qua it canh, co the mat cau truc phong
    - 5-15%: Tot de dung voi ControlNet
    - > 15%: Qua nhieu canh / nhieu

    Args:
        edges: Anh canh binary (uint8), pixel=255 la canh

    Returns:
        density: Phan tram pixel la canh (0.0 - 100.0)
    """
    if edges.size == 0:
        return 0.0
    return float(np.count_nonzero(edges) / edges.size * 100)


def edges_to_rgb(edges: np.ndarray) -> np.ndarray:
    """
    Chuyen anh edge (grayscale) sang RGB de hien thi hoac luu thanh JPEG/PNG.

    Args:
        edges: Anh canh grayscale (uint8)

    Returns:
        Anh RGB (uint8, shape HxWx3)
    """
    if len(edges.shape) == 2:
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return edges


def validate_image_for_edges(image: np.ndarray) -> Tuple[bool, str]:
    """
    Kiem tra anh dau vao co hop le de xu ly edge detection khong.

    Args:
        image: Anh numpy array

    Returns:
        (is_valid, error_message): Neu hop le thi error_message = ""
    """
    if image is None:
        return False, "Anh la None"

    if not isinstance(image, np.ndarray):
        return False, f"Anh phai la numpy array, nhan duoc: {type(image)}"

    if image.ndim not in (2, 3):
        return False, f"Anh phai la 2D (grayscale) hoac 3D (RGB), nhan duoc: {image.ndim}D"

    if image.dtype != np.uint8:
        return False, f"Anh phai la uint8, nhan duoc: {image.dtype}"

    h, w = image.shape[:2]
    if h < 64 or w < 64:
        return False, f"Anh qua nho: {w}x{h}. Toi thieu 64x64 pixels"

    return True, ""
