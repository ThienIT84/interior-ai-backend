"""
Generation API Endpoints
========================
Endpoint cho viec tao thiet ke noi that moi su dung ControlNet.

Week 3 Tasks:
  - 3.1.3: POST /preview-edges  <- Hien tai implement
  - 3.2:   POST /generate-design (TODO - Week 3 Day 16-17)
  - 3.4.2: GET  /styles          (TODO - Week 3 Day 18)
  - 3.4.3: POST /generate-batch  (TODO - Week 3 Day 18)
"""

import io
import time
import uuid
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from PIL import Image

from app.config import settings
from app.core.edge_detection import (
    auto_canny_edges,
    detect_canny_edges,
    edges_to_rgb,
    get_edge_density,
    validate_image_for_edges,
)
from app.utils.logger import logger

router = APIRouter()


# ---------------------------------------------------------------------------
# Task 3.1.3 - Edge Preview Endpoint
# ---------------------------------------------------------------------------

@router.post("/preview-edges")
async def preview_edges(
    image_id: str = Query(..., description="ID anh da upload truoc do"),
    mode: str = Query("auto", description="Che do phat hien canh: 'auto' hoac 'manual'"),
    sigma: float = Query(0.33, ge=0.05, le=1.0, description="He so sigma cho auto mode (default: 0.33)"),
    low_threshold: int = Query(50, ge=0, le=255, description="Nguong duoi cho manual mode (default: 50)"),
    high_threshold: int = Query(150, ge=0, le=255, description="Nguong tren cho manual mode (default: 150)"),
):
    """
    Tao va tra ve anh edge (Canny) de debug / preview truoc khi dung ControlNet.

    **Auto mode** (khuyen nghi):
    - Tu dong tinh threshold dua tren median do sang cua anh
    - Dieu chinh `sigma` de kiem soat do nhi (default sigma=0.33 tot cho anh noi that)

    **Manual mode**:
    - Tu chi dinh `low_threshold` va `high_threshold`
    - Phu hop khi anh co dac diem dac biet (qua sang/qua toi)

    **Edge density ideal range**: 5-15% cho anh noi that
    - < 5%: Co the mat cau truc phong, thu giam sigma
    - > 15%: Qua nhieu nhieu, thu tang sigma

    Args:
        image_id: ID cua anh da upload qua /api/v1/segmentation/upload
        mode: "auto" (default) hoac "manual"
        sigma: He so auto threshold (chi dung khi mode=auto)
        low_threshold: Nguong Canny thap (chi dung khi mode=manual)
        high_threshold: Nguong Canny cao (chi dung khi mode=manual)

    Returns:
        PNG edge image + header `X-Edge-Density` (% pixel la canh)

    Example:
        ```
        POST /api/v1/generation/preview-edges?image_id=abc123&mode=auto&sigma=0.33
        POST /api/v1/generation/preview-edges?image_id=abc123&mode=manual&low_threshold=30&high_threshold=100
        ```
    """
    start_time = time.time()

    # --- Validate mode ---
    if mode not in ("auto", "manual"):
        raise HTTPException(
            status_code=400,
            detail=f"mode phai la 'auto' hoac 'manual', nhan duoc: '{mode}'"
        )

    if mode == "manual" and low_threshold >= high_threshold:
        raise HTTPException(
            status_code=400,
            detail=f"low_threshold ({low_threshold}) phai nho hon high_threshold ({high_threshold})"
        )

    # --- Tim file anh ---
    # Ho tro ca JPEG (inputs) va PNG (outputs / inpainted result)
    image_files = list(settings.INPUTS_DIR.glob(f"*_{image_id}.jpg"))
    if not image_files:
        image_files = list(settings.INPUTS_DIR.glob(f"*_{image_id}.png"))
    if not image_files:
        image_files = list(settings.OUTPUTS_DIR.glob(f"*_{image_id}.*"))

    if not image_files:
        raise HTTPException(
            status_code=404,
            detail=f"Khong tim thay anh voi image_id='{image_id}'. "
                   f"Hay upload anh truoc qua POST /api/v1/segmentation/upload"
        )

    image_path = image_files[0]
    logger.info(f"📂 Loading image for edge detection: {image_path.name}")

    # --- Load anh ---
    try:
        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil, dtype=np.uint8)
    except Exception as e:
        logger.error(f"❌ Khong the doc anh: {e}")
        raise HTTPException(status_code=500, detail=f"Khong the doc anh: {str(e)}")

    # --- Validate anh ---
    is_valid, error_msg = validate_image_for_edges(image_np)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Anh khong hop le: {error_msg}")

    h, w = image_np.shape[:2]
    logger.info(f"🖼️  Image size: {w}x{h} | Edge detection mode: {mode}")

    # --- Chay edge detection ---
    try:
        if mode == "auto":
            edges = auto_canny_edges(image_np, sigma=sigma)
        else:  # manual
            edges = detect_canny_edges(
                image_np,
                low_threshold=low_threshold,
                high_threshold=high_threshold,
            )
    except Exception as e:
        logger.error(f"❌ Edge detection that bai: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Edge detection that bai: {str(e)}")

    # --- Tinh edge density ---
    density = get_edge_density(edges)
    processing_time = time.time() - start_time

    logger.info(
        f"✅ Edge detection hoan tat | "
        f"mode={mode} | density={density:.2f}% | "
        f"time={processing_time:.2f}s"
    )

    # --- Luu file va tra ve ---
    # Chuyen edges sang RGB de luu thành PNG
    edge_rgb = edges_to_rgb(edges)
    edge_pil = Image.fromarray(edge_rgb)

    # Luu vao OUTPUTS_DIR de co the truy cap lai
    edge_filename = f"edges_{mode}_{int(time.time())}_{image_id[:8]}.png"
    edge_path = settings.OUTPUTS_DIR / edge_filename
    edge_pil.save(edge_path, format="PNG")

    logger.info(f"💾 Saved edge image: {edge_filename}")

    # Tra ve file PNG voi cac header thong tin
    return FileResponse(
        path=str(edge_path),
        media_type="image/png",
        filename=edge_filename,
        headers={
            "X-Edge-Density": f"{density:.4f}",
            "X-Edge-Mode": mode,
            "X-Processing-Time": f"{processing_time:.3f}",
            "X-Image-Size": f"{w}x{h}",
            "X-Sigma": str(sigma) if mode == "auto" else "N/A",
            "X-Low-Threshold": "N/A" if mode == "auto" else str(low_threshold),
            "X-High-Threshold": "N/A" if mode == "auto" else str(high_threshold),
        },
    )
