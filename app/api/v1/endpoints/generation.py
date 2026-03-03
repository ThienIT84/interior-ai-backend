"""
Generation API Endpoints
========================
Endpoint cho viec tao thiet ke noi that moi su dung ControlNet.

Week 3 Tasks:
  - 3.1.3: POST /preview-edges       ✅ DONE
  - 3.2.1: POST /generate-design     ✅ DONE (async job)
  - 3.2.2: GET  /job-status/{job_id} ✅ DONE
  - 3.4.2: GET  /styles              ✅ DONE
  - 3.4.3: POST /generate-batch      TODO - Week 3 Day 18
"""

import io
import time
import uuid
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from PIL import Image

from app.config import settings
from app.core.edge_detection import (
    auto_canny_edges,
    detect_canny_edges,
    edges_to_rgb,
    get_edge_density,
    validate_image_for_edges,
)
from app.core.prompts import list_styles, AVAILABLE_STYLES
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


# ---------------------------------------------------------------------------
# Task 3.4.2 - List available styles
# ---------------------------------------------------------------------------

@router.get("/styles")
async def get_styles():
    """
    Lay danh sach cac style thiet ke noi that co the ap dung.

    Returns:
        List cac style voi ten hien thi va mo ta

    Example response:
        ```json
        {
          "styles": [
            {"name": "modern", "display_name": "Modern", "description": "..."},
            ...
          ],
          "count": 3
        }
        ```
    """
    styles = list_styles()
    return {"styles": styles, "count": len(styles)}


# ---------------------------------------------------------------------------
# Task 3.2.1 + 3.4.1 - Generate design (async)
# ---------------------------------------------------------------------------

class GenerateDesignRequest(BaseModel):
    """Request de tao thiet ke noi that moi."""
    image_id: str = Field(..., description="ID anh da upload (hoac result_id cua inpainting)")
    style: str = Field(..., description=f"Ten style. Cac gia tri hop le: {AVAILABLE_STYLES}")
    guidance_scale: Optional[float] = Field(None, ge=1.0, le=20.0, description="Guidance scale (default: 9.0)")
    steps: Optional[int] = Field(None, ge=10, le=50, description="So buoc inference (default: 20)")
    seed: Optional[int] = Field(None, description="Random seed de reproduce ket qua")


class GenerateDesignResponse(BaseModel):
    """Response sau khi submit async generation job."""
    job_id: str
    status: str
    style: str
    message: str


class JobStatusResponse(BaseModel):
    """Response khi poll job status."""
    job_id: str
    status: str  # "pending" | "processing" | "completed" | "failed"
    style: str
    result_url: Optional[str] = None
    result_id: Optional[str] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None
    metadata: Optional[dict] = None


@router.post("/generate-design", response_model=GenerateDesignResponse)
async def generate_design(request: GenerateDesignRequest):
    """
    Tao thiet ke noi that moi dua tren phong hien tai va style da chon.

    **Quy trinh**:
    1. Goi endpoint nay -> nhan `job_id`
    2. Poll `GET /generation/job-status/{job_id}` moi 3-5 giay
    3. Khi `status == "completed"` -> lay `result_url` de xem ket qua

    **Style hop le**: modern, minimalist, industrial
    (Lay danh sach day du qua `GET /generation/styles`)

    **Thoi gian xu ly**: ~15-30 giay (Replicate ControlNet)

    Args:
        request.image_id: ID anh goc (sau khi upload) hoac ID ket qua inpainting
        request.style: Ten style thiet ke muon tao
        request.guidance_scale: Muc do bam sat prompt (default: 9.0)
        request.steps: So buoc inference (default: 20, max: 50)
        request.seed: Random seed de co the tao lai ket qua tuong tu

    Returns:
        job_id de poll trang thai

    Example:
        ```
        POST /api/v1/generation/generate-design
        {
          "image_id": "abc123",
          "style": "modern",
          "guidance_scale": 9.0,
          "steps": 20
        }
        ```
    """
    # Lazy import de tranh loi khi token chua set
    try:
        from app.core.controlnet_generation import get_controlnet_service
        svc = get_controlnet_service()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"ControlNet service chua san sang: {str(e)}")

    # Validate style
    if request.style.lower() not in AVAILABLE_STYLES:
        raise HTTPException(
            status_code=400,
            detail=f"Style '{request.style}' khong hop le. Cac style ho tro: {AVAILABLE_STYLES}"
        )

    # Tim file anh (ho tro ca inputs/ va outputs/)
    image_path = None
    for pattern in [
        f"*_{request.image_id}.jpg",
        f"*_{request.image_id}.png",
        f"*_{request.image_id}.jpeg",
    ]:
        candidates = list(settings.INPUTS_DIR.glob(pattern))
        if candidates:
            image_path = candidates[0]
            break

    if image_path is None:
        # Thu tim trong outputs (inpainting result)
        for pattern in [
            f"*_{request.image_id}.*",
            f"inpainted_*_{request.image_id}.*",
        ]:
            candidates = list(settings.OUTPUTS_DIR.glob(pattern))
            if candidates:
                image_path = candidates[0]
                break

    if image_path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Khong tim thay anh voi image_id='{request.image_id}'. "
                f"Hay upload anh truoc qua POST /api/v1/segmentation/upload"
            )
        )

    logger.info(f"🎨 Generate design | image={image_path.name} | style={request.style}")

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Khong the doc anh: {str(e)}")

    # Submit async job
    try:
        job = svc.submit_job(
            image=image,
            style=request.style,
            guidance_scale=request.guidance_scale,
            steps=request.steps,
            seed=request.seed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Khong the submit job: {str(e)}")

    return GenerateDesignResponse(
        job_id=job.job_id,
        status=job.status,
        style=job.style,
        message=(
            f"Job da duoc dang ky. "
            f"Poll GET /api/v1/generation/job-status/{job.job_id} de kiem tra trang thai."
        ),
    )


@router.get("/job-status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Kiem tra trang thai cua mot generation job.

    **Status values**:
    - `pending`: Dang cho trong hang doi
    - `processing`: Dang xu ly tren Replicate
    - `completed`: Hoan tat, co the lay ket qua qua `result_url`
    - `failed`: That bai, xem `error` de biet nguyen nhan

    Args:
        job_id: ID nhan duoc tu POST /generate-design

    Returns:
        Trang thai job + result_url (khi completed)
    """
    try:
        from app.core.controlnet_generation import get_controlnet_service
        svc = get_controlnet_service()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    job = svc.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Khong tim thay job_id='{job_id}'. "
                   f"Job co the da het han (server restart) hoac ID sai."
        )

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        style=job.style,
        result_url=job.result_url,
        result_id=job.result_id,
        processing_time=job.processing_time,
        error=job.error,
        metadata=job.metadata if job.status == "completed" else None,
    )


@router.get("/result/{result_id}")
async def get_result(result_id: str):
    """
    Lay anh ket qua generation theo result_id.

    Args:
        result_id: ID nhan duoc trong job status khi completed

    Returns:
        PNG image cua thiet ke da tao
    """
    # Tim file trong outputs/
    candidates = list(settings.OUTPUTS_DIR.glob(f"design_*_{result_id}.png"))
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=f"Khong tim thay result_id='{result_id}'."
        )

    return FileResponse(
        path=str(candidates[0]),
        media_type="image/png",
        filename=candidates[0].name,
    )
