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
import threading
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw

from app.config import settings
from app.core.edge_detection import (
    auto_canny_edges,
    detect_canny_edges,
    edges_to_rgb,
    get_edge_density,
    validate_image_for_edges,
)
from app.core.prompts import list_styles, AVAILABLE_STYLES, get_furniture_placement_prompt
from app.services.job_service import get_job_service, JobStatus
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
        image_files = list(settings.INPUTS_DIR.glob(f"*_{image_id}.jpeg"))
    if not image_files:
        for _ext in ("png", "jpg", "jpeg"):
            image_files = list(settings.OUTPUTS_DIR.glob(f"*_{image_id}.{_ext}"))
            if image_files:
                break

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

# Supported model identifiers
SUPPORTED_MODELS = ["controlnet", "flux-pro"]


class GenerateDesignRequest(BaseModel):
    """Request de tao thiet ke noi that moi."""
    image_id: str = Field(..., description="ID anh da upload (hoac result_id cua inpainting)")
    style: str = Field(..., description=f"Ten style. Cac gia tri hop le: {AVAILABLE_STYLES}")
    model_id: str = Field(
        "controlnet",
        description=f"Model AI su dung. Cac gia tri hop le: {SUPPORTED_MODELS}",
    )
    guidance_scale: Optional[float] = Field(None, ge=1.0, le=100.0, description="Guidance scale")
    steps: Optional[int] = Field(None, ge=10, le=50, description="So buoc inference")
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
    # Validate model_id
    if request.model_id not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"model_id '{request.model_id}' khong hop le. Cac model ho tro: {SUPPORTED_MODELS}",
        )

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
        # Thu tim trong outputs (inpainting result) - chi match image files, tranh .json
        for pattern in [
            f"*_{request.image_id}.png",
            f"*_{request.image_id}.jpg",
            f"*_{request.image_id}.jpeg",
            f"inpainted_*_{request.image_id}.png",
            f"inpainted_*_{request.image_id}.jpg",
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
            model_id=request.model_id,
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
        metadata=job.metadata, # Luon tra ve metadata de lay status_message
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


# ---------------------------------------------------------------------------
# Option 1 – Targeted Furniture Placement
# ---------------------------------------------------------------------------

# JobService will handle placement jobs (Task 3.5.1 fix)


class PlaceFurnitureRequest(BaseModel):
    """Request cho Option 1: dat do noi that vao vung da chon qua bounding box."""
    image_id: str = Field(..., description="ID anh phong (thuong la inpainting result)")
    # Normalized coordinates (0.0 – 1.0) relative to displayed image size
    bbox_x: float = Field(..., ge=0.0, le=1.0, description="Left edge (normalized)")
    bbox_y: float = Field(..., ge=0.0, le=1.0, description="Top edge (normalized)")
    bbox_w: float = Field(..., gt=0.0, le=1.0, description="Width (normalized)")
    bbox_h: float = Field(..., gt=0.0, le=1.0, description="Height (normalized)")
    furniture_description: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="Mo ta do noi that muon dat (VD: 'a modern leather sofa')",
    )


class PlaceFurnitureResponse(BaseModel):
    job_id: str
    status: str
    message: str


class PlacementJobStatusResponse(BaseModel):
    job_id: str
    status: str
    furniture_description: str
    result_url: Optional[str] = None
    result_id: Optional[str] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None


def _run_placement(
    job_id: str,
    image: Image.Image,
    bbox_x: float,
    bbox_y: float,
    bbox_w: float,
    bbox_h: float,
    furniture_description: str,
) -> None:
    """Background thread: tao mask tu bbox, chay inpainting, cap nhat job trong Redis."""
    job_service = get_job_service()
    job_service.update_job(job_id, status=JobStatus.PROCESSING.value, progress=0.2)
    start_time = time.time()
    logger.info(
        f"⏳ [placement job={job_id}] Processing | '{furniture_description}' "
        f"at bbox=({bbox_x:.2f},{bbox_y:.2f},{bbox_w:.2f},{bbox_h:.2f})"
    )

    try:
        import replicate
        import os
        import base64
        import io as _io
        import urllib.request

        w, h = image.size

        # --- Tao rectangular mask tu normalized bbox ---
        px = int(bbox_x * w)
        py = int(bbox_y * h)
        pw = int(bbox_w * w)
        ph = int(bbox_h * h)
        # Clamp
        px = max(0, min(px, w - 1))
        py = max(0, min(py, h - 1))
        pw = min(pw, w - px)
        ph = min(ph, h - py)

        mask = Image.new("L", (w, h), 0)  # Black background
        draw = ImageDraw.Draw(mask)
        draw.rectangle([px, py, px + pw, py + ph], fill=255)  # White rect
        mask_rgb = mask.convert("RGB")

        # --- Convert to data URIs ---
        def to_data_uri(img: Image.Image, mode: str = "RGB") -> str:
            buf = _io.BytesIO()
            img.convert(mode).save(buf, format="PNG")
            buf.seek(0)
            return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

        positive_prompt, negative_prompt = get_furniture_placement_prompt(furniture_description)

        # --- Replicate SD inpainting ---
        MODEL = (
            "stability-ai/stable-diffusion-inpainting:"
            "95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3"
        )

        api_token = os.getenv("REPLICATE_API_TOKEN") or settings.REPLICATE_API_TOKEN
        if not api_token:
            raise RuntimeError("REPLICATE_API_TOKEN not set")
        os.environ["REPLICATE_API_TOKEN"] = api_token

        output = replicate.run(
            MODEL,
            input={
                "image": to_data_uri(image),
                "mask": to_data_uri(mask_rgb),
                "prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": 50,
                "guidance_scale": 9.0,
                "prompt_strength": 1.0,
            },
        )

        result_url_remote = str(output[0]) if isinstance(output, list) else str(output)

        # --- Download & save ---
        result_id = uuid.uuid4().hex[:16]
        filename = f"placement_{result_id}.png"
        output_path = settings.OUTPUTS_DIR / filename
        urllib.request.urlretrieve(result_url_remote, output_path)

        processing_time = time.time() - start_time
        result_url_local = f"/api/v1/generation/placement-result/{result_id}"
        
        job_service.update_job(
            job_id,
            status=JobStatus.COMPLETED.value,
            progress=1.0,
            result_url=result_url_local,
            metadata={
                "result_id": result_id,
                "processing_time": processing_time
            }
        )
        
        logger.info(
            f"✅ [placement job={job_id}] Done | result_id={result_id} "
            f"| time={processing_time:.1f}s"
        )

    except Exception as e:
        processing_time = time.time() - start_time
        job_service.update_job(
            job_id,
            status=JobStatus.FAILED.value,
            error=str(e),
            metadata={"processing_time": processing_time}
        )
        logger.error(f"❌ [placement job={job_id}] Failed: {e}", exc_info=True)


@router.post("/place-furniture", response_model=PlaceFurnitureResponse)
async def place_furniture(request: PlaceFurnitureRequest):
    """
    Option 1 – Targeted Placement: dat do noi that vao vung nguoi dung chon.

    Quy trinh:
    1. Nhan bounding box (normalized 0-1) va mo ta do noi that
    2. Tao rectangular mask tai vi tri do
    3. Chay Stable Diffusion Inpainting voi prompt mo ta do noi that
    4. Tra ve job_id, poll GET /placement-job-status/{job_id}

    Args:
        request.image_id: ID anh phong (thuong la inpainting result = phong trong)
        request.bbox_x/y/w/h: Vi tri va kich thuoc vung dat do (0.0 - 1.0)
        request.furniture_description: Mo ta do noi that (vi du: 'a modern leather sofa')
    """
    # Tim file anh
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
        # Chi match image files, tranh .json
        for pattern in [
            f"*_{request.image_id}.png",
            f"*_{request.image_id}.jpg",
            f"*_{request.image_id}.jpeg",
        ]:
            candidates = list(settings.OUTPUTS_DIR.glob(pattern))
            if candidates:
                image_path = candidates[0]
                break
    if image_path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Khong tim thay anh voi image_id='{request.image_id}'.",
        )

    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Khong the doc anh: {e}")

    job_service = get_job_service()
    job_id = job_service.create_job(
        job_type="placement",
        payload={
            "image_id": request.image_id,
            "furniture_description": request.furniture_description,
            "bbox": [request.bbox_x, request.bbox_y, request.bbox_w, request.bbox_h]
        }
    )

    t = threading.Thread(
        target=_run_placement,
        args=(
            job_id, image,
            request.bbox_x, request.bbox_y, request.bbox_w, request.bbox_h,
            request.furniture_description,
        ),
        daemon=True,
    )
    t.start()

    logger.info(
        f"📋 Placement job submitted | job_id={job_id} "
        f"| furniture='{request.furniture_description}'"
    )
    return PlaceFurnitureResponse(
        job_id=job_id,
        status="pending",
        message=(
            f"Job da dang ky. "
            f"Poll GET /api/v1/generation/placement-job-status/{job_id} de kiem tra."
        ),
    )


@router.get("/placement-job-status/{job_id}", response_model=PlacementJobStatusResponse)
async def get_placement_job_status(job_id: str):
    """Kiem tra trang thai cua mot furniture placement job."""
    job_service = get_job_service()
    job_data = job_service.get_job(job_id)
    
    if job_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Khong tim thay placement job_id='{job_id}'.",
        )
        
    payload = job_data.get("payload", {})
    metadata = job_data.get("metadata", {})
    
    return PlacementJobStatusResponse(
        job_id=job_id,
        status=job_data.get("status"),
        furniture_description=payload.get("furniture_description", "unknown"),
        result_url=job_data.get("result_url"),
        result_id=metadata.get("result_id"),
        processing_time=metadata.get("processing_time"),
        error=job_data.get("error"),
    )


@router.get("/placement-result/{result_id}")
async def get_placement_result(result_id: str):
    """Lay anh ket qua furniture placement theo result_id."""
    candidates = list(settings.OUTPUTS_DIR.glob(f"placement_*_{result_id}.png"))
    # Also try without prefix pattern to handle uuid hex filenames
    if not candidates:
        candidates = list(settings.OUTPUTS_DIR.glob(f"placement_{result_id}.png"))
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=f"Khong tim thay placement result_id='{result_id}'.",
        )
    return FileResponse(
        path=str(candidates[0]),
        media_type="image/png",
        filename=candidates[0].name,
    )
