"""
Inpainting API Endpoints
Handles object removal and room inpainting requests

Pipeline:
  1. Validate mask coverage (gate)
  2. Preprocess mask (morphology + feather)
  3. LaMa object removal (primary)
  4. Artifact score post-check
  5. Pass-2 SD inpainting on artifact region (if needed)
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
from pathlib import Path
import uuid
import json
import time
from PIL import Image
import logging

from app.config import settings
from app.core.inpainting_service import get_inpainting_service
from app.utils.mask_quality import (
    validate_mask_coverage,
    preprocess_mask_for_removal,
    compute_artifact_score,
    build_artifact_repair_mask,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models
class InpaintRequest(BaseModel):
    """Request to remove object and inpaint"""
    image_id: str = Field(..., description="ID of uploaded image")
    mask_id: str = Field(..., description="ID of generated mask")
    # ── Removal backend ────────────────────────────────────────────────
    method: Optional[str] = Field(
        None,
        description=(
            'Removal method: "lama" (default, best for clean removal), '
            '"replicate" (SD inpainting), "local" (local GPU), "auto"'
        ),
    )
    # ── SD-specific params (ignored when method=lama) ─────────────────
    prompt: Optional[str] = Field(None, description="Custom prompt (optional, SD only)")
    negative_prompt: Optional[str] = Field(None, description="Custom negative prompt (SD only)")
    steps: Optional[int] = Field(None, ge=10, le=100, description="Inference steps (SD only)")
    guidance_scale: Optional[float] = Field(None, ge=1.0, le=20.0, description="Guidance scale (SD only)")
    strength: Optional[float] = Field(None, ge=0.1, le=1.0, description="Inpainting strength (SD only)")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    # ── Mask preprocessing ────────────────────────────────────────
    mask_padding: Optional[int] = Field(
        None, ge=0, le=50,
        description=f"Mask dilation pixels (default: {settings.MASK_DILATION_DEFAULT})",
    )
    skip_mask_gate: bool = Field(
        False, description="Skip coverage validation (debug use only)"
    )
    skip_artifact_check: bool = Field(
        False, description="Skip post-removal artifact check (debug use only)"
    )


class InpaintResponse(BaseModel):
    """Response with inpainting result"""
    result_id: str
    result_url: str
    processing_time: float
    method_used: str
    fallback_used: Optional[str] = None
    mask_coverage_pct: Optional[float] = None
    artifact_score: Optional[float] = None
    pass2_applied: bool = False
    metadata: dict


class InpaintJobResponse(BaseModel):
    """Response for async job submission"""
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    """Response for job status check"""
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: Optional[float] = None
    result_url: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[dict] = None


# In-memory job storage (for MVP, use Redis/DB in production)
_jobs = {}


@router.post("/remove-object", response_model=InpaintResponse)
async def remove_object(request: InpaintRequest):
    """
    Remove object from image (synchronous).

    Pipeline:
      1. Load image + mask
      2. Validate mask coverage (gate)
      3. Preprocess mask (morphology + feather)
      4. LaMa / SD / local removal
      5. Artifact score check + optional pass-2 repair
    """
    logger.info(f"Inpainting request: image={request.image_id}, mask={request.mask_id}")

    try:
        # ─ Load files ──────────────────────────────────────────────────
        image_files = list(settings.INPUTS_DIR.glob(f"*_{request.image_id}.*"))
        if not image_files:
            raise HTTPException(status_code=404, detail=f"Image not found: {request.image_id}")

        mask_files = list(settings.MASKS_DIR.glob(f"*_{request.mask_id}.png"))
        if not mask_files:
            raise HTTPException(status_code=404, detail=f"Mask not found: {request.mask_id}")

        image = Image.open(image_files[0]).convert("RGB")
        mask_raw = Image.open(mask_files[0]).convert("L")

        # ─ 1. Mask coverage gate ────────────────────────────────────────
        if not request.skip_mask_gate:
            validation = validate_mask_coverage(
                mask_raw,
                min_fraction=settings.MASK_COVERAGE_MIN,
                max_fraction=settings.MASK_COVERAGE_MAX,
            )
            if not validation.valid:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "mask_invalid",
                        "message": validation.reason,
                        "coverage_pct": round(validation.coverage_pct, 2),
                    },
                )
            mask_coverage_pct = validation.coverage_pct
        else:
            import numpy as np
            _arr = np.array(mask_raw)
            mask_coverage_pct = float((_arr > 127).sum()) / _arr.size * 100

        # ─ 2. Preprocess mask ─────────────────────────────────────────
        dilation = request.mask_padding if request.mask_padding is not None else settings.MASK_DILATION_DEFAULT
        prep = preprocess_mask_for_removal(
            mask_raw,
            dilation_px=dilation,
            feather_px=settings.MASK_FEATHER_DEFAULT,
        )
        mask_ready = prep.mask

        # ─ 3. Remove object ───────────────────────────────────────────
        # Resolve effective method: request > config > auto
        effective_method = (
            request.method
            or settings.REMOVE_OBJECT_METHOD
            or "auto"
        ).lower()

        inpainting = get_inpainting_service()
        start_time = time.time()

        result, metadata = inpainting.inpaint(
            image=image,
            mask=mask_ready,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            strength=request.strength,
            seed=request.seed,
            mask_padding=0,   # Dilation đã xử lý trong preprocess; không dilate lần 2
            method=effective_method,
        )

        processing_time = time.time() - start_time

        # ─ 4. Artifact check + pass-2 repair ──────────────────────────
        artifact_score = 0.0
        pass2_applied = False

        if not request.skip_artifact_check:
            artifact_score = compute_artifact_score(image, result, mask_ready)
            metadata["artifact_score"] = round(artifact_score, 4)

            if artifact_score > settings.ARTIFACT_SCORE_THRESHOLD:
                logger.warning(
                    f"⚠️  Artifact score {artifact_score:.3f} > threshold "
                    f"{settings.ARTIFACT_SCORE_THRESHOLD}. Running pass-2 repair..."
                )
                repair_mask = build_artifact_repair_mask(image, result, mask_ready)
                if repair_mask is not None:
                    try:
                        repair_svc = get_inpainting_service()
                        result, repair_meta = repair_svc.inpaint(
                            image=result,
                            mask=repair_mask,
                            method="replicate" if inpainting.replicate_available else "local",
                            steps=20,   # Faster pass
                            mask_padding=0,
                        )
                        pass2_applied = True
                        metadata["pass2_repair"] = {
                            "method": repair_meta.get("method"),
                            "reason": f"artifact_score={artifact_score:.3f}",
                        }
                        logger.info("✅ Pass-2 repair completed")
                    except Exception as repair_err:
                        logger.error(f"❌ Pass-2 failed (non-blocking): {repair_err}")

        # ─ Save result ─────────────────────────────────────────────────
        result_id = str(uuid.uuid4())[:8]
        timestamp = int(time.time())
        result_filename = f"inpaint_{timestamp}_{result_id}.png"
        result_path = settings.OUTPUTS_DIR / result_filename
        result.save(result_path)
        logger.info(f"Saved result: {result_filename}")

        metadata_full = {
            **metadata,
            "result_id": result_id,
            "image_id": request.image_id,
            "mask_id": request.mask_id,
            "timestamp": timestamp,
            "mask_coverage_pct": round(mask_coverage_pct, 2),
            "artifact_score": round(artifact_score, 4),
            "pass2_applied": pass2_applied,
        }
        with open(result_path.with_suffix(".json"), "w") as f:
            json.dump(metadata_full, f, indent=2)

        return InpaintResponse(
            result_id=result_id,
            result_url=f"/api/v1/inpainting/result/{result_id}",
            processing_time=processing_time,
            method_used=metadata.get("method", effective_method),
            fallback_used=metadata.get("fallback_used"),
            mask_coverage_pct=round(mask_coverage_pct, 2),
            artifact_score=round(artifact_score, 4),
            pass2_applied=pass2_applied,
            metadata=metadata_full,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inpainting failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inpainting failed: {str(e)}")


@router.post("/remove-object-async", response_model=InpaintJobResponse)
async def remove_object_async(request: InpaintRequest, background_tasks: BackgroundTasks):
    """
    Remove object from image using inpainting (asynchronous)
    
    This endpoint submits the job and returns immediately.
    Use /job-status/{job_id} to check progress.
    """
    job_id = str(uuid.uuid4())[:8]
    
    # Initialize job status
    _jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "result_url": None,
        "error": None,
        "metadata": None,
        "created_at": time.time(),
    }
    
    # Add background task
    background_tasks.add_task(
        _process_inpainting_job,
        job_id=job_id,
        request=request,
    )
    
    logger.info(f"Created async job: {job_id}")
    
    return InpaintJobResponse(
        job_id=job_id,
        status="pending",
        message="Job submitted successfully. Use /job-status/{job_id} to check progress.",
    )


@router.get("/job-status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of async inpainting job"""
    logger.info(f"🔍 Checking job status: {job_id}")
    logger.info(f"📊 Available jobs: {list(_jobs.keys())}")
    
    if job_id not in _jobs:
        logger.warning(f"❌ Job not found: {job_id}")
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    
    job = _jobs[job_id]
    logger.info(f"✅ Job {job_id} found: status={job['status']}, progress={job.get('progress', 0)*100:.1f}%")
    
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress"),
        result_url=job.get("result_url"),
        error=job.get("error"),
        metadata=job.get("metadata"),
    )


@router.get("/result/{result_id}")
async def get_result_image(result_id: str):
    """Get inpainting result image"""
    from fastapi.responses import FileResponse
    
    # Find result file
    result_files = list(settings.OUTPUTS_DIR.glob(f"*_{result_id}.png"))
    if not result_files:
        raise HTTPException(status_code=404, detail=f"Result not found: {result_id}")
    
    result_path = result_files[0]
    
    return FileResponse(
        result_path,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


async def _process_inpainting_job(job_id: str, request: InpaintRequest):
    """Background task — same LaMa pipeline as synchronous endpoint."""
    logger.info(f"🎬 Starting background job: {job_id}")
    try:
        _jobs[job_id]["status"] = "processing"
        _jobs[job_id]["progress"] = 0.05

        # ── Load files ────────────────────────────────────────────────────
        image_files = list(settings.INPUTS_DIR.glob(f"*_{request.image_id}.*"))
        if not image_files:
            raise ValueError(f"Image not found: {request.image_id}")
        mask_files = list(settings.MASKS_DIR.glob(f"*_{request.mask_id}.png"))
        if not mask_files:
            raise ValueError(f"Mask not found: {request.mask_id}")

        _jobs[job_id]["progress"] = 0.15
        image = Image.open(image_files[0]).convert("RGB")
        mask_raw = Image.open(mask_files[0]).convert("L")

        # ── 1. Mask coverage gate ─────────────────────────────────────────
        mask_coverage_pct = 0.0
        if not request.skip_mask_gate:
            validation = validate_mask_coverage(
                mask_raw,
                min_fraction=settings.MASK_COVERAGE_MIN,
                max_fraction=settings.MASK_COVERAGE_MAX,
            )
            if not validation.valid:
                raise ValueError(f"Mask invalid: {validation.reason}")
            mask_coverage_pct = validation.coverage_pct
        else:
            import numpy as np
            _arr = np.array(mask_raw)
            mask_coverage_pct = float((_arr > 127).sum()) / _arr.size * 100

        # ── 2. Preprocess mask ────────────────────────────────────────────
        dilation = request.mask_padding if request.mask_padding is not None else settings.MASK_DILATION_DEFAULT
        prep = preprocess_mask_for_removal(
            mask_raw,
            dilation_px=dilation,
            feather_px=settings.MASK_FEATHER_DEFAULT,
        )
        mask_ready = prep.mask
        _jobs[job_id]["progress"] = 0.25

        # ── 3. Remove object ──────────────────────────────────────────────
        effective_method = (
            request.method or settings.REMOVE_OBJECT_METHOD or "auto"
        ).lower()

        inpainting = get_inpainting_service()
        _jobs[job_id]["progress"] = 0.30
        logger.info(f"📊 Job {job_id}: Starting removal [{effective_method}]")

        result, metadata = inpainting.inpaint(
            image=image,
            mask=mask_ready,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            strength=request.strength,
            seed=request.seed,
            mask_padding=0,
            method=effective_method,
        )
        _jobs[job_id]["progress"] = 0.80

        # ── 4. Artifact check + pass-2 ────────────────────────────────────
        artifact_score = 0.0
        pass2_applied = False
        if not request.skip_artifact_check:
            artifact_score = compute_artifact_score(image, result, mask_ready)
            metadata["artifact_score"] = round(artifact_score, 4)
            if artifact_score > settings.ARTIFACT_SCORE_THRESHOLD:
                logger.warning(f"⚠️  Job {job_id}: artifact={artifact_score:.3f}, running pass-2...")
                repair_mask = build_artifact_repair_mask(image, result, mask_ready)
                if repair_mask is not None:
                    try:
                        result, repair_meta = inpainting.inpaint(
                            image=result,
                            mask=repair_mask,
                            method="replicate" if inpainting.replicate_available else "local",
                            steps=20,
                            mask_padding=0,
                        )
                        pass2_applied = True
                        metadata["pass2_repair"] = {
                            "method": repair_meta.get("method"),
                            "reason": f"artifact_score={artifact_score:.3f}",
                        }
                    except Exception as repair_err:
                        logger.error(f"❌ Pass-2 failed (non-blocking): {repair_err}")

        _jobs[job_id]["progress"] = 0.95

        # ── Save ──────────────────────────────────────────────────────────
        result_id = str(uuid.uuid4())[:8]
        timestamp = int(time.time())
        result_filename = f"inpaint_{timestamp}_{result_id}.png"
        result_path = settings.OUTPUTS_DIR / result_filename
        result.save(result_path)

        metadata_full = {
            **metadata,
            "result_id": result_id,
            "image_id": request.image_id,
            "mask_id": request.mask_id,
            "timestamp": timestamp,
            "job_id": job_id,
            "mask_coverage_pct": round(mask_coverage_pct, 2),
            "artifact_score": round(artifact_score, 4),
            "pass2_applied": pass2_applied,
        }
        with open(result_path.with_suffix(".json"), "w") as f:
            json.dump(metadata_full, f, indent=2)

        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["progress"] = 1.0
        _jobs[job_id]["result_url"] = f"/api/v1/inpainting/result/{result_id}"
        _jobs[job_id]["metadata"] = metadata_full
        logger.info(f"✅ Job {job_id} completed")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)
