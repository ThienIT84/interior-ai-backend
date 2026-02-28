"""
Inpainting API Endpoints
Handles object removal and room inpainting requests
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
from app.core.diffusion_inpainting import get_inpainting_service

logger = logging.getLogger(__name__)

router = APIRouter()


# Request/Response Models
class InpaintRequest(BaseModel):
    """Request to remove object and inpaint"""
    image_id: str = Field(..., description="ID of uploaded image")
    mask_id: str = Field(..., description="ID of generated mask")
    prompt: Optional[str] = Field(None, description="Custom prompt (optional)")
    negative_prompt: Optional[str] = Field(None, description="Custom negative prompt (optional)")
    steps: Optional[int] = Field(None, ge=10, le=100, description="Inference steps (default: 50)")
    guidance_scale: Optional[float] = Field(None, ge=1.0, le=20.0, description="Guidance scale (default: 11.0)")
    strength: Optional[float] = Field(None, ge=0.1, le=1.0, description="Inpainting strength (default: 0.99)")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")


class InpaintResponse(BaseModel):
    """Response with inpainting result"""
    result_id: str
    result_url: str
    processing_time: float
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
    Remove object from image using inpainting (synchronous)
    
    This endpoint processes the request immediately and returns the result.
    Processing time: ~13-15 minutes on GTX 1650 4GB
    
    For async processing, use /remove-object-async instead.
    """
    logger.info(f"Inpainting request: image={request.image_id}, mask={request.mask_id}")
    
    try:
        # Find image file
        image_files = list(settings.INPUTS_DIR.glob(f"*_{request.image_id}.*"))
        if not image_files:
            raise HTTPException(status_code=404, detail=f"Image not found: {request.image_id}")
        
        image_path = image_files[0]
        
        # Find mask file
        mask_files = list(settings.MASKS_DIR.glob(f"*_{request.mask_id}.png"))
        if not mask_files:
            raise HTTPException(status_code=404, detail=f"Mask not found: {request.mask_id}")
        
        mask_path = mask_files[0]
        
        logger.info(f"Loading image: {image_path.name}")
        logger.info(f"Loading mask: {mask_path.name}")
        
        # Load images
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # Get inpainting service
        inpainting = get_inpainting_service()
        
        # Perform inpainting
        start_time = time.time()
        
        result, metadata = inpainting.inpaint(
            image=image,
            mask=mask,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            strength=request.strength,
            seed=request.seed,
        )
        
        processing_time = time.time() - start_time
        
        # Save result
        result_id = str(uuid.uuid4())[:8]
        timestamp = int(time.time())
        result_filename = f"inpaint_{timestamp}_{result_id}.png"
        result_path = settings.OUTPUTS_DIR / result_filename
        
        result.save(result_path)
        logger.info(f"Saved result: {result_filename}")
        
        # Save metadata
        metadata_path = result_path.with_suffix(".json")
        metadata_full = {
            **metadata,
            "result_id": result_id,
            "image_id": request.image_id,
            "mask_id": request.mask_id,
            "timestamp": timestamp,
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata_full, f, indent=2)
        
        # Generate URL
        result_url = f"/api/v1/inpainting/result/{result_id}"
        
        return InpaintResponse(
            result_id=result_id,
            result_url=result_url,
            processing_time=processing_time,
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
    """Background task to process inpainting job"""
    logger.info(f"🎬 Starting background job: {job_id}")
    try:
        # Update status
        _jobs[job_id]["status"] = "processing"
        _jobs[job_id]["progress"] = 0.1
        logger.info(f"📊 Job {job_id}: status=processing, progress=10%")
        
        # Find files
        image_files = list(settings.INPUTS_DIR.glob(f"*_{request.image_id}.*"))
        if not image_files:
            raise ValueError(f"Image not found: {request.image_id}")
        
        mask_files = list(settings.MASKS_DIR.glob(f"*_{request.mask_id}.png"))
        if not mask_files:
            raise ValueError(f"Mask not found: {request.mask_id}")
        
        image_path = image_files[0]
        mask_path = mask_files[0]
        
        # Load images
        _jobs[job_id]["progress"] = 0.2
        logger.info(f"📊 Job {job_id}: Loading images, progress=20%")
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        # Get inpainting service
        inpainting = get_inpainting_service()
        
        # Perform inpainting
        _jobs[job_id]["progress"] = 0.3
        logger.info(f"📊 Job {job_id}: Starting inpainting, progress=30%")
        
        result, metadata = inpainting.inpaint(
            image=image,
            mask=mask,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            strength=request.strength,
            seed=request.seed,
        )
        
        _jobs[job_id]["progress"] = 0.9
        
        # Save result
        result_id = str(uuid.uuid4())[:8]
        timestamp = int(time.time())
        result_filename = f"inpaint_{timestamp}_{result_id}.png"
        result_path = settings.OUTPUTS_DIR / result_filename
        
        result.save(result_path)
        
        # Save metadata
        metadata_path = result_path.with_suffix(".json")
        metadata_full = {
            **metadata,
            "result_id": result_id,
            "image_id": request.image_id,
            "mask_id": request.mask_id,
            "timestamp": timestamp,
            "job_id": job_id,
        }
        
        with open(metadata_path, "w") as f:
            json.dump(metadata_full, f, indent=2)
        
        # Update job status
        result_url = f"/api/v1/inpainting/result/{result_id}"
        
        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["progress"] = 1.0
        _jobs[job_id]["result_url"] = result_url
        _jobs[job_id]["metadata"] = metadata_full
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)
