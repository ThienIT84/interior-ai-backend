"""
Health check endpoints
"""
from fastapi import APIRouter, Depends
from app.dependencies import get_model_manager, ModelManager
from app.config import settings

router = APIRouter()


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running"
    }


@router.get("/health")
async def health_check(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Health check endpoint
    Returns system status and loaded models
    """
    default_backend = settings.SEGMENTATION_BACKEND.strip().lower()
    default_model = (
        settings.SAM3_REPLICATE_MODEL
        if default_backend == "sam3_replicate"
        else f"local_sam:vit_b:{settings.SAM_CHECKPOINT}"
    )

    return {
        "status": "healthy",
        "device": model_manager.device,
        "segmentation_backend": default_backend,
        "segmentation_model": default_model,
        "models": {
            "sam": "loaded" if model_manager.is_sam_loaded else "not loaded"
        },
        "message": "Hệ thống AI Interior đã sẵn sàng!"
    }
