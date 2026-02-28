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
    return {
        "status": "healthy",
        "device": model_manager.device,
        "models": {
            "sam": "loaded" if model_manager.sam_predictor else "not loaded"
        },
        "message": "Hệ thống AI Interior đã sẵn sàng!"
    }
