"""
FastAPI application entry point
Refactored with clean architecture
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.config import settings, ensure_directories
from app.dependencies import get_model_manager
from app.api.v1.router import api_router
from app.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events - startup and shutdown
    """
    # Startup
    logger.info("🚀 Starting AI Interior Design Backend...")
    ensure_directories()
    
    # Initialize models (this will load SAM)
    model_manager = get_model_manager()
    logger.info(f"✅ Models loaded on device: {model_manager.device}")
    
    # Note: Stable Diffusion models are now lazy-loaded by hybrid inpainting service
    # - Replicate API: No local model needed
    # - Local GPU: Model loads on first request (saves RAM and startup time)
    logger.info("✅ Hybrid inpainting service ready (Replicate API + Local GPU fallback)")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

# Legacy endpoint for backward compatibility
@app.post("/predict")
async def predict_legacy(file):
    """
    Legacy endpoint - redirects to new API
    Kept for backward compatibility with existing Flutter app
    """
    from fastapi import UploadFile, File
    from app.api.v1.endpoints.segmentation import segment_image
    
    return await segment_image(file, get_model_manager())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
