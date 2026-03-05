"""
API v1 router - combines all endpoint routers
"""
from fastapi import APIRouter

from app.api.v1.endpoints import health, segmentation, inpainting, generation

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(health.router, tags=["health"])
api_router.include_router(segmentation.router, prefix="/segmentation", tags=["segmentation"])
api_router.include_router(inpainting.router, prefix="/inpainting", tags=["inpainting"])
api_router.include_router(generation.router, prefix="/generation", tags=["generation"])
