"""
Pydantic models for segmentation endpoints
"""
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional


class PointPrompt(BaseModel):
    """Point prompt for segmentation"""
    x: int = Field(..., description="X coordinate")
    y: int = Field(..., description="Y coordinate")
    label: int = Field(..., description="1 for foreground, 0 for background")


class SegmentationRequest(BaseModel):
    """Request model for segmentation with points"""
    image_id: str = Field(..., description="ID of previously uploaded image")
    points: List[PointPrompt] = Field(..., description="List of point prompts")
    segmentation_backend: Optional[str] = Field(
        default=None,
        description='Optional backend override: "local" or "sam3_replicate"'
    )
    text_prompt: Optional[str] = Field(
        default=None,
        description='Text description of the object to segment — used by SAM3 to guide detection. '
                    'E.g. "sofa", "chair", "table lamp". Falls back to "object" if omitted.'
    )

    class Config:
        json_schema_extra = {
            "example": {
                "image_id": "abc123-def456",
                "points": [
                    {"x": 100, "y": 150, "label": 1},
                ],
                "segmentation_backend": "sam3_replicate",
                "text_prompt": "sofa"
            }
        }


class BoxPrompt(BaseModel):
    """Bounding box prompt for segmentation"""
    x1: int = Field(..., description="Top-left X")
    y1: int = Field(..., description="Top-left Y")
    x2: int = Field(..., description="Bottom-right X")
    y2: int = Field(..., description="Bottom-right Y")


class SegmentationResponse(BaseModel):
    """Response model for segmentation"""
    status: str = Field(..., description="Status of the operation")
    message: str = Field(..., description="Human-readable message")
    mask_url: Optional[str] = Field(None, description="URL to download mask")
    image_shape: Tuple[int, int] = Field(..., description="(height, width) of image")
    confidence: float = Field(..., description="Confidence score of segmentation")
