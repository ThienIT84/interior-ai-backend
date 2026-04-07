# SAM3 Migration Design Document

## Overview

This design document outlines the migration from SAM v1 (local model, 4GB VRAM) to SAM3 (cloud-based via Replicate API) for the AI Interior Design project. This migration is critical for **Mục tiêu 2: Semantic Inpainting** - the highest priority feature for the Computer Vision capstone project.

### Business Context

**Problem**: SAM v1 requires local GPU inference on GTX 1650 4GB, which:
- Consumes limited VRAM needed for other models (Stable Diffusion)
- Adds complexity to model management and dependencies
- Slower inference compared to cloud APIs
- Difficult to scale for multiple concurrent users

**Solution**: Migrate to SAM3 via Replicate API:
- Zero VRAM usage (cloud-based)
- Faster inference (~2-5 seconds)
- Dual-mode support: text prompts + point-based segmentation
- Cost-effective: $0.00098 per request
- Simplifies deployment and maintenance

### Key Design Decisions

1. **Complete Replacement**: SAM3 replaces SAM v1 entirely (no hybrid mode)
2. **API Compatibility**: Maintain existing endpoint signatures for Flutter app compatibility
3. **Dual-Mode Support**: Support both text prompts (new) and point-based (existing) segmentation
4. **Backward Compatibility**: Existing Flutter code continues to work without changes
5. **Migration Strategy**: Update backend core logic, keep API layer stable

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Flutter Mobile App                       │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐ │
│  │ Camera Screen  │  │ Segmentation   │  │ Inpainting    │ │
│  │                │→ │ Screen         │→ │ Screen        │ │
│  └────────────────┘  └────────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              ↓ HTTP/JSON
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Backend (WSL)                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              API Layer (app/api/v1/)                  │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │  │
│  │  │ /upload      │  │ /segment-    │  │ /segment-  │ │  │
│  │  │              │  │  points      │  │  box       │ │  │
│  │  └──────────────┘  └──────────────┘  └────────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
│                              ↓                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Core Logic (app/core/)                      │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │         SAM3Segmentation                        │  │  │
│  │  │  - segment_by_text(image, prompt)              │  │  │
│  │  │  - segment_by_points(image, points, labels)    │  │  │
│  │  │  - _encode_image_to_bytes()                    │  │  │
│  │  │  - _download_mask()                            │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                              ↓                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │        Services (app/services/)                       │  │
│  │  ┌────────────────────────────────────────────────┐  │  │
│  │  │      Replicate Client (existing)                │  │  │
│  │  │  - API token management                         │  │  │
│  │  │  - Error handling & retries                     │  │  │
│  │  └────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓ HTTPS
┌─────────────────────────────────────────────────────────────┐
│                    Replicate API (Cloud)                     │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  SAM3 Model: mattsays/sam3-image:d73db077...         │ │
│  │  - Text-based segmentation                            │ │
│  │  - Point-based segmentation                           │ │
│  │  - Returns: PNG mask URL                             │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

#### Flow 1: Text-Based Segmentation (New Feature)
```
User → Flutter → POST /segment-text → SAM3Segmentation.segment_by_text()
                                    → Replicate API (text prompt)
                                    → Download mask PNG
                                    → Save to storage
                                    ← Return mask_id + URL
```

#### Flow 2: Point-Based Segmentation (Existing, Migrated)
```
User → Flutter → POST /segment-points → SAM3Segmentation.segment_by_points()
                                      → Replicate API (point coords)
                                      → Download mask PNG
                                      → Save to storage
                                      ← Return mask_id + URL
```

---

## Components and Interfaces

### 1. Core Component: SAM3Segmentation

**Location**: `backend/app/core/sam3_segmentation.py`

**Responsibilities**:
- Encapsulate all SAM3 API interactions
- Handle image encoding (numpy → bytes)
- Handle mask decoding (URL → numpy array)
- Provide unified interface for both segmentation modes
- Error handling and retry logic

**Class Structure**:

```python
class SAM3Segmentation:
    """
    Wrapper for SAM3 segmentation via Replicate API
    Replaces local SAM v1 model with cloud-based inference
    """
    
    def __init__(self):
        """Initialize SAM3 with Replicate model reference"""
        self.model = "mattsays/sam3-image:d73db077..."
        # No local model loading - zero VRAM usage
    
    def segment_by_text(
        self, 
        image: np.ndarray,
        text_prompt: str,
        box_threshold: float = 0.3,
        iou_threshold: float = 0.5
    ) -> Tuple[np.ndarray, float]:
        """
        Segment objects using natural language description
        
        Args:
            image: Input image (H, W, 3) RGB numpy array
            text_prompt: Natural language description (e.g., "sofa, chair")
            box_threshold: Confidence threshold for object detection
            iou_threshold: IoU threshold for NMS
            
        Returns:
            mask: Binary mask (H, W) uint8 numpy array
            confidence: Confidence score (0.0-1.0)
            
        Raises:
            ValueError: If text_prompt is empty
            RuntimeError: If API call fails
        """
    
    def segment_by_points(
        self,
        image: np.ndarray,
        point_coords: List[Tuple[int, int]],
        point_labels: List[int]
    ) -> Tuple[np.ndarray, float]:
        """
        Segment objects using point prompts (SAM v1 compatibility)
        
        Args:
            image: Input image (H, W, 3) RGB numpy array
            point_coords: List of (x, y) coordinates
            point_labels: List of labels (1=foreground, 0=background)
            
        Returns:
            mask: Binary mask (H, W) uint8 numpy array
            confidence: Confidence score (0.0-1.0)
            
        Raises:
            ValueError: If points/labels mismatch or empty
            RuntimeError: If API call fails
        """
    
    def _encode_image_to_bytes(self, image: np.ndarray) -> bytes:
        """Convert numpy array to PNG bytes for API upload"""
    
    def _download_mask(self, mask_url: str) -> np.ndarray:
        """Download mask PNG from URL and convert to numpy array"""
    
    def _validate_mask(self, mask: np.ndarray, image_shape: Tuple) -> bool:
        """Validate mask dimensions match input image"""
```

**Key Design Patterns**:
- **Adapter Pattern**: Wraps Replicate API to match SAM v1 interface
- **Fail-Fast**: Validate inputs before expensive API calls
- **Separation of Concerns**: Image encoding/decoding isolated in private methods

### 2. API Endpoints

**Location**: `backend/app/api/v1/endpoints/segmentation.py`

#### Endpoint 1: Upload Image (Unchanged)
```python
POST /api/v1/segmentation/upload
Content-Type: multipart/form-data

Request:
  file: <image file>

Response:
  {
    "status": "success",
    "image_id": "abc123",
    "image_shape": {"width": 1920, "height": 1080}
  }
```

#### Endpoint 2: Segment by Points (Updated Implementation)
```python
POST /api/v1/segmentation/segment-points
Content-Type: application/json

Request:
  {
    "image_id": "abc123",
    "points": [
      {"x": 100, "y": 200, "label": 1},  // foreground
      {"x": 500, "y": 600, "label": 0}   // background
    ]
  }

Response:
  {
    "status": "success",
    "message": "Segmentation completed with confidence 0.95",
    "mask_id": "mask_xyz",
    "mask_url": "/data/masks/mask_xyz.png",
    "confidence": 0.95,
    "mask_area_percentage": 23.5,
    "processing_time_ms": 2340
  }
```

**Implementation Changes**:
- Replace `SAMSegmentation` with `SAM3Segmentation`
- Remove `model_manager` dependency (no local model)
- Remove multi-object segmentation logic (SAM3 handles internally)
- Simplify to single API call

#### Endpoint 3: Segment by Text (New)
```python
POST /api/v1/segmentation/segment-text
Content-Type: application/json

Request:
  {
    "image_id": "abc123",
    "text_prompt": "sofa, chair, table",
    "box_threshold": 0.3,  // optional
    "iou_threshold": 0.5   // optional
  }

Response:
  {
    "status": "success",
    "message": "Text-based segmentation completed",
    "mask_id": "mask_xyz",
    "mask_url": "/data/masks/mask_xyz.png",
    "confidence": 0.92,
    "detected_objects": ["sofa", "chair"],
    "mask_area_percentage": 35.2,
    "processing_time_ms": 3120
  }
```

#### Endpoint 4: Segment by Box (Updated Implementation)
```python
POST /api/v1/segmentation/segment-box
Content-Type: application/json

Request:
  {
    "image_id": "abc123",
    "x1": 100, "y1": 100,
    "x2": 500, "y2": 400
  }

Response:
  {
    "status": "success",
    "message": "Box segmentation completed",
    "mask_id": "mask_xyz",
    "mask_url": "/data/masks/mask_xyz.png",
    "confidence": 0.98,
    "mask_area_percentage": 18.3,
    "processing_time_ms": 2100
  }
```

**Note**: SAM3 doesn't natively support box prompts. Implementation options:
1. Convert box to 4 corner points (recommended)
2. Use text prompt with spatial description
3. Return error and guide user to use points/text

### 3. Dependency Injection Updates

**Location**: `backend/app/dependencies.py`

**Current State** (SAM v1):
```python
class ModelManager:
    def __init__(self):
        self.sam_predictor = load_sam_model()  # 4GB VRAM
        
def get_model_manager() -> ModelManager:
    return _model_manager
```

**New State** (SAM3):
```python
class ModelManager:
    def __init__(self):
        # SAM3 doesn't need local model
        self.sam3 = SAM3Segmentation()  # Zero VRAM
        
def get_model_manager() -> ModelManager:
    return _model_manager
```

**Migration Impact**:
- Remove SAM v1 checkpoint loading (~2.4GB file)
- Remove `segment_anything` library dependency
- Faster startup time (no model loading)
- Reduced memory footprint

---

## Data Models

### Request Models

**Location**: `backend/app/models/segmentation.py`

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class PointPrompt(BaseModel):
    """Point prompt for segmentation"""
    x: int = Field(..., description="X coordinate", ge=0)
    y: int = Field(..., description="Y coordinate", ge=0)
    label: int = Field(..., description="1=foreground, 0=background", ge=0, le=1)

class SegmentationRequest(BaseModel):
    """Request for point-based segmentation"""
    image_id: str = Field(..., description="ID of uploaded image")
    points: List[PointPrompt] = Field(..., min_items=1, description="Point prompts")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_id": "img_abc123",
                "points": [
                    {"x": 100, "y": 200, "label": 1},
                    {"x": 500, "y": 600, "label": 0}
                ]
            }
        }

class TextSegmentationRequest(BaseModel):
    """Request for text-based segmentation (NEW)"""
    image_id: str = Field(..., description="ID of uploaded image")
    text_prompt: str = Field(..., min_length=1, description="Natural language description")
    box_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Detection confidence threshold")
    iou_threshold: float = Field(0.5, ge=0.0, le=1.0, description="NMS IoU threshold")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_id": "img_abc123",
                "text_prompt": "sofa, chair, coffee table",
                "box_threshold": 0.3,
                "iou_threshold": 0.5
            }
        }

class BoxSegmentationRequest(BaseModel):
    """Request for box-based segmentation"""
    image_id: str = Field(..., description="ID of uploaded image")
    x1: int = Field(..., ge=0, description="Top-left X")
    y1: int = Field(..., ge=0, description="Top-left Y")
    x2: int = Field(..., ge=0, description="Bottom-right X")
    y2: int = Field(..., ge=0, description="Bottom-right Y")
```

### Response Models

```python
class SegmentationResponse(BaseModel):
    """Response for all segmentation types"""
    status: str = Field(..., description="success or error")
    message: str = Field(..., description="Human-readable message")
    mask_id: str = Field(..., description="Unique mask identifier")
    mask_url: str = Field(..., description="URL to download mask PNG")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Segmentation confidence")
    mask_area_percentage: float = Field(..., ge=0.0, le=100.0, description="% of image covered")
    processing_time_ms: int = Field(..., ge=0, description="Processing time in milliseconds")
    
    # Optional fields for text-based segmentation
    detected_objects: Optional[List[str]] = Field(None, description="Objects found by text prompt")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Segmentation completed with confidence 0.95",
                "mask_id": "mask_20240115_123456",
                "mask_url": "/data/masks/mask_20240115_123456.png",
                "confidence": 0.95,
                "mask_area_percentage": 23.5,
                "processing_time_ms": 2340
            }
        }
```

### Error Response Model

```python
class ErrorResponse(BaseModel):
    """Standard error response"""
    status: str = Field(default="error")
    error_type: str = Field(..., description="Error category")
    message: str = Field(..., description="Error description")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "error",
                "error_type": "ValidationError",
                "message": "Invalid point coordinates",
                "details": {"point_index": 2, "reason": "x coordinate out of bounds"}
            }
        }
```

---

## Integration Strategy

### Phase 1: Core Migration (Priority 1)

**Goal**: Replace SAM v1 with SAM3 in core logic

**Tasks**:
1. Update `SAM3Segmentation` class with complete implementation
2. Add comprehensive error handling and logging
3. Add unit tests for image encoding/decoding
4. Test with sample images (furniture, rooms)

**Success Criteria**:
- SAM3 can segment objects via text prompts
- SAM3 can segment objects via point prompts
- Masks are correctly downloaded and validated
- Processing time < 5 seconds per image

### Phase 2: API Layer Update (Priority 2)

**Goal**: Update endpoints to use SAM3

**Tasks**:
1. Update `/segment-points` endpoint
2. Add `/segment-text` endpoint
3. Update `/segment-box` endpoint (convert box → points)
4. Remove `model_manager.sam_predictor` dependency
5. Update error handling for API failures

**Success Criteria**:
- All endpoints return correct response format
- Backward compatibility maintained for Flutter app
- API errors are user-friendly
- Response times logged for monitoring

### Phase 3: Dependency Cleanup (Priority 3)

**Goal**: Remove SAM v1 artifacts

**Tasks**:
1. Remove `segment_anything` from requirements.txt
2. Delete SAM v1 checkpoint file (~2.4GB)
3. Remove `sam_segmentation.py` (old implementation)
4. Update `dependencies.py` to remove SAM v1 loading
5. Update documentation

**Success Criteria**:
- Backend starts without loading SAM v1
- Disk space freed (~2.4GB)
- No import errors
- Startup time reduced

### Phase 4: Flutter Integration (Priority 4)

**Goal**: Add text-based segmentation UI

**Tasks**:
1. Add text input field to segmentation screen
2. Add mode toggle (points vs text)
3. Update API service to call `/segment-text`
4. Add UI feedback for text segmentation
5. Test end-to-end flow

**Success Criteria**:
- User can switch between point and text modes
- Text segmentation works from Flutter app
- UI is intuitive and responsive
- Error messages are clear

---

## Migration Path & Backward Compatibility

### Compatibility Matrix

| Component | SAM v1 | SAM3 | Compatibility |
|-----------|--------|------|---------------|
| `/upload` endpoint | ✅ | ✅ | 100% compatible |
| `/segment-points` endpoint | ✅ | ✅ | 100% compatible (response format unchanged) |
| `/segment-box` endpoint | ✅ | ✅ | 95% compatible (box→points conversion) |
| `/segment-text` endpoint | ❌ | ✅ | New feature |
| Flutter app (existing) | ✅ | ✅ | No changes needed |
| Mask storage format | PNG | PNG | 100% compatible |
| Inpainting pipeline | ✅ | ✅ | No changes needed |

### Breaking Changes

**None** - This is a drop-in replacement. Existing Flutter code continues to work without modifications.

### Deprecation Plan

1. **Week 1**: Deploy SAM3 alongside SAM v1 (both available)
2. **Week 2**: Monitor SAM3 performance and fix issues
3. **Week 3**: Make SAM3 default, keep SAM v1 as fallback
4. **Week 4**: Remove SAM v1 completely

**Rollback Strategy**: Keep SAM v1 code in git history for 1 month in case rollback is needed.

---

