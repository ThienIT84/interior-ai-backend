"""
Segmentation endpoints using SAM
"""
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from pathlib import Path
import uuid
from datetime import datetime
from PIL import Image

from app.dependencies import get_model_manager, ModelManager
from app.core.sam_segmentation import SAMSegmentation
from app.core.sam3_replicate_segmentation import get_sam3_replicate_service
from app.utils.image_processing import load_image_from_bytes
from app.utils.logger import logger
from app.utils.mask_storage import MaskStorage
from app.config import settings
from app.models.segmentation import SegmentationRequest

router = APIRouter()


def _is_sam3_backend() -> bool:
    return settings.SEGMENTATION_BACKEND.strip().lower() == "sam3_replicate"


def _normalize_backend_name(backend: str | None) -> str:
    if not backend:
        return settings.SEGMENTATION_BACKEND.strip().lower()

    normalized = backend.strip().lower()
    if normalized not in {"local", "sam3_replicate"}:
        raise HTTPException(
            status_code=400,
            detail='Invalid segmentation_backend. Use "local" or "sam3_replicate".',
        )
    return normalized


def _resolve_effective_backend(requested_backend: str | None) -> str:
    return _normalize_backend_name(requested_backend)


def _get_model_name_for_backend(backend: str) -> str:
    if backend == "sam3_replicate":
        return settings.SAM3_REPLICATE_MODEL
    return f"local_sam:vit_b:{settings.SAM_CHECKPOINT}"


@router.get("/debug/backend")
async def get_segmentation_backend_debug(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Debug endpoint to inspect active segmentation backend and model selection."""
    default_backend = _normalize_backend_name(settings.SEGMENTATION_BACKEND)

    return JSONResponse(content={
        "status": "success",
        "segmentation": {
            "default_backend": default_backend,
            "default_model": _get_model_name_for_backend(default_backend),
            "fallback_to_local": settings.SEGMENTATION_FALLBACK_TO_LOCAL,
            "supported_backends": ["local", "sam3_replicate"],
            "local_sam_loaded": model_manager.is_sam_loaded,
            "local_sam_checkpoint": settings.SAM_CHECKPOINT,
            "sam3_replicate_model": settings.SAM3_REPLICATE_MODEL,
            "per_request_override_supported": True,
        }
    })


def _segment_points_local(
    image_np: np.ndarray,
    point_coords: list[tuple[int, int]],
    point_labels: list[int],
    model_manager: ModelManager,
):
    """Local SAM point segmentation flow."""
    sam_seg = SAMSegmentation(model_manager.sam_predictor)
    sam_seg.set_image(image_np)

    n_foreground = sum(1 for l in point_labels if l == 1)
    if n_foreground >= 2:
        logger.info(
            f"🔀 {n_foreground} foreground points → using multi-object segmentation "
            f"(segment each separately then merge)"
        )
        best_mask, best_score = sam_seg.segment_multiple_objects(point_coords, point_labels)
        all_masks_info = []
    else:
        masks, scores, _ = sam_seg.segment_by_points(point_coords, point_labels)
        best_mask = sam_seg.get_best_mask(masks, scores, prefer_larger=True)
        best_score = float(np.max(scores))
        all_masks_info = sam_seg.get_all_masks_with_scores(masks, scores)

        logger.info(f"📊 Generated {len(masks)} masks:")
        for info in all_masks_info:
            logger.info(f"   Mask {info['index']}: score={info['score']:.3f}, area={info['area_percentage']:.1f}%")

    return best_mask, best_score, all_masks_info


@router.post("/upload")
async def upload_image(
    file: UploadFile = File(...),
):
    """
    Upload image and return image_id for later segmentation
    
    This is a simple upload endpoint that just saves the image
    and returns an ID. The actual segmentation happens in /segment-points
    """
    try:
        # Read and process image
        contents = await file.read()
        image_pil, image_np = load_image_from_bytes(contents)
        
        # Save input image
        image_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = f"{timestamp}_{image_id}.jpg"
        input_path = settings.INPUTS_DIR / input_filename
        
        image_pil.save(input_path, quality=settings.JPEG_QUALITY)
        logger.info(f"💾 Saved input image: {input_path}")
        
        h, w = image_np.shape[:2]
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Image uploaded successfully: {w}x{h}",
            "image_id": image_id,
            "image_shape": {"width": w, "height": h},
        })
        
    except Exception as e:
        logger.error(f"❌ Error uploading image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/segment")
async def segment_image(
    file: UploadFile = File(...),
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Upload image and prepare for segmentation
    This endpoint sets the image in SAM and returns basic info
    """
    try:
        # Read and process image
        contents = await file.read()
        image_pil, image_np = load_image_from_bytes(contents)
        
        backend_used = _resolve_effective_backend(None)
        if backend_used == "local":
            # Warm local SAM for faster first point-segmentation call
            sam_seg = SAMSegmentation(model_manager.sam_predictor)
            sam_seg.set_image(image_np)
        
        # Save input image
        image_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_filename = f"{timestamp}_{image_id}.jpg"
        input_path = settings.INPUTS_DIR / input_filename
        
        image_pil.save(input_path, quality=settings.JPEG_QUALITY)
        logger.info(f"💾 Saved input image: {input_path}")
        
        h, w = image_np.shape[:2]
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Ông họa sĩ AI đã nhìn thấy ảnh {w}x{h} của em.",
            "image_id": image_id,
            "image_shape": {"width": w, "height": h},
            "ready_for_segmentation": True,
            "segmentation_backend": backend_used,
            "segmentation_model": _get_model_name_for_backend(backend_used),
            "note": "Gửi point hoặc box prompts để thực hiện segmentation"
        })
        
    except Exception as e:
        logger.error(f"❌ Error in segmentation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/segment-points")
async def segment_with_points_json(
    request: SegmentationRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Segment image using point prompts (JSON body)
    
    This is the preferred endpoint with proper request validation
    
    Request body:
    {
        "image_id": "abc123-def456",
        "points": [
            {"x": 100, "y": 150, "label": 1},
            {"x": 200, "y": 250, "label": 1}
        ]
    }
    
    Returns:
        Segmentation result with mask URL, confidence, and all mask options
    """
    try:
        # Extract data from request
        image_id = request.image_id
        point_coords = [(p.x, p.y) for p in request.points]
        point_labels = [p.label for p in request.points]
        requested_backend = _resolve_effective_backend(request.segmentation_backend)
        text_prompt = (request.text_prompt or "").strip() or "object"

        logger.info(f"🎯 Segmenting image {image_id} with {len(point_coords)} points"
                    + (f" | text='{text_prompt}'" if requested_backend == "sam3_replicate" else ""))
        
        # Load image
        input_files = list(settings.INPUTS_DIR.glob(f"*_{image_id}.jpg"))
        if not input_files:
            raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
        
        image_path = input_files[0]
        image_pil = Image.open(image_path)
        image_np = np.array(image_pil)
        
        backend_used = requested_backend
        all_masks_info = []

        if requested_backend == "sam3_replicate":
            try:
                sam3_service = get_sam3_replicate_service()
                best_mask, best_score, _ = sam3_service.segment_by_points(
                    image_path=image_path,
                    point_coords=point_coords,
                    point_labels=point_labels,
                    text_prompt=text_prompt,
                )
            except Exception as sam3_error:
                if settings.SEGMENTATION_FALLBACK_TO_LOCAL:
                    logger.warning(
                        "⚠️ SAM3 Replicate failed, falling back to local SAM: "
                        f"{sam3_error}"
                    )
                    best_mask, best_score, all_masks_info = _segment_points_local(
                        image_np=image_np,
                        point_coords=point_coords,
                        point_labels=point_labels,
                        model_manager=model_manager,
                    )
                    backend_used = "local_fallback"
                else:
                    raise RuntimeError(f"SAM3 Replicate segmentation failed: {sam3_error}")
        else:
            best_mask, best_score, all_masks_info = _segment_points_local(
                image_np=image_np,
                point_coords=point_coords,
                point_labels=point_labels,
                model_manager=model_manager,
            )
        
        # Save mask with metadata
        mask_id = str(uuid.uuid4())
        metadata = {
            "confidence": best_score,
            "segmentation_backend": backend_used,
            "segmentation_model": _get_model_name_for_backend(
                "local" if backend_used == "local_fallback" else backend_used
            ),
            "text_prompt": text_prompt if requested_backend == "sam3_replicate" else None,
            "num_points": len(point_coords),
            "points": [{"x": int(x), "y": int(y), "label": int(l)} 
                      for (x, y), l in zip(point_coords, point_labels)],
            "mask_area_percentage": float((best_mask.sum() / best_mask.size) * 100),
            "all_masks": [
                {
                    "index": info["index"],
                    "score": info["score"],
                    "area_percentage": info["area_percentage"],
                    "type": "whole" if info["index"] == 0 else ("part" if info["index"] == 1 else "sub-part")
                }
                for info in all_masks_info
            ]
        }
        
        mask_path, metadata_path = MaskStorage.save_mask_with_metadata(
            mask=best_mask,
            mask_id=mask_id,
            image_id=image_id,
            metadata=metadata
        )
        
        # Generate mask URL (relative path)
        mask_url = f"/data/masks/{mask_path.name}"
        
        h, w = image_np.shape[:2]
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Segmentation completed with confidence {best_score:.2f}",
            "mask_id": mask_id,
            "mask_url": mask_url,
            "image_shape": {"width": w, "height": h},
            "confidence": best_score,
            "segmentation_backend": backend_used,
            "segmentation_model": metadata["segmentation_model"],
            "num_points": len(point_coords),
            "mask_area_percentage": round(metadata["mask_area_percentage"], 2),
            "all_masks": metadata["all_masks"]
        })
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"❌ Error in point-based segmentation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/segment-box")
async def segment_with_box(
    image_id: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Segment image using bounding box
    
    This is MORE ACCURATE than points for furniture!
    
    Args:
        image_id: ID of previously uploaded image
        x1, y1: Top-left corner
        x2, y2: Bottom-right corner
        
    Returns:
        Segmentation result with mask
    """
    try:
        logger.info(f"📦 Segmenting image {image_id} with box: ({x1},{y1}) to ({x2},{y2})")
        
        # Load image
        input_files = list(settings.INPUTS_DIR.glob(f"*_{image_id}.jpg"))
        if not input_files:
            raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
        
        image_path = input_files[0]
        image_pil = Image.open(image_path)
        image_np = np.array(image_pil)
        
        # SAM3 endpoint is currently wired for point prompts.
        # Box mode uses local SAM and can still be reached when fallback is enabled.
        if _resolve_effective_backend(None) == "sam3_replicate" and not settings.SEGMENTATION_FALLBACK_TO_LOCAL:
            raise HTTPException(
                status_code=400,
                detail="segment-box is only available with local SAM currently. "
                       "Set SEGMENTATION_FALLBACK_TO_LOCAL=true or switch SEGMENTATION_BACKEND=local",
            )

        # Initialize local SAM and set image
        sam_seg = SAMSegmentation(model_manager.sam_predictor)
        sam_seg.set_image(image_np)
        
        # Perform segmentation with box
        box = (x1, y1, x2, y2)
        masks, scores, logits = sam_seg.segment_by_box(box)
        
        # Box segmentation returns single mask
        best_mask = masks[0]
        best_score = float(scores[0])
        
        logger.info(f"✅ Box segmentation score: {best_score:.3f}")
        
        # Save mask with metadata
        mask_id = str(uuid.uuid4())
        metadata = {
            "confidence": best_score,
            "segmentation_backend": "local",
            "segmentation_type": "box",
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "mask_area_percentage": float((best_mask.sum() / best_mask.size) * 100)
        }
        
        mask_path, metadata_path = MaskStorage.save_mask_with_metadata(
            mask=best_mask,
            mask_id=mask_id,
            image_id=image_id,
            metadata=metadata
        )
        
        # Generate mask URL
        mask_url = f"/data/masks/{mask_path.name}"
        
        h, w = image_np.shape[:2]
        
        return JSONResponse(content={
            "status": "success",
            "message": f"Box segmentation completed with confidence {best_score:.2f}",
            "mask_id": mask_id,
            "mask_url": mask_url,
            "image_shape": {"width": w, "height": h},
            "confidence": best_score,
            "segmentation_backend": "local",
            "mask_area_percentage": round(metadata["mask_area_percentage"], 2),
            "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        })
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"❌ Error in box segmentation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mask/{mask_id}")
async def get_mask_info(mask_id: str):
    """
    Get mask metadata by mask_id
    
    Args:
        mask_id: Unique mask identifier
        
    Returns:
        Mask metadata including confidence, points, area, etc.
    """
    try:
        metadata = MaskStorage.load_metadata(mask_id)
        
        if metadata is None:
            raise HTTPException(status_code=404, detail=f"Mask {mask_id} not found")
        
        return JSONResponse(content={
            "status": "success",
            "mask": metadata
        })
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"❌ Error retrieving mask: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/masks/image/{image_id}")
async def list_masks_for_image(image_id: str):
    """
    List all masks for a given image
    
    Args:
        image_id: Image identifier
        
    Returns:
        List of all masks created for this image
    """
    try:
        masks = MaskStorage.list_masks_for_image(image_id)
        
        return JSONResponse(content={
            "status": "success",
            "image_id": image_id,
            "count": len(masks),
            "masks": masks
        })
        
    except Exception as e:
        logger.error(f"❌ Error listing masks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualize/{mask_id}")
async def visualize_mask(
    mask_id: str,
    style: str = "overlay"  # overlay, contour, side-by-side, isolated
):
    """
    Visualize mask overlaid on original image
    
    Args:
        mask_id: Unique mask identifier
        style: Visualization style
            - overlay: Semi-transparent mask overlay
            - contour: Mask with contour outline
            - side-by-side: Original and masked side by side
            - isolated: Only masked region on white background
            
    Returns:
        Visualization image
    """
    from fastapi.responses import FileResponse
    from app.utils.mask_utils import (
        overlay_mask_on_image,
        overlay_mask_with_contour,
        create_side_by_side_comparison,
        apply_mask_to_image
    )
    
    try:
        # Load mask and metadata
        mask = MaskStorage.load_mask(mask_id)
        metadata = MaskStorage.load_metadata(mask_id)
        
        if mask is None or metadata is None:
            raise HTTPException(status_code=404, detail=f"Mask {mask_id} not found")
        
        # Load original image
        image_id = metadata["image_id"]
        input_files = list(settings.INPUTS_DIR.glob(f"*_{image_id}.jpg"))
        if not input_files:
            raise HTTPException(status_code=404, detail=f"Original image not found")
        
        image_pil = Image.open(input_files[0])
        image_np = np.array(image_pil)
        
        # Convert mask to boolean
        mask_bool = mask > 127
        
        # Generate visualization based on style
        if style == "overlay":
            result = overlay_mask_on_image(image_np, mask_bool, color=(255, 0, 0), alpha=0.4)
        elif style == "contour":
            result = overlay_mask_with_contour(
                image_np, mask_bool,
                fill_color=(255, 0, 0),
                contour_color=(0, 255, 0),
                fill_alpha=0.3,
                contour_width=3
            )
        elif style == "side-by-side":
            masked = overlay_mask_on_image(image_np, mask_bool, color=(255, 0, 0), alpha=0.4)
            result = create_side_by_side_comparison(image_np, masked)
        elif style == "isolated":
            result = apply_mask_to_image(image_np, mask_bool, background_color=(255, 255, 255))
        else:
            raise HTTPException(status_code=400, detail=f"Unknown style: {style}")
        
        # Save visualization
        vis_filename = f"vis_{mask_id}_{style}.jpg"
        vis_path = settings.OUTPUTS_DIR / vis_filename
        Image.fromarray(result).save(vis_path, quality=95)
        
        logger.info(f"🎨 Created visualization: {vis_path}")
        
        # Return image file
        return FileResponse(
            vis_path,
            media_type="image/jpeg",
            filename=vis_filename
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"❌ Error visualizing mask: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/image/{image_id}")
async def get_image(image_id: str):
    """
    Get uploaded image by image_id
    
    Args:
        image_id: Image identifier
        
    Returns:
        Image file
    """
    from fastapi.responses import FileResponse
    
    try:
        # Find image file
        input_files = list(settings.INPUTS_DIR.glob(f"*_{image_id}.jpg"))
        if not input_files:
            raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
        
        image_path = input_files[0]
        
        return FileResponse(
            image_path,
            media_type="image/jpeg",
            filename=f"{image_id}.jpg"
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"❌ Error retrieving image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/mask-image/{mask_id}")
async def get_mask_image(mask_id: str):
    """
    Get mask image file by mask_id
    
    Args:
        mask_id: Mask identifier
        
    Returns:
        Mask PNG file
    """
    from fastapi.responses import FileResponse
    
    try:
        # Find mask file
        mask_files = list(settings.MASKS_DIR.glob(f"*_{mask_id}.png"))
        if not mask_files:
            raise HTTPException(status_code=404, detail=f"Mask {mask_id} not found")
        
        mask_path = mask_files[0]
        
        return FileResponse(
            mask_path,
            media_type="image/png",
            filename=f"{mask_id}.png"
        )
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"❌ Error retrieving mask image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

