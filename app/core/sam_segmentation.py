"""
SAM (Segment Anything Model) segmentation logic
"""
import numpy as np
from typing import List, Tuple, Optional
from segment_anything import SamPredictor

from app.utils.logger import logger


class SAMSegmentation:
    """Wrapper for SAM segmentation operations"""
    
    def __init__(self, predictor: SamPredictor):
        """
        Initialize SAM segmentation
        
        Args:
            predictor: SAM predictor instance
        """
        self.predictor = predictor
        self._current_image = None
    
    def set_image(self, image: np.ndarray) -> None:
        """
        Set image for segmentation
        
        Args:
            image: RGB image as numpy array (H, W, 3)
        """
        logger.info(f"🖼️  Setting image for SAM: {image.shape}")
        self.predictor.set_image(image)
        self._current_image = image
    
    def segment_by_points(
        self,
        point_coords: List[Tuple[int, int]],
        point_labels: List[int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment object using point prompts
        
        Args:
            point_coords: List of (x, y) coordinates
            point_labels: List of labels (1 = foreground, 0 = background)
            
        Returns:
            Tuple of (masks, scores, logits)
            - masks: (N, H, W) boolean array
            - scores: (N,) confidence scores
            - logits: (N, H, W) raw logits
        """
        if self._current_image is None:
            raise ValueError("No image set. Call set_image() first.")
        
        point_coords_np = np.array(point_coords)
        point_labels_np = np.array(point_labels)
        
        logger.info(f"🎯 Segmenting with {len(point_coords)} points")
        
        masks, scores, logits = self.predictor.predict(
            point_coords=point_coords_np,
            point_labels=point_labels_np,
            multimask_output=True  # Generate 3 masks with different quality
        )
        
        logger.info(f"✅ Generated {len(masks)} masks with scores: {scores}")
        return masks, scores, logits
    
    def segment_by_box(
        self,
        box: Tuple[int, int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Segment object using bounding box
        
        Args:
            box: (x1, y1, x2, y2) bounding box coordinates
            
        Returns:
            Tuple of (masks, scores, logits)
        """
        if self._current_image is None:
            raise ValueError("No image set. Call set_image() first.")
        
        box_np = np.array(box)
        
        logger.info(f"📦 Segmenting with box: {box}")
        
        masks, scores, logits = self.predictor.predict(
            box=box_np,
            multimask_output=False  # Single mask for box
        )
        
        logger.info(f"✅ Generated mask with score: {scores[0]:.3f}")
        return masks, scores, logits
    
    def get_best_mask(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
        prefer_larger: bool = True
    ) -> np.ndarray:
        """
        Get the best mask based on confidence scores and size
        
        Args:
            masks: (N, H, W) array of masks
            scores: (N,) array of scores
            prefer_larger: If True, prefer larger masks (whole object vs sub-part)
            
        Returns:
            Best mask (H, W)
        """
        if prefer_larger:
            # Calculate mask areas
            areas = np.array([mask.sum() for mask in masks])
            
            # Normalize scores and areas
            norm_scores = scores / scores.max() if scores.max() > 0 else scores
            norm_areas = areas / areas.max() if areas.max() > 0 else areas
            
            # Combined score: 60% confidence + 40% size
            combined_scores = 0.6 * norm_scores + 0.4 * norm_areas
            best_idx = np.argmax(combined_scores)
            
            logger.info(f"🏆 Best mask index: {best_idx}")
            logger.info(f"   - Confidence: {scores[best_idx]:.3f}")
            logger.info(f"   - Area: {areas[best_idx]} pixels ({areas[best_idx]/(masks[0].size)*100:.1f}% of image)")
            logger.info(f"   - Combined score: {combined_scores[best_idx]:.3f}")
        else:
            best_idx = np.argmax(scores)
            logger.info(f"🏆 Best mask index: {best_idx} with score: {scores[best_idx]:.3f}")
        
        return masks[best_idx]
    
    def get_all_masks_with_scores(
        self,
        masks: np.ndarray,
        scores: np.ndarray
    ) -> list:
        """
        Get all masks with their scores and metadata
        
        Args:
            masks: (N, H, W) array of masks
            scores: (N,) array of scores
            
        Returns:
            List of dicts with mask info
        """
        results = []
        for idx, (mask, score) in enumerate(zip(masks, scores)):
            area = mask.sum()
            results.append({
                "index": idx,
                "mask": mask,
                "score": float(score),
                "area": int(area),
                "area_percentage": float(area / mask.size * 100)
            })
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
