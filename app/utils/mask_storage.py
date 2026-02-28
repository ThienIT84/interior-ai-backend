"""
Mask storage utilities - save and retrieve masks with metadata
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import numpy as np
from PIL import Image

from app.utils.logger import logger
from app.config import settings


class MaskStorage:
    """Handle mask saving and retrieval with metadata"""
    
    @staticmethod
    def save_mask_with_metadata(
        mask: np.ndarray,
        mask_id: str,
        image_id: str,
        metadata: Dict
    ) -> tuple[Path, Path]:
        """
        Save mask as PNG and metadata as JSON
        
        Args:
            mask: Binary mask (H, W) boolean or uint8 array
            mask_id: Unique mask identifier
            image_id: Associated image identifier
            metadata: Additional metadata (confidence, points, etc.)
            
        Returns:
            Tuple of (mask_path, metadata_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{timestamp}_{image_id}_{mask_id}"
        
        # Save mask as PNG
        mask_filename = f"{base_filename}.png"
        mask_path = settings.MASKS_DIR / mask_filename
        
        # Convert to uint8 if needed
        if mask.dtype == bool:
            mask_uint8 = (mask * 255).astype(np.uint8)
        else:
            mask_uint8 = mask
        
        mask_pil = Image.fromarray(mask_uint8)
        mask_pil.save(mask_path)
        logger.info(f"💾 Saved mask: {mask_path}")
        
        # Save metadata as JSON
        metadata_filename = f"{base_filename}.json"
        metadata_path = settings.MASKS_DIR / metadata_filename
        
        full_metadata = {
            "mask_id": mask_id,
            "image_id": image_id,
            "timestamp": timestamp,
            "mask_filename": mask_filename,
            "mask_shape": {"height": mask.shape[0], "width": mask.shape[1]},
            **metadata
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
        logger.info(f"📝 Saved metadata: {metadata_path}")
        
        return mask_path, metadata_path
    
    @staticmethod
    def load_mask(mask_id: str) -> Optional[np.ndarray]:
        """
        Load mask by mask_id
        
        Args:
            mask_id: Unique mask identifier
            
        Returns:
            Mask as numpy array or None if not found
        """
        # Find mask file
        mask_files = list(settings.MASKS_DIR.glob(f"*_{mask_id}.png"))
        
        if not mask_files:
            logger.warning(f"⚠️  Mask {mask_id} not found")
            return None
        
        mask_path = mask_files[0]
        mask_pil = Image.open(mask_path)
        mask_np = np.array(mask_pil)
        
        logger.info(f"📂 Loaded mask: {mask_path}")
        return mask_np
    
    @staticmethod
    def load_metadata(mask_id: str) -> Optional[Dict]:
        """
        Load metadata by mask_id
        
        Args:
            mask_id: Unique mask identifier
            
        Returns:
            Metadata dict or None if not found
        """
        # Find metadata file
        metadata_files = list(settings.MASKS_DIR.glob(f"*_{mask_id}.json"))
        
        if not metadata_files:
            logger.warning(f"⚠️  Metadata for mask {mask_id} not found")
            return None
        
        metadata_path = metadata_files[0]
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"📂 Loaded metadata: {metadata_path}")
        return metadata
    
    @staticmethod
    def get_mask_path(mask_id: str) -> Optional[Path]:
        """
        Get full path to mask file
        
        Args:
            mask_id: Unique mask identifier
            
        Returns:
            Path to mask file or None if not found
        """
        mask_files = list(settings.MASKS_DIR.glob(f"*_{mask_id}.png"))
        return mask_files[0] if mask_files else None
    
    @staticmethod
    def list_masks_for_image(image_id: str) -> list[Dict]:
        """
        List all masks for a given image
        
        Args:
            image_id: Image identifier
            
        Returns:
            List of metadata dicts for all masks of this image
        """
        metadata_files = list(settings.MASKS_DIR.glob(f"*_{image_id}_*.json"))
        
        masks_info = []
        for metadata_path in metadata_files:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                masks_info.append(metadata)
        
        # Sort by timestamp descending (newest first)
        masks_info.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        logger.info(f"📋 Found {len(masks_info)} masks for image {image_id}")
        return masks_info
