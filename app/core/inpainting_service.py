"""
Hybrid Inpainting Service
Supports multiple backends: Local GPU, Replicate API
Auto-fallback and configurable priority
"""
import os
import logging
from typing import Optional, Tuple, Literal
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

InpaintingMethod = Literal["local", "replicate", "auto"]


class InpaintingService:
    """
    Unified inpainting service with multiple backends
    
    Backends:
    - Local: GTX 1650 (15 min, FREE, always available)
    - Replicate: Cloud API (10 sec, $0.01/image, requires token)
    
    Strategy:
    - Development: Use local (free, no limits)
    - Demo: Use Replicate (fast, impressive)
    - Production: Configurable via env var
    """
    
    def __init__(self, default_method: InpaintingMethod = "auto"):
        """
        Initialize inpainting service
        
        Args:
            default_method: "local", "replicate", or "auto"
                - "local": Always use local GPU
                - "replicate": Always use Replicate API
                - "auto": Use Replicate if available, fallback to local
        """
        self.default_method = default_method
        
        # Check available backends
        self.local_available = True  # Always available
        # Dùng settings thay vì os.getenv vì pydantic_settings không populate os.environ
        self.replicate_available = bool(settings.REPLICATE_API_TOKEN)
        
        logger.info("🔧 Inpainting Service initialized")
        logger.info(f"   Default method: {default_method}")
        logger.info(f"   Local GPU: {'✅ Available' if self.local_available else '❌ Not available'}")
        logger.info(f"   Replicate API: {'✅ Available' if self.replicate_available else '❌ Token not set'}")
        
        # Lazy load backends
        self._local_service = None
        self._replicate_service = None
    
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        method: Optional[InpaintingMethod] = None,
        **kwargs
    ) -> Tuple[Image.Image, dict]:
        """
        Perform inpainting using specified or auto-selected method
        
        Args:
            image: Original RGB image
            mask: Grayscale mask (white=object to remove)
            prompt: Custom prompt
            negative_prompt: Custom negative prompt
            method: Override default method ("local", "replicate", "auto")
            **kwargs: Additional parameters for specific backend
        
        Returns:
            Tuple of (result_image, metadata_dict)
        """
        # Determine method
        method = method or self.default_method
        
        logger.info(f"🔧 Inpainting request received")
        logger.info(f"   Method requested: {method}")
        logger.info(f"   Replicate available: {self.replicate_available}")
        
        if method == "auto":
            # Auto-select: Prefer Replicate if available
            if self.replicate_available:
                method = "replicate"
                logger.info("🤖 Auto-selected: Replicate API (fast)")
            else:
                method = "local"
                logger.info("🤖 Auto-selected: Local GPU (Replicate token not set)")
        
        # Execute with selected method
        try:
            if method == "replicate":
                return self._inpaint_replicate(image, mask, prompt, negative_prompt, **kwargs)
            elif method == "local":
                return self._inpaint_local(image, mask, prompt, negative_prompt, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")
        
        except Exception as e:
            logger.error(f"❌ Inpainting failed with {method}: {e}")
            
            # Auto-fallback if using "auto" mode
            if self.default_method == "auto" and method == "replicate":
                logger.warning("⚠️  Falling back to local GPU...")
                return self._inpaint_local(image, mask, prompt, negative_prompt, **kwargs)
            else:
                raise
    
    def _inpaint_local(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: Optional[str],
        negative_prompt: Optional[str],
        **kwargs
    ) -> Tuple[Image.Image, dict]:
        """Inpaint using local GPU (GTX 1650)"""
        logger.info("🖥️  Using LOCAL GPU inpainting (15 min, FREE)")
        
        # Lazy load local service
        if self._local_service is None:
            from app.core.diffusion_inpainting import get_inpainting_service
            self._local_service = get_inpainting_service()
        
        result, metadata = self._local_service.inpaint(
            image=image,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            **kwargs
        )
        
        metadata["method"] = "local_gpu"
        metadata["cost"] = 0.0
        
        return result, metadata
    
    def _inpaint_replicate(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: Optional[str],
        negative_prompt: Optional[str],
        **kwargs
    ) -> Tuple[Image.Image, dict]:
        """Inpaint using Replicate API (cloud)"""
        logger.info("☁️  Using REPLICATE API inpainting (10 sec, $0.01)")
        
        if not self.replicate_available:
            raise ValueError(
                "Replicate API token not set! "
                "Set REPLICATE_API_TOKEN in .env or use method='local'"
            )
        
        # Lazy load Replicate service
        if self._replicate_service is None:
            from app.core.replicate_inpainting import get_replicate_inpainting_service
            self._replicate_service = get_replicate_inpainting_service()
        
        # Dịch tham số từ API chung sang Replicate format:
        # steps -> num_inference_steps, strength -> prompt_strength
        replicate_kwargs = {}
        if "steps" in kwargs and kwargs["steps"] is not None:
            replicate_kwargs["num_inference_steps"] = kwargs["steps"]
        if "guidance_scale" in kwargs and kwargs["guidance_scale"] is not None:
            replicate_kwargs["guidance_scale"] = kwargs["guidance_scale"]
        if "strength" in kwargs and kwargs["strength"] is not None:
            replicate_kwargs["prompt_strength"] = kwargs["strength"]
        if "seed" in kwargs and kwargs["seed"] is not None:
            replicate_kwargs["seed"] = kwargs["seed"]
        if "mask_padding" in kwargs and kwargs["mask_padding"] is not None:
            replicate_kwargs["mask_padding"] = kwargs["mask_padding"]

        result, metadata = self._replicate_service.inpaint(
            image=image,
            mask=mask,
            prompt=prompt,
            negative_prompt=negative_prompt,
            **replicate_kwargs
        )
        
        metadata["method"] = "replicate_api"
        metadata["cost"] = 0.01  # Approximate cost per image
        
        return result, metadata
    
    def get_available_methods(self) -> list[str]:
        """Get list of available inpainting methods"""
        methods = []
        if self.local_available:
            methods.append("local")
        if self.replicate_available:
            methods.append("replicate")
        return methods
    
    def get_recommended_method(self) -> str:
        """Get recommended method based on availability and use case"""
        if self.replicate_available:
            return "replicate"  # Fast, good for demo
        else:
            return "local"  # Free, always available


# Global instance (singleton pattern)
_inpainting_service_instance: Optional[InpaintingService] = None


def get_inpainting_service(default_method: Optional[InpaintingMethod] = None) -> InpaintingService:
    """
    Get or create global inpainting service instance
    
    Args:
        default_method: Override default method from env var
    """
    global _inpainting_service_instance
    
    if _inpainting_service_instance is None:
        # Read default method từ env - os.getenv hoạt động nếu được set trực tiếp,
        # fallback về settings nếu cần
        env_method = os.getenv("INPAINTING_METHOD") or getattr(settings, "INPAINTING_METHOD", "auto") or "auto"
        method = default_method or env_method
        
        _inpainting_service_instance = InpaintingService(default_method=method)
    
    return _inpainting_service_instance
