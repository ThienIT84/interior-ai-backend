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

# method="lama"        → allenhooo/lama via Replicate (~3s, $0.00057) — dùng cho remove-object
# method="replicate"   → stability-ai/sd-inpainting via Replicate (~10s, $0.01)
# method="local"       → local GTX 1650 (~15 phút, free)
# method="auto"        → lama nếu có token, rồi replicate, rồi local
InpaintingMethod = Literal["lama", "replicate", "local", "auto"]


class InpaintingService:
    """
    Unified inpainting service with multiple backends.

    Backends:
    - lama:      allenhooo/lama via Replicate (~3s, $0.00057) -- BEST for object removal
    - replicate: stability-ai/sd-inpainting via Replicate (~10s, $0.01)
    - local:     GTX 1650 diffusion (~15 min, FREE)

    Auto fallback chain: lama -> replicate -> local
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
        # Dùng settings vì pydantic_settings không tự populate os.environ
        self.replicate_available = bool(settings.REPLICATE_API_TOKEN)
        # LaMa cũng cần Replicate token
        self.lama_available = self.replicate_available

        logger.info("🔧 Inpainting Service initialized")
        logger.info(f"   Default method: {default_method}")
        logger.info(f"   LaMa (Replicate): {'✅ Available' if self.lama_available else '❌ Token not set'}")
        logger.info(f"   SD Replicate:     {'✅ Available' if self.replicate_available else '❌ Token not set'}")
        logger.info(f"   Local GPU:        ✅ Available")

        # Lazy load backends
        self._local_service = None
        self._replicate_service = None
        self._lama_service = None
    
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
            # Auto-select: lama > replicate > local
            if self.lama_available:
                method = "lama"
                logger.info("🤖 Auto-selected: LaMa (fast + cheap object removal)")
            elif self.replicate_available:
                method = "replicate"
                logger.info("🤖 Auto-selected: SD Replicate")
            else:
                method = "local"
                logger.info("🤖 Auto-selected: Local GPU")

        # Execute with fallback chain
        try:
            if method == "lama":
                return self._inpaint_lama(image, mask, **kwargs)
            elif method == "replicate":
                return self._inpaint_replicate(image, mask, prompt, negative_prompt, **kwargs)
            elif method == "local":
                return self._inpaint_local(image, mask, prompt, negative_prompt, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")

        except Exception as e:
            logger.error(f"❌ Inpainting failed with {method}: {e}")

            # Fallback chain
            if method == "lama" and self.replicate_available:
                logger.warning("⚠️  LaMa failed → falling back to SD Replicate...")
                try:
                    result, meta = self._inpaint_replicate(image, mask, prompt, negative_prompt, **kwargs)
                    meta["fallback_used"] = "replicate_sd"
                    return result, meta
                except Exception as e2:
                    logger.error(f"❌ SD Replicate also failed: {e2}")

            if method in ("lama", "replicate"):
                logger.warning("⚠️  Cloud failed → falling back to Local GPU...")
                result, meta = self._inpaint_local(image, mask, prompt, negative_prompt, **kwargs)
                meta["fallback_used"] = "local_gpu"
                return result, meta

            raise
    
    def _inpaint_lama(
        self,
        image: Image.Image,
        mask: Image.Image,
        **kwargs,
    ) -> Tuple[Image.Image, dict]:
        """Remove object using LaMa — fills with surrounding texture."""
        logger.info("🦙 Using LaMa inpainting (fast + cheap, ~3s)")

        if not self.lama_available:
            raise ValueError(
                "Replicate API token not set. "
                "LaMa requires REPLICATE_API_TOKEN in .env"
            )

        if self._lama_service is None:
            from app.core.lama_inpainting import get_lama_inpainting_service
            self._lama_service = get_lama_inpainting_service()

        mask_padding = kwargs.get("mask_padding", None)
        result, metadata = self._lama_service.remove_object(
            image=image,
            mask=mask,
            mask_padding=mask_padding,
        )

        metadata["method"] = "lama"
        metadata["fallback_used"] = None
        return result, metadata

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
        # Primary config is REMOVE_OBJECT_METHOD, keep INPAINTING_METHOD as legacy alias.
        env_method = (
            os.getenv("REMOVE_OBJECT_METHOD")
            or getattr(settings, "REMOVE_OBJECT_METHOD", None)
            or os.getenv("INPAINTING_METHOD")
            or getattr(settings, "INPAINTING_METHOD", None)
            or "auto"
        )
        method = default_method or env_method
        
        _inpainting_service_instance = InpaintingService(default_method=method)
    
    return _inpainting_service_instance
