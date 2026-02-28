"""
Stable Diffusion Inpainting Module
Handles object removal and room inpainting using local SD model
"""
import torch
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import time
import logging

logger = logging.getLogger(__name__)


class DiffusionInpainting:
    """Stable Diffusion Inpainting wrapper for local GPU inference"""
    
    def __init__(self):
        self.pipe = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_loaded = False
        
        # Optimized parameters for OBJECT REMOVAL
        self.default_params = {
            "steps": 50,
            "guidance": 8.5,   # Higher guidance for stronger prompt adherence
            "strength": 1.0,   # Maximum - starts from pure noise, fully replaces object
            "target_size": 512,
        }
        
        # Prompts optimized for empty room generation
        self.default_prompt = (
            "clean wall texture, matching floor pattern, seamless blend, "
            "natural lighting, photorealistic interior, high quality"
        )
        
        self.default_negative_prompt = (
            "furniture, objects, decorations, people, text, watermark, "
            "blur, distortion, artifacts, unrealistic, low quality"
        )
    
    def load_model(self):
        """Load Stable Diffusion Inpainting model (float32 for GTX 1650)"""
        if self.model_loaded:
            logger.info("Model already loaded")
            return
        
        logger.info("Loading Stable Diffusion Inpainting model...")
        start_time = time.time()
        
        try:
            from diffusers import StableDiffusionInpaintPipeline
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            model_id = "runwayml/stable-diffusion-inpainting"
            
            # Use float32 (NOT fp16) for GTX 1650 compatibility
            self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                safety_checker=None,  # Disable for speed
            )
            
            self.pipe = self.pipe.to(self.device)
            
            # Enable memory optimizations
            self.pipe.enable_attention_slicing()
            self.pipe.enable_vae_slicing()
            
            self.model_loaded = True
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Model loaded in {elapsed:.1f}s")
            logger.info(f"   Device: {self.device}")
            logger.info(f"   Dtype: float32")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def prepare_images(
        self,
        image: Image.Image,
        mask: Image.Image,
        target_size: Optional[int] = None
    ) -> Tuple[Image.Image, Image.Image]:
        """
        Prepare image and mask for inpainting
        
        Args:
            image: Original RGB image
            mask: Grayscale mask (SAM output: white=object to remove)
            target_size: Target size for processing (default: 512)
        
        Returns:
            Tuple of (resized_image, processed_mask)
        """
        if target_size is None:
            target_size = self.default_params["target_size"]
        
        # Resize image
        image_resized = image.resize(
            (target_size, target_size),
            Image.Resampling.LANCZOS
        )
        
        # Convert mask to numpy for processing
        mask_np = np.array(mask.convert("L"))
        
        # Step 1: Binarize mask (remove any anti-aliasing from SAM)
        mask_np = (mask_np > 127).astype(np.uint8) * 255
        
        # Step 2: Dilate mask to expand beyond object edges
        # This ensures the inpainting covers slightly beyond the object boundary
        from PIL import ImageFilter
        mask_pil = Image.fromarray(mask_np)
        # Dilate by applying MaxFilter multiple times (expands white area)
        for _ in range(3):  # ~15px dilation
            mask_pil = mask_pil.filter(ImageFilter.MaxFilter(size=5))
        
        # Step 3: Resize with NEAREST (NOT LANCZOS!) to keep sharp binary edges
        # LANCZOS creates soft edges (values like 128, 200) which let the
        # original object bleed through the semi-transparent mask boundary
        mask_resized = mask_pil.resize(
            (target_size, target_size),
            Image.Resampling.NEAREST
        )
        
        # Step 4: Final binarization after resize
        mask_final = np.array(mask_resized)
        mask_final = (mask_final > 127).astype(np.uint8) * 255
        mask_resized = Image.fromarray(mask_final)
        
        logger.info(f"  Mask stats: min={mask_final.min()}, max={mask_final.max()}, "
                     f"white_pct={mask_final.sum()/255/mask_final.size*100:.1f}%")
        
        return image_resized, mask_resized
    
    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        strength: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Image.Image, dict]:
        """
        Perform inpainting to remove objects and generate empty room
        
        Args:
            image: Original RGB image
            mask: Grayscale mask from SAM (white=object to remove)
            prompt: Custom prompt (default: empty room prompt)
            negative_prompt: Custom negative prompt
            steps: Number of inference steps (default: 50)
            guidance_scale: Guidance scale (default: 11.0)
            strength: Inpainting strength (default: 0.99)
            seed: Random seed for reproducibility
        
        Returns:
            Tuple of (result_image, metadata_dict)
        """
        if not self.model_loaded:
            self.load_model()
        
        # Use defaults if not specified
        prompt = prompt or self.default_prompt
        negative_prompt = negative_prompt or self.default_negative_prompt
        steps = steps or self.default_params["steps"]
        guidance_scale = guidance_scale or self.default_params["guidance"]
        strength = strength or self.default_params["strength"]
        
        logger.info("Starting inpainting...")
        logger.info(f"  Steps: {steps}, Guidance: {guidance_scale}, Strength: {strength}")
        
        start_time = time.time()
        
        try:
            # Prepare images
            image_prep, mask_prep = self.prepare_images(image, mask)
            
            # Setup generator for reproducibility
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Run inpainting
            with torch.no_grad():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=image_prep,
                    mask_image=mask_prep,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    generator=generator,
                ).images[0]
            
            # Handle NaN values (safety check)
            result_np = np.array(result)
            result_np = np.nan_to_num(result_np, nan=128.0, posinf=255.0, neginf=0.0)
            result = Image.fromarray(result_np.astype(np.uint8))
            
            elapsed = time.time() - start_time
            
            # Collect metadata
            metadata = {
                "processing_time": elapsed,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "strength": strength,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "input_size": image.size,
                "output_size": result.size,
                "device": self.device,
            }
            
            logger.info(f"✅ Inpainting completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
            
            return result, metadata
            
        except Exception as e:
            logger.error(f"Inpainting failed: {e}")
            raise
    
    def unload_model(self):
        """Unload model to free VRAM"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            self.model_loaded = False
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Model unloaded, VRAM freed")


# Global instance (singleton pattern)
_inpainting_instance: Optional[DiffusionInpainting] = None


def get_inpainting_service() -> DiffusionInpainting:
    """Get or create global inpainting service instance"""
    global _inpainting_instance
    
    if _inpainting_instance is None:
        _inpainting_instance = DiffusionInpainting()
    
    return _inpainting_instance
