"""
Replicate API Client for Stable Diffusion Inpainting
Fast cloud-based inpainting (~10 seconds, $0.01/image)
More reliable than HuggingFace free tier
"""
import replicate
import io
import time
import logging
from PIL import Image, ImageFilter
from typing import Optional, Tuple
import os
import base64

from app.config import settings

logger = logging.getLogger(__name__)


class ReplicateInpainting:
    """Replicate API wrapper for fast cloud inpainting"""
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize Replicate API client
        
        Args:
            api_token: Replicate API token (or set REPLICATE_API_TOKEN env var)
        """
        # Dùng settings để lấy token vì pydantic_settings không populate os.environ
        self.api_token = api_token or os.getenv("REPLICATE_API_TOKEN") or settings.REPLICATE_API_TOKEN
        
        if not self.api_token:
            raise ValueError(
                "Replicate API token required! "
                "Set REPLICATE_API_TOKEN environment variable or pass api_token parameter. "
                "Get token at: https://replicate.com/account/api-tokens"
            )
        
        # Set token for replicate client
        os.environ["REPLICATE_API_TOKEN"] = self.api_token
        
        # Model: Stability AI SD Inpainting
        self.model_version = "stability-ai/stable-diffusion-inpainting:95b7223104132402a9ae91cc677285bc5eb997834bd2349fa486f53910fd68b3"
        
        # Optimized parameters
        self.default_params = {
            "num_inference_steps": 50,
            "guidance_scale": 8.5,
            "prompt_strength": 1.0,
        }

        # Max dimension gửi lên Replicate - SD Inpainting tối ưu ở 512/768
        self.max_inpaint_size = 768

        # Mask dilation mặc định (px) - mở rộng mask để phủ viền đối tượng
        self.default_mask_padding = 15
    
        # Prompt tập trung vào sự liền mạch, bề mặt và bối cảnh chung (Không gán cứng Tường/Sàn)
        self.default_prompt = (
            "empty space, seamless background texture, continuous surface, "
            "perfect blend with surrounding environment, natural lighting, "
            "photorealistic interior, 8k resolution, architectural photography"
        )
        
        # Negative Prompt bổ sung các lỗi "kinh điển" của Inpainting (bóng đổ thừa, vết cắt)
        self.default_negative_prompt = (
            "leftover shadows, floating objects, ghosting, mismatched patterns, "
            "harsh edges, cut seams, furniture, people, pets, text, watermark, "
            "blurry, deformed, cartoon, illustration, low quality"
        )
        
        logger.info("✅ Replicate API client initialized")
    
    def _image_to_data_uri(self, image: Image.Image) -> str:
        """Convert PIL Image to data URI for Replicate API"""
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Save to bytes buffer
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        # Create data URI
        data_uri = f"data:image/png;base64,{image_base64}"
        
        return data_uri
    
    def _dilate_mask(self, mask: Image.Image, padding: int) -> Image.Image:
        """
        Mở rộng mask ra ngoài `padding` pixels để phủ viền đối tượng.
        Tránh artifact đường viền còn sót sau inpainting.
        """
        if padding <= 0:
            return mask
        # MaxFilter size phải là số lẻ, dilation = size // 2
        filter_size = padding * 2 + 1
        return mask.filter(ImageFilter.MaxFilter(size=filter_size))

    def _resize_for_inpainting(
        self, image: Image.Image, mask: Image.Image, max_size: int
    ) -> Tuple[Image.Image, Image.Image, Tuple[int, int]]:
        """
        Resize image và mask về max_size (giữ tỉ lệ, chia hết cho 8).
        Trả về (image_resized, mask_resized, original_size).
        SD Inpainting tối ưu ở 512px hoặc 768px.
        """
        original_size = image.size  # (W, H)
        w, h = original_size

        if max(w, h) <= max_size:
            return image, mask, original_size

        # Scale xuống giữ tỉ lệ
        scale = max_size / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Làm tròn về bội số của 8 (yêu cầu của SD)
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8

        logger.info(f"  Resize: {w}x{h} → {new_w}x{new_h} (max={max_size})")

        image_resized = image.resize((new_w, new_h), Image.LANCZOS)
        mask_resized = mask.resize((new_w, new_h), Image.NEAREST)

        return image_resized, mask_resized, original_size

    def inpaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        prompt_strength: Optional[float] = None,
        seed: Optional[int] = None,
        mask_padding: Optional[int] = None,
    ) -> Tuple[Image.Image, dict]:
        """
        Perform inpainting using Replicate API

        Args:
            image: Original RGB image
            mask: Grayscale mask (white=object to remove)
            prompt: Custom prompt (default: empty room prompt)
            negative_prompt: Custom negative prompt
            num_inference_steps: Number of steps (default: 50)
            guidance_scale: Guidance scale (default: 8.5)
            prompt_strength: Prompt strength (default: 1.0)
            seed: Random seed
            mask_padding: Pixels để mở rộng mask (default: 15)

        Returns:
            Tuple of (result_image, metadata_dict)
        """
        # Use defaults if not specified
        prompt = prompt or self.default_prompt
        negative_prompt = negative_prompt or self.default_negative_prompt
        num_inference_steps = num_inference_steps or self.default_params["num_inference_steps"]
        guidance_scale = guidance_scale or self.default_params["guidance_scale"]
        prompt_strength = prompt_strength or self.default_params["prompt_strength"]
        if mask_padding is None:
            mask_padding = self.default_mask_padding

        logger.info("🚀 Starting Replicate API inpainting...")
        logger.info(f"  Prompt: {prompt[:50]}...")
        logger.info(f"  Steps: {num_inference_steps}, Guidance: {guidance_scale}")
        logger.info(f"  Mask padding: {mask_padding}px, Max size: {self.max_inpaint_size}px")

        start_time = time.time()

        try:
            # --- Fix #1: Dilate mask để phủ viền đối tượng ---
            mask = mask.convert("L")
            mask_dilated = self._dilate_mask(mask, mask_padding)

            # --- Fix #2: Resize về max 768px để SD inpaint tối ưu ---
            image_proc, mask_proc, original_size = self._resize_for_inpainting(
                image, mask_dilated, self.max_inpaint_size
            )

            # Convert images to data URIs
            image_uri = self._image_to_data_uri(image_proc)
            mask_uri = self._image_to_data_uri(mask_proc)
            
            # Prepare input
            input_data = {
                "image": image_uri,
                "mask": mask_uri,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "prompt_strength": prompt_strength,
            }
            if seed is not None:
                input_data["seed"] = seed
            
            logger.info("  Sending request to Replicate...")
            
            # Run prediction
            output = replicate.run(
                self.model_version,
                input=input_data
            )
            
            # Output is a list of URLs
            if isinstance(output, list) and len(output) > 0:
                result_url = output[0]
            else:
                result_url = output
            
            logger.info(f"  Downloading result from: {result_url}")
            
            # Download result image
            import requests
            response = requests.get(result_url, timeout=30)
            response.raise_for_status()
            
            result_image = Image.open(io.BytesIO(response.content))

            # --- Resize kết quả về đúng size ảnh gốc ---
            if result_image.size != original_size:
                logger.info(f"  Resize result: {result_image.size} → {original_size}")
                result_image = result_image.resize(original_size, Image.LANCZOS)

            elapsed = time.time() - start_time

            # Collect metadata
            metadata = {
                "processing_time": elapsed,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "prompt_strength": prompt_strength,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "input_size": image.size,
                "output_size": result_image.size,
                "processed_size": image_proc.size,
                "mask_padding": mask_padding,
                "method": "replicate_api",
                "model": "stability-ai/stable-diffusion-inpainting",
                "cost_usd": 0.01,
            }
            
            logger.info(f"✅ Inpainting completed in {elapsed:.1f}s")
            logger.info(f"   Cost: ~$0.01")
            
            return result_image, metadata
        
        except Exception as e:
            logger.error(f"❌ Replicate API inpainting failed: {e}")
            raise


# Global instance (singleton pattern)
_replicate_inpainting_instance: Optional[ReplicateInpainting] = None


def get_replicate_inpainting_service(api_token: Optional[str] = None) -> ReplicateInpainting:
    """Get or create global Replicate inpainting service instance"""
    global _replicate_inpainting_instance
    
    if _replicate_inpainting_instance is None:
        _replicate_inpainting_instance = ReplicateInpainting(api_token=api_token)
    
    return _replicate_inpainting_instance
