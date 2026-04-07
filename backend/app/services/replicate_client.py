"""
Replicate API client for Stable Diffusion Inpainting
"""
import os
import replicate
from pathlib import Path
from typing import Optional, Dict, Any
import requests
from PIL import Image
import io

from app.config import settings


class ReplicateClient:
    """Client for Replicate API - Stable Diffusion Inpainting"""
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize Replicate client
        
        Args:
            api_token: Replicate API token (defaults to settings.REPLICATE_API_TOKEN)
        """
        self.api_token = api_token or settings.REPLICATE_API_TOKEN
        
        if not self.api_token:
            raise ValueError(
                "Replicate API token not found. "
                "Set REPLICATE_API_TOKEN in .env file"
            )
        
        # Set token for replicate library
        os.environ["REPLICATE_API_TOKEN"] = self.api_token
    
    def inpaint(
        self,
        image_path: Path,
        mask_path: Path,
        prompt: str = "empty room, clean floor, white walls, natural lighting",
        negative_prompt: str = "furniture, objects, clutter, people, low quality",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
    ) -> Dict[str, Any]:
        """
        Inpaint image using Stable Diffusion
        
        Args:
            image_path: Path to input image
            mask_path: Path to mask image (white = inpaint, black = keep)
            prompt: Text prompt for inpainting
            negative_prompt: Negative prompt (what to avoid)
            num_inference_steps: Number of denoising steps (higher = better quality, slower)
            guidance_scale: How closely to follow the prompt (7-15 recommended)
        
        Returns:
            Dict with:
                - output_url: URL of inpainted image
                - processing_time: Time taken in seconds
                - model: Model used
        """
        try:
            # Open and validate images
            image = Image.open(image_path)
            mask = Image.open(mask_path)
            
            # Ensure mask is binary (black and white)
            mask = mask.convert("L")
            
            # Convert to file-like objects
            image_bytes = io.BytesIO()
            mask_bytes = io.BytesIO()
            
            image.save(image_bytes, format="PNG")
            mask.save(mask_bytes, format="PNG")
            
            image_bytes.seek(0)
            mask_bytes.seek(0)
            
            # Call Replicate API
            # Using stability-ai/stable-diffusion-inpainting
            output = replicate.run(
                "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
                input={
                    "image": image_bytes,
                    "mask": mask_bytes,
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                }
            )
            
            # Output is a list of URLs
            if isinstance(output, list) and len(output) > 0:
                output_url = output[0]
            else:
                output_url = str(output)
            
            return {
                "success": True,
                "output_url": output_url,
                "model": "stability-ai/stable-diffusion-inpainting",
                "prompt": prompt,
                "negative_prompt": negative_prompt,
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": "stability-ai/stable-diffusion-inpainting",
            }
    
    def download_result(self, url: str, output_path: Path) -> bool:
        """
        Download inpainted image from URL
        
        Args:
            url: URL of inpainted image
            output_path: Where to save the image
        
        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save image
            with open(output_path, "wb") as f:
                f.write(response.content)
            
            return True
            
        except Exception as e:
            print(f"Error downloading result: {e}")
            return False


# Global client instance
_replicate_client: Optional[ReplicateClient] = None


def get_replicate_client() -> ReplicateClient:
    """Get or create global Replicate client instance"""
    global _replicate_client
    
    if _replicate_client is None:
        _replicate_client = ReplicateClient()
    
    return _replicate_client
