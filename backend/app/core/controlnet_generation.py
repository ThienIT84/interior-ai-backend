"""
ControlNet Generation Module (Task 3.2.1)
==========================================
Tao thiet ke noi that moi dua tren cau truc phong hien tai su dung ControlNet.

Architecture:
    - Backend: Replicate API (jagilley/controlnet-canny)
    - Control signal: Anh goc (Replicate tu tinh Canny edge noi bo)
    - Edge preview: Dung rieng qua /preview-edges endpoint (edge_detection.py)

Flow:
    1. Client goi POST /generate-design  -> tra ve job_id
    2. ControlNetGeneration.generate_async() chay trong background thread
    3. Client poll GET /generation/job-status/{job_id} de lay ket qua

Usage:
    from app.core.controlnet_generation import get_controlnet_service, GenerationJob

    svc = get_controlnet_service()
    job = svc.submit_job(image, style="modern")
    # ... poll job.status ...
"""

import io
import os
import time
import uuid
import base64
import threading
import logging
from typing import Optional, Tuple
from PIL import Image
import numpy as np

from app.config import settings
from app.core.prompts import get_style_prompts, AVAILABLE_STYLES
from app.services.job_service import get_job_service, JobStatus

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------
# Model configurations – add new models here
# -------------------------------------------------------------------------------------

MODEL_CONFIGS = {
    "controlnet": {
        "replicate_id": (
            "jagilley/controlnet-canny:"
            "aff48af9c68d162388d230a2ab003f68d2638d88307bdaf1c2f1ac95079c9613"
        ),
        "display_name": "Standard (SD 1.5)",
        "default_steps": 20,
        "default_guidance": 9.0,
        "resolution": 512,
    },
    "flux-pro": {
        "replicate_id": "black-forest-labs/flux-canny-pro",
        "display_name": "Professional (Flux)",
        "default_steps": 28,
        "default_guidance": 30,
        "resolution": 1024,
    },
}

DEFAULT_MODEL_ID = "controlnet"


# -------------------------------------------------------------------------------------
# Job model
# -------------------------------------------------------------------------------------


class GenerationJob:
    """Trang thai cua mot async generation job."""

    def __init__(self, job_id: str, style: str, status: str = "pending"):
        self.job_id: str = job_id
        self.style: str = style
        self.status: str = status
        self.result_url: Optional[str] = None
        self.result_id: Optional[str] = None
        self.error: Optional[str] = None
        self.processing_time: Optional[float] = None
        self.metadata: dict = {}
        self.created_at: float = time.time()

    @classmethod
    def from_redis(cls, data: dict) -> "GenerationJob":
        """Reconstruct GenerationJob from Redis data."""
        payload = data.get("payload", {})
        job = cls(
            job_id=data.get("job_id"),
            style=payload.get("style", "unknown"),
            status=data.get("status", "pending")
        )
        job.result_url = data.get("result_url")
        job.result_id = data.get("metadata", {}).get("result_id")
        job.error = data.get("error")
        job.metadata = data.get("metadata", {})
        
        # Parse processing time if available
        if "processing_time" in job.metadata:
            job.processing_time = job.metadata["processing_time"]
            
        return job


# -------------------------------------------------------------------------------------
# ControlNet Generation Service (Task 3.2.1 + 3.2.2)
# -------------------------------------------------------------------------------------

class ControlNetGeneration:
    """
    Multi-model Replicate wrapper de tao thiet ke noi that.

    Supported models:
      - controlnet  : jagilley/controlnet-canny (SD 1.5) – fast, cheap
      - flux-pro    : black-forest-labs/flux-canny-pro  – high quality

    Note: Edge preview rieng qua /preview-edges endpoint (edge_detection.py).
    """

    # -------------------------------------------------------------------------------------

    def __init__(self, api_token: Optional[str] = None):
        """
        Args:
            api_token: Replicate API token. Mac dinh doc tu env/settings.
        """
        import replicate  # lazy import - khong can khi khong dung Replicate

        self.api_token = (
            api_token
            or os.getenv("REPLICATE_API_TOKEN")
            or settings.REPLICATE_API_TOKEN
        )

        if not self.api_token:
            raise ValueError(
                "Replicate API token is required. "
                "Set REPLICATE_API_TOKEN in .env or pass api_token parameter. "
                "Get token at: https://replicate.com/account/api-tokens"
            )

        os.environ["REPLICATE_API_TOKEN"] = self.api_token

        # Redis-based job record (Task 3.5.1 fix)
        self.job_service = get_job_service()

        logger.info("✅ ControlNet Generation service initialized")
        for mid, cfg in MODEL_CONFIGS.items():
            logger.info(f"   [{mid}] {cfg['display_name']} → {cfg['replicate_id'].split(':')[0]}")

    # -------------------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------------------

    def submit_job(
        self,
        image: Image.Image,
        style: str,
        model_id: str = DEFAULT_MODEL_ID,
        guidance_scale: Optional[float] = None,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> GenerationJob:
        """
        Dang ky mot async generation job va chay trong background thread.

        Args:
            image: Anh phong goc (RGB PIL Image)
            style: Ten style ("modern", "minimalist", "industrial", ...)
            model_id: Model identifier ("controlnet" or "flux-pro")
            guidance_scale: Guidance scale (auto-adjusted per model if None)
            steps: So buoc inference (auto-adjusted per model if None)
            seed: Random seed de co the reproduce ket qua

        Returns:
            GenerationJob voi status="pending"

        Raises:
            ValueError: Neu style hoac model_id khong hop le
        """
        # Validate style
        if style.lower() not in AVAILABLE_STYLES:
            raise ValueError(
                f"Style '{style}' khong hop le. "
                f"Cac style ho tro: {AVAILABLE_STYLES}"
            )

        # Validate model_id
        if model_id not in MODEL_CONFIGS:
            raise ValueError(
                f"Model '{model_id}' khong hop le. "
                f"Cac model ho tro: {list(MODEL_CONFIGS.keys())}"
            )

        # Create job in Redis via JobService
        job_id = self.job_service.create_job(
            job_type="generation",
            payload={
                "style": style,
                "model_id": model_id,
                "guidance_scale": guidance_scale,
                "steps": steps,
                "seed": seed,
            }
        )

        # Legacy object mapping for the background thread
        job = GenerationJob(job_id=job_id, style=style)

        # Chay trong daemon thread de khong block request
        t = threading.Thread(
            target=self._run_generation,
            args=(job, image, style, model_id, guidance_scale, steps, seed),
            daemon=True,
        )
        t.start()

        model_display = MODEL_CONFIGS[model_id]["display_name"]
        logger.info(f"📋 Generation job submitted | job_id={job_id} | style={style} | model={model_display}")
        return job

    def get_job(self, job_id: str) -> Optional[GenerationJob]:
        """Tra ve GenerationJob tu Redis theo job_id, hoac None neu khong tim thay."""
        job_data = self.job_service.get_job(job_id)
        if not job_data:
            return None
        return GenerationJob.from_redis(job_data)

    def generate_with_controlnet(
        self,
        image: Image.Image,
        edges: Optional[np.ndarray],
        style: str,
        model_id: str = DEFAULT_MODEL_ID,
        guidance_scale: Optional[float] = None,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[Image.Image, dict]:
        """
        Tao thiet ke noi that (SYNCHRONOUS) dung model duoc chon.

        NOTE: Su dung submit_job() + poll cho async (khuyen nghi cho endpoint).
              Ham nay dung cho testing va noi bo.

        Args:
            image: Anh phong goc (RGB PIL Image)
            edges: Pre-computed Canny edge map (unused – kept for API compat)
            style: Ten style
            model_id: "controlnet" hoac "flux-pro"
            guidance_scale: Guidance scale (auto per model if None)
            steps: So buoc inference (auto per model if None)
            seed: Random seed

        Returns:
            Tuple (result_image: PIL.Image, metadata: dict)

        Raises:
            ValueError: Neu style khong hop le
            RuntimeError: Neu Replicate API that bai
        """
        cfg = MODEL_CONFIGS[model_id]
        positive_prompt, negative_prompt = get_style_prompts(style)

        # Apply model-specific defaults
        guidance_scale = guidance_scale or cfg["default_guidance"]
        steps = steps or cfg["default_steps"]

        # Prepare image data
        image_uri = self._image_to_data_uri(image)

        # Build model-specific input
        if model_id == "flux-pro":
            input_data = self._build_flux_input(
                image_uri, positive_prompt, steps, guidance_scale, seed,
            )
        else:
            input_data = self._build_controlnet_input(
                image_uri, positive_prompt, negative_prompt,
                steps, guidance_scale, cfg["resolution"], seed,
            )

        model_display = cfg["display_name"]
        replicate_id = cfg["replicate_id"]
        logger.info(
            f"🎨 Generation | model={model_display} | style={style} "
            f"| steps={steps} | guidance={guidance_scale}"
        )
        logger.info(f"   Prompt: {positive_prompt[:80]}...")

        start_time = time.time()

        try:
            import replicate

            output = replicate.run(replicate_id, input=input_data)

            # Parse output – ControlNet returns list [edge, result]; Flux returns single URL
            if isinstance(output, list) and len(output) > 0:
                result_url = str(output[-1])
            else:
                result_url = str(output)

            logger.info(f"   Downloading result from: {result_url}")

            result_image, result_id = self._download_and_save(result_url, style)

            processing_time = time.time() - start_time
            logger.info(
                f"✅ Generation done | model={model_display} | style={style} "
                f"| time={processing_time:.1f}s | id={result_id}"
            )

            metadata = {
                "style": style,
                "model_id": model_id,
                "model_display": model_display,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "processing_time": processing_time,
                "model": replicate_id.split(":")[0],
                "resolution": cfg["resolution"],
                "result_id": result_id,
            }

            return result_image, metadata

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(
                f"❌ Generation failed | model={model_display} | style={style} "
                f"| time={processing_time:.1f}s | error={e}",
                exc_info=True,
            )
            raise RuntimeError(f"Generation that bai ({model_display}): {str(e)}") from e

    # -------------------------------------------------------------------------------------
    # Model-specific input builders
    # -------------------------------------------------------------------------------------

    @staticmethod
    def _build_controlnet_input(
        image_uri: str,
        positive_prompt: str,
        negative_prompt: str,
        steps: int,
        guidance_scale: float,
        resolution: int,
        seed: Optional[int],
    ) -> dict:
        """Build input dict for jagilley/controlnet-canny."""
        data = {
            "image": image_uri,
            "prompt": positive_prompt,
            "a_prompt": "best quality, extremely detailed",
            "n_prompt": negative_prompt,
            "num_samples": "1",
            "image_resolution": str(resolution),
            "detect_resolution": resolution,
            "ddim_steps": steps,
            "scale": guidance_scale,
            "eta": 0.0,
        }
        if seed is not None:
            data["seed"] = seed
        return data

    @staticmethod
    def _build_flux_input(
        image_uri: str,
        positive_prompt: str,
        steps: int,
        guidance: float,
        seed: Optional[int],
    ) -> dict:
        """Build input dict for black-forest-labs/flux-canny-pro."""
        data = {
            "control_image": image_uri,
            "prompt": positive_prompt,
            "steps": steps,
            "guidance": guidance,
            "output_format": "png",
            "safety_tolerance": 5,
        }
        if seed is not None:
            data["seed"] = seed
        return data

    # -------------------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------------------

    def _run_generation(
        self,
        job: GenerationJob,
        image: Image.Image,
        style: str,
        model_id: str,
        guidance_scale: Optional[float],
        steps: Optional[int],
        seed: Optional[int],
    ) -> None:
        """Background thread: thuc hien generation va cap nhat job status trong Redis."""
        model_display = MODEL_CONFIGS[model_id]["display_name"]
        start_time = time.time()
        
        self.job_service.update_job(
            job.job_id, 
            status=JobStatus.PROCESSING.value, 
            progress=0.1,
            metadata={"status_message": f"Đang chuẩn bị ({model_display})..."}
        )
        logger.info(f"⏳ [job={job.job_id}] Preparation | style={style} | model={model_display}")

        try:
            self.job_service.update_job(
                job.job_id,
                progress=0.2,
                metadata={"status_message": f"Đang gửi tới {model_display}..."}
            )

            # Step 3: Run generation (Blocking call)
            self.job_service.update_job(
                job.job_id,
                progress=0.3,
                metadata={"status_message": f"AI đang vẽ thiết kế ({model_display})..."}
            )
            
            result_image, metadata = self.generate_with_controlnet(
                image=image,
                edges=None,
                style=style,
                model_id=model_id,
                guidance_scale=guidance_scale,
                steps=steps,
                seed=seed,
            )

            # Step 4: Finalizing
            self.job_service.update_job(
                job.job_id,
                progress=0.9,
                metadata={"status_message": "Đang hoàn thiện và lưu kết quả..."}
            )

            # Lay result_id tu output da save
            result_id = metadata.get("result_id", str(uuid.uuid4()).replace("-", "")[:16])
            result_url = f"/api/v1/generation/result/{result_id}"

            self.job_service.update_job(
                job.job_id,
                status=JobStatus.COMPLETED.value,
                progress=1.0,
                result_url=result_url,
                metadata={
                    "result_id": result_id, 
                    "status_message": "Hoàn tất!",
                    **metadata
                }
            )

            logger.info(f"✅ [job={job.job_id}] Completed | result_id={result_id}")

        except Exception as e:
            self.job_service.update_job(
                job.job_id,
                status=JobStatus.FAILED.value,
                error=str(e),
                metadata={"status_message": f"Lỗi: {str(e)}"}
            )
            logger.error(f"❌ [job={job.job_id}] Failed: {e}")

    def _image_to_data_uri(self, image: Image.Image) -> str:
        """Chuyen PIL Image thanh data URI de gui len Replicate API."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        image_b64 = base64.b64encode(buffer.read()).decode("utf-8")
        return f"data:image/png;base64,{image_b64}"

    def _download_and_save(self, url: str, style: str) -> Tuple[Image.Image, str]:
        """
        Download result tu Replicate URL, save vao outputs/, tra ve (image, result_id).
        """
        import urllib.request

        result_id = str(uuid.uuid4()).replace("-", "")[:16]
        filename = f"design_{style}_{result_id}.png"
        output_path = settings.OUTPUTS_DIR / filename

        try:
            urllib.request.urlretrieve(url, output_path)
            result_image = Image.open(output_path).convert("RGB")
            logger.info(f"   💾 Saved: {filename}")
        except Exception as e:
            # Fallback: download qua urllib neu co loi
            logger.warning(f"   urlretrieve failed, trying fallback: {e}")
            import urllib.request
            with urllib.request.urlopen(url) as resp:
                data = resp.read()
            result_image = Image.open(io.BytesIO(data)).convert("RGB")
            result_image.save(output_path)

        return result_image, result_id


# -------------------------------------------------------------------------------------
# Singleton factory
# -------------------------------------------------------------------------------------

_controlnet_service: Optional[ControlNetGeneration] = None


def get_controlnet_service() -> ControlNetGeneration:
    """
    Tra ve singleton ControlNetGeneration instance.
    Khoi tao lan dau khi duoc goi.

    Returns:
        ControlNetGeneration instance

    Raises:
        RuntimeError: Neu REPLICATE_API_TOKEN chua duoc set
    """
    global _controlnet_service
    if _controlnet_service is None:
        try:
            _controlnet_service = ControlNetGeneration()
        except ValueError as e:
            raise RuntimeError(str(e)) from e
    return _controlnet_service
