"""
SAM3 segmentation via Replicate API.
"""
import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import replicate
import requests
from PIL import Image

from app.config import settings
from app.utils.logger import logger


class Sam3ReplicateSegmentation:
    """Wrapper for SAM3 cloud segmentation using Replicate."""

    def __init__(self, api_token: Optional[str] = None, model_version: Optional[str] = None):
        self.api_token = api_token or os.getenv("REPLICATE_API_TOKEN") or settings.REPLICATE_API_TOKEN
        if not self.api_token:
            raise ValueError(
                "Replicate API token is required for SAM3 backend. "
                "Set REPLICATE_API_TOKEN in .env"
            )

        os.environ["REPLICATE_API_TOKEN"] = self.api_token
        self.model_version = model_version or settings.SAM3_REPLICATE_MODEL

    def segment_by_points(
        self,
        image_path: Path,
        point_coords: List[Tuple[int, int]],
        point_labels: List[int],
        text_prompt: str = "",
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """Run SAM3 segmentation with text prompt and return binary mask.

        [CẢNH BÁO]: mattsays/sam3-image KHÔNG hỗ trợ tọa độ điểm (point_coords).
        Nó là model Grounded-SAM chỉ dùng text prompt.
        """
        # Xác định Prompt cuối cùng
        # Nếu user không nhập gì, đành dùng "object" để nó tự tìm vật thể chung chung
        final_prompt = text_prompt.strip() if text_prompt and text_prompt.strip() else "object"

        logger.info(
            f"☁️ SAM3 Replicate segmentation: {image_path.name} | "
            f"Model mattsays/sam3 ignores points. Using prompt: '{final_prompt}'"
        )

        # 1. Khởi tạo payload đúng chuẩn schema của mattsays/sam3-image
        replicate_input = {
            "prompt": final_prompt,  # <--- Đã sửa key thành "prompt"
            "mask_only": True,
            "return_zip": False,
            "save_overlay": False,
        }

        with open(image_path, "rb") as image_file:
            replicate_input["image"] = image_file
            
            # 2. Gửi request
            output = replicate.run(
                self.model_version,
                input=replicate_input,
            )

        mask_url = self._extract_mask_url(output)
        if not mask_url:
            raise RuntimeError(
                f"SAM3 Replicate response does not contain a mask URL. output={str(output)[:300]}"
            )

        mask = self._download_mask(mask_url)
        score = self._extract_score(output)
        mask_coverage = float(mask.sum() / mask.size * 100)

        logger.info(
            f"☁️ SAM3 mask received: coverage={mask_coverage:.2f}% | url={mask_url}"
        )

        metadata = {
            "backend": "sam3_replicate",
            "model": self.model_version,
            "prompt_used": final_prompt,
            "mask_source_url": mask_url,
            "mask_coverage_percentage": round(mask_coverage, 4),
            "raw_output_preview": str(output)[:500],
        }
        return mask, score, metadata

    def _download_mask(self, mask_url: str) -> np.ndarray:
        try:
            response = requests.get(mask_url, timeout=60)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Lỗi khi tải mask từ Replicate ({mask_url}): {e}")
            raise RuntimeError(f"Không thể tải mask từ Replicate: {e}")

        mask_pil = Image.open(io.BytesIO(response.content)).convert("L")
        mask_np = np.array(mask_pil)
        return mask_np > 127

    def _extract_mask_url(self, output: Any) -> Optional[str]:
        if isinstance(output, str):
            return output if output.startswith("http") else None

        if isinstance(output, list):
            for item in output:
                url = self._extract_mask_url(item)
                if url:
                    return url
            return None

        if isinstance(output, dict):
            candidate_keys = ["mask", "mask_url", "output", "image", "result", "segmentation", "masks"]
            for key in candidate_keys:
                if key in output:
                    url = self._extract_mask_url(output[key])
                    if url:
                        return url

            for value in output.values():
                url = self._extract_mask_url(value)
                if url:
                    return url

        return None

    def _extract_score(self, output: Any) -> float:
        if isinstance(output, dict):
            for key in ["score", "confidence", "iou", "predicted_iou"]:
                value = output.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
        return 1.0


_sam3_instance: Optional[Sam3ReplicateSegmentation] = None

def get_sam3_replicate_service() -> Sam3ReplicateSegmentation:
    global _sam3_instance
    if _sam3_instance is None:
        _sam3_instance = Sam3ReplicateSegmentation()
    return _sam3_instance