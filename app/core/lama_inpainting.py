"""
LaMa (Large Mask Inpainting) Module — IOPaint local backend
============================================================
Ưu tiên chạy LOCAL bằng IOPaint (đã cài trong conda interior_ai).
Fallback sang Replicate cloud khi không có GPU / IOPaint không load được.

IOPaint API:
    from iopaint.model.lama import LaMa
    from iopaint.schema import InpaintRequest, HDStrategy
    model = LaMa(device="cuda")
    result_np = model(image_np_BGR, mask_np, config)

Replicate fallback:
    allenhooo/lama (pinned version)
    ~$0.00057/image, ~3 seconds
"""
import io
import time
import numpy as np
from PIL import Image
from typing import Optional, Tuple

from app.config import settings
from app.utils.logger import logger


class LamaInpainting:
    """
    LaMa wrapper — IOPaint local (ưu tiên) + Replicate fallback.

    Khởi tạo lazy: model chỉ load lần đầu khi gọi remove_object().
    """

    # Replicate pinned version (fallback)
    _REPLICATE_MODEL = "allenhooo/lama:cdac78a1bec5b23c07fd29692fb70baa513ea403a39e643c48ec5edadb15fe72"

    def __init__(self, device: Optional[str] = None):
        """
        Args:
            device: "cuda" | "cpu" | None (auto-detect từ settings)
        """
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Lazy-loaded IOPaint model
        self._iopaint_model = None

        # Check IOPaint availability
        try:
            from iopaint.model.lama import LaMa  # noqa: F401
            self._iopaint_available = True
        except ImportError:
            self._iopaint_available = False
            logger.warning("⚠️  IOPaint not found. Will use Replicate fallback.")

        # Check Replicate token (fallback)
        import os
        self._replicate_token = (
            os.getenv("REPLICATE_API_TOKEN") or settings.REPLICATE_API_TOKEN
        )

        logger.info(
            f"🦙 LaMa service init | device={self.device} | "
            f"iopaint={self._iopaint_available} | "
            f"replicate_fallback={bool(self._replicate_token)}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def remove_object(
        self,
        image: Image.Image,
        mask: Image.Image,
        mask_padding: Optional[int] = None,   # already handled by preprocess; kept for compat
    ) -> Tuple[Image.Image, dict]:
        """
        Remove object using LaMa.

        Note: mask_padding ở đây KHÔNG dilate thêm nữa vì caller
        (endpoint) đã chạy preprocess_mask_for_removal() trước.
        Tham số giữ lại chỉ để log metadata.

        Args:
            image: RGB PIL Image
            mask:  Grayscale PIL Image (trắng = vùng cần xóa)
            mask_padding: chỉ dùng để ghi vào metadata

        Returns:
            (result_image, metadata_dict)
        """
        logger.info(
            f"🦙 LaMa remove_object | size={image.size} | "
            f"iopaint={self._iopaint_available}"
        )

        if self._iopaint_available:
            try:
                return self._run_iopaint(image, mask, mask_padding)
            except Exception as e:
                logger.error(f"❌ IOPaint LaMa failed: {e}. Trying Replicate...")

        if self._replicate_token:
            return self._run_replicate(image, mask, mask_padding)

        raise RuntimeError(
            "Không thể chạy LaMa: IOPaint failed và không có REPLICATE_API_TOKEN. "
            "Kiểm tra lại GPU hoặc thêm token vào .env"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # IOPaint backend (local)
    # ──────────────────────────────────────────────────────────────────────────

    def _load_iopaint_model(self):
        """Lazy-load IOPaint LaMa model."""
        if self._iopaint_model is not None:
            return
        logger.info(f"📦 Loading IOPaint LaMa on {self.device}...")
        from iopaint.model.lama import LaMa
        self._iopaint_model = LaMa(device=self.device)
        logger.info("✅ IOPaint LaMa loaded")

    def _run_iopaint(
        self,
        image: Image.Image,
        mask: Image.Image,
        mask_padding: Optional[int],
    ) -> Tuple[Image.Image, dict]:
        from iopaint.schema import InpaintRequest, HDStrategy

        self._load_iopaint_model()
        start = time.time()

        # PIL → numpy
        # IOPaint LaMa expects input image as RGB uint8 and returns BGR uint8.
        import cv2
        image_rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        mask_np = np.array(mask.convert("L"), dtype=np.uint8)

        # Mask stats
        coverage_pct = float((mask_np > 127).sum()) / mask_np.size * 100

        # IOPaint config — dùng ORIGINAL để giữ nguyên resolution
        config = InpaintRequest(
            image=b"",   # không dùng, model nhận numpy trực tiếp
            mask=b"",
            hd_strategy=HDStrategy.ORIGINAL,
        )

        # Inference
        result_bgr = self._iopaint_model(image_rgb, mask_np, config)

        # IOPaint có thể trả về float32/float64 [0-255] — cast về uint8 trước cvtColor
        if result_bgr.dtype != np.uint8:
            result_bgr = np.clip(result_bgr, 0, 255).astype(np.uint8)

        # BGR → RGB → PIL
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(result_rgb)

        # Đảm bảo kích thước khớp với ảnh gốc
        if result_image.size != image.size:
            result_image = result_image.resize(image.size, Image.LANCZOS)

        elapsed = time.time() - start
        logger.info(f"✅ IOPaint LaMa done in {elapsed:.1f}s")

        return result_image, {
            "method": "lama",
            "backend": "iopaint_local",
            "device": self.device,
            "input_size": list(image.size),
            "output_size": list(result_image.size),
            "mask_coverage_pct": round(coverage_pct, 2),
            "mask_padding": mask_padding,
            "processing_time": round(elapsed, 2),
            "cost_usd": 0.0,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Replicate fallback (cloud)
    # ──────────────────────────────────────────────────────────────────────────

    def _run_replicate(
        self,
        image: Image.Image,
        mask: Image.Image,
        mask_padding: Optional[int],
    ) -> Tuple[Image.Image, dict]:
        import os, base64, replicate, requests

        os.environ["REPLICATE_API_TOKEN"] = self._replicate_token
        start = time.time()

        def _to_data_uri(img: Image.Image, mode: str) -> str:
            buf = io.BytesIO()
            img.convert(mode).save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f"data:image/png;base64,{b64}"

        mask_np = np.array(mask.convert("L"))
        coverage_pct = float((mask_np > 127).sum()) / mask_np.size * 100

        logger.info("☁️  LaMa fallback → Replicate cloud...")
        output = replicate.run(
            self._REPLICATE_MODEL,
            input={
                "image": _to_data_uri(image, "RGB"),
                "mask": _to_data_uri(mask, "L"),
            },
        )

        result_url = output[0] if isinstance(output, list) else output
        resp = requests.get(str(result_url), timeout=60)
        resp.raise_for_status()
        result_image = Image.open(io.BytesIO(resp.content)).convert("RGB")

        if result_image.size != image.size:
            result_image = result_image.resize(image.size, Image.LANCZOS)

        elapsed = time.time() - start
        logger.info(f"✅ Replicate LaMa done in {elapsed:.1f}s")

        return result_image, {
            "method": "lama",
            "backend": "replicate_cloud",
            "model": self._REPLICATE_MODEL,
            "input_size": list(image.size),
            "output_size": list(result_image.size),
            "mask_coverage_pct": round(coverage_pct, 2),
            "mask_padding": mask_padding,
            "processing_time": round(elapsed, 2),
            "cost_usd": 0.00057,
        }


# ────────────────────────────────────────────────────────────────────────────
# Singleton factory
# ────────────────────────────────────────────────────────────────────────────

_lama_instance: Optional[LamaInpainting] = None


def get_lama_inpainting_service(device: Optional[str] = None) -> LamaInpainting:
    """Get or create global LaMa service instance (lazy-loads IOPaint model)."""
    global _lama_instance
    if _lama_instance is None:
        _lama_instance = LamaInpainting(device=device)
    return _lama_instance
