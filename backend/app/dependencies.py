"""
Dependency injection for FastAPI
Models are loaded once and injected into endpoints
"""
import torch
from pathlib import Path
from functools import lru_cache
from segment_anything import sam_model_registry, SamPredictor

from app.config import settings
from app.utils.logger import logger


class ModelManager:
    """Singleton class to manage AI models"""
    
    _instance = None
    _sam_predictor = None
    _device = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._device is None:
            self._initialize_runtime()
    
    def _initialize_runtime(self):
        """Initialize runtime info without loading heavy models."""
        # Detect device
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 Runtime initialized on device: {self._device}")

    def _resolve_checkpoint_path(self) -> Path:
        """Resolve SAM checkpoint path from config."""
        if settings.SAM_CHECKPOINT_PATH:
            configured_path = Path(settings.SAM_CHECKPOINT_PATH)
            if configured_path.is_absolute():
                return configured_path
            return settings.BASE_DIR / configured_path

        # Load SAM
        checkpoint_path = settings.WEIGHTS_DIR / settings.SAM_CHECKPOINT
        if not checkpoint_path.exists():
            # Fallback to old location for backward compatibility
            checkpoint_path = settings.BASE_DIR / settings.SAM_CHECKPOINT

        return checkpoint_path

    def _load_sam_if_needed(self):
        """Lazy-load local SAM predictor only when required."""
        if self._sam_predictor is not None:
            return

        checkpoint_path = self._resolve_checkpoint_path()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found at: {checkpoint_path}. "
                "Set SAM_CHECKPOINT_PATH or place checkpoint in backend/weights/."
            )
        
        logger.info(f"📦 Loading local SAM from: {checkpoint_path}")
        sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))
        sam.to(device=self._device)
        self._sam_predictor = SamPredictor(sam)
        logger.info("✅ Local SAM model loaded successfully")

    @property
    def is_sam_loaded(self) -> bool:
        """Whether local SAM is already loaded in memory."""
        return self._sam_predictor is not None
    
    @property
    def sam_predictor(self) -> SamPredictor:
        """Get SAM predictor instance (loads lazily on first use)."""
        self._load_sam_if_needed()
        return self._sam_predictor
    
    @property
    def device(self) -> str:
        """Get current device"""
        return self._device


@lru_cache()
def get_model_manager() -> ModelManager:
    """
    Dependency function to get ModelManager instance
    Cached to ensure singleton behavior
    """
    return ModelManager()
