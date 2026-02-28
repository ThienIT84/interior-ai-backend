"""
Dependency injection for FastAPI
Models are loaded once and injected into endpoints
"""
import torch
from functools import lru_cache
from segment_anything import sam_model_registry, SamPredictor

from app.config import settings


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
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all AI models"""
        # Detect device
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing models on device: {self._device}")
        
        # Load SAM
        checkpoint_path = settings.WEIGHTS_DIR / settings.SAM_CHECKPOINT
        if not checkpoint_path.exists():
            # Fallback to old location for backward compatibility
            checkpoint_path = settings.BASE_DIR / settings.SAM_CHECKPOINT
        
        print(f"📦 Loading SAM from: {checkpoint_path}")
        sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))
        sam.to(device=self._device)
        self._sam_predictor = SamPredictor(sam)
        print("✅ SAM model loaded successfully")
    
    @property
    def sam_predictor(self) -> SamPredictor:
        """Get SAM predictor instance"""
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
