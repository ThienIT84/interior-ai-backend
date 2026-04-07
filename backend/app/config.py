"""
Configuration management for the application
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # App Info
    APP_NAME: str = "AI Interior Design Backend"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = True
    
    # Backend Server Settings
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    
    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list = ["*"]
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    WEIGHTS_DIR: Path = BASE_DIR / "weights"
    DATA_DIR: Path = BASE_DIR / "data"
    INPUTS_DIR: Path = DATA_DIR / "inputs"
    OUTPUTS_DIR: Path = DATA_DIR / "outputs"
    MASKS_DIR: Path = DATA_DIR / "masks"
    TEMP_DIR: Path = DATA_DIR / "temp"
    
    # Model Settings
    SAM_CHECKPOINT: str = "sam_vit_b_01ec64.pth"
    SAM_CHECKPOINT_PATH: Optional[str] = None  # Full path from .env (optional)
    DEVICE: str = "cuda"  # Will be auto-detected
    # Segmentation backend: "local" (SAM on this machine) or "sam3_replicate" (cloud)
    SEGMENTATION_BACKEND: str = "local"
    # Fallback to local SAM when SAM3 Replicate fails
    SEGMENTATION_FALLBACK_TO_LOCAL: bool = True
    # Pinned Replicate model version for SAM3
    SAM3_REPLICATE_MODEL: str = "mattsays/sam3-image:d73db077226443ba4fafd34e233b3626b552eac2a433f90c7c32a9ac89bd9e72"
    
    # External API Keys (for Replicate, HuggingFace)
    REPLICATE_API_TOKEN: Optional[str] = None
    HUGGINGFACE_API_TOKEN: Optional[str] = None  # For Inference API
    HUGGINGFACE_API_KEY: Optional[str] = None  # Legacy name
    
    # ── Remove-object Pipeline Settings ──────────────────────────────────────
    # Primary method for object removal: "lama" | "replicate" | "local" | "auto"
    REMOVE_OBJECT_METHOD: str = "lama"
    # Backward-compatible legacy alias. Prefer REMOVE_OBJECT_METHOD.
    INPAINTING_METHOD: Optional[str] = None
    # Fallback when primary fails
    REMOVE_OBJECT_FALLBACK: str = "replicate_sd"
    # Mask coverage gate: reject if fraction outside (0-1 float, not %)
    MASK_COVERAGE_MIN: float = 0.003   # 0.3%
    MASK_COVERAGE_MAX: float = 0.45    # 45%
    # Mask preprocessing defaults
    MASK_DILATION_DEFAULT: int = 15    # pixels
    MASK_FEATHER_DEFAULT: int = 4      # pixels
    # Artifact score threshold (0-1 scale) above which pass-2 repair is triggered
    ARTIFACT_SCORE_THRESHOLD: float = 0.15

    # ── Job Processing & Persistence Settings ────────────────────────────────
    # Redis configuration for job storage
    REDIS_URL: str = "redis://localhost:6379/0"
    # Redis key prefix for job storage
    REDIS_KEY_PREFIX: str = "interior_job:"
    # Job expiry time in seconds (24 hours)
    JOB_EXPIRY_SECONDS: int = 86400
    # Maximum number of retries for failed jobs
    MAX_RETRIES: int = 3
    # Request timeout for external API calls (seconds)
    REQUEST_TIMEOUT_SECONDS: int = 300  # 5 minutes
    # Retry delay between attempts (seconds)
    RETRY_DELAY_SECONDS: int = 2

    # Processing Settings
    MAX_IMAGE_SIZE: int = 2048  # Max dimension for processing
    JPEG_QUALITY: int = 90
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env


# Global settings instance
settings = Settings()


def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        settings.WEIGHTS_DIR,
        settings.INPUTS_DIR,
        settings.OUTPUTS_DIR,
        settings.MASKS_DIR,
        settings.TEMP_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
