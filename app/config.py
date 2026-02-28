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
    
    # External API Keys (for Replicate, HuggingFace)
    REPLICATE_API_TOKEN: Optional[str] = None
    HUGGINGFACE_API_TOKEN: Optional[str] = None  # For Inference API
    HUGGINGFACE_API_KEY: Optional[str] = None  # Legacy name
    
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
