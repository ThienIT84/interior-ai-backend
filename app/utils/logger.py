"""
Logging configuration
"""
import logging
import sys
from pathlib import Path

from app.config import settings


def setup_logger(name: str = "interior_ai") -> logging.Logger:
    """
    Setup application logger
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set level based on debug mode
    level = logging.DEBUG if settings.DEBUG else logging.INFO
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger


# Global logger instance
logger = setup_logger()
