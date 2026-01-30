from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Security
    API_KEY: str = "dev-key-change-in-production"
    MAX_REQUEST_SIZE: int = 10 * 1024 * 1024  # 10MB
    
    # Performance
    MAX_CONCURRENT_REQUESTS: int = 5
    REQUEST_TIMEOUT: int = 30
    WORKERS: int = 1  # Adjust based on CPU cores
    
    # Audio Processing
    SAMPLE_RATE: int = 16000  # Resample to 16kHz for model
    MAX_AUDIO_DURATION: float = 60.0  # seconds
    MIN_AUDIO_DURATION: float = 0.5  # seconds
    SUPPORTED_FORMATS: List[str] = ["mp3"]
    
    # Model Settings
    MODEL_PATH: str = "./models/voice_classifier.pkl"
    DEVICE: str = "cpu"  # or "cuda" if available
    CONFIDENCE_THRESHOLD: float = 0.5
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()


settings = get_settings()
