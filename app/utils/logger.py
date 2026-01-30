import logging
import sys
import json
from typing import Any, Dict

from app.config import settings


def setup_logging() -> logging.Logger:
    """Configure structured logging for the application."""
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper()))
    root_logger.addHandler(console_handler)
    
    # Silence noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    return root_logger


def log_request(logger: logging.Logger, data: Dict[str, Any]) -> None:
    """Log a structured request event."""
    logger.info(json.dumps({
        "event": "request",
        **data
    }))


def log_prediction(
    logger: logging.Logger,
    language: str,
    classification: str,
    confidence: float,
    processing_time: float,
    client_ip: str = None
) -> None:
    """Log a prediction event with metrics."""
    logger.info(json.dumps({
        "event": "prediction",
        "language": language,
        "classification": classification,
        "confidence": round(confidence, 3),
        "processing_time_ms": round(processing_time * 1000, 2),
        "client_ip": client_ip or "unknown"
    }))
