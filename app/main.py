from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging
from contextlib import asynccontextmanager

from app.config import settings
from app.models.request import VoiceDetectionRequest
from app.models.response import VoiceDetectionResponse, ErrorResponse
from app.middleware.auth import APIKeyMiddleware
from app.middleware.rate_limit import RateLimitMiddleware
from app.services.audio_processor import AudioProcessor
from app.services.feature_extractor import FeatureExtractor
from app.services.classifier import VoiceClassifier
from app.services.explanation import generate_explanation
from app.utils.logger import setup_logging, log_prediction

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global service instances (initialized in lifespan)
audio_processor = None
feature_extractor = None
classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    global audio_processor, feature_extractor, classifier
    
    logger.info("=" * 50)
    logger.info("Voice Detection API Starting...")
    logger.info(f"Log Level: {settings.LOG_LEVEL}")
    logger.info(f"Sample Rate: {settings.SAMPLE_RATE} Hz")
    logger.info(f"Max Audio Duration: {settings.MAX_AUDIO_DURATION}s")
    logger.info("=" * 50)
    
    # Initialize services
    logger.info("Initializing audio processor...")
    audio_processor = AudioProcessor(
        target_sr=settings.SAMPLE_RATE,
        max_duration=settings.MAX_AUDIO_DURATION,
        min_duration=settings.MIN_AUDIO_DURATION
    )
    
    logger.info("Initializing feature extractor...")
    feature_extractor = FeatureExtractor(sr=settings.SAMPLE_RATE)
    
    logger.info("Initializing classifier...")
    classifier = VoiceClassifier(model_path=settings.MODEL_PATH)
    
    logger.info("✅ All services initialized successfully")
    
    yield
    
    logger.info("Shutting down Voice Detection API...")


app = FastAPI(
    title="AI-Generated Voice Detection API",
    description="Detects whether a voice sample is AI-generated or human across 5 Indian languages (Tamil, English, Hindi, Malayalam, Telugu)",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware (order matters - first added = last executed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(APIKeyMiddleware)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper JSON format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "message": str(exc.detail) if isinstance(exc.detail, str) else exc.detail.get("message", str(exc.detail)),
            "errorCode": exc.detail.get("errorCode") if isinstance(exc.detail, dict) else None
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global error handler to ensure all responses are JSON."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "errorCode": "INTERNAL_ERROR"
        }
    )


@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    Does not require authentication.
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "services": {
            "audio_processor": audio_processor is not None,
            "feature_extractor": feature_extractor is not None,
            "classifier": classifier is not None
        }
    }


@app.post(
    "/api/voice-detection",
    response_model=VoiceDetectionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication error"},
        413: {"model": ErrorResponse, "description": "Audio too large"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Detect AI-generated voice",
    description="Analyzes an MP3 audio sample and classifies it as AI_GENERATED or HUMAN"
)
async def detect_voice(request: VoiceDetectionRequest, req: Request):
    """
    Main voice detection endpoint.
    
    - Validates API key via middleware
    - Processes base64 encoded MP3 audio
    - Extracts acoustic features (jitter, shimmer, HNR, etc.)
    - Classifies as AI_GENERATED or HUMAN
    - Returns structured JSON response with explanation
    """
    import asyncio
    import uuid
    
    start_time = time.time()
    client_ip = req.client.host if req.client else "unknown"
    request_id = str(uuid.uuid4())[:8]  # Short request ID for tracing
    
    try:
        logger.info(f"[{request_id}] Processing request for language: {request.language} from {client_ip}")
        
        # Step 1: Process audio with 25s timeout (Railway kills at 30s)
        try:
            waveform = await asyncio.wait_for(
                audio_processor.process(request.audioBase64),
                timeout=25.0
            )
        except asyncio.TimeoutError:
            logger.warning(f"[{request_id}] Processing timeout after 25s")
            raise HTTPException(
                status_code=504,
                detail={
                    "status": "error",
                    "message": "Processing timeout - audio too long or complex",
                    "errorCode": "PROCESSING_TIMEOUT"
                }
            )
        
        logger.debug(f"[{request_id}] Audio processed: {len(waveform)} samples @ {settings.SAMPLE_RATE}Hz")
        
        # Step 2: Extract features
        features = feature_extractor.extract(waveform)
        logger.debug(f"[{request_id}] Features extracted: jitter={features.get('jitter', 0):.4f}, "
                    f"shimmer={features.get('shimmer', 0):.4f}, "
                    f"hnr_db={features.get('hnr_db', 0):.2f}")
        
        # Step 3: Classify
        classification, confidence, reasons = classifier.classify(features)
        
        # Step 4: Generate explanation
        explanation = generate_explanation(features, classification, reasons)
        
        processing_time = time.time() - start_time
        
        # Log prediction for monitoring
        log_prediction(
            logger,
            language=request.language,
            classification=classification,
            confidence=confidence,
            processing_time=processing_time,
            client_ip=client_ip
        )
        
        logger.info(f"[{request_id}] Classification: {classification}, Confidence: {confidence:.2f}, Time: {processing_time:.2f}s")
        
        return VoiceDetectionResponse(
            status="success",
            language=request.language,
            classification=classification,
            confidenceScore=round(confidence, 2),
            explanation=explanation,
            processingTime=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Processing error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": "Failed to process audio",
                "errorCode": "PROCESSING_ERROR"
            }
        )


if __name__ == "__main__":
    import uvicorn
    import os
    
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        workers=settings.WORKERS,
        timeout_keep_alive=30,
        log_level=settings.LOG_LEVEL.lower()
    )

