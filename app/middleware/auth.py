from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time
import logging
import secrets

from app.config import settings

logger = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Middleware for API key authentication."""
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth for health check endpoint
        if request.url.path == "/health":
            return await call_next(request)
        
        # Skip auth for docs endpoints
        if request.url.path in ["/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Extract API key from header
        api_key = request.headers.get("x-api-key")
        
        if not api_key:
            logger.warning(f"Missing API key from {request.client.host if request.client else 'unknown'}")
            return JSONResponse(
                status_code=401,
                content={
                    "status": "error",
                    "message": "Missing API key. Include 'x-api-key' header.",
                    "errorCode": "AUTH_MISSING"
                }
            )
        
        # Use constant-time comparison to prevent timing attacks
        if not secrets.compare_digest(api_key, settings.API_KEY):
            # Log failed attempt for security monitoring
            logger.warning(f"Invalid API key attempt from {request.client.host if request.client else 'unknown'}")
            return JSONResponse(
                status_code=401,
                content={
                    "status": "error",
                    "message": "Invalid API key",
                    "errorCode": "AUTH_INVALID"
                }
            )
        
        # Add request metadata
        request.state.authenticated = True
        request.state.timestamp = time.time()
        
        response = await call_next(request)
        return response
