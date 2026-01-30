from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import time
from collections import defaultdict
import asyncio
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Window-based rate limiting middleware."""
    
    def __init__(self, app, max_requests: int = None, window: int = None):
        super().__init__(app)
        self.max_requests = max_requests or settings.RATE_LIMIT_PER_MINUTE
        self.window = window or settings.RATE_LIMIT_WINDOW
        self.requests: dict = defaultdict(list)
        self.lock = asyncio.Lock()
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for health check
        if request.url.path == "/health":
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        async with self.lock:
            # Clean old requests outside the window
            self.requests[client_ip] = [
                req_time for req_time in self.requests[client_ip]
                if current_time - req_time < self.window
            ]
            
            # Check if rate limit exceeded
            if len(self.requests[client_ip]) >= self.max_requests:
                logger.warning(f"Rate limit exceeded for {client_ip}")
                return JSONResponse(
                    status_code=429,
                    content={
                        "status": "error",
                        "message": f"Rate limit exceeded. Max {self.max_requests} requests per minute.",
                        "errorCode": "RATE_LIMIT_EXCEEDED"
                    }
                )
            
            # Record this request
            self.requests[client_ip].append(current_time)
        
        return await call_next(request)


class CircuitBreaker:
    """
    Circuit breaker pattern for model inference.
    Prevents cascade failures when the model is overloaded or failing.
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = asyncio.Lock()
    
    async def call(self, func, *args, **kwargs):
        """Execute a function with circuit breaker protection."""
        async with self._lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                else:
                    raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            async with self._lock:
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info("Circuit breaker recovered to CLOSED")
            
            return result
            
        except Exception as e:
            async with self._lock:
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    self.last_failure_time = time.time()
                    logger.error(f"Circuit breaker OPENED after {self.failure_count} failures")
            raise e
    
    def reset(self):
        """Manually reset the circuit breaker."""
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = None
