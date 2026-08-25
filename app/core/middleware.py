import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Rate limiter instance
limiter = Limiter(key_func=get_remote_address)

class RequestCorrelationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        
        start_time = time.time()
        
        # Add request ID to logger context if using structlog or similar,
        # For standard logging, we can just log it here
        logger.info(f"⇒ [{request_id}] {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"❌ [{request_id}] Unhandled error: {str(e)}")
            response = JSONResponse(
                status_code=500,
                content={"detail": "Internal Server Error", "request_id": request_id}
            )
            
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(elapsed_ms)
        
        logger.info(f"⇐ [{request_id}] {request.method} {request.url.path} → {response.status_code} ({elapsed_ms}ms)")
        
        return response

def setup_middlewares(app):
    app.add_middleware(RequestCorrelationMiddleware)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

