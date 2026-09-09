"""
Photo Organizer Backend - FastAPI Application Entry Point.
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.routes import (
    health_router,
    upload_router,
    jobs_router,
    results_router
)
from app.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("Starting Photo Organizer Backend")
    logger.info(f"CORS origins: {settings.cors_origins}")
    logger.info(f"Max file size: {settings.max_file_size / 1024 / 1024:.0f}MB")
    
    # Initialize services (lazy loading happens on first request)
    logger.info("Backend started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Photo Organizer Backend")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Photo Organizer API",
        description="API for photo upload, face detection, and person grouping",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )
    
    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"]
    )
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_ERROR",
                "detail": "An unexpected error occurred"
            }
        )
    
    # Include routers
    app.include_router(health_router, prefix="/api")
    app.include_router(upload_router, prefix="/api")
    app.include_router(jobs_router, prefix="/api")
    app.include_router(results_router, prefix="/api")
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level=settings.log_level.lower()
    )