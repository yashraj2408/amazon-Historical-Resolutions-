"""
API Routes package.
"""
from .health import router as health_router
from .upload import router as upload_router
from .jobs import router as jobs_router
from .results import router as results_router

__all__ = [
    "health_router",
    "upload_router",
    "jobs_router",
    "results_router",
]