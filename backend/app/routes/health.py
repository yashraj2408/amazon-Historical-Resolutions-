"""
Health check endpoint.
"""
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    timestamp: datetime = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "version": "1.0.0",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Render and monitoring."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        timestamp=datetime.utcnow()
    )