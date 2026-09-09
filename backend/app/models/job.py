"""
Job models for the Photo Organizer API.
"""
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict


class JobStatus(str, Enum):
    """Job processing status."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobBase(BaseModel):
    """Base job model with common fields."""
    job_id: UUID = Field(default_factory=uuid4)
    status: JobStatus = JobStatus.QUEUED
    progress: int = Field(default=0, ge=0, le=100)
    total_files: int = Field(ge=1)
    processed_files: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    error: Optional[str] = None


class JobCreate(BaseModel):
    """Request model for creating a job."""
    total_files: int = Field(ge=1, le=50)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"total_files": 5}
        }
    )


class Job(JobBase):
    """Complete job model with all fields."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "processing",
                "progress": 60,
                "total_files": 5,
                "processed_files": 3,
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:30:15Z",
                "error": None
            }
        }
    )


class JobListResponse(BaseModel):
    """Response model for job listing."""
    jobs: list["Job"]
    total: int
    limit: int
    offset: int
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "jobs": [
                    {
                        "job_id": "550e8400-e29b-41d4-a716-446655440000",
                        "status": "completed",
                        "progress": 100,
                        "total_files": 5,
                        "processed_files": 5,
                        "created_at": "2024-01-15T10:30:00Z",
                        "updated_at": "2024-01-15T10:30:45Z"
                    }
                ],
                "total": 42,
                "limit": 20,
                "offset": 0
            }
        }
    )