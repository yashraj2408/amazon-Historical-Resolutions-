"""
Job status and management endpoints.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Query, Depends
from uuid import UUID

from app.models.job import Job, JobListResponse, JobStatus
from app.services.job_service import JobService

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get(
    "/{job_id}",
    response_model=Job,
    responses={
        404: {"description": "Job not found"}
    }
)
async def get_job_status(
    job_id: UUID,
    job_service: JobService = Depends()
):
    """
    Get the current status of a processing job.
    
    Returns job status, progress, and any error messages.
    """
    job_service = JobService()
    job = await job_service.get_job(job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    return job


@router.get(
    "/",
    response_model=JobListResponse,
    responses={
        400: {"description": "Invalid query parameters"}
    }
)
async def list_jobs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: Optional[str] = Query(None, description="Filter by status"),
    job_service: JobService = Depends()
):
    """
    List all jobs with pagination and optional status filter.
    """
    job_service = JobService()
    
    # Validate status filter
    if status and status not in [s.value for s in JobStatus]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid status. Must be one of: {[s.value for s in JobStatus]}"
        )
    
    jobs, total = await job_service.list_jobs(limit, offset, status)
    
    return JobListResponse(
        jobs=jobs,
        total=total,
        limit=limit,
        offset=offset
    )


@router.delete(
    "/{job_id}",
    status_code=status.HTTP_200_OK,
    responses={
        404: {"description": "Job not found"}
    }
)
async def delete_job(
    job_id: UUID,
    job_service: JobService = Depends()
):
    """
    Delete a job and its associated data.
    """
    success = await job_service.delete_job(job_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    return {"message": "Job deleted successfully", "job_id": str(job_id)}