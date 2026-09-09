"""
Results endpoint for completed jobs.
"""
from fastapi import APIRouter, HTTPException, status, Depends
from uuid import UUID

from app.models.photo import JobResults
from app.services.job_service import JobService

router = APIRouter(prefix="/results", tags=["results"])


@router.get(
    "/{job_id}",
    response_model=JobResults,
    responses={
        404: {"description": "Job not found"},
        400: {"description": "Job not completed yet"}
    }
)
async def get_job_results(
    job_id: UUID,
    job_service: JobService = Depends()
):
    """
    Get the processed results for a completed job.
    
    Returns person groups, face detections, and photo metadata.
    Only available for completed jobs.
    """
    # Check job exists and is completed
    job = await job_service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job not completed yet"
        )
    
    results = await job_service.get_results(job_id)
    
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Results not found"
        )
    
    return results