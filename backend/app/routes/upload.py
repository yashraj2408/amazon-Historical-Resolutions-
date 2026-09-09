"""
Upload endpoint for photo processing jobs.
"""
import uuid
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, status, BackgroundTasks, Depends
from pydantic import BaseModel

from app.models.job import JobCreate, Job
from app.services.job_service import JobService
from app.services.image_service import validate_upload_files
from app.config import get_settings

router = APIRouter(prefix="/upload", tags=["upload"])


class UploadResponse(BaseModel):
    job_id: str
    status: str
    total_files: int
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "queued",
                "total_files": 5,
                "message": "Upload successful, processing started"
            }
        }


@router.post(
    "/",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"description": "Invalid file type or too many files"},
        413: {"description": "File too large"},
        415: {"description": "Unsupported media type"},
        422: {"description": "Validation error"}
    }
)
async def upload_photos(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    job_service: JobService = Depends()
):
    """
    Upload photos for face detection and person grouping.
    
    - Maximum 50 files per request
    - Maximum 10MB per file
    - Supported formats: JPEG, PNG, WebP
    - Returns job_id for status tracking
    """
    settings = get_settings()
    
    # Validate files
    try:
        validated_files = await validate_upload_files(files, settings)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    # Create job
    job = JobCreate(total_files=len(validated_files))
    job_id = await job_service.create_job(job)
    
    # Queue background processing
    background_tasks.add_task(
        process_job_background,
        job_id,
        validated_files,
        job_service
    )
    
    return UploadResponse(
        job_id=str(job_id),
        status="queued",
        total_files=len(validated_files),
        message="Upload successful, processing started"
    )


async def process_job_background(job_id: uuid.UUID, files: List[UploadFile], job_service: JobService = Depends()):
    """Background task to process uploaded photos."""
    # This will be implemented in job_service
    from app.services.job_service import JobService
    from app.services.face_service import FaceService
    from app.services.clustering_service import ClusteringService
    from app.config import get_settings
    
    settings = get_settings()
    face_service = FaceService(settings)
    clustering_service = ClusteringService(settings)
    
    try:
        await job_service.update_job_status(
            job_id, 
            status="processing",
            progress=0,
            processed_files=0
        )
        
        # Process each file
        all_faces = []
        photo_results = []
        
        for idx, file in enumerate(files):
            # Read and process image
            content = await file.read()
            await file.seek(0)
            
            # Detect faces
            faces = await face_service.detect_faces(content)
            
            # Generate embeddings for each face
            for face in faces:
                embedding = await face_service.get_embedding(content, face.bbox)
                face.embedding = embedding
            
            all_faces.extend(faces)
            photo_results.append({
                "photo_id": file.filename,
                "faces": faces
            })
            
            # Update progress
            progress = int(((idx + 1) / len(files)) * 100)
            await job_service.update_job_status(
                job_id,
                progress=progress,
                processed_files=idx + 1
            )
        
        # Cluster faces
        person_groups = await clustering_service.cluster_faces(all_faces)
        
        # Assign person IDs to faces
        for face in all_faces:
            for group in person_groups:
                if face.embedding in group.embeddings:
                    face.person_id = group.person_id
                    break
        
        # Save results
        await job_service.save_results(job_id, person_groups, photo_results)
        await job_service.update_job_status(job_id, status="completed", progress=100)
        
    except Exception as e:
        await job_service.update_job_status(
            job_id,
            status="failed",
            error=str(e)
        )
        raise


# Dependency injection for services
def get_job_service() -> JobService:
    return JobService()