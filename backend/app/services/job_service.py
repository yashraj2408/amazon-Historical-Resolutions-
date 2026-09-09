"""
Job service for managing photo processing jobs.
Uses in-memory storage (suitable for Render Free tier).
"""
import uuid
from datetime import datetime
from typing import List, Optional, Tuple
from uuid import UUID

from app.models.job import Job, JobCreate, JobStatus
from app.models.photo import JobResults, PersonGroup, PhotoResult


class JobService:
    """
    In-memory job service for managing photo processing jobs.
    
    For Render Free tier, uses in-memory storage.
    For production, replace with Redis/PostgreSQL.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._jobs: dict[UUID, Job] = {}
        self._results: dict[UUID, JobResults] = {}
        self._initialized = True
    
    async def create_job(self, job_create: JobCreate) -> UUID:
        """Create a new job and return its ID."""
        job = Job(
            job_id=uuid.uuid4(),
            status=JobStatus.QUEUED,
            total_files=job_create.total_files
        )
        self._jobs[job.job_id] = job
        return job.job_id
    
    async def get_job(self, job_id: UUID) -> Optional[Job]:
        """Get job by ID."""
        return self._jobs.get(job_id)
    
    async def update_job_status(
        self,
        job_id: UUID,
        status: Optional[JobStatus] = None,
        progress: Optional[int] = None,
        processed_files: Optional[int] = None,
        error: Optional[str] = None
    ) -> bool:
        """Update job status and progress."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        
        if status is not None:
            job.status = status
        if progress is not None:
            job.progress = max(0, min(100, progress))
        if processed_files is not None:
            job.processed_files = processed_files
        if error is not None:
            job.error = error
        
        job.updated_at = datetime.utcnow()
        return True
    
    async def save_results(
        self,
        job_id: UUID,
        person_groups: List["PersonGroup"],
        photo_results: List[dict]
    ) -> bool:
        """Save processing results for a completed job."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        
        # Convert photo results
        photo_results_obj = []
        for pr in photo_results:
            faces = pr.get("faces", [])
            photo_results_obj.append(
                PhotoResult(
                    photo_id=pr.get("photo_id", ""),
                    faces=pr.get("faces", [])
                )
            )
        
        # Create person groups with representative faces
        person_groups_obj = []
        for group in person_groups:
            person_groups_obj.append(PersonGroup(
                person_id=group.get("person_id", ""),
                face_count=group.get("face_count", 0),
                representative_face=group.get("representative_face", ""),
                photo_ids=group.get("photo_ids", [])
            ))
        
        # Calculate totals
        total_faces = sum(g.get("face_count", 0) for g in person_groups)
        
        results = JobResults(
            job_id=str(job_id),
            status="completed",
            total_photos=job.total_files,
            total_faces=total_faces,
            person_groups=person_groups_obj,
            photos=[]
        )
        
        self._results[job_id] = results
        return True
    
    async def get_results(self, job_id: UUID) -> Optional[JobResults]:
        """Get results for a completed job."""
        return self._results.get(job_id)
    
    async def list_jobs(
        self,
        limit: int = 20,
        offset: int = 0,
        status_filter: Optional[str] = None
    ) -> Tuple[List[Job], int]:
        """List jobs with pagination and optional status filter."""
        jobs = list(self._jobs.values())
        
        # Filter by status if provided
        if status_filter:
            jobs = [j for j in jobs if j.status.value == status_filter]
        
        # Sort by created_at descending (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        
        total = len(jobs)
        paginated = jobs[offset:offset + limit]
        
        return paginated, total
    
    async def delete_job(self, job_id: UUID) -> bool:
        """Delete a job and its results."""
        if job_id in self._jobs:
            del self._jobs[job_id]
        if job_id in self._results:
            del self._results[job_id]
        return True
    
    async def cleanup_old_jobs(self) -> int:
        """Remove jobs older than TTL. Returns count of cleaned jobs."""
        from app.config import get_settings
        settings = get_settings()
        cutoff = datetime.utcnow() - datetime.timedelta(hours=settings.job_ttl_hours)
        
        to_delete = [
            job_id for job_id, job in self._jobs.items()
            if job.updated_at < cutoff
        ]
        
        for job_id in to_delete:
            if job_id in self._jobs:
                del self._jobs[job_id]
            if job_id in self._results:
                del self._results[job_id]
        
        return len(to_delete)