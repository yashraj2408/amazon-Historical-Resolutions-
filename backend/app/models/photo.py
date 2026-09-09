"""
Photo and face detection models for the Photo Organizer API.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class FaceDetection(BaseModel):
    """Detected face in a photo."""
    person_id: str = Field(description="Assigned person identifier")
    bbox: List[int] = Field(
        description="Bounding box [x, y, width, height]",
        min_length=4,
        max_length=4
    )
    confidence: float = Field(
        ge=0.0, 
        le=1.0,
        description="Detection confidence score"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "person_id": "person_1",
                "bbox": [100, 150, 80, 80],
                "confidence": 0.95
            }
        }
    )


class PhotoResult(BaseModel):
    """Processed photo with detected faces."""
    photo_id: str = Field(description="Original filename or generated ID")
    faces: List[FaceDetection] = Field(default_factory=list)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "photo_id": "photo_1.jpg",
                "faces": [
                    {
                        "person_id": "person_1",
                        "bbox": [100, 150, 80, 80],
                        "confidence": 0.95
                    }
                ]
            }
        }
    )


class PersonGroup(BaseModel):
    """Group of photos belonging to the same person."""
    person_id: str = Field(description="Unique person identifier")
    face_count: int = Field(ge=1, description="Number of faces in this group")
    representative_face: str = Field(
        description="Base64 encoded thumbnail of representative face"
    )
    photo_ids: List[str] = Field(
        description="List of photo IDs containing this person"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "person_id": "person_1",
                "face_count": 5,
                "representative_face": "data:image/jpeg;base64,/9j/4AAQ...",
                "photo_ids": ["photo_1.jpg", "photo_3.jpg", "photo_5.jpg"]
            }
        }
    )


class JobResults(BaseModel):
    """Complete results for a completed job."""
    job_id: str = Field(description="Job identifier")
    status: str = Field(description="Job status (completed/failed)")
    total_photos: int = Field(ge=0)
    total_faces: int = Field(ge=0)
    person_groups: List["PersonGroup"] = Field(default_factory=list)
    photos: List["PhotoResult"] = Field(default_factory=list)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "completed",
                "total_photos": 5,
                "total_faces": 12,
                "person_groups": [
                    {
                        "person_id": "person_1",
                        "face_count": 5,
                        "representative_face": "data:image/jpeg;base64,/9j/4AAQ...",
                        "photo_ids": ["photo_1.jpg", "photo_3.jpg", "photo_5.jpg"]
                    },
                    {
                        "person_id": "person_2",
                        "face_count": 4,
                        "representative_face": "data:image/jpeg;base64,/9j/4AAQ...",
                        "photo_ids": ["photo_2.jpg", "photo_4.jpg"]
                    }
                ],
                "photos": [
                    {
                        "photo_id": "photo_1.jpg",
                        "faces": [
                            {
                                "person_id": "person_1",
                                "bbox": [100, 150, 80, 80],
                                "confidence": 0.95
                            }
                        ]
                    }
                ]
            }
        }
    )


# Resolve forward references
PhotoResult.model_rebuild()
PersonGroup.model_rebuild()
JobResults.model_rebuild()