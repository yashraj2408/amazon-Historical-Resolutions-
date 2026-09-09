"""
Configuration management for Photo Organizer Backend.
Uses Pydantic Settings for environment variable validation.
"""
from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from urllib.parse import urlparse


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Server
    port: int = Field(default=8000, env="PORT")
    host: str = Field(default="0.0.0.0", env="HOST")
    
    # CORS - Comma-separated origins
    cors_origins: List[str] = Field(default=["http://localhost:5173"], env="CORS_ORIGINS")
    
    # File Upload Limits
    max_file_size: int = Field(default=10485760, env="MAX_FILE_SIZE")  # 10MB
    max_files_per_job: int = Field(default=50, env="MAX_FILES_PER_JOB")
    allowed_extensions: List[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".webp"],
        env="ALLOWED_EXTENSIONS"
    )
    
    # Image Processing
    max_image_dimension: int = Field(default=1920, env="MAX_IMAGE_DIMENSION")
    face_detection_model: str = Field(default="hog", env="FACE_DETECTION_MODEL")
    
    # Face Clustering (DBSCAN)
    clustering_eps: float = Field(default=0.5, env="CLUSTERING_EPS")
    clustering_min_samples: int = Field(default=2, env="CLUSTERING_MIN_SAMPLES")
    
    # Job Management
    job_ttl_hours: int = Field(default=24, env="JOB_TTL_HOURS")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse comma-separated CORS origins."""
        if isinstance(v, str):
            origins = [o.strip() for o in v.split(",") if o.strip()]
            for origin in origins:
                if not origin:
                    continue
                parsed = urlparse(origin)
                if not parsed.scheme or not parsed.netloc:
                    raise ValueError(f"Invalid CORS origin: {origin}")
                if origin.endswith("/"):
                    raise ValueError(f"CORS origin should not have trailing slash: {origin}")
            return origins
        return v
    
    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def parse_allowed_extensions(cls, v):
        """Parse comma-separated allowed extensions."""
        if isinstance(v, str):
            return [e.strip().lower() for e in v.split(",") if e.strip()]
        return v
    
    @field_validator("face_detection_model")
    @classmethod
    def validate_face_model(cls, v):
        """Validate face detection model."""
        if v not in ("hog", "cnn"):
            raise ValueError("face_detection_model must be 'hog' or 'cnn'")
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()
    
    @field_validator("max_file_size")
    @classmethod
    def validate_max_file_size(cls, v):
        if v <= 0 or v > 100_000_000:  # Max 100MB
            raise ValueError("max_file_size must be between 1 and 100000000")
        return v
    
    @field_validator("max_files_per_job")
    @classmethod
    def validate_max_files(cls, v):
        if v <= 0 or v > 100:
            raise ValueError("max_files_per_job must be between 1 and 100")
        return v
    
    @field_validator("max_image_dimension")
    @classmethod
    def validate_max_dimension(cls, v):
        if v < 100 or v > 4096:
            raise ValueError("max_image_dimension must be between 100 and 4096")
        return v
    
    @field_validator("clustering_eps")
    @classmethod
    def validate_clustering_eps(cls, v):
        if v <= 0 or v > 1.0:
            raise ValueError("clustering_eps must be between 0.1 and 1.0")
        return v
    
    @field_validator("clustering_min_samples")
    @classmethod
    def validate_clustering_min_samples(cls, v):
        if v < 1 or v > 10:
            raise ValueError("clustering_min_samples must be between 1 and 10")
        return v
    
    @field_validator("job_ttl_hours")
    @classmethod
    def validate_job_ttl(cls, v):
        if v < 1 or v > 168:  # Max 1 week
            raise ValueError("job_ttl_hours must be between 1 and 168")
        return v


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Dependency injection for FastAPI."""
    return settings