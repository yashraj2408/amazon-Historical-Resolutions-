# Photo Organizer - API Specification

## Overview

REST API for the Photo Organizer application. Provides endpoints for photo upload, face detection, person grouping, and job management.

## Base URLs

- **Development**: `http://localhost:8000`
- **Production**: `https://photo-organizer-api.onrender.com`

All endpoints are prefixed with `/api`.

---

## Endpoints

### 1. Health Check

Check if the API is running and healthy.

**Endpoint**: `GET /api/health`

**Response** (200 OK):
```json
{
  "status": "ok",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

### 2. Upload Photos

Upload photos for processing. Returns a job ID for tracking.

**Endpoint**: `POST /api/upload`

**Headers**:
```
Content-Type: multipart/form-data
```

**Request Body**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| files | file[] | Yes | Array of image files (max 50, 10MB each) |

**Supported Formats**: JPEG, PNG, WebP

**Constraints**:
- Max 50 files per request
- Max 10MB per file
- Max total: 100MB per request

**Success Response** (202 Accepted):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "total_files": 5,
  "message": "Upload successful, processing started"
}
```

**Error Responses**:

| Status | Code | Description |
|--------|------|-------------|
| 400 | INVALID_FILE_TYPE | File type not allowed |
| 400 | FILE_TOO_LARGE | File exceeds size limit |
| 400 | TOO_MANY_FILES | Exceeds max files per job |
| 413 | PAYLOAD_TOO_LARGE | Total payload exceeds limit |
| 415 | UNSUPPORTED_MEDIA_TYPE | Invalid content type |

---

### 3. Get Job Status

Check the status of a processing job.

**Endpoint**: `GET /api/jobs/{job_id}`

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| job_id | string (UUID) | Yes | Job identifier |

**Response** (200 OK):
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 60,
  "total_files": 5,
  "processed_files": 3,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:15Z",
  "error": null
}
```

**Status Values**:
| Status | Description |
|--------|-------------|
| `queued` | Job waiting to start |
| `processing` | Actively processing |
| `completed` | Finished successfully |
| `failed` | Processing failed |

**Error Responses**:
| Status | Description |
|--------|-------------|
| 404 | Job not found |

---

### 4. Get Job Results

Retrieve processed results for a completed job.

**Endpoint**: `GET /api/results/{job_id}`

**Path Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| job_id | string (UUID) | Yes | Job identifier |

**Response** (200 OK):
```json
{
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
    },
    {
      "person_id": "person_3",
      "face_count": 3,
      "representative_face": "data:image/jpeg;base64,/9j/4AAQ...",
      "photo_ids": ["photo_1.jpg", "photo_2.jpg"]
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
        },
        {
          "person_id": "person_3",
          "bbox": [300, 200, 75, 75],
          "confidence": 0.92
        }
      ]
    }
  ]
}
```

**Error Responses**:
| Status | Code | Description |
|--------|------|-------------|
| 404 | JOB_NOT_FOUND | Job not found |
| 400 | JOB_NOT_COMPLETED | Job not finished yet |

---

### 5. List Jobs (Optional)

List all jobs with pagination.

**Endpoint**: `GET /api/jobs`

**Query Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| limit | integer | No | 20 | Max items to return (max 100) |
| offset | integer | No | 0 | Number of items to skip |
| status | string | No | - | Filter by status |

**Response** (200 OK):
```json
{
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
```

---

### 6. Delete Job (Optional)

Delete job data and results.

**Endpoint**: `DELETE /api/jobs/{job_id}`

**Response** (200 OK):
```json
{
  "message": "Job deleted successfully",
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

---

## Data Models

### Job
```json
{
  "job_id": "string (UUID)",
  "status": "queued|processing|completed|failed",
  "progress": 0-100,
  "total_files": 5,
  "processed_files": 3,
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-01-15T10:30:15Z",
  "error": null
}
```

### PersonGroup
```json
{
  "person_id": "person_1",
  "face_count": 5,
  "representative_face": "data:image/jpeg;base64,...",
  "photo_ids": ["photo_1.jpg", "photo_3.jpg"]
}
```

### PhotoResult
```json
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
```

### Face Detection
```json
{
  "person_id": "person_1",
  "bbox": [x, y, width, height],
  "confidence": 0.95
}
```

---

## Error Response Format

All errors follow this format:

```json
{
  "error": "SHORT_ERROR_CODE",
  "detail": "Human-readable description",
  "code": "OPTIONAL_INTERNAL_CODE"
}
```

### Common Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 400 | VALIDATION_ERROR | Request validation failed |
| 400 | INVALID_FILE_TYPE | Unsupported file format |
| 400 | FILE_TOO_LARGE | File exceeds 10MB limit |
| 400 | TOO_MANY_FILES | Exceeds 50 files per job |
| 404 | JOB_NOT_FOUND | Job ID not found |
| 400 | JOB_NOT_COMPLETED | Results not ready |
| 413 | PAYLOAD_TOO_LARGE | Total upload > 100MB |
| 415 | UNSUPPORTED_MEDIA_TYPE | Not multipart/form-data |
| 422 | UNPROCESSABLE_ENTITY | Validation failed |
| 500 | INTERNAL_ERROR | Server error |
| 503 | SERVICE_UNAVAILABLE | Model loading / overloaded |

### Example Error Response

```json
{
  "error": "FILE_TOO_LARGE",
  "detail": "File 'photo.jpg' exceeds maximum size of 10MB",
  "code": "FILE_SIZE_EXCEEDED"
}
```

---

## File Upload Constraints

| Constraint | Value |
|------------|-------|
| Max files per request | 50 |
| Max file size | 10 MB |
| Total request size | 100 MB |
| Allowed MIME types | image/jpeg, image/png, image/webp |
| Allowed extensions | .jpg, .jpeg, .png, .webp |

---

## Rate Limiting

Render Free tier provides basic protection. Recommended client-side:
- Max 10 requests/minute for upload
- Max 30 requests/minute for status/results
- Implement exponential backoff on 429/503

---

## CORS Configuration

Allowed origins (configured via `CORS_ORIGINS` env var):
- `https://username.github.io`
- `http://localhost:5173` (development)

---

## Versioning

Current version: **v1**

Future versions will use URL prefix: `/api/v2/`

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2024-01-15 | Initial release |

---

*Generated for Photo Organizer API v1.0.0*