# Photo Organizer - Architecture Documentation

## System Overview

A full-stack photo organizer application with React frontend on GitHub Pages and FastAPI backend on Render Free tier.

## Architecture Diagram

```
┌─────────────┐     HTTPS/REST      ┌─────────────┐
│   GitHub    │ ──────────────────▶ │    Render   │
│   Pages     │                     │  (Free)     │
│  (React)    │ ◀────────────────── │  (FastAPI)  │
└─────────────┘                     └──────┬──────┘
                                           │
                              ┌────────────┴───────────┐
                              ▼                       ▼
                        Lightweight              AI Processing
                        Processing               (Replaceable)
```

## Technology Stack

### Frontend
- **React 18** - Component library
- **Vite 5** - Build tool & dev server
- **Tailwind CSS 3** - Utility-first styling
- **React Router 6** - Client-side routing
- **Axios** - HTTP client
- **React Dropzone** - File upload UX

### Backend
- **Python 3.11+**
- **FastAPI** - Modern async web framework
- **Uvicorn** - ASGI server
- **Pydantic** - Data validation
- **Python-multipart** - File upload handling

### AI/ML (Lightweight, Replaceable)
- **OpenCV** - Image processing
- **face-recognition** or **insightface** - Face detection/embeddings
- **scikit-learn** - Clustering (DBSCAN/KMeans)
- **NumPy** - Numerical operations

### Infrastructure
- **GitHub Pages** - Static hosting (free)
- **Render Free Web Service** - Backend hosting (512MB RAM, spins down after inactivity)
- **GitHub Actions** - CI/CD
- **Environment Variables** - Configuration

## Data Flow

### Upload Flow
1. User selects photos in React frontend
2. Frontend uploads to `/api/upload` (multipart/form-data)
3. Backend validates, generates job ID, returns immediately
4. Backend queues background processing task
5. Frontend polls `/api/jobs/{job_id}` for status
6. When complete, frontend fetches `/api/results/{job_id}`

### Processing Pipeline (Background)
1. **Load Images** - Decode, resize for processing
2. **Face Detection** - Locate faces in each image
3. **Face Embeddings** - Generate 128-d vectors per face
4. **Clustering** - Group similar faces (DBSCAN)
5. **Person Groups** - Assign cluster IDs to "Person 1", "Person 2", etc.
7. **Store Results** - JSON metadata per image + person groups

## API Specification

### Base URL
- Development: `http://localhost:8000`
- Production: `https://photo-organizer-api.onrender.com`

### Endpoints

#### Health Check
```
GET /api/health
Response: { "status": "ok", "version": "1.0.0" }
```

#### Upload Photos
```
POST /api/upload
Content-Type: multipart/form-data

Request:
- files: List[UploadFile] (max 50, max 10MB each, image/*)

Response: 
{
  "job_id": "uuid",
  "status": "queued",
  "total_files": 5,
  "message": "Upload successful, processing started"
}
```

#### Job Status
```
GET /api/jobs/{job_id}
Response:
{
  "job_id": "uuid",
  "status": "processing|completed|failed",
  "progress": 0-100,
  "total_files": 5,
  "processed_files": 3,
  "error": null|string
}
```

#### Job Results
```
GET /api/results/{job_id}
Response:
{
  "job_id": "uuid",
  "status": "completed",
  "total_photos": 5,
  "total_faces": 12,
  "person_groups": [
    {
      "person_id": "person_1",
      "face_count": 5,
      "representative_face": "base64_thumbnail",
      "photo_ids": ["photo_1.jpg", "photo_3.jpg"]
    }
  ],
  "photos": [
    {
      "photo_id": "photo_1.jpg",
      "faces": [
        {"person_id": "person_1", "bbox": [x,y,w,h], "confidence": 0.95}
      ]
    }
  ]
}
```

#### List Jobs (Optional)
```
GET /api/jobs
Query: ?limit=20&offset=0
Response: { "jobs": [...], "total": 42 }
```

## Data Models

### Job
```python
class Job(BaseModel):
    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    created_at: datetime
    updated_at: datetime
    total_files: int
    processed_files: int = 0
    progress: int = 0
    error: Optional[str] = None
```

### Face Detection Result
```python
class FaceDetection(BaseModel):
    bbox: List[int]  # [x, y, width, height]
    confidence: float
    embedding: Optional[List[float]] = None
```

### Photo Result
```python
class PhotoResult(BaseModel):
    photo_id: str
    faces: List[FaceDetection]
```

### Person Group
```python
class PersonGroup(BaseModel):
    person_id: str
    face_count: int
    representative_face: str  # base64 encoded thumbnail
    photo_ids: List[str]
```

### Job Results
```python
class JobResults(BaseModel):
    job_id: str
    status: Literal["completed", "failed"]
    total_photos: int
    total_faces: int
    person_groups: List[PersonGroup]
    photos: List[PhotoResult]
```

## Error Responses

```python
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
```

HTTP Status Codes:
- 200 - Success
- 400 - Bad Request (validation error)
- 404 - Not Found (job not found)
- 413 - Payload Too Large
- 415 - Unsupported Media Type
- 422 - Unprocessable Entity
- 500 - Internal Server Error
- 503 - Service Unavailable (model loading)

## Configuration (Environment Variables)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PORT` | No | 8000 | Server port |
| `HOST` | No | 0.0.0.0 | Bind address |
| `CORS_ORIGINS` | Yes | - | Comma-separated allowed origins |
| `MAX_FILE_SIZE` | No | 10485760 | Max file size in bytes (10MB) |
| `MAX_FILES_PER_JOB` | No | 50 | Max files per upload |
| `ALLOWED_EXTENSIONS` | No | .jpg,.jpeg,.png,.webp | Allowed file extensions |
| `MAX_IMAGE_DIMENSION` | No | 1920 | Max width/height for processing |
| `FACE_DETECTION_MODEL` | No | hog | `hog` or `cnn` |
| `CLUSTERING_EPS` | No | 0.5 | DBSCAN epsilon |
| `CLUSTERING_MIN_SAMPLES` | No | 2 | DBSCAN min samples |
| `JOB_TTL_HOURS` | No | 24 | Job retention |
| `LOG_LEVEL` | No | INFO | Logging level |

## Repository Structure

```
/photo-organizer/
├── .github/
│   └── workflows/
│       ├── backend-deploy.yml
│       └── frontend-deploy.yml
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── job.py
│   │   │   ├── photo.py
│   │   │   └── person.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── health.py
│   │   │   ├── upload.py
│   │   │   ├── jobs.py
│   │   │   └── results.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── image_service.py
│   │   │   ├── face_service.py
│   │   │   └── clustering_service.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── logging.py
│   │       └── validation.py
│   ├── requirements.txt
│   ├── .env.example
│   ├── .gitignore
│   └── README.md
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ui/
│   │   │   ├── upload/
│   │   │   ├── results/
│   │   │   └── layout/
│   │   ├── pages/
│   │   ├── services/
│   │   │   ├── api.js
│   │   │   └── storage.js
│   │   ├── hooks/
│   │   ├── utils/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── .env.example
│   └── .gitignore
├── architecture.md
├── API_SPEC.md
├── DEPLOYMENT.md
├── AGENTS.md
└── README.md
```

## Deployment Plan

### Backend (Render Free)
1. Connect GitHub repo to Render
2. Create Web Service:
   - Root Directory: `backend`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Plan: Free
3. Add Environment Variables in Render Dashboard
4. Enable Auto-Deploy on push to main

### Frontend (GitHub Pages)
1. Configure Vite base path: `/photo-organizer/`
2. GitHub Action builds on push to main
3. Deploys `dist/` to `gh-pages` branch
4. Access at `https://username.github.io/photo-organizer/`

### CI/CD
- **Backend**: Render auto-deploys on push to main
- **Frontend**: GitHub Actions builds and deploys on push to main

## Security Considerations

1. **CORS**: Restricted to GitHub Pages origin only
2. **File Validation**: MIME type + extension + size validation
3. **Rate Limiting**: Built-in via Render (consider adding slowapi)
3. **No Secrets in Frontend**: All secrets in Render environment variables
4. **Input Validation**: Pydantic models for all inputs
5. **Error Sanitization**: No stack traces in production responses

## Performance Considerations (Render Free: 512MB RAM)

1. **Lazy Model Loading**: Load ML models on first request
5. **Image Resizing**: Downscale to max 1920px before processing
6. **Streaming Uploads**: Use StreamingResponse for large files
6. **Memory Cleanup**: Explicit `del` and `gc.collect()` after processing
7. **Batch Processing**: Process images sequentially, not all at once
8. **Job Cleanup**: Auto-delete job data after 24 hours

## Future Extensibility

The AI processing layer is isolated in `services/` and can be replaced:
- Swap `face-recognition` for `insightface` or cloud API
- Add object detection, scene classification
- Support video processing
- Add facial recognition (known persons)

## Monitoring & Observability

- Structured JSON logging
- Request/response logging middleware
- Health endpoint for Render health checks
- Job status tracking for debugging

---

*Document Version: 1.0*
*Last Updated: 2024*