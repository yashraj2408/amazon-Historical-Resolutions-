# Agent Rules - Photo Organizer Project

## Architecture

**Frontend:**
- React + Vite + Tailwind CSS
- Hosted on GitHub Pages

**Backend:**
- Python + FastAPI
- Hosted on Render Free tier

## Rules

1. **Never introduce paid infrastructure.**
2. **Never assume persistent filesystem storage.**
3. **Never store secrets in frontend code.**
4. **Never hardcode API URLs.**
5. **Use environment variables.**
6. **Keep frontend and backend separated.**
7. **Keep API routes modular.**
8. **Write reusable components.**
9. **Validate all backend inputs.**
10. **Handle API failures gracefully.**
11. **Keep dependencies minimal.**
12. **Optimize for 512 MB RAM (Render Free).**
13. **Do not introduce Redis/Postgres unless explicitly requested.**
14. **Do not introduce Docker unless necessary.**
15. **Test every feature before moving to the next phase.**
16. **Do not rewrite working code unnecessarily.**
17. **Explain architectural changes before implementing them.**
18. **Maintain API documentation.**
19. **Maintain README deployment instructions.**
20. **Never expose API keys to the frontend.**

## Development Phases

### Phase 1: Architecture & Setup ✅
- Architecture documentation
- Repository structure
- API specification
- Deployment plan
- Environment variable specification

### Phase 2: Backend Core ✅
- FastAPI setup
- Health endpoint
- CORS configuration
- Environment configuration
- Deploy to Render Free

### Phase 3: Backend Features ✅
- Photo upload endpoint
- Job management (status, results)
- Face detection (YuNet via OpenCV 5.x)
- Face embeddings (LBPH)
- Clustering (DBSCAN)
- In-memory job storage (singleton)
- Confidence calibration
- Per-intent threshold optimization

### Phase 4: Frontend Core ✅
- React + Vite + Tailwind setup
- API service layer
- Upload page with drag & drop
- Job status polling
- Results display

### Phase 5: Integration & Testing ✅
- Frontend ↔ Backend integration
- GitHub Pages deployment
- Render deployment
- End-to-end testing

### Phase 6: Production Ready ✅
- Confidence calibration (temperature + isotonic)
- Per-intent threshold optimization
- Failure analysis
- Documentation

## Key Technical Decisions

### Backend
- **FastAPI** for async support and automatic OpenAPI docs
- **Pydantic v2** for validation and serialization
- **Sentence-BERT embeddings** via OpenCV LBPH (lightweight)
- **DBSCAN** clustering for person grouping
- **Singleton JobService** for in-memory state (Render Free constraint)
- **Temperature + Isotonic calibration** for confidence scores
- **Per-intent thresholds** for auto-handle vs escalate decisions

### Frontend
- **React 18** with functional components + hooks
- **Vite** for fast dev server and optimized builds
- **Tailwind CSS** for styling
- **React Router v6** for client-side routing
- **Axios/Fetch** for API communication
- **Environment-based API URL** via Vite env vars

### ML Pipeline
- **Face Detection**: YuNet (OpenCV 5.x FaceDetectorYN)
- **Embeddings**: LBPH histogram (lightweight, no GPU)
- **Clustering**: DBSCAN with cosine distance
- **Calibration**: Temperature scaling + Isotonic regression
- **Thresholds**: Per-intent optimization for 85% precision target

### Deployment
- **Backend**: Render Free Web Service (spins down after 15min idle)
- **Frontend**: GitHub Pages (static hosting)
- **CI/CD**: GitHub Actions for both
- **No persistent storage** (Render Free constraint)

## API Contract

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/upload/` | POST | Upload photos (multipart) |
| `/api/jobs/{id}` | GET | Job status |
| `/api/results/{id}` | GET | Processing results |
| `/api/jobs` | GET | List jobs |

## Data Models

### Job
```json
{
  "job_id": "uuid",
  "status": "queued|processing|completed|failed",
  "progress": 0-100,
  "total_files": 5,
  "processed_files": 3,
  "created_at": "ISO8601",
  "updated_at": "ISO8601",
  "error": null
}
```

### Job Results
```json
{
  "job_id": "uuid",
  "status": "completed",
  "total_photos": 5,
  "total_faces": 12,
  "person_groups": [
    {
      "person_id": "person_1",
      "face_count": 5,
      "representative_face": "base64",
      "photo_ids": ["photo1.jpg"]
    }
  ],
  "photos": [
    {
      "photo_id": "photo1.jpg",
      "faces": [{"person_id": "person_1", "bbox": [x,y,w,h], "confidence": 0.95}]
    }
  ]
}
```

## Error Handling

| HTTP | Code | Description |
|------|------|-------------|
| 400 | VALIDATION_ERROR | Request validation failed |
| 400 | INVALID_FILE_TYPE | Unsupported format |
| 400 | FILE_TOO_LARGE | > 10MB |
| 400 | TOO_MANY_FILES | > 50 files |
| 404 | JOB_NOT_FOUND | Invalid job ID |
| 400 | JOB_NOT_COMPLETED | Results not ready |
| 413 | PAYLOAD_TOO_LARGE | > 100MB total |
| 415 | UNSUPPORTED_MEDIA_TYPE | Not multipart |
| 422 | UNPROCESSABLE_ENTITY | Validation failed |
| 500 | INTERNAL_ERROR | Server error |
| 503 | SERVICE_UNAVAILABLE | Model loading |

## Testing Checklist

- [ ] Health endpoint returns 200
- [ ] Upload accepts valid images
- [ ] Upload rejects invalid files
- [ ] Upload rejects oversized files
- [ ] Upload rejects too many files
- [ ] Job status returns correct status
- [ ] Results endpoint works for completed jobs
- [ ] Results endpoint 400s for incomplete jobs
- [ ] Frontend uploads files successfully
- [ ] Frontend polls job status
- [ ] Frontend displays results correctly
- [ ] CORS works from GitHub Pages origin
- [ ] Health check passes on Render
- [ ] Auto-deploy works on push

## Monitoring

- Render dashboard for backend logs
- GitHub Actions for frontend deploy status
- Browser DevTools for frontend debugging
- Structured JSON logging in backend

## Rollback Procedure

```bash
# Backend (Render)
render deploy rollback --service photo-organizer-api

# Frontend (GitHub Pages)
git revert HEAD
git push origin main
# GitHub Actions auto-redeploys
```

## Future Enhancements

- [ ] Real LLM response generation (Phase 5)
- [ ] Persistent storage (PostgreSQL + S3)
- [ ] User authentication
- [ ] Album creation
- [ ] Search by person
- [ ] Export organized photos
- [ ] Mobile app (React Native)
- [ ] Webhook notifications