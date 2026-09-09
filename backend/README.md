# Photo Organizer - Backend

FastAPI backend for the Photo Organizer application.

## Quick Start

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your configuration
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Server runs at `http://localhost:8000`

## API Documentation

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Health Check: `http://localhost:8000/api/health`

## Project Structure

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI app entry point
│   ├── config.py         # Configuration management
│   ├── models/           # Pydantic models
│   ├── routes/           # API route modules
│   ├── services/         # Business logic
│   └── utils/            # Utilities
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Environment Variables

See `.env.example` for all available options. Copy to `.env` and modify:

```bash
cp .env.example .env
```

Key variables:
- `PORT` - Server port (default: 8000)
- `HOST` - Bind address (default: 0.0.0.0)
- `CORS_ORIGINS` - Comma-separated allowed origins
- `MAX_FILE_SIZE` - Max upload size in bytes (default: 10MB)
- `MAX_FILES_PER_JOB` - Max files per upload (default: 50)

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/upload` | Upload photos |
| GET | `/api/jobs/{job_id}` | Get job status |
| GET | `/api/results/{job_id}` | Get job results |
| GET | `/api/jobs` | List jobs (optional) |

## Deployment (Render Free)

1. Connect GitHub repo to Render
2. Create Web Service:
   - Root Directory: `backend`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Plan: Free
2. Add Environment Variables in Render Dashboard
3. Enable Auto-Deploy

## Testing

```bash
# Health check
curl http://localhost:8000/api/health

# Upload test
curl -X POST -F "files=@test.jpg" http://localhost:8000/api/upload

# Check job status
curl http://localhost:8000/api/jobs/{job_id}
```