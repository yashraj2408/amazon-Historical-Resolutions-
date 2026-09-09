# Photo Organizer - Environment Variables Specification

## Overview

This document specifies all environment variables used across the application. Variables are separated by service (backend/frontend) and environment (development/production).

---

## Backend Environment Variables

### Required Variables

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `PORT` | Server port (Render sets automatically) | `8000` | Yes (local) |
| `HOST` | Bind address | `0.0.0.0` | Yes |
| `CORS_ORIGINS` | Comma-separated allowed origins | `https://user.github.io,http://localhost:5173` | Yes |

### Optional Variables (with defaults)

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_FILE_SIZE` | `10485760` | Max file size in bytes (10MB) |
| `MAX_FILES_PER_JOB` | `50` | Maximum files per upload job |
| `ALLOWED_EXTENSIONS` | `.jpg,.jpeg,.png,.webp` | Comma-separated extensions |
| `MAX_IMAGE_DIMENSION` | `1920` | Max width/height for processing |
| `FACE_DETECTION_MODEL` | `hog` | `hog` (fast) or `cnn` (accurate) |
| `CLUSTERING_EPS` | `0.5` | DBSCAN epsilon parameter |
| `CLUSTERING_MIN_SAMPLES` | `2` | DBSCAN min samples |
| `JOB_TTL_HOURS` | `24` | Job retention before cleanup |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Render-Specific (Auto-set)

| Variable | Description |
|----------|-------------|
| `PORT` | Set by Render (do not override) |
| `RENDER` | Set to `true` by Render |

### Local Development Only

```env
# .env.local (not committed)
PORT=8000
HOST=0.0.0.0
CORS_ORIGINS=http://localhost:5173
LOG_LEVEL=DEBUG
```

---

## Frontend Environment Variables

### Build-Time Variables (Vite)

Vite exposes variables prefixed with `VITE_` to the client bundle.

| Variable | Development | Production | Description |
|----------|-------------|------------|-------------|
| `VITE_API_URL` | `http://localhost:8000` | `https://your-app.onrender.com` | Backend API base URL |

### Files

**`.env.development`** (committed):
```env
VITE_API_URL=http://localhost:8000
```

**`.env.production`** (committed):
```env
VITE_API_URL=https://photo-organizer-api.onrender.com
```

**`.env.local`** (gitignored, for local overrides):
```env
VITE_API_URL=http://localhost:8000
```

### Access in Code

```javascript
// src/services/api.js
const API_URL = import.meta.env.VITE_API_URL;

// Usage
const response = await fetch(`${API_URL}/api/health`);
```

---

## Environment Variable Validation

### Backend Startup Validation

The backend validates required variables on startup:

```python
# app/config.py
class Settings(BaseSettings):
    port: int = Field(default=8000, env="PORT")
    host: str = Field(default="0.0.0.0", env="HOST")
    cors_origins: List[str] = Field(env="CORS_ORIGINS")
    
    # Optional with defaults
    max_file_size: int = Field(default=10485760, env="MAX_FILE_SIZE")
    max_files_per_job: int = Field(default=50, env="MAX_FILES_PER_JOB")
    # ... etc
    
    class Config:
        env_file = ".env"
        case_sensitive = False
```

### Validation Rules

| Variable | Validation |
|----------|------------|
| `CORS_ORIGINS` | Must be valid URLs, no trailing slashes |
| `MAX_FILE_SIZE` | Positive integer, max 100MB |
| `MAX_FILES_PER_JOB` | Integer 1-100 |
| `MAX_IMAGE_DIMENSION` | Integer 100-4096 |
| `CLUSTERING_EPS` | Float 0.1-1.0 |
| `CLUSTERING_MIN_SAMPLES` | Integer 1-10 |
| `JOB_TTL_HOURS` | Integer 1-168 (1 week max) |
| `LOG_LEVEL` | One of: DEBUG, INFO, WARNING, ERROR |

---

## Render Dashboard Configuration

### Web Service Environment Variables

Set these in Render Dashboard → Environment:

```
PORT                    # Auto-set by Render
HOST                    0.0.0.0
CORS_ORIGINS            https://yourusername.github.io
MAX_FILE_SIZE           10485760
MAX_FILES_PER_JOB       50
ALLOWED_EXTENSIONS      .jpg,.jpeg,.png,.webp
MAX_IMAGE_DIMENSION     1920
FACE_DETECTION_MODEL    hog
CLUSTERING_EPS          0.5
CLUSTERING_MIN_SAMPLES  2
JOB_TTL_HOURS           24
LOG_LEVEL               INFO
```

### Adding Secrets (if needed later)

For future features requiring secrets (API keys, etc.):

1. Go to Render Dashboard → Environment
2. Add as **Secret** (encrypted, not visible in logs)
3. Reference in code: `os.getenv("SECRET_NAME")`

---

## Local Development Setup

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with local values
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.development
npm run dev
```

### Docker Compose (Optional)

```yaml
# docker-compose.yml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - PORT=8000
      - HOST=0.0.0.0
      - CORS_ORIGINS=http://localhost:5173
      - LOG_LEVEL=DEBUG
    volumes:
      - ./backend:/app
    command: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

  frontend:
    build: ./frontend
    ports:
      - "5173:5173"
    environment:
      - VITE_API_URL=http://localhost:8000
    volumes:
      - ./frontend:/app
      - /app/node_modules
    command: npm run dev -- --host 0.0.0.0
```

---

## Validation Script

```python
# scripts/validate_env.py
import os
from urllib.parse import urlparse

def validate_env():
    errors = []
    
    # Required
    cors = os.getenv("CORS_ORIGINS", "")
    if not cors:
        errors.append("CORS_ORIGINS is required")
    else:
        for origin in cors.split(","):
            origin = origin.strip()
            if not origin:
                continue
            parsed = urlparse(origin)
            if not parsed.scheme or not parsed.netloc:
                errors.append(f"Invalid CORS origin: {origin}")
            if origin.endswith("/"):
                errors.append(f"CORS origin should not have trailing slash: {origin}")
    
    # Optional numeric validations
    numeric_vars = {
        "MAX_FILE_SIZE": (1, 100_000_000),
        "MAX_FILES_PER_JOB": (1, 100),
        "MAX_IMAGE_DIMENSION": (100, 4096),
        "CLUSTERING_EPS": (0.1, 1.0),
        "CLUSTERING_MIN_SAMPLES": (1, 10),
        "JOB_TTL_HOURS": (1, 168),
    }
    
    for var, (min_val, max_val) in numeric_vars.items():
        val = os.getenv(var)
        if val:
            try:
                num = float(var) if var == "CLUSTERING_EPS" else int(val)
                if not (min_val <= num <= max_val):
                    errors.append(f"{var} must be between {min_val} and {max_val}")
            except ValueError:
                errors.append(f"{var} must be a number")
    
    if errors:
        print("❌ Environment validation failed:")
        for err in errors:
            print(f"  - {err}")
        return False
    
    print("✅ Environment validation passed")
    return True

if __name__ == "__main__":
    exit(0 if validate_env() else 1)
```

---

## Security Notes

1. **Never commit `.env.local` or `.env.production` with real secrets**
2. **Never put API keys in frontend `.env` files** (they're bundled)
3. **Use Render Secret Files** for sensitive backend config
3. **Rotate secrets regularly** if compromised
4. **Audit `.gitignore`** includes all `.env*` files except `.env.example`

---

## Migration Guide

### Adding New Variables

1. Add to `.env.example` (backend) or `.env.example` (frontend)
2. Add validation in `config.py` or Vite config
3. Update this document
4. Add to Render Dashboard / GitHub Actions secrets
5. Deploy

---

*Last Updated: 2024*
*Version: 1.0*