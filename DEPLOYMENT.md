# Photo Organizer - Deployment Plan

## Overview

This document describes the deployment strategy for the Photo Organizer application with React frontend on GitHub Pages and FastAPI backend on Render Free tier.

## Architecture Summary

- **Frontend**: React + Vite + Tailwind → GitHub Pages (static hosting)
- **Backend**: FastAPI + Uvicorn → Render Free Web Service
- **Communication**: HTTPS REST API
- **Storage**: No persistent local filesystem (Render Free constraint)

---

## Prerequisites

- GitHub account
- Render account (free)
- Node.js 18+ and npm
- Python 3.11+
- Git

---

## Backend Deployment (Render Free)

### 1. Repository Setup

Ensure your backend code is in the `backend/` directory at the root of your repository.

### 2. Create Render Web Service

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click **New** → **Web Service**
3. Connect your GitHub repository
4. Configure:
   - **Name**: `photo-organizer-api` (or your preferred name)
   - **Region**: Choose closest to your users
   - **Branch**: `main`
   - **Root Directory**: `backend`
   - **Runtime**: `Docker` (or Python 3.11+)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: **Free**

### 3. Environment Variables

Add these in Render Dashboard → Environment:

| Key | Value | Notes |
|-----|-------|-------|
| `PORT` | (auto) | Render sets this automatically |
| `HOST` | `0.0.0.0` | Required for Render |
| `CORS_ORIGINS` | `https://yourusername.github.io` | Your GitHub Pages URL |
| `MAX_FILE_SIZE` | `10485760` | 10MB in bytes |
| `MAX_FILES_PER_JOB` | `50` | |
| `ALLOWED_EXTENSIONS` | `.jpg,.jpeg,.png,.webp` | |
| `MAX_IMAGE_DIMENSION` | `1920` | For processing |
| `FACE_DETECTION_MODEL` | `hog` | `hog` is faster, `cnn` more accurate |
| `CLUSTERING_EPS` | `0.5` | DBSCAN parameter |
| `CLUSTERING_MIN_SAMPLES` | `2` | DBSCAN parameter |
| `JOB_TTL_HOURS` | `24` | Auto-cleanup |
| `LOG_LEVEL` | `INFO` | |

### 4. Deploy

1. Click **Create Web Service**
2. Render will build and deploy automatically
3. Note your service URL: `https://your-app.onrender.com`

### 5. Verify Deployment

```bash
curl https://your-app.onrender.com/api/health
# Expected: {"status":"ok","version":"1.0.0"}
```

---

## Frontend Deployment (GitHub Pages)

### 1. Repository Setup

Ensure your frontend code is in the `frontend/` directory.

### 2. Configure Vite for GitHub Pages

In `frontend/vite.config.js`:

```javascript
export default defineConfig({
  base: '/photo-organizer/',  // Must match your repo name
  // ... rest of config
})
```

### 3. Environment Files

Create `frontend/.env.development`:
```env
VITE_API_URL=http://localhost:8000
```

Create `frontend/.env.production`:
```env
VITE_API_URL=https://your-app.onrender.com
```

### 5. GitHub Actions Workflow

Create `.github/workflows/frontend-deploy.yml`:

```yaml
name: Deploy Frontend to GitHub Pages

on:
  push:
    branches: [main]
    paths:
      - 'frontend/**'

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Node
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json
      
      - name: Install dependencies
        run: cd frontend && npm ci
      
      - name: Build
        run: cd frontend && npm run build
      
      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: frontend/dist

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

### 6. Enable GitHub Pages

1. Go to Repository Settings → Pages
2. Source: **GitHub Actions**
3. Custom domain (optional): Configure if needed

### 6. Verify Deployment

Visit: `https://yourusername.github.io/photo-organizer/`

---

## Local Development

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # Edit with local values
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Opens http://localhost:5173
```

### Full Stack Local

1. Start backend: `cd backend && uvicorn app.main:app --reload`
2. Start frontend: `cd frontend && npm run dev`
3. Frontend at `http://localhost:5173` proxies to `http://localhost:8000`

---

## Environment Variable Reference

### Backend (`.env`)

```env
# Server
PORT=8000
HOST=0.0.0.0

# CORS - comma separated, no trailing slashes
CORS_ORIGINS=http://localhost:5173,https://yourusername.github.io

# File upload limits
MAX_FILE_SIZE=10485760
MAX_FILES_PER_JOB=50
ALLOWED_EXTENSIONS=.jpg,.jpeg,.png,.webp

# Processing
MAX_IMAGE_DIMENSION=1920
FACE_DETECTION_MODEL=hog
CLUSTERING_EPS=0.5
CLUSTERING_MIN_SAMPLES=2

# Job management
JOB_TTL_HOURS=24

# Logging
LOG_LEVEL=INFO
```

### Frontend (`.env.development` / `.env.production`)

```env
# Development
VITE_API_URL=http://localhost:8000

# Production
VITE_API_URL=https://your-app.onrender.com
```

---

## CI/CD Pipeline

### Backend (Render)
- Automatic on push to `main` branch
- Render handles build + deploy
- Zero config needed after initial setup

### Frontend (GitHub Actions → GitHub Pages)
- Triggered on push to `main` with frontend changes
- Builds with Vite
- Deploys to `gh-pages` branch
- Available at `https://username.github.io/repo-name/`

---

## Testing Deployment

### Backend Health Check
```bash
curl https://your-app.onrender.com/api/health
```

### Frontend Access
```bash
# Open in browser
open https://yourusername.github.io/photo-organizer/
```

### Integration Test
1. Open frontend
2. Upload test images
3. Verify job creation and status polling
5. Verify results display

---

## Rollback Procedure

### Backend (Render)
1. Go to Render Dashboard → Service → Deploys
2. Click **Rollback** on previous successful deploy

### Frontend (GitHub Pages)
1. Go to Actions → Select failed run
2. Re-run previous successful workflow
6. Or: `git revert <commit>` and push

---

## Monitoring & Debugging

### Render Logs
- Dashboard → Service → Logs
- Real-time streaming available

### GitHub Actions Logs
- Actions tab → Workflow run details

### Common Issues

| Issue | Solution |
|-------|----------|
| CORS errors | Check `CORS_ORIGINS` matches exactly (no trailing slash) |
| 503 on Render | Free tier spins down; first request wakes it (30s delay) |
| Large file upload fails | Check `MAX_FILE_SIZE` and Render body limits |
| Model loading OOM | Use `hog` model, reduce `MAX_IMAGE_DIMENSION` |

---

## Cost Summary

| Component | Tier | Cost |
|-----------|------|------|
| Render Backend | Free | $0/month |
| GitHub Pages | Free | $0/month |
| GitHub Actions | Free (public repo) | $0/month |
| **Total** | | **$0/month** |

---

## Rollback Commands

```bash
# Backend - Render CLI
render deploy rollback --service photo-organizer-api

# Frontend - Manual
git revert HEAD
git push origin main
# GitHub Actions auto-redeploys
```

---

## Security Checklist

- [ ] No secrets in frontend code
- [ ] CORS restricted to GitHub Pages domain
- [ ] File type validation on backend
- [ ] File size limits enforced
- [ ] No secrets in repository
- [ ] HTTPS enforced (Render + GitHub Pages provide this)
- [ ] Environment variables used for all config

---

## Scaling Considerations (Future)

When outgrowing Free tier:
1. **Render Starter** ($7/mo): 512MB RAM, no spin-down
2. **Render Standard** ($25/mo): 1GB RAM, custom domains
3. **External Storage**: S3 for original images
6. **Database**: PostgreSQL for job metadata
7. **Queue**: Redis + Celery for background jobs

---

*Last Updated: 2024*
*Version: 1.0*