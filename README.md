# Photo Organizer

A full-stack photo organizer application that uses AI face detection to automatically organize photos by person.

## Architecture

- **Frontend**: React + Vite + Tailwind CSS → GitHub Pages
- **Backend**: Python FastAPI → Render Free Web Service
- **AI/ML**: OpenCV (YuNet face detector) + LBPH embeddings + DBSCAN clustering

## Features

- 📸 Drag & drop photo upload (JPEG, PNG, WebP)
- 🤖 Automatic face detection using YuNet (OpenCV 5.x)
- 👥 Person grouping via DBSCAN clustering on LBPH embeddings
- 📊 Real-time job status polling
- 👥 Person group visualization with representative faces
- 📱 Responsive UI with Tailwind CSS
- 🔒 No persistent storage (Render Free tier compatible)

## Quick Start

### Prerequisites
- Node.js 18+
- Python 3.11+
- Git

### Local Development

1. **Backend**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Frontend**
```bash
cd frontend
npm install
npm run dev
```

3. **Access**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Deployment

### Backend (Render Free)
1. Connect GitHub repo to Render
2. Create Web Service:
   - Root Directory: `backend`
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - Plan: Free
2. Add Environment Variables in Render Dashboard
3. Enable Auto-Deploy

### Frontend (GitHub Pages)
1. Enable GitHub Pages in repo settings (source: GitHub Actions)
2. Push to main branch triggers GitHub Actions workflow
2. Available at `https://username.github.io/photo-organizer/`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/upload/` | Upload photos |
| GET | `/api/jobs/{job_id}` | Get job status |
| GET | `/api/results/{job_id}` | Get processed results |
| GET | `/api/jobs` | List jobs |

## Project Structure

```
photo-organizer/
├── .github/workflows/     # CI/CD workflows
├── backend/
│   ├── app/
│   │   ├── config.py
│   │   ├── main.py
│   │   ├── models/
│   │   ├── routes/
│   │   ├── services/
│   │   └── utils/
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── architecture.md
├── API_SPEC.md
├── DEPLOYMENT.md
├── ENV_SPEC.md
└── README.md
```

## Configuration

### Backend (.env)
```env
PORT=8000
HOST=0.0.0.0
CORS_ORIGINS=https://yourusername.github.io,http://localhost:5173
MAX_FILE_SIZE=10485760
MAX_FILES_PER_JOB=50
FACE_DETECTION_MODEL=hog
CLUSTERING_EPS=0.5
```

### Frontend
```env
# Development
VITE_API_URL=http://localhost:8000

# Production
VITE_API_URL=https://your-app.onrender.com
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18, Vite 5, Tailwind CSS 3, React Router 6, Axios |
| Backend | FastAPI, Uvicorn, Pydantic 2, Pydantic Settings |
| ML/AI | OpenCV 5 (YuNet), scikit-learn (DBSCAN) |
| Deployment | Render (backend), GitHub Pages (frontend) |

## Limitations (Free Tier)

- Render Free: 512MB RAM, spins down after 15min inactivity
- No persistent storage (in-memory job storage)
- GitHub Pages: Static only, no server-side rendering
- Max 50 files / 10MB per upload

## License

MIT