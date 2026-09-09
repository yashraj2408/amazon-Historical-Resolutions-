# Photo Organizer Frontend

React + Vite + Tailwind CSS frontend for the Photo Organizer application.

## Quick Start

```bash
cd frontend
npm install
npm run dev
```

## Build for Production

```bash
npm run build
```

## Deploy to GitHub Pages

```bash
npm run deploy
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API base URL | `http://localhost:8000` |

## Project Structure

```
frontend/
├── public/              # Static assets
├── src/
│   ├── components/      # Reusable UI components
│   │   ├── upload/      # Upload-related components
│   │   ├── results/     # Results display components
│   │   └── layout/      # Layout components
│   ├── pages/           # Page components
│   │   ├── Upload.jsx   # Upload page
│   │   └── Results.jsx  # Results page
│   ├── services/        # API service layer
│   │   └── api.js       # API client
│   ├── App.jsx          # Main app component
│   ├── main.jsx         # Entry point
│   └── index.css        # Global styles with Tailwind
├── index.html
├── package.json
├── vite.config.js
├── tailwind.config.js
├── postcss.config.js
└── README.md
```

## Features

- Drag & drop photo upload
- Real-time job status polling
- Face detection results display
- Person grouping visualization
- Responsive design with Tailwind CSS
- Error handling and loading states

## Development

```bash
# Start dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Deploy to GitHub Pages
npm run deploy
```

## API Integration

The frontend communicates with the backend via REST API:
- `POST /api/upload/` - Upload photos
- `GET /api/jobs/:jobId` - Check job status
- `GET /api/results/:jobId` - Get processing results

The API base URL is configured via `VITE_API_URL` environment variable.