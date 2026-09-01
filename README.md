# Viral Clipper

AI Viral Clipper — MVP

This repository contains an MVP scaffold for a web application that generates short social clips from YouTube or local videos. The project is designed to run as a PWA on Android and includes a FastAPI backend with Celery workers to download videos (yt-dlp), extract audio (ffmpeg), transcribe (faster-whisper), and produce clip metadata.

Contents
- frontend/: Next.js PWA minimal (installable on Android)
- backend/: FastAPI app, Celery tasks, Dockerfile
- docker-compose.yml: orchestrates redis, backend, worker

Quick start (local)
1. Clone the repo
2. Create storage folder at project root:
   mkdir -p storage
3. Build and run with Docker Compose:
   docker-compose up --build

The backend will be available at http://localhost:8000

API endpoints (MVP)
- POST /api/jobs/analyze -> { "url": "https://www.youtube.com/..." } -> { "job_id": N }
- GET /api/jobs/{job_id}/status -> job status and metadata

Notes
- This is an MVP scaffold. After starting, the worker will download the YouTube video, extract audio, transcribe with faster-whisper, and store transcript.json under ./storage/job_{id}/
- Ensure you have sufficient resources for transcription (model size impacts CPU/GPU requirements).

Next steps
- Add segment generation, viral scoring (LLM) and clip rendering
- Integrate frontend to call backend endpoints and display clips
- Add authentication, quotas and production hardening
