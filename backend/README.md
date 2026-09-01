# Viral Clipper — Backend MVP

Contenu:
- FastAPI backend (uvicorn)
- Celery worker (Redis broker)
- Task: download YouTube via yt-dlp → extract audio (ffmpeg) → transcribe via faster-whisper → store transcript.json

Prerequis:
- Docker & Docker Compose installed
- (Optionnel) GPU + drivers if you want faster `faster-whisper` inference
- Connexion internet (téléchargement YouTube & modèles whisper)

Structure:
- docker-compose.yml (racine)
- backend/ (contient Dockerfile & app/)

Quick start:
1. Copier les fichiers fournis (voir structure).
2. Créer dossier `storage` à la racine du projet:
   mkdir -p storage
3. Lancer:
   docker-compose up --build

Le backend sera accessible sur http://localhost:8000

Endpoints:
- POST /api/jobs/analyze  -> body JSON { "url": "https://www.youtube.com/..." } ; retourne {"job_id": N}
- GET /api/jobs/{job_id}/status -> retourne status et metadata

Notes:
- ffmpeg est installé dans l'image.
- Le modèle Whisper utilisé par faster-whisper est défini par la variable d'environnement WHISPER_MODEL (par défaut "small"). Sur CPU, `small` peut être lent pour les longues vidéos. Ajuster selon ta machine.
- Les fichiers téléchargés et résultats sont stockés dans ./storage/job_{id}/
- Pour production: sécuriser l'API, ajout d'auth, quotas, validation des droits d'usage des vidéos.
