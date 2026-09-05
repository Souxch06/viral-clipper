import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from sqlmodel import select
from app.database import init_db, get_session, engine
from app.models import Job
from app.schemas import CreateJob, JobStatus
from app.tasks import download_and_transcribe_task
from sqlalchemy.orm import Session
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Viral Clipper Backend (MVP)")

# CORS (for development; lock down in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STORAGE_PATH = os.getenv("STORAGE_PATH", "/storage")
# Mount storage to serve generated files (clips, previews)
if os.path.exists(STORAGE_PATH):
    app.mount("/storage", StaticFiles(directory=STORAGE_PATH), name="storage")

@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/", response_class=FileResponse)
def studio():
    """Serve the same studio when Render is opened directly or in the iframe."""
    return FileResponse(os.path.join(os.path.dirname(__file__), "termux_ui.html"))

@app.post("/api/jobs/analyze", response_model=dict)
def create_job(payload: CreateJob, background_tasks: BackgroundTasks):
    url = payload.url
    if not url:
        raise HTTPException(status_code=400, detail="Missing url")

    # Basic validation
    if "youtube.com" not in url and "youtu.be" not in url:
        raise HTTPException(status_code=400, detail="Only YouTube URLs supported in MVP")

    # create job
    from sqlmodel import Session
    with Session(engine) as session:
        job = Job(original_url=url, status="pending", step="queued")
        session.add(job)
        session.commit()
        session.refresh(job)

        # Termux n'a généralement ni Redis ni worker séparé : exécuter la tâche
        # dans le processus FastAPI évite une infrastructure serveur obligatoire.
        if os.getenv("TERMUX_MODE", "0") == "1":
            background_tasks.add_task(download_and_transcribe_task.run, job.id)
        else:
            download_and_transcribe_task.delay(job.id)
        return {"job_id": job.id}

@app.get("/api/jobs/{job_id}/status", response_model=dict)
def job_status(job_id: int):
    from sqlmodel import Session
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return {
            "id": job.id,
            "status": job.status,
            "step": job.step,
            "storage_path": job.storage_path,
            "metadata": job.get_metadata()
        }

@app.get("/api/jobs/{job_id}/clips")
def job_clips(job_id: int):
    """Return list of generated clips for a job (if any)."""
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        meta = job.get_metadata() or {}
        clips = meta.get("clips") or []
        # add full URLs for frontend
        base = os.getenv("PUBLIC_BASE_URL", None)
        if base:
            for c in clips:
                if c.get("path") and not c.get("download_url"):
                    c["download_url"] = base.rstrip('/') + c["path"]
                    c["preview_url"] = base.rstrip('/') + c["path"]
        else:
            # serve from same origin via /storage
            for c in clips:
                if c.get("path") and not c.get("download_url"):
                    c["download_url"] = c["path"]
                    c["preview_url"] = c["path"]
        return {"clips": clips}
