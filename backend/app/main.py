import os
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
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
import shutil
import uuid

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

def studio_file():
    return FileResponse(os.path.join(os.path.dirname(__file__), "termux_ui.html"))

@app.get("/", response_class=FileResponse)
def studio():
    """Main landing page: the URL YouTube workflow."""
    return studio_file()

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
        job = Job(original_url=url, status="pending", step="queued", progress=2)
        job.set_metadata(payload.dict())
        session.add(job)
        session.commit()
        session.refresh(job)

        # Termux n'a généralement ni Redis ni worker séparé : exécuter la tâche
        # dans le processus FastAPI évite une infrastructure serveur obligatoire.
        # Render runs a single web process in the free setup; use FastAPI background
        # tasks by default instead of requiring a separate Redis/Celery worker.
        if os.getenv("TERMUX_MODE", "1") == "1":
            background_tasks.add_task(download_and_transcribe_task.run, job.id)
        else:
            download_and_transcribe_task.delay(job.id)
        return {"job_id": job.id}

@app.post("/api/jobs/upload", response_model=dict)
async def create_job_from_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    subtitle: UploadFile = File(None),
    max_clips: int = Form(4),
    clip_duration: int = Form(20),
    quality: str = Form("720"),
    captions: bool = Form(True),
    zoom: bool = Form(True),
    caption_style: str = Form("bold"),
    caption_position: str = Form("bottom"),
    caption_size: int = Form(8),
    aspect: str = Form("9:16"),
    fps: int = Form(30),
    bitrate: str = Form("auto"),
):
    """Accept a video file uploaded directly from the phone/PC, bypassing
    YouTube entirely. This is the fix for the 'Fichier vidéo' mode, which the
    frontend already calls but which previously had no matching backend route."""
    allowed_ext = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
    original_ext = os.path.splitext(file.filename or "")[1].lower() or ".mp4"
    if original_ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"Format vidéo non supporté: {original_ext}")

    from sqlmodel import Session
    with Session(engine) as session:
        job = Job(original_url=None, status="pending", step="queued", progress=2)
        session.add(job)
        session.commit()
        session.refresh(job)

        project_dir = os.path.join(STORAGE_PATH, f"job_{job.id}")
        os.makedirs(project_dir, exist_ok=True)

        # Save the uploaded video to disk
        video_path = os.path.join(project_dir, f"source{original_ext}")
        with open(video_path, "wb") as out:
            shutil.copyfileobj(file.file, out)

        # Save the optional subtitle file, if one was provided
        subtitle_path = None
        if subtitle is not None and subtitle.filename:
            sub_ext = os.path.splitext(subtitle.filename)[1].lower() or ".srt"
            subtitle_path = os.path.join(project_dir, f"uploaded_subtitle{sub_ext}")
            with open(subtitle_path, "wb") as out:
                shutil.copyfileobj(subtitle.file, out)

        payload = {
            "source_type": "upload",
            "uploaded_path": video_path,
            "uploaded_subtitle_path": subtitle_path,
            "max_clips": max_clips,
            "clip_duration": clip_duration,
            "quality": quality,
            "captions": captions,
            "zoom": zoom,
            "caption_style": caption_style,
            "caption_position": caption_position,
            "caption_size": caption_size,
            "aspect": aspect,
            "fps": fps,
            "bitrate": bitrate,
        }
        job.set_metadata(payload)
        job.storage_path = project_dir
        session.add(job)
        session.commit()
        session.refresh(job)

        if os.getenv("TERMUX_MODE", "1") == "1":
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
        step_progress = {
            "queued": 2,
            "downloading": 15,
            "extract_audio": 30,
            "transcribing": 50,
            "segmenting": 65,
            "scoring": 72,
            "rendering": 85,
            "finished": 100,
        }
        current_step = job.step or "queued"
        progress = 0 if job.status == "error" else (job.progress or step_progress.get(current_step, 5))
        if job.status == "done":
            progress = 100
        elapsed = max(0, (datetime.utcnow() - job.created_at).total_seconds())
        eta = None
        if 0 < progress < 100:
            eta = int(max(1, elapsed * (100 - progress) / progress))
        return {
            "id": job.id,
            "status": job.status,
            "step": job.step,
            "progress": progress,
            "elapsed_seconds": int(elapsed),
            "eta_seconds": eta,
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
