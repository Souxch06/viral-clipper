import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlmodel import select
from app.database import init_db, get_session, engine
from app.models import Job
from app.schemas import CreateJob, JobStatus
from app.tasks import download_and_transcribe_task
from sqlalchemy.orm import Session
from datetime import datetime

app = FastAPI(title="Viral Clipper Backend (MVP)")

@app.on_event("startup")
def on_startup():
    init_db()

@app.post("/api/jobs/analyze", response_model=dict)
def create_job(payload: CreateJob):
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

        # enqueue celery task
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
