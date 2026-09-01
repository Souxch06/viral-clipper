import os
from celery import Celery
from sqlmodel import Session
from app.database import engine
from app.models import Job
import yt_dlp
import subprocess
import json
import traceback
from datetime import datetime

# Celery config
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
celery_app = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)
STORAGE_PATH = os.getenv("STORAGE_PATH", "/storage")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# helper to update job
def update_job(job_id: int, **kwargs):
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            return
        for k, v in kwargs.items():
            setattr(job, k, v)
        job.updated_at = datetime.utcnow()
        session.add(job)
        session.commit()

@celery_app.task(bind=True, name="download_and_transcribe")
def download_and_transcribe_task(self, job_id: int):
    try:
        update_job(job_id, status="processing", step="downloading")
        with Session(engine) as session:
            job = session.get(Job, job_id)
            if not job:
                return

            project_dir = os.path.join(STORAGE_PATH, f"job_{job_id}")
            os.makedirs(project_dir, exist_ok=True)
            video_out = os.path.join(project_dir, "source.%(ext)s")
            ydl_opts = {
                "format": "best",
                "outtmpl": video_out,
                "quiet": True,
                "no_warnings": True
            }
            # download
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(job.original_url, download=True)
                # get downloaded filename
                downloaded_file = ydl.prepare_filename(info)

            update_job(job_id, step="extract_audio", storage_path=project_dir)

            # extract audio (wav 16k mono)
            audio_path = os.path.join(project_dir, "audio.wav")
            cmd = [
                "ffmpeg", "-y",
                "-i", downloaded_file,
                "-vn", "-ac", "1", "-ar", "16000",
                "-f", "wav", audio_path
            ]
            subprocess.run(cmd, check=True)

            update_job(job_id, step="transcribing")

            # transcribe with faster-whisper
            try:
                from faster_whisper import WhisperModel
                model = WhisperModel(WHISPER_MODEL, device="cpu")
                segments, info = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
                # build transcript structure
                transcript = []
                for seg in segments:
                    words = []
                    if hasattr(seg, "words") and seg.words:
                        for w in seg.words:
                            words.append({"word": w.word, "start": w.start, "end": w.end})
                    transcript.append({"start": seg.start, "end": seg.end, "text": seg.text, "words": words})
                transcript_path = os.path.join(project_dir, "transcript.json")
                with open(transcript_path, "w", encoding="utf-8") as f:
                    json.dump({"segments": transcript}, f, ensure_ascii=False, indent=2)
            except Exception as e:
                # faster-whisper not available or failed
                transcript_path = os.path.join(project_dir, "transcript_failed.txt")
                with open(transcript_path, "w") as f:
                    f.write("Transcription failed: " + str(e))
                raise

            # update job metadata and status
            meta = {
                "video_filename": downloaded_file,
                "audio": audio_path,
                "transcript": transcript_path,
            }
            with Session(engine) as session2:
                job2 = session2.get(Job, job_id)
                job2.status = "done"
                job2.step = "finished"
                job2.storage_path = project_dir
                job2.set_metadata(meta)
                session2.add(job2)
                session2.commit()

    except Exception as exc:
        traceback.print_exc()
        update_job(job_id, status="error", step=f"error: {str(exc)[:200]}")
        raise
