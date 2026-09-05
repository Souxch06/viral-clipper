import os
import re
from celery import Celery
from sqlmodel import Session
from app.database import engine
from app.models import Job
import yt_dlp
import subprocess
import json
import traceback
import time
from datetime import datetime

# Celery config
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
celery_app = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)
STORAGE_PATH = os.getenv("STORAGE_PATH", "/storage")
# tiny is the practical default on Render's free CPU; users can select base
# or small through the environment when they prefer accuracy over speed.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")

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


def generate_segments_from_transcript(transcript_path: str, video_duration: float, prefer_clip_durations=[15,30,45]):
    """Generate candidate segments from transcript.json using a simple heuristic.
    Returns list of segments: {start,end,score,reason}
    """
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return []

    words = []
    for seg in data.get('segments', []):
        for w in seg.get('words', []):
            words.append(w)

    # fallback: create sliding windows every 15s with length 30s
    candidates = []
    step = 15
    window = 30
    t = 0.0
    while t < max(0, video_duration - 1):
        start = t
        end = min(t + window, video_duration)
        # compute words/sec in window
        cnt = sum(1 for w in words if w['start'] >= start and w['end'] <= end)
        density = cnt / max(1.0, end - start)
        # simple heuristic score: density * 50 + presence of exclamations in text etc (not available), clamp
        score = int(min(95, density * 20 + 50))
        reason = f"Density {density:.2f} words/sec"
        candidates.append({"start": round(start,3), "end": round(end,3), "score": score, "reason": reason})
        t += step
    # sort by score desc
    candidates.sort(key=lambda x: x['score'], reverse=True)
    # deduplicate by overlap picking top 6
    selected = []
    for c in candidates:
        if len(selected) >= 6:
            break
        overlap = False
        for s in selected:
            if not (c['end'] <= s['start'] or c['start'] >= s['end']):
                overlap = True
                break
        if not overlap:
            selected.append(c)
    return selected


def render_clip(downloaded_file: str, start: float, end: float, out_path: str):
    """Render vertical 9:16 clip 1080x1920 from downloaded_file between start and end.
    This uses ffmpeg to scale height to 1920 then center-crop width 1080.
    """
    try:
        # Use ffmpeg to trim and re-encode with vertical crop
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start),
            '-to', str(end),
            '-i', downloaded_file,
            '-vf', "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920:(in_w-1080)/2:(in_h-1920)/2",
            '-c:v', 'libx264', '-crf', '23', '-preset', 'faster', '-threads', '0',
            '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart',
            out_path
        ]
        subprocess.run(cmd, check=True)
        return True
    except Exception as e:
        print('render_clip error', e)
        return False

@celery_app.task(bind=True, name="download_and_transcribe")
def download_and_transcribe_task(self, job_id: int):
    try:
        update_job(job_id, status="processing", step="downloading", progress=3)
        with Session(engine) as session:
            job = session.get(Job, job_id)
            if not job:
                return

            project_dir = os.path.join(STORAGE_PATH, f"job_{job_id}")
            os.makedirs(project_dir, exist_ok=True)
            video_out = os.path.join(project_dir, "source.%(ext)s")
            ydl_opts = {
                # Prefer YouTube's mobile/web clients: the default web client
                # is frequently challenged as a bot from cloud IP ranges.
                # Avoid downloading 1080p+ sources on Render unless no smaller
                # representation is available; this greatly reduces wait time.
                "format": "best[height<=720][ext=mp4]/best[height<=720]/best",
                "outtmpl": video_out,
                "noplaylist": True,
                "quiet": True,
                "no_warnings": True,
                "extractor_args": {
                    "youtube": {
                        "player_client": ["android", "web_safari"],
                    }
                },
            }
            # An optional base64 encoded cookies.txt can be supplied in Render
            # for videos that still require an authenticated YouTube session.
            # It is deliberately opt-in; public videos should work without it.
            cookies_b64 = os.getenv("YOUTUBE_COOKIES_B64")
            cookies_text = os.getenv("YOUTUBE_COOKIES_TEXT")
            cookie_header = os.getenv("YOUTUBE_COOKIE_HEADER")
            if cookies_b64 or cookies_text or cookie_header:
                import base64
                cookies_path = os.path.join(project_dir, "youtube-cookies.txt")
                if cookies_b64:
                    cookie_bytes = base64.b64decode(cookies_b64)
                elif cookies_text:
                    cookie_bytes = cookies_text.encode("utf-8")
                else:
                    # Also accept a copied browser Cookie header. This is
                    # useful when Android cannot export a cookies.txt file.
                    header = cookie_header.removeprefix("Cookie:").strip()
                    lines = ["# Netscape HTTP Cookie File"]
                    for item in re.split(r"[;\\r\\n]+", header):
                        if "=" in item:
                            name, value = item.strip().split("=", 1)
                            lines.append(f".youtube.com\\tTRUE\\t/\\tTRUE\\t0\\t{name}\\t{value}")
                    cookie_bytes = ("\\n".join(lines) + "\\n").encode("utf-8")
                with open(cookies_path, "wb") as cookies_file:
                    cookies_file.write(cookie_bytes)
                ydl_opts["cookiefile"] = cookies_path
            # YouTube changes the available player clients frequently and cloud
            # IPs are sometimes challenged. Retry compatible clients before
            # failing the job; cookies, when configured, are applied to each try.
            client_profiles = [
                ["android", "web_safari"],
                ["tv", "web_safari"],
                ["mweb", "web_safari"],
            ]
            last_download_error = None
            for clients in client_profiles:
                try:
                    attempt_opts = dict(ydl_opts)
                    attempt_opts["extractor_args"] = {"youtube": {"player_client": clients}}
                    last_progress_write = [0.0]
                    def download_progress(status):
                        now = time.monotonic()
                        if status.get("status") == "downloading":
                            total = status.get("total_bytes") or status.get("total_bytes_estimate")
                            current = status.get("downloaded_bytes", 0)
                            if total and (now - last_progress_write[0] >= 1.0):
                                last_progress_write[0] = now
                                update_job(job_id, progress=min(25, 3 + int(22 * current / total)))
                        elif status.get("status") == "finished":
                            update_job(job_id, progress=25)
                    attempt_opts["progress_hooks"] = [download_progress]
                    with yt_dlp.YoutubeDL(attempt_opts) as ydl:
                        info = ydl.extract_info(job.original_url, download=True)
                        downloaded_file = ydl.prepare_filename(info)
                        duration = info.get('duration', None) or 0
                    break
                except Exception as download_error:
                    last_download_error = download_error
                    # Remove a partial file before trying the next client.
                    for partial in os.listdir(project_dir):
                        if partial.startswith("source"):
                            try:
                                os.remove(os.path.join(project_dir, partial))
                            except OSError:
                                pass
            else:
                raise RuntimeError(
                    "YouTube bloque le téléchargement depuis Render. "
                    "Ajoutez YOUTUBE_COOKIES_B64 dans Render ou importez directement une vidéo. "
                    f"Détail: {str(last_download_error)[:240]}"
                )

            update_job(job_id, step="extract_audio", progress=30, storage_path=project_dir)

            # extract audio (wav 16k mono)
            audio_path = os.path.join(project_dir, "audio.wav")
            cmd = [
                "ffmpeg", "-y",
                "-i", downloaded_file,
                "-vn", "-ac", "1", "-ar", "16000",
                "-f", "wav", audio_path
            ]
            subprocess.run(cmd, check=True)

            update_job(job_id, step="transcribing", progress=40)

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
                # Android/Termux peut ne pas disposer d'une wheel faster-whisper.
                # On conserve un transcript vide afin de permettre au moteur FFmpeg
                # de produire des clips par fenêtres temporelles.
                if os.getenv("TERMUX_MODE", "0") != "1":
                    transcript_path = os.path.join(project_dir, "transcript_failed.txt")
                    with open(transcript_path, "w") as f:
                        f.write("Transcription failed: " + str(e))
                    raise
                transcript_path = os.path.join(project_dir, "transcript.json")
                with open(transcript_path, "w", encoding="utf-8") as f:
                    json.dump({"segments": []}, f)

            update_job(job_id, step="segmenting", progress=65)
            # generate candidate segments
            candidates = generate_segments_from_transcript(transcript_path, duration)

            update_job(job_id, step="scoring", progress=72)
            # for MVP we use the candidate score as-is
            selected = candidates[:4]

            update_job(job_id, step="rendering", progress=75)
            clips_meta = []
            total_selected = max(1, len(selected))
            for idx, seg in enumerate(selected):
                update_job(job_id, progress=75 + int(20 * idx / total_selected))
                out_mp4 = os.path.join(project_dir, f"clip_{idx+1:02d}.mp4")
                ok = render_clip(downloaded_file, seg['start'], seg['end'], out_mp4)
                if ok:
                    # create a simple srt
                    srt_path = os.path.join(project_dir, f"clip_{idx+1:02d}.srt")
                    with open(srt_path, 'w', encoding='utf-8') as srtf:
                        srtf.write("1\n")
                        srtf.write(f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n")
                        # try to pick transcript text inside the segment
                        txt = collect_text_between(transcript_path, seg['start'], seg['end'])
                        srtf.write(txt + "\n")

                    clip_path_url = f"/storage/job_{job_id}/clip_{idx+1:02d}.mp4"
                    clips_meta.append({
                        "id": f"clip_{idx+1:02d}",
                        "start": seg['start'],
                        "end": seg['end'],
                        "duration": round(seg['end'] - seg['start'], 2),
                        "score": seg['score'],
                        "reason": seg.get('reason',''),
                        "path": clip_path_url
                    })

            # update job metadata and status
            meta = {
                "video_filename": downloaded_file,
                "audio": audio_path,
                "transcript": transcript_path,
                "clips": clips_meta,
            }
            with Session(engine) as session2:
                job2 = session2.get(Job, job_id)
                job2.status = "done"
                job2.step = "finished"
                job2.progress = 100
                job2.storage_path = project_dir
                job2.set_metadata(meta)
                session2.add(job2)
                session2.commit()

    except Exception as exc:
        traceback.print_exc()
        update_job(job_id, status="error", step=f"error: {str(exc)[:200]}")
        raise


# utility functions

def format_srt_time(t: float) -> str:
    hrs = int(t // 3600)
    mins = int((t % 3600) // 60)
    secs = int(t % 60)
    ms = int((t - int(t)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def collect_text_between(transcript_path: str, start: float, end: float) -> str:
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return ""
    parts = []
    for seg in data.get('segments', []):
        if seg['end'] < start or seg['start'] > end:
            continue
        parts.append(seg.get('text',''))
    return ' '.join(parts).strip()
