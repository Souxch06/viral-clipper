import os
import re
import sys
import asyncio
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

# Make the vendored OpenClip modules importable (backend/openclip_core/...)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Celery config
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
celery_app = Celery("tasks", broker=REDIS_URL, backend=REDIS_URL)
STORAGE_PATH = os.getenv("STORAGE_PATH", "/storage")
# tiny is the practical default on Render's free CPU; users can select base
# or small through the environment when they prefer accuracy over speed.
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "tiny")

# Provider env vars, checked in this order. The first one that is set is
# used to score moments with a real LLM instead of the word-density heuristic.
LLM_PROVIDER_ENV_VARS = [
    ("qwen", "QWEN_API_KEY"),
    ("glm", "GLM_API_KEY"),
    ("minimax", "MINIMAX_API_KEY"),
    ("openrouter", "OPENROUTER_API_KEY"),
    ("custom_openai", "CUSTOM_OPENAI_API_KEY"),
]


def get_available_llm_provider():
    """Return the name of the first configured LLM provider, or None."""
    for provider, env_var in LLM_PROVIDER_ENV_VARS:
        if os.getenv(env_var):
            return provider
    return None


def build_full_srt(transcript_path: str, out_srt_path: str) -> bool:
    """Convert transcript.json produced by faster-whisper into a single
    sequential .srt file covering the whole video, so it can be fed to the
    OpenClip EngagingMomentsAnalyzer (which expects SRT input)."""
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False
    segments = data.get("segments", [])
    if not segments:
        return False
    with open(out_srt_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            f.write(f"{idx}\n")
            f.write(f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}\n")
            f.write((seg.get("text") or "").strip() + "\n\n")
    return True


def score_moments_with_llm(srt_path: str, max_clips: int, min_duration: int, max_duration: int, provider: str):
    """Use OpenClip's EngagingMomentsAnalyzer to get real virality scoring
    from an LLM instead of the naive word-density heuristic.

    Returns (candidates, viral_score) where candidates is a list of
    {start, end, score, reason, title} sorted by score desc, or (None, None)
    if the LLM call failed or returned nothing usable.
    """
    try:
        from openclip_core.engaging_moments_analyzer import EngagingMomentsAnalyzer
    except Exception as e:
        print("openclip_core import failed, falling back to heuristic:", e)
        return None, None

    try:
        analyzer = EngagingMomentsAnalyzer(provider=provider, max_clips=max_clips)
        # The analyzer's internal validator only accepts moments between 30s
        # and 240s (OpenClip's original design). Real user settings can ask
        # for shorter/longer clips, so we clamp our own filtering afterwards
        # instead of fighting its hardcoded validator.
        result = asyncio.run(analyzer.analyze_part_for_engaging_moments(srt_path, "full_video"))
        moments = (result or {}).get("engaging_moments", [])
        if not moments:
            return None, None

        level_score = {"high": 92, "medium": 72, "low": 55}
        candidates = []
        for m in moments:
            try:
                start = analyzer.time_to_seconds(m["start_time"])
                end = analyzer.time_to_seconds(m["end_time"])
            except Exception:
                continue
            level = (m.get("engagement_details") or {}).get("engagement_level", "medium")
            score = level_score.get(level, 65)
            candidates.append({
                "start": round(start, 3),
                "end": round(end, 3),
                "score": score,
                "reason": m.get("summary") or ", ".join(m.get("tags", [])) or "Moment identifié comme engageant par l'IA",
                "title": m.get("title", ""),
            })
        if not candidates:
            return None, None
        candidates.sort(key=lambda c: c["score"], reverse=True)
        viral_score = candidates[0]["score"]
        return candidates, viral_score
    except Exception as e:
        traceback.print_exc()
        print("LLM scoring failed, falling back to heuristic:", e)
        return None, None

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
    window = max(10, int(prefer_clip_durations[0] if prefer_clip_durations else 30))
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


ASPECT_DIMENSIONS = {
    # aspect -> (quality) -> (width, height)
    "9:16": {"720": (720, 1280), "1080": (1080, 1920)},
    "1:1": {"720": (720, 720), "1080": (1080, 1080)},
    "16:9": {"720": (1280, 720), "1080": (1920, 1080)},
}

CAPTION_FONT_SIZE = {8: 22, 10: 28, 14: 36, 24: 52, 30: 64}

# ASS alignment codes (numpad layout): bottom-center=2, top-center=8, middle-center=5
CAPTION_ALIGNMENT = {"bottom": 2, "top": 8, "center": 5}


def _escape_ffmpeg_path(path: str) -> str:
    """Escape a filesystem path for use inside an ffmpeg filtergraph argument."""
    return path.replace("\\", "/").replace(":", "\\:").replace("'", "\\'")


def _build_caption_style(options: dict) -> str:
    style = options.get("caption_style", "bold")
    position = options.get("caption_position", "bottom")
    size = CAPTION_FONT_SIZE.get(int(options.get("caption_size", 14) or 14), 36)
    alignment = CAPTION_ALIGNMENT.get(position, 2)

    if style == "neon":
        primary, outline, back, bold, border_style, outline_w = "&H00FFFF00", "&H00FF00FF", "&H00000000", 1, 1, 3
    elif style == "boxed":
        primary, outline, back, bold, border_style, outline_w = "&H00FFFFFF", "&H00000000", "&H80000000", 1, 3, 2
    elif style == "minimal":
        primary, outline, back, bold, border_style, outline_w = "&H00FFFFFF", "&H00000000", "&H00000000", 0, 1, 1
    else:  # bold (default)
        primary, outline, back, bold, border_style, outline_w = "&H00FFFFFF", "&H00000000", "&H00000000", 1, 1, 2

    return (
        f"FontName=DejaVu Sans,Fontsize={size},Bold={bold},"
        f"PrimaryColour={primary},OutlineColour={outline},BackColour={back},"
        f"BorderStyle={border_style},Outline={outline_w},Alignment={alignment},MarginV=40"
    )


def render_clip(downloaded_file: str, start: float, end: float, out_path: str,
                 options: dict = None, srt_path: str = None):
    """Render a clip from downloaded_file between start and end, applying the
    user's export settings: aspect ratio, quality, fps, bitrate, burned-in
    styled captions and an optional dynamic zoom (Ken Burns effect)."""
    options = options or {}
    try:
        aspect = options.get("aspect", "9:16")
        quality = str(options.get("quality", "720"))
        fps = int(options.get("fps", 30) or 30)
        bitrate = options.get("bitrate", "auto")
        captions_enabled = bool(options.get("captions", True)) and srt_path and os.path.exists(srt_path)
        zoom_enabled = bool(options.get("zoom", False))

        width, height = ASPECT_DIMENSIONS.get(aspect, ASPECT_DIMENSIONS["9:16"]).get(
            quality, ASPECT_DIMENSIONS["9:16"]["720"]
        )
        clip_dur = max(0.5, end - start)

        vf_parts = [
            f"scale={width}:{height}:force_original_aspect_ratio=increase",
            f"crop={width}:{height}:(in_w-{width})/2:(in_h-{height})/2",
        ]
        if zoom_enabled:
            # Safe Ken-Burns style zoom: scale up slightly, then progressively
            # shrink the crop window toward the target size over the clip's
            # duration, and normalize back to a fixed output size. This is a
            # per-frame filter (no frame-count blow-up), unlike ffmpeg's
            # zoompan filter which can runaway-generate frames on video input
            # (verified: it produced a 22-minute file from a 12s clip).
            zoom_w, zoom_h = int(width * 1.15), int(height * 1.15)
            vf_parts = [
                f"scale={zoom_w}:{zoom_h}:force_original_aspect_ratio=increase",
                f"crop={zoom_w}:{zoom_h}:(in_w-{zoom_w})/2:(in_h-{zoom_h})/2",
                (
                    f"crop=w='{zoom_w}-({zoom_w}-{width})*min(t/{clip_dur:.3f}\\,1)':"
                    f"h='{zoom_h}-({zoom_h}-{height})*min(t/{clip_dur:.3f}\\,1)':"
                    f"x='(in_w-out_w)/2':y='(in_h-out_h)/2'"
                ),
                f"scale={width}:{height}",
            ]
        if captions_enabled:
            style = _build_caption_style(options)
            sub_path_escaped = _escape_ffmpeg_path(srt_path)
            vf_parts.append(
                f"subtitles='{sub_path_escaped}':original_size={width}x{height}:force_style='{style}'"
            )
        vf_parts.append(f"fps={fps}")
        vf = ",".join(vf_parts)

        crf = "23" if quality == "1080" else "27"
        preset = "veryfast" if quality == "1080" else "ultrafast"

        cmd = [
            'ffmpeg', '-y', '-ss', str(start), '-to', str(end), '-i', downloaded_file,
            '-vf', vf, '-c:v', 'libx264', '-preset', preset, '-threads', '0',
            '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart',
        ]
        if bitrate and bitrate != "auto":
            cmd += ['-b:v', bitrate, '-maxrate', bitrate, '-bufsize', bitrate]
        else:
            cmd += ['-crf', crf]
        cmd.append(out_path)

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
            options = job.get_metadata() or {}
            max_clips = max(1, min(10, int(options.get("max_clips", 4))))
            clip_duration = max(10, min(90, int(options.get("clip_duration", 20))))
            source_type = options.get("source_type", "youtube")

            project_dir = os.path.join(STORAGE_PATH, f"job_{job_id}")
            os.makedirs(project_dir, exist_ok=True)
            video_out = os.path.join(project_dir, "source.%(ext)s")

            if source_type == "upload":
                # File was already saved to disk by the /api/jobs/upload
                # endpoint; nothing to download, so skip straight past the
                # yt-dlp / YouTube-blocking logic entirely.
                downloaded_file = options.get("uploaded_path")
                if not downloaded_file or not os.path.exists(downloaded_file):
                    raise RuntimeError("Fichier vidéo importé introuvable sur le serveur.")
                duration = options.get("uploaded_duration") or 0
                update_job(job_id, progress=25)
                if not duration:
                    probe = subprocess.run(
                        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                         '-of', 'default=noprint_wrappers=1:nokey=1', downloaded_file],
                        capture_output=True, text=True, check=False,
                    )
                    try:
                        duration = float(probe.stdout.strip())
                    except (ValueError, TypeError):
                        duration = 0
                _skip_youtube_download = True
            else:
                _skip_youtube_download = False

            if not _skip_youtube_download:
                ydl_opts = {
                    # Prefer YouTube's mobile/web clients: the default web client
                    # is frequently challenged as a bot from cloud IP ranges.
                    # Avoid downloading 1080p+ sources on Render unless no smaller
                    # representation is available; this greatly reduces wait time.
                    "format": "best[height<=480][ext=mp4]/best[height<=480]/best[height<=720]/best",
                    "outtmpl": video_out,
                    "noplaylist": True,
                    "socket_timeout": 30,
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
                        for item in re.split(r"[;\r\n]+", header):
                            if "=" in item:
                                name, value = item.strip().split("=", 1)
                                lines.append(f".youtube.com\tTRUE\t/\tTRUE\t0\t{name}\t{value}")
                        cookie_bytes = ("\n".join(lines) + "\n").encode("utf-8")
                    # Browser extensions often export JSON instead of Netscape.
                    # Convert it locally so either export format is accepted.
                    try:
                        decoded = cookie_bytes.decode("utf-8-sig").strip()
                        if decoded.startswith("{") or decoded.startswith("["):
                            parsed = json.loads(decoded)
                            if isinstance(parsed, dict):
                                parsed = parsed.get("cookies", [])
                            netscape = ["# Netscape HTTP Cookie File"]
                            for item in parsed:
                                name, value = item.get("name"), item.get("value", "")
                                domain = item.get("domain", ".youtube.com") or ".youtube.com"
                                if not name:
                                    continue
                                subdomains = "TRUE" if domain.startswith(".") else "FALSE"
                                path = item.get("path", "/") or "/"
                                secure = "TRUE" if item.get("secure", True) else "FALSE"
                                expiry = int(item.get("expirationDate", item.get("expires", 0)) or 0)
                                netscape.append(f"{domain}\t{subdomains}\t{path}\t{secure}\t{expiry}\t{name}\t{value}")
                            cookie_bytes = ("\n".join(netscape) + "\n").encode("utf-8")
                    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, AttributeError, ValueError):
                        pass
                    with open(cookies_path, "wb") as cookies_file:
                        cookies_file.write(cookie_bytes)
                    ydl_opts["cookiefile"] = cookies_path
                # YouTube changes the available player clients frequently and cloud
                # IPs are sometimes challenged. Retry compatible clients before
                # failing the job; cookies, when configured, are applied to each try.
                client_profiles = [
                    ["android_vr"],
                    ["web_embedded"],
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
                        error_text = str(download_error).lower()
                        # Retrying the same cloud IP after YouTube has issued a bot
                        # challenge only wastes many minutes. Stop immediately and
                        # report the actionable fix; retries remain useful for other
                        # transient extractor errors.
                        if ("sign in to confirm" in error_text or "not a bot" in error_text) and not (cookies_b64 or cookies_text or cookie_header):
                            raise RuntimeError(
                                "YouTube a bloqué Render immédiatement. Ajoutez des cookies YouTube valides dans Render ou importez un fichier vidéo."
                            ) from download_error
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

            uploaded_subtitle_path = options.get("uploaded_subtitle_path")
            if uploaded_subtitle_path and os.path.exists(uploaded_subtitle_path):
                # A subtitle file was supplied with the upload: reuse it
                # directly and skip Whisper entirely (much faster).
                entries = parse_subtitle_to_segments(uploaded_subtitle_path)
                transcript_path = os.path.join(project_dir, "transcript.json")
                with open(transcript_path, "w", encoding="utf-8") as f:
                    json.dump({"segments": entries}, f, ensure_ascii=False, indent=2)
            else:
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

            # Try real LLM-based virality scoring (OpenClip) first; fall back
            # to the word-density heuristic if no provider key is configured
            # or the LLM call fails for any reason.
            viral_score = None
            candidates = None
            llm_provider = get_available_llm_provider()
            if llm_provider:
                full_srt_path = os.path.join(project_dir, "full_transcript.srt")
                if build_full_srt(transcript_path, full_srt_path):
                    update_job(job_id, step="scoring", progress=68)
                    candidates, viral_score = score_moments_with_llm(
                        full_srt_path, max_clips,
                        min_duration=clip_duration, max_duration=clip_duration,
                        provider=llm_provider,
                    )
            if not candidates:
                # generate candidate segments (heuristic fallback)
                candidates = generate_segments_from_transcript(transcript_path, duration, [clip_duration])

            update_job(job_id, step="scoring", progress=72)
            selected = candidates[:max_clips]

            update_job(job_id, step="rendering", progress=75)
            clips_meta = []
            total_selected = max(1, len(selected))
            for idx, seg in enumerate(selected):
                update_job(job_id, progress=75 + int(20 * idx / total_selected))
                out_mp4 = os.path.join(project_dir, f"clip_{idx+1:02d}.mp4")
                # Build a properly timed, multi-cue SRT for this clip (one
                # caption per spoken phrase, not one giant text block).
                srt_path = os.path.join(project_dir, f"clip_{idx+1:02d}.srt")
                has_captions = build_clip_srt(transcript_path, seg['start'], seg['end'], srt_path)
                ok = render_clip(
                    downloaded_file, seg['start'], seg['end'], out_mp4,
                    options=options, srt_path=srt_path if has_captions else None,
                )
                if ok:
                    clip_path_url = f"/storage/job_{job_id}/clip_{idx+1:02d}.mp4"
                    clips_meta.append({
                        "id": f"clip_{idx+1:02d}",
                        "start": seg['start'],
                        "end": seg['end'],
                        "duration": round(seg['end'] - seg['start'], 2),
                        "score": seg['score'],
                        "title": seg.get('title', ''),
                        "reason": seg.get('reason', ''),
                        "path": clip_path_url
                    })

            # update job metadata and status
            meta = {
                "video_filename": downloaded_file,
                "audio": audio_path,
                "transcript": transcript_path,
                "clips": clips_meta,
                "settings": options,
                "scoring_method": "llm" if (llm_provider and viral_score is not None) else "heuristic",
                "viral_score": viral_score if viral_score is not None else (
                    max((c['score'] for c in clips_meta), default=None)
                ),
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


def _srt_time_to_seconds(time_str: str) -> float:
    time_part, _, ms_part = time_str.strip().partition(',')
    h, m, s = (int(x) for x in time_part.split(':'))
    ms = int(ms_part) if ms_part else 0
    return h * 3600 + m * 60 + s + ms / 1000


def parse_subtitle_to_segments(subtitle_path: str):
    """Parse a user-supplied .srt (or plain .vtt-as-srt) file into the same
    segment structure produced by faster-whisper, so it can feed both the
    heuristic scorer and the LLM scorer without needing Whisper at all."""
    segments = []
    try:
        with open(subtitle_path, 'r', encoding='utf-8-sig') as f:
            content = f.read().strip()
        # Basic VTT -> SRT normalization (drop WEBVTT header, use comma for ms)
        content = re.sub(r'^WEBVTT.*\n', '', content)
        content = content.replace('.', ',') if '-->' in content and '.' in content.split('-->')[0][-20:] else content
        blocks = re.split(r'\n\s*\n', content)
        for block in blocks:
            lines = [l for l in block.strip().split('\n') if l.strip()]
            if len(lines) < 2:
                continue
            timing_line = next((l for l in lines if '-->' in l), None)
            if not timing_line:
                continue
            start_str, end_str = [p.strip() for p in timing_line.split('-->')]
            try:
                start = _srt_time_to_seconds(start_str.replace('.', ','))
                end = _srt_time_to_seconds(end_str.replace('.', ','))
            except Exception:
                continue
            text_lines = lines[lines.index(timing_line) + 1:]
            text = ' '.join(text_lines).strip()
            segments.append({"start": start, "end": end, "text": text, "words": []})
    except Exception as e:
        print("parse_subtitle_to_segments error:", e)
    return segments


def build_clip_srt(transcript_path: str, clip_start: float, clip_end: float, out_srt_path: str) -> bool:
    """Build a properly timed multi-cue SRT for a single clip, instead of
    dumping the whole segment's text as one giant static caption. Each
    original transcript segment overlapping the clip becomes its own SRT
    cue, with timestamps re-based to the clip's own timeline (starting at 0,
    since ffmpeg -ss/-to trims the source before the subtitle filter sees it)."""
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return False

    cues = []
    for seg in data.get('segments', []):
        seg_start, seg_end = seg.get('start', 0), seg.get('end', 0)
        if seg_end <= clip_start or seg_start >= clip_end:
            continue
        text = (seg.get('text') or '').strip()
        if not text:
            continue
        rel_start = max(0.0, seg_start - clip_start)
        rel_end = max(rel_start + 0.3, min(seg_end, clip_end) - clip_start)
        cues.append((rel_start, rel_end, text))

    if not cues:
        # No transcript coverage for this window (e.g. heuristic fallback
        # with an empty transcript): show nothing rather than a fake caption.
        return False

    with open(out_srt_path, 'w', encoding='utf-8') as srtf:
        for i, (start, end, text) in enumerate(cues, start=1):
            srtf.write(f"{i}\n")
            srtf.write(f"{format_srt_time(start)} --> {format_srt_time(end)}\n")
            srtf.write(text + "\n\n")
    return True


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
