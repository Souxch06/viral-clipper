from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from openclip_core.clip_generator import ClipGenerator
from openclip_core.config import API_KEY_ENV_VARS, DEFAULT_LLM_PROVIDER
from openclip_core.cover_image_generator import COVER_COLORS, CoverImageGenerator
from openclip_core.editor.manifest import (
    discover_manifest_by_project_id,
    format_seconds_as_timecode,
    list_manifest_paths,
    load_manifest,
    parse_timecode_to_seconds,
    reconcile_manifest,
    save_manifest,
)
from openclip_core.editor.models import EditorClip, EditorManifest, SubtitleSegment, utc_now_iso
from openclip_core.subtitle_burner import SubtitleBurner, SubtitleStyleConfig
from openclip_core.title_adder import TitleAdder
from job_manager import JobManager

logger = logging.getLogger(__name__)


def _read_effective_subtitle_text(path: str | None) -> str:
    if not path:
        return ''
    subtitle_path = Path(path)
    if not subtitle_path.exists():
        return ''
    try:
        content = subtitle_path.read_text(encoding='utf-8')
    except Exception:
        return ''
    lines = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.isdigit() or '-->' in line:
            continue
        lines.append(line)
    return '\n'.join(lines).strip()


def _subtitle_segments_to_text(segments: list[dict[str, str]]) -> str:
    lines = [str(segment.get("text", "")).strip() for segment in segments]
    return "\n".join(line for line in lines if line).strip()


def _serialize_subtitle_segments(segments: list[Any]) -> list[dict[str, str]]:
    serialized: list[dict[str, str]] = []
    for index, segment in enumerate(segments, start=1):
        if hasattr(segment, "to_dict"):
            item = segment.to_dict()
        else:
            item = dict(segment or {})
        serialized.append(
            {
                "index": index,
                "start_time": str(item.get("start_time", "00:00:00,000")),
                "end_time": str(item.get("end_time", "00:00:00,500")),
                "text": str(item.get("text", "")),
            }
        )
    return serialized


def _write_subtitle_segments(path: Path, segments: list[dict[str, str]]) -> None:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{segment['start_time']} --> {segment['end_time']}",
                    segment["text"],
                ]
            )
        )
    path.write_text("\n\n".join(blocks).strip() + "\n", encoding="utf-8")


def _legacy_override_segments_for_clip(clip: EditorClip) -> list[dict[str, str]]:
    if not (clip.subtitle_recipe.override_text or "").strip():
        return []
    override_lines = [line.strip() for line in (clip.subtitle_recipe.override_text or "").splitlines() if line.strip()]
    timed_segments = _derive_subtitle_segments_for_bounds(clip)
    if timed_segments:
        remapped_segments: list[dict[str, str]] = []
        for index, segment in enumerate(timed_segments, start=1):
            if index - 1 < len(override_lines):
                text = override_lines[index - 1]
            else:
                text = segment["text"]
            remapped_segments.append(
                {
                    "index": index,
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "text": text,
                }
            )
        if len(override_lines) > len(remapped_segments):
            overflow = "\n".join(override_lines[len(remapped_segments):]).strip()
            if overflow:
                last_segment = remapped_segments[-1]
                last_segment["text"] = f"{last_segment['text']}\n{overflow}".strip()
        return remapped_segments

    clip_duration = max(parse_timecode_to_seconds(clip.end_time) - parse_timecode_to_seconds(clip.start_time), 0.5)
    return [
        {
            "index": 1,
            "start_time": "00:00:00,000",
            "end_time": format_seconds_as_timecode(clip_duration).replace(".", ","),
            "text": clip.subtitle_recipe.override_text or "",
        }
    ]


def _parse_subtitle_segments_from_path(path: str | None) -> list[dict[str, str]]:
    if not path:
        return []
    subtitle_path = Path(path)
    if not subtitle_path.exists():
        return []
    try:
        generator = ClipGenerator(output_dir=str(subtitle_path.parent))
        return [
            {
                "index": index,
                "start_time": str(segment.get("start_time", "00:00:00,000")),
                "end_time": str(segment.get("end_time", "00:00:00,500")),
                "text": str(segment.get("text", "")),
            }
            for index, segment in enumerate(generator._parse_srt_file(str(subtitle_path)), start=1)
        ]
    except Exception:
        return []


def _derive_subtitle_segments_for_bounds(clip: EditorClip) -> list[dict[str, str]]:
    source_subtitle_path = clip.metadata.get("source_subtitle_path")
    if not source_subtitle_path:
        return _parse_subtitle_segments_from_path(clip.asset_registry.subtitle_active)

    subtitle_path = Path(source_subtitle_path)
    if not subtitle_path.exists():
        return _parse_subtitle_segments_from_path(clip.asset_registry.subtitle_active)

    try:
        generator = ClipGenerator(output_dir=str(subtitle_path.parent))
        segments = generator._parse_srt_file(str(subtitle_path))
    except Exception:
        return _parse_subtitle_segments_from_path(clip.asset_registry.subtitle_active)

    clip_start = parse_timecode_to_seconds(clip.start_time)
    clip_end = parse_timecode_to_seconds(clip.end_time)
    speed = max(float(getattr(clip, "speed", 1.0) or 1.0), 0.001)
    derived_segments: list[dict[str, str]] = []
    for segment in segments:
        seg_start = generator._time_to_seconds_srt(str(segment.get("start_time")))
        seg_end = generator._time_to_seconds_srt(str(segment.get("end_time")))
        if seg_end > clip_start and seg_start < clip_end:
            new_start = max(0.0, seg_start - clip_start) / speed
            new_end = max(new_start + 0.1, (seg_end - clip_start) / speed)
            derived_segments.append(
                {
                    "index": len(derived_segments) + 1,
                    "start_time": generator._seconds_to_time_srt(new_start),
                    "end_time": generator._seconds_to_time_srt(new_end),
                    "text": str(segment.get("text", "")),
                }
            )
    return derived_segments


def _effective_subtitle_segments_for_clip(clip: EditorClip) -> list[dict[str, str]]:
    override_segments = _serialize_subtitle_segments(clip.subtitle_recipe.override_segments)
    if override_segments:
        return override_segments
    legacy_override_segments = _legacy_override_segments_for_clip(clip)
    if legacy_override_segments:
        return legacy_override_segments
    return _derive_subtitle_segments_for_bounds(clip)


def _derive_subtitle_text_for_bounds(clip: EditorClip) -> str:
    return _subtitle_segments_to_text(_effective_subtitle_segments_for_clip(clip))


def _remap_text_segments_onto_timings(
    text_segments: list[dict[str, str]],
    timed_segments: list[dict[str, str]],
    *,
    fill_from_timed_segments: bool = False,
) -> list[dict[str, str]]:
    if not timed_segments:
        return []

    remapped_segments: list[dict[str, str]] = []
    for index, segment in enumerate(timed_segments, start=1):
        if index - 1 < len(text_segments):
            text = str(text_segments[index - 1].get("text", "")).strip()
        elif fill_from_timed_segments:
            text = str(segment.get("text", "")).strip()
        else:
            text = ""
        remapped_segments.append(
            {
                "index": index,
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "text": text,
            }
        )

    if len(text_segments) > len(remapped_segments):
        overflow = "\n".join(
            str(segment.get("text", "")).strip()
            for segment in text_segments[len(remapped_segments):]
            if str(segment.get("text", "")).strip()
        ).strip()
        if overflow:
            remapped_segments[-1]["text"] = "\n".join(
                part for part in [remapped_segments[-1]["text"], overflow] if part
            ).strip()

    return remapped_segments


def _resolve_cover_color(value: Any, fallback_name: str) -> Any:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return tuple(int(channel) for channel in value)
    return COVER_COLORS.get(str(value), COVER_COLORS[fallback_name])


class EditorService:
    def __init__(self, projects_root: str | Path = "processed_videos", jobs_dir: str | Path = "jobs"):
        self.projects_root = Path(projects_root)
        self.jobs_dir = Path(jobs_dir)
        self.job_manager = JobManager(str(self.jobs_dir))

    def list_projects(self) -> list[dict[str, Any]]:
        projects: list[dict[str, Any]] = []
        for manifest_path in list_manifest_paths(self.projects_root):
            try:
                manifest = load_manifest(manifest_path)
            except Exception:
                continue
            projects.append(
                {
                    "project_id": manifest.project_id,
                    "source_video_title": manifest.source_video_title,
                    "project_root": manifest.project_root,
                    "clip_count": len(manifest.clips),
                    "updated_at": manifest.updated_at,
                    "manifest_path": str(manifest_path),
                }
            )
        return projects

    def _manifest_path(self, project_id: str) -> Path:
        path = discover_manifest_by_project_id(self.projects_root, project_id)
        if path is None:
            raise KeyError(project_id)
        return path

    def _load_manifest(self, project_id: str) -> tuple[EditorManifest, Path]:
        manifest_path = self._manifest_path(project_id)
        manifest = load_manifest(manifest_path)
        if reconcile_manifest(manifest, job_manager=self.job_manager, jobs_dir=self.jobs_dir):
            save_manifest(manifest, manifest_path)
        return manifest, manifest_path

    def _save_manifest(self, manifest: EditorManifest, manifest_path: Path) -> None:
        manifest.updated_at = utc_now_iso()
        save_manifest(manifest, manifest_path)

    def _resolve_media_path(self, path: str | None) -> Path | None:
        if not path:
            return None
        candidate = Path(path)
        candidates = [candidate]
        if not candidate.is_absolute():
            candidates.extend(
                [
                    self.projects_root.parent / candidate,
                    self.projects_root / candidate,
                ]
            )
        for media_path in candidates:
            if media_path.exists():
                return media_path
        return None

    def _project_media_url(self, project_id: str, media_kind: str) -> str:
        return f"/api/projects/{project_id}/media/{media_kind}"

    def _clip_media_url(self, project_id: str, clip_id: str, media_kind: str) -> str:
        return f"/api/projects/{project_id}/clips/{clip_id}/media/{media_kind}"

    def _serialize_clip(self, clip: EditorClip) -> dict[str, Any]:
        payload = clip.to_dict()
        subtitle_segments = _effective_subtitle_segments_for_clip(clip)
        translated_subtitle_segments = _parse_subtitle_segments_from_path(clip.asset_registry.subtitle_translated)
        payload["active_subtitle_path"] = clip.asset_registry.subtitle_active
        payload["subtitle_segments"] = subtitle_segments
        payload["effective_subtitle_text"] = _subtitle_segments_to_text(subtitle_segments)
        payload["translated_subtitle_segments"] = translated_subtitle_segments
        payload["translated_subtitle_text"] = _subtitle_segments_to_text(translated_subtitle_segments)
        payload["has_translated_subtitles"] = bool(translated_subtitle_segments)
        payload["has_manual_subtitle_override"] = clip.subtitle_recipe.has_override
        payload["recovery_state"] = clip.recovery.recovery_state
        payload["last_error"] = clip.recovery.last_error
        payload["pending_job_id"] = clip.recovery.pending_job_id
        payload["pending_operation"] = clip.recovery.pending_operation
        return payload

    def _serialize_project(self, manifest: EditorManifest) -> dict[str, Any]:
        clips = []
        for clip in manifest.clips:
            serialized_clip = self._serialize_clip(clip)
            serialized_clip["source_video_url"] = self._project_media_url(manifest.project_id, "source")
            serialized_clip["raw_clip_url"] = self._clip_media_url(manifest.project_id, clip.clip_id, "raw")
            serialized_clip["current_composed_clip_url"] = self._clip_media_url(manifest.project_id, clip.clip_id, "current")
            serialized_clip["horizontal_cover_url"] = self._clip_media_url(manifest.project_id, clip.clip_id, "horizontal_cover")
            serialized_clip["vertical_cover_url"] = self._clip_media_url(manifest.project_id, clip.clip_id, "vertical_cover")
            clips.append(serialized_clip)
        return {
            "project_id": manifest.project_id,
            "schema_version": manifest.schema_version,
            "source_video_title": manifest.source_video_title,
            "source_video_path": manifest.source_video_path,
            "source_video_url": self._project_media_url(manifest.project_id, "source"),
            "source_video_duration": manifest.source_video_duration,
            "project_root": manifest.project_root,
            "created_at": manifest.created_at,
            "updated_at": manifest.updated_at,
            "active_clip_id": clips[0]["clip_id"] if clips else None,
            "clips": clips,
        }

    def load_project(self, project_id: str) -> dict[str, Any]:
        manifest, _ = self._load_manifest(project_id)
        return self._serialize_project(manifest)

    def get_clip(self, project_id: str, clip_id: str) -> dict[str, Any]:
        manifest, _ = self._load_manifest(project_id)
        return self._serialize_clip(manifest.clip_by_id(clip_id))

    def update_clip_bounds(
        self,
        project_id: str,
        clip_id: str,
        start: str | float,
        end: str | float,
        speed: float | None = None,
    ) -> dict[str, Any]:
        manifest, manifest_path = self._load_manifest(project_id)
        clip = manifest.clip_by_id(clip_id)
        absolute_start_seconds = parse_timecode_to_seconds(start)
        absolute_end_seconds = parse_timecode_to_seconds(end)
        next_speed = float(speed if speed is not None else clip.speed)
        max_duration = manifest.source_video_duration or clip.source_video_duration
        if absolute_start_seconds < 0:
            raise ValueError("Clip start must be >= 0")
        if absolute_end_seconds <= absolute_start_seconds:
            raise ValueError("Clip end must be greater than start")
        if next_speed <= 0:
            raise ValueError("Clip speed must be greater than 0")
        if max_duration is not None and absolute_end_seconds > float(max_duration):
            raise ValueError("Clip end exceeds source duration")
        local_start_seconds = absolute_start_seconds - float(clip.part_offset_seconds or 0.0)
        local_end_seconds = absolute_end_seconds - float(clip.part_offset_seconds or 0.0)
        if local_start_seconds < 0:
            raise ValueError("Clip start cannot move before the start of its source part")
        if local_end_seconds <= local_start_seconds:
            raise ValueError("Clip end must remain after clip start within its source part")
        if clip.part_duration_seconds is not None and local_end_seconds > float(clip.part_duration_seconds):
            raise ValueError("Clip end cannot move past the end of its source part")

        clip.start_time = format_seconds_as_timecode(local_start_seconds)
        clip.end_time = format_seconds_as_timecode(local_end_seconds)
        clip.time_range = f"{clip.start_time} - {clip.end_time}"
        clip.absolute_start_time = format_seconds_as_timecode(absolute_start_seconds)
        clip.absolute_end_time = format_seconds_as_timecode(absolute_end_seconds)
        clip.absolute_time_range = f"{clip.absolute_start_time} - {clip.absolute_end_time}"
        clip.duration = round(absolute_end_seconds - absolute_start_seconds, 3)
        clip.speed = next_speed
        clip.recovery.dirty = True
        clip.recovery.cover_dirty = True
        clip.recovery.recovery_state = "dirty"
        clip.recovery.last_error = None
        clip.updated_at = utc_now_iso()
        clip.metadata["cover_dirty"] = True
        self._save_manifest(manifest, manifest_path)
        return self._serialize_clip(clip)

    def update_clip_subtitles(
        self,
        project_id: str,
        clip_id: str,
        subtitle_text: str = "",
        subtitle_segments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        manifest, manifest_path = self._load_manifest(project_id)
        clip = manifest.clip_by_id(clip_id)
        normalized_segments = _serialize_subtitle_segments(subtitle_segments or [])
        if normalized_segments:
            clip.subtitle_recipe.override_segments = [SubtitleSegment.from_dict(segment) for segment in normalized_segments]
            clip.subtitle_recipe.override_text = _subtitle_segments_to_text(normalized_segments)
        else:
            clip.subtitle_recipe.override_segments = []
            clip.subtitle_recipe.override_text = subtitle_text
        clip.recovery.dirty = True
        clip.recovery.recovery_state = "dirty"
        clip.recovery.last_error = None
        clip.updated_at = utc_now_iso()
        self._save_manifest(manifest, manifest_path)
        return self._serialize_clip(clip)

    def update_clip_translated_subtitles(
        self,
        project_id: str,
        clip_id: str,
        subtitle_text: str = "",
        subtitle_segments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        manifest, manifest_path = self._load_manifest(project_id)
        clip = manifest.clip_by_id(clip_id)
        normalized_segments = _serialize_subtitle_segments(subtitle_segments or [])
        if normalized_segments:
            translated_segments = normalized_segments
        else:
            translated_segments = _remap_text_segments_onto_timings(
                [{"text": line.strip()} for line in str(subtitle_text or "").splitlines() if line.strip()],
                _effective_subtitle_segments_for_clip(clip),
                fill_from_timed_segments=False,
            )
        translated_path = self._translated_subtitle_path(manifest, clip)
        translated_path.parent.mkdir(parents=True, exist_ok=True)
        _write_subtitle_segments(translated_path, translated_segments)
        clip.asset_registry.subtitle_sidecars["translated"] = str(translated_path)
        clip.recovery.dirty = True
        clip.recovery.recovery_state = "dirty"
        clip.recovery.last_error = None
        clip.updated_at = utc_now_iso()
        self._save_manifest(manifest, manifest_path)
        return self._serialize_clip(clip)

    def update_cover_title(self, project_id: str, clip_id: str, title_text: str) -> dict[str, Any]:
        manifest, manifest_path = self._load_manifest(project_id)
        clip = manifest.clip_by_id(clip_id)
        clip.cover_recipe.text = title_text
        clip.recovery.cover_dirty = True
        clip.recovery.dirty = True
        clip.recovery.recovery_state = "dirty"
        clip.recovery.last_error = None
        clip.updated_at = utc_now_iso()
        clip.metadata["cover_dirty"] = True
        self._save_manifest(manifest, manifest_path)
        return self._serialize_clip(clip)

    def preview_bounds(self, project_id: str, clip_id: str, start: str | float, end: str | float) -> dict[str, Any]:
        manifest, _ = self._load_manifest(project_id)
        clip = manifest.clip_by_id(clip_id)
        absolute_start_seconds = parse_timecode_to_seconds(start)
        absolute_end_seconds = parse_timecode_to_seconds(end)
        local_start_seconds = absolute_start_seconds - float(clip.part_offset_seconds or 0.0)
        local_end_seconds = absolute_end_seconds - float(clip.part_offset_seconds or 0.0)
        if local_start_seconds < 0:
            raise ValueError("Preview start cannot move before the start of its source part")
        if clip.part_duration_seconds is not None and local_end_seconds > float(clip.part_duration_seconds):
            raise ValueError("Preview end cannot move past the end of its source part")
        return {
            "project_id": project_id,
            "clip_id": clip_id,
            "source_video_path": manifest.source_video_path or clip.source_video_path,
            "source_video_url": self._project_media_url(project_id, "source"),
            "preview_start_time": format_seconds_as_timecode(absolute_start_seconds),
            "preview_end_time": format_seconds_as_timecode(absolute_end_seconds),
            "current_time_range": clip.absolute_time_range,
            "duration": round(max(0.0, absolute_end_seconds - absolute_start_seconds), 3),
        }

    def _queue_job(self, project_id: str, clip_id: str, operation: str, worker) -> dict[str, Any]:
        manifest, manifest_path = self._load_manifest(project_id)
        clip = manifest.clip_by_id(clip_id)
        if clip.recovery.pending_job_id:
            raise ValueError(f"Clip {clip_id} already has a pending rerender job")
        job_id = self.job_manager.create_job(
            manifest.source_video_path or clip.source_video_path or clip.asset_registry.raw_clip or "",
            {
                "kind": "editor_rerender",
                "project_id": project_id,
                "projects_root": str(self.projects_root.resolve()),
                "clip_id": clip_id,
                "operation": operation,
            },
        )
        clip.recovery.pending_job_id = job_id
        clip.recovery.pending_operation = operation
        clip.recovery.pending_assets = {}
        clip.recovery.last_good_assets = clip.snapshot_assets()
        clip.recovery.dirty = True
        clip.recovery.last_error = None
        clip.recovery.recovery_state = "pending"
        if operation == "boundary":
            clip.subtitle_recipe.override_text = None
            clip.subtitle_recipe.override_segments = []
        clip.updated_at = utc_now_iso()
        self._save_manifest(manifest, manifest_path)
        self.job_manager.start_job(job_id, lambda job, progress: worker(manifest_path, clip_id, job, progress))
        return {"job_id": job_id, "project_id": project_id, "clip_id": clip_id, "operation": operation, "status": "pending"}

    @staticmethod
    def _translated_subtitle_path(manifest: EditorManifest, clip: EditorClip) -> Path:
        override_dir = Path(manifest.project_root) / "editor_overrides"
        return override_dir / f"{clip.clip_id}.translated.srt"

    def _build_subtitle_burner(self, clip: EditorClip, *, bootstrap_translation: bool) -> SubtitleBurner:
        burner_kwargs: dict[str, Any] = {
            "subtitle_style_config": SubtitleStyleConfig(
                preset=clip.subtitle_recipe.style_preset,
                font_size=clip.subtitle_recipe.style_font_size,
                vertical_position=clip.subtitle_recipe.style_vertical_position,
                bilingual_layout=clip.subtitle_recipe.style_bilingual_layout,
                background_style=clip.subtitle_recipe.style_background_style,
            )
        }
        if bootstrap_translation and clip.subtitle_recipe.translation:
            provider = DEFAULT_LLM_PROVIDER
            api_key_env_var = API_KEY_ENV_VARS.get(provider)
            api_key = os.getenv(api_key_env_var) if api_key_env_var else None
            if api_key or provider == "custom_openai":
                burner_kwargs.update(
                    {
                        "enable_llm": True,
                        "provider": provider,
                        "api_key": api_key,
                    }
                )
            else:
                logger.warning(
                    "Skipping translated subtitle bootstrap for clip %s: missing %s.",
                    clip.clip_id,
                    api_key_env_var,
                )
        return SubtitleBurner(**burner_kwargs)

    def _boundary_worker(self, manifest_path: Path, clip_id: str, _job, progress_callback) -> dict[str, Any]:
        manifest = load_manifest(manifest_path)
        clip = manifest.clip_by_id(clip_id)
        raw_clip_path = Path(clip.asset_registry.raw_clip or "")
        if not clip.source_video_path or not raw_clip_path:
            raise RuntimeError("Missing source video or raw clip target for boundary rerender")
        raw_clip_path.parent.mkdir(parents=True, exist_ok=True)
        generator = ClipGenerator(output_dir=str(raw_clip_path.parent))
        progress_callback("Recutting clip", 25)
        if not generator._create_clip(
            clip.source_video_path,
            clip.start_time,
            clip.end_time,
            str(raw_clip_path),
            clip.title,
            speed=clip.speed,
        ):
            raise RuntimeError("Failed to recut clip")
        source_subtitle_path = clip.metadata.get("source_subtitle_path")
        subtitle_sidecars = dict(clip.asset_registry.subtitle_sidecars)
        progress_callback("Refreshing subtitle sidecars", 55)
        if source_subtitle_path:
            original_path = Path(subtitle_sidecars.get("original") or raw_clip_path.with_suffix('.srt'))
            if generator._extract_subtitle_from_file(
                source_subtitle_path,
                clip.start_time,
                clip.end_time,
                str(original_path),
                speed=clip.speed,
            ):
                subtitle_sidecars["original"] = str(original_path)
            whisper_path = subtitle_sidecars.get("whisper")
            if whisper_path:
                whisper_target = Path(whisper_path)
                if generator._extract_subtitle_from_file(
                    source_subtitle_path,
                    clip.start_time,
                    clip.end_time,
                    str(whisper_target),
                    speed=clip.speed,
                ):
                    subtitle_sidecars["whisper"] = str(whisper_target)
            subtitle_sidecars["active"] = subtitle_sidecars.get("whisper") or subtitle_sidecars.get("original")
        translated_path = subtitle_sidecars.get("translated")
        if translated_path and subtitle_sidecars.get("active"):
            translated_segments = _parse_subtitle_segments_from_path(translated_path)
            refreshed_segments = _parse_subtitle_segments_from_path(subtitle_sidecars.get("active"))
            if translated_segments and refreshed_segments:
                remapped_translated_segments = _remap_text_segments_onto_timings(
                    translated_segments,
                    refreshed_segments,
                    fill_from_timed_segments=False,
                )
                translated_target = Path(translated_path)
                translated_target.parent.mkdir(parents=True, exist_ok=True)
                _write_subtitle_segments(translated_target, remapped_translated_segments)
                subtitle_sidecars["translated"] = str(translated_target)
        clip.asset_registry.raw_clip = str(raw_clip_path)
        clip.asset_registry.subtitle_sidecars = subtitle_sidecars
        should_refresh_composed_clip = not clip.subtitle_recipe.has_override
        active_subtitle_path = Path(clip.asset_registry.subtitle_active or "")
        if should_refresh_composed_clip and active_subtitle_path.exists():
            progress_callback("Refreshing post-processed clip", 75)
            current_composed_clip = self._render_current_composed_clip(
                manifest,
                clip,
                progress_callback,
                prepare_progress=82,
                render_progress=90,
            )
        else:
            current_composed_clip = str(raw_clip_path)
            clip.asset_registry.current_composed_clip = current_composed_clip
        clip.recovery.pending_assets = {
            "raw_clip": clip.asset_registry.raw_clip,
            "current_composed_clip": current_composed_clip,
            "subtitle_sidecars": dict(clip.asset_registry.subtitle_sidecars),
        }
        clip.recovery.cover_dirty = True
        clip.metadata["cover_dirty"] = True
        save_manifest(manifest, manifest_path)
        progress_callback("Boundary rerender complete", 95)
        return {
            "raw_clip": clip.asset_registry.raw_clip,
            "current_composed_clip": current_composed_clip,
            "subtitle_sidecars": dict(clip.asset_registry.subtitle_sidecars),
        }

    def _render_current_composed_clip(
        self,
        manifest: EditorManifest,
        clip: EditorClip,
        progress_callback,
        *,
        prepare_progress: int = 30,
        render_progress: int = 70,
    ) -> str:
        raw_clip_path = Path(clip.asset_registry.raw_clip or "")
        if not raw_clip_path.exists():
            raise RuntimeError("Raw clip missing for subtitle rerender")
        output_dir = Path(manifest.project_root) / "clips_post_processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / raw_clip_path.name
        subtitle_path = Path(clip.asset_registry.subtitle_active or raw_clip_path.with_suffix('.srt'))
        translated_sidecar_path = None
        override_segments = _serialize_subtitle_segments(clip.subtitle_recipe.override_segments)
        if override_segments:
            override_dir = Path(manifest.project_root) / "editor_overrides"
            override_dir.mkdir(parents=True, exist_ok=True)
            subtitle_path = override_dir / f"{clip.clip_id}.srt"
            _write_subtitle_segments(subtitle_path, override_segments)
            clip.asset_registry.subtitle_sidecars["override"] = str(subtitle_path)
            clip.asset_registry.subtitle_sidecars["active"] = str(subtitle_path)
        elif clip.subtitle_recipe.override_text:
            override_dir = Path(manifest.project_root) / "editor_overrides"
            override_dir.mkdir(parents=True, exist_ok=True)
            subtitle_path = override_dir / f"{clip.clip_id}.srt"
            _write_subtitle_segments(subtitle_path, _legacy_override_segments_for_clip(clip))
            clip.asset_registry.subtitle_sidecars["override"] = str(subtitle_path)
            clip.asset_registry.subtitle_sidecars["active"] = str(subtitle_path)
        translated_sidecar = clip.asset_registry.subtitle_sidecars.get("translated")
        if translated_sidecar and Path(translated_sidecar).exists():
            translated_sidecar_path = Path(translated_sidecar)
        elif clip.subtitle_recipe.translation:
            translated_sidecar_path = self._translated_subtitle_path(manifest, clip)
            translated_sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        should_bootstrap_translation = bool(
            clip.subtitle_recipe.translation
            and translated_sidecar_path
            and not translated_sidecar_path.exists()
        )
        burner = self._build_subtitle_burner(clip, bootstrap_translation=should_bootstrap_translation)
        title_overlay_enabled = bool(clip.metadata.get("title_overlay_enabled"))
        if title_overlay_enabled:
            ass_path = output_dir / f"{clip.clip_id}.ass"
            progress_callback("Preparing subtitle overlay", prepare_progress)
            burner.prepare_ass_for_clip(
                subtitle_path,
                ass_path,
                subtitle_translation=clip.subtitle_recipe.translation,
                translated_srt_path=translated_sidecar_path if translated_sidecar_path and translated_sidecar_path.exists() else None,
                translated_output_path=translated_sidecar_path if translated_sidecar_path and not translated_sidecar_path.exists() else None,
            )
            progress_callback("Compositing title and subtitle", render_progress)
            adder = TitleAdder(output_dir=str(output_dir), language='zh')
            if not adder._add_artistic_title(
                str(raw_clip_path),
                clip.title_recipe.text,
                str(output_path),
                clip.title_recipe.style,
                clip.title_recipe.font_size,
                ass_path=str(ass_path) if ass_path.exists() else None,
            ):
                raise RuntimeError("Failed to compose subtitle/title output")
            ass_path.unlink(missing_ok=True)
        else:
            progress_callback("Burning subtitles", render_progress)
            if not burner._process_clip(
                raw_clip_path,
                subtitle_path,
                output_path,
                clip.subtitle_recipe.translation,
                translated_srt_path=translated_sidecar_path if translated_sidecar_path and translated_sidecar_path.exists() else None,
                translated_output_path=translated_sidecar_path if translated_sidecar_path and not translated_sidecar_path.exists() else None,
            ):
                raise RuntimeError("Failed to burn subtitle-only output")
        if translated_sidecar_path and translated_sidecar_path.exists():
            clip.asset_registry.subtitle_sidecars["translated"] = str(translated_sidecar_path)
        clip.asset_registry.current_composed_clip = str(output_path)
        return str(output_path)

    def _subtitle_worker(self, manifest_path: Path, clip_id: str, _job, progress_callback) -> dict[str, Any]:
        manifest = load_manifest(manifest_path)
        clip = manifest.clip_by_id(clip_id)
        output_path = self._render_current_composed_clip(manifest, clip, progress_callback)
        clip.recovery.pending_assets = {
            "current_composed_clip": output_path,
            "subtitle_sidecars": dict(clip.asset_registry.subtitle_sidecars),
        }
        save_manifest(manifest, manifest_path)
        return {"current_composed_clip": output_path}

    def _cover_worker(self, manifest_path: Path, clip_id: str, _job, progress_callback) -> dict[str, Any]:
        manifest = load_manifest(manifest_path)
        clip = manifest.clip_by_id(clip_id)
        source_clip = Path(clip.asset_registry.raw_clip or clip.asset_registry.current_composed_clip or "")
        if not source_clip.exists():
            raise RuntimeError("No clip asset available for cover rerender")
        cover_dir = Path(manifest.project_root) / "covers"
        cover_dir.mkdir(parents=True, exist_ok=True)
        output_path = cover_dir / f"cover_{clip.clip_id}.jpg"
        generator = CoverImageGenerator(language='zh')
        progress_callback("Rendering covers", 60)
        if not generator.generate_cover(
            str(source_clip),
            clip.cover_recipe.text,
            str(output_path),
            frame_time=0.0,
            text_location=clip.cover_recipe.text_location,
            fill_color=_resolve_cover_color(clip.cover_recipe.fill_color, 'yellow'),
            outline_color=_resolve_cover_color(clip.cover_recipe.outline_color, 'black'),
        ):
            raise RuntimeError("Failed to rerender covers")
        vertical_path = output_path.with_name(output_path.stem + '_vertical' + output_path.suffix)
        clip.asset_registry.horizontal_cover = str(output_path)
        if vertical_path.exists():
            clip.asset_registry.vertical_cover = str(vertical_path)
        clip.recovery.pending_assets = {"horizontal_cover": clip.asset_registry.horizontal_cover, "vertical_cover": clip.asset_registry.vertical_cover}
        clip.recovery.cover_dirty = False
        clip.metadata["cover_dirty"] = False
        save_manifest(manifest, manifest_path)
        return clip.recovery.pending_assets

    def request_rerender(self, project_id: str, clip_id: str, operation: str) -> dict[str, Any]:
        if operation not in {"boundary", "subtitles", "cover", "subtitle"}:
            raise ValueError(f"Unsupported rerender operation: {operation}")
        if operation == 'subtitle':
            operation = 'subtitles'
        worker = {
            'boundary': self._boundary_worker,
            'subtitles': self._subtitle_worker,
            'cover': self._cover_worker,
        }[operation]
        return self._queue_job(project_id, clip_id, operation, worker)

    def resume_rerender(self, project_id: str, clip_id: str) -> dict[str, Any]:
        manifest, _ = self._load_manifest(project_id)
        clip = manifest.clip_by_id(clip_id)
        operation = clip.recovery.pending_operation
        if clip.recovery.recovery_state != "recoverable" or not operation:
            raise ValueError("Clip is not in a recoverable rerender state")
        return self.request_rerender(project_id, clip_id, operation)

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        job = self.job_manager.get_job(job_id)
        if job is not None:
            return job.to_dict()
        job_path = self.jobs_dir / f"{job_id}.json"
        if not job_path.exists():
            raise KeyError(job_id)
        return json.loads(job_path.read_text(encoding="utf-8"))

    def get_project_media(self, project_id: str, media_kind: str) -> FileResponse:
        manifest, _ = self._load_manifest(project_id)
        if media_kind != "source":
            raise KeyError(media_kind)
        source_path = self._resolve_media_path(manifest.source_video_path)
        if source_path is None:
            raise FileNotFoundError("Source media is unavailable")
        return FileResponse(source_path)

    def get_clip_media(self, project_id: str, clip_id: str, media_kind: str) -> FileResponse:
        manifest, _ = self._load_manifest(project_id)
        clip = manifest.clip_by_id(clip_id)
        media_map = {
            "raw": clip.asset_registry.raw_clip,
            "current": clip.asset_registry.current_composed_clip,
            "horizontal_cover": clip.asset_registry.horizontal_cover,
            "vertical_cover": clip.asset_registry.vertical_cover,
        }
        media_path = self._resolve_media_path(media_map.get(media_kind))
        if media_path is None:
            raise FileNotFoundError(f"Clip media is unavailable for {media_kind}")
        return FileResponse(media_path)


def create_app(projects_root: str | Path = "processed_videos", jobs_dir: str | Path = "jobs") -> FastAPI:
    service = EditorService(projects_root=projects_root, jobs_dir=jobs_dir)
    app = FastAPI(title="OpenClip Editor Service", version="0.1.0")
    app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
    app.state.editor_service = service

    dist_dir = Path('editor_frontend/dist')
    assets_dir = dist_dir / 'assets'
    if assets_dir.exists():
        app.mount('/assets', StaticFiles(directory=assets_dir), name='assets')

    def serve_spa(project_id: str | None = None):
        index = dist_dir / 'index.html'
        if index.exists():
            return FileResponse(
                index,
                headers={'Cache-Control': 'no-store, no-cache, must-revalidate'},
            )
        project_note = f"<p>Project: <code>{project_id}</code></p>" if project_id else ''
        return HTMLResponse(
            '<html><body><h1>OpenClip Editor</h1><p>The editor frontend has not been built yet.</p>' + project_note + '</body></html>'
        )

    @app.get('/')
    def root():
        return {'service': 'openclip-editor', 'status': 'ok'}

    @app.get('/healthz')
    def healthz() -> dict[str, str]:
        return {'status': 'ok'}

    @app.get('/projects')
    @app.get('/api/projects')
    def list_projects() -> list[dict[str, Any]]:
        return service.list_projects()

    @app.get('/projects/{project_id}')
    def spa_project(project_id: str):
        return serve_spa(project_id)

    @app.get('/api/projects/{project_id}')
    @app.get('/projects/{project_id}/data')
    def load_project(project_id: str) -> dict[str, Any]:
        try:
            return service.load_project(project_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f'Unknown project_id: {project_id}') from None

    @app.get('/api/projects/{project_id}/clips/{clip_id}')
    @app.get('/projects/{project_id}/clips/{clip_id}')
    def get_clip(project_id: str, clip_id: str) -> dict[str, Any]:
        try:
            return service.get_clip(project_id, clip_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None

    @app.post('/api/projects/{project_id}/clips/{clip_id}/preview-bounds')
    def preview_bounds(project_id: str, clip_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            return service.preview_bounds(project_id, clip_id, payload.get('start_time') or payload.get('start'), payload.get('end_time') or payload.get('end'))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None

    @app.patch('/api/projects/{project_id}/clips/{clip_id}/bounds')
    @app.post('/projects/{project_id}/clips/{clip_id}/bounds')
    def update_bounds(project_id: str, clip_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            return service.update_clip_bounds(
                project_id,
                clip_id,
                payload.get('start_time') or payload.get('start'),
                payload.get('end_time') or payload.get('end'),
                payload.get('speed'),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @app.patch('/api/projects/{project_id}/clips/{clip_id}/subtitle')
    @app.post('/projects/{project_id}/clips/{clip_id}/subtitle-override')
    def update_subtitle_override(project_id: str, clip_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            return service.update_clip_subtitles(
                project_id,
                clip_id,
                payload.get('subtitle_text', ''),
                payload.get('subtitle_segments'),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None

    @app.patch('/api/projects/{project_id}/clips/{clip_id}/translated-subtitle')
    def update_translated_subtitle(project_id: str, clip_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            return service.update_clip_translated_subtitles(
                project_id,
                clip_id,
                payload.get('subtitle_text', ''),
                payload.get('subtitle_segments'),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None

    @app.patch('/api/projects/{project_id}/clips/{clip_id}/cover-title')
    @app.post('/projects/{project_id}/clips/{clip_id}/cover-title')
    def update_cover_title(project_id: str, clip_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            return service.update_cover_title(project_id, clip_id, payload.get('title_text', ''))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None

    @app.post('/api/projects/{project_id}/clips/{clip_id}/rerender/{operation}')
    @app.post('/projects/{project_id}/clips/{clip_id}/rerender/{operation}')
    def rerender_clip(project_id: str, clip_id: str, operation: str) -> dict[str, Any]:
        try:
            return service.request_rerender(project_id, clip_id, operation)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @app.post('/api/projects/{project_id}/clips/{clip_id}/resume')
    @app.post('/projects/{project_id}/clips/{clip_id}/resume')
    def resume_clip(project_id: str, clip_id: str) -> dict[str, Any]:
        try:
            return service.resume_rerender(project_id, clip_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None

    @app.get('/api/jobs/{job_id}')
    @app.get('/jobs/{job_id}')
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return service.get_job_status(job_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f'Unknown job_id: {job_id}') from None

    @app.get('/api/projects/{project_id}/media/{media_kind}')
    def get_project_media(project_id: str, media_kind: str):
        try:
            return service.get_project_media(project_id, media_kind)
        except KeyError:
            raise HTTPException(status_code=404, detail=f'Unknown media kind: {media_kind}') from None
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None

    @app.get('/api/projects/{project_id}/clips/{clip_id}/media/{media_kind}')
    def get_clip_media(project_id: str, clip_id: str, media_kind: str):
        try:
            return service.get_clip_media(project_id, clip_id, media_kind)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from None

    return app
