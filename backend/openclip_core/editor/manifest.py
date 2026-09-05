from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from openclip_core.editor.models import (
    EDITOR_MANIFEST_VERSION,
    CoverRecipe,
    EditorAssetRegistry,
    EditorClip,
    EditorManifest,
    EditorRecoveryState,
    SubtitleRecipe,
    TitleRecipe,
    new_clip_id,
    new_project_id,
    utc_now_iso,
)
from openclip_core.file_string_utils import FileStringUtils

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "editor_project.json"


def manifest_path_for_project_root(project_root: str | Path) -> Path:
    return Path(project_root) / MANIFEST_FILENAME


def load_manifest(path: str | Path) -> EditorManifest:
    manifest_path = Path(path)
    return EditorManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))


def save_manifest(manifest: EditorManifest, path: str | Path | None = None) -> Path:
    manifest.updated_at = utc_now_iso()
    manifest_path = Path(path) if path else manifest.manifest_path
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=manifest_path.parent,
        delete=False,
        prefix=f".{manifest_path.name}.",
        suffix=".tmp",
    ) as handle:
        json.dump(manifest.to_dict(), handle, ensure_ascii=False, indent=2)
        handle.flush()
        temp_path = Path(handle.name)
    temp_path.replace(manifest_path)
    return manifest_path


def discover_manifest_by_project_id(output_dir: str | Path, project_id: str) -> Optional[Path]:
    matches: list[tuple[str, Path]] = []
    for path in Path(output_dir).rglob(MANIFEST_FILENAME):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if data.get("project_id") == project_id:
            matches.append((str(data.get("updated_at") or data.get("created_at") or ""), path))
    if not matches:
        return None
    matches.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    return matches[0][1]


def list_manifest_paths(output_dir: str | Path) -> list[Path]:
    return sorted(Path(output_dir).rglob(MANIFEST_FILENAME))


def parse_timecode_to_seconds(value: str | float | int | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    raw_value = str(value).strip()
    if not raw_value:
        return 0.0

    parts = raw_value.replace(",", ".").split(":")
    if len(parts) == 1:
        return float(parts[0])
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"Unsupported timecode: {value}")


def format_seconds_as_timecode(value: float) -> str:
    total_ms = int(round(float(value) * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    if milliseconds:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _extract_time_range_parts(value: str | None) -> tuple[str, str]:
    if not value or " - " not in value:
        return "00:00:00", "00:00:00"
    start, end = value.split(" - ", 1)
    return start.strip(), end.strip()


def _part_suffix_map(paths: list[str] | None) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for raw_path in paths or []:
        path = Path(raw_path)
        stem = path.stem
        if "_" in stem:
            mapping[stem.rsplit("_", 1)[-1]] = str(path.resolve())
    return mapping


def _normalize_part_offsets(raw_offsets: Dict[str, Any] | None) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for key, value in (raw_offsets or {}).items():
        try:
            normalized[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _derive_part_duration_seconds(source_subtitle_path: str | None) -> Optional[float]:
    if not source_subtitle_path:
        return None
    subtitle_path = Path(source_subtitle_path)
    if not subtitle_path.exists():
        return None
    try:
        content = subtitle_path.read_text(encoding="utf-8")
    except Exception:
        return None
    max_end = 0.0
    for line in content.splitlines():
        if "-->" not in line:
            continue
        try:
            _start, end = [part.strip() for part in line.split("-->", 1)]
            max_end = max(max_end, parse_timecode_to_seconds(end))
        except (TypeError, ValueError):
            continue
    return max_end if max_end > 0 else None


def _normalize_path(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    return value


def _snapshot_asset_registry(asset_registry: EditorAssetRegistry) -> Dict[str, Any]:
    return {
        "raw_clip": asset_registry.raw_clip,
        "current_composed_clip": asset_registry.current_composed_clip,
        "subtitle_sidecars": dict(asset_registry.subtitle_sidecars),
        "horizontal_cover": asset_registry.horizontal_cover,
        "vertical_cover": asset_registry.vertical_cover,
    }


def _apply_asset_updates(asset_registry: EditorAssetRegistry, updates: Dict[str, Any]) -> None:
    if not updates:
        return
    if "raw_clip" in updates:
        asset_registry.raw_clip = updates["raw_clip"]
    if "current_composed_clip" in updates:
        asset_registry.current_composed_clip = updates["current_composed_clip"]
    if "subtitle_sidecars" in updates and isinstance(updates["subtitle_sidecars"], dict):
        asset_registry.subtitle_sidecars = dict(updates["subtitle_sidecars"])
    if "horizontal_cover" in updates:
        asset_registry.horizontal_cover = updates["horizontal_cover"]
    if "vertical_cover" in updates:
        asset_registry.vertical_cover = updates["vertical_cover"]


def build_manifest(
    *,
    video_root_dir: Path,
    result: Any,
    title_style: str,
    title_font_size: int,
    subtitle_translation: Optional[str],
    subtitle_style_preset: str,
    subtitle_style_font_size: str,
    subtitle_style_vertical_position: str,
    subtitle_style_bilingual_layout: str,
    subtitle_style_background_style: str,
    cover_text_location: str,
    cover_fill_color: Any,
    cover_outline_color: Any,
    existing_manifest: EditorManifest | None = None,
    existing_project_id: Optional[str] = None,
) -> EditorManifest:
    clip_generation = getattr(result, "clip_generation", None) or {}
    if not clip_generation.get("clips_info"):
        raise ValueError("Cannot build editor manifest without clip_generation.clips_info")

    project_id = (
        existing_project_id
        or (existing_manifest.project_id if existing_manifest else None)
        or new_project_id(video_root_dir)
    )
    source_title = (getattr(result, "video_info", {}) or {}).get("title", video_root_dir.name)
    source_video_duration = (getattr(result, "video_info", {}) or {}).get("duration")
    default_source_video_path = getattr(result, "source_video_path", None) or getattr(result, "video_path", None) or None

    part_video_map = _part_suffix_map(getattr(result, "video_parts", None))
    part_subtitle_map = _part_suffix_map(getattr(result, "transcript_parts", None))
    part_offset_map = _normalize_part_offsets(getattr(result, "part_offsets", None))

    post_processing = getattr(result, "post_processing", None) or {}
    cover_generation = getattr(result, "cover_generation", None) or {}

    existing_by_clip_id = {clip.clip_id: clip for clip in (existing_manifest.clips if existing_manifest else [])}

    covers_by_rank: Dict[int, Dict[str, str]] = {}
    for cover in cover_generation.get("covers", []) or []:
        rank = cover.get("rank")
        path = cover.get("path")
        if rank is None or not path:
            continue
        slot = covers_by_rank.setdefault(int(rank), {})
        path_obj = Path(path)
        slot["horizontal_cover"] = str(path_obj.resolve())
        vertical_path = cover.get("vertical_path")
        if vertical_path:
            slot["vertical_cover"] = str(Path(vertical_path).resolve())
        else:
            vertical = path_obj.with_name(path_obj.stem + "_vertical" + path_obj.suffix)
            if vertical.exists():
                slot["vertical_cover"] = str(vertical.resolve())

    clips: list[EditorClip] = []
    created_at = existing_manifest.created_at if existing_manifest else utc_now_iso()

    for clip in clip_generation.get("clips_info", []):
        rank = int(clip.get("rank", 0))
        raw_filename = clip.get("filename") or ""
        clip_output_dir = Path(clip_generation.get("output_dir", video_root_dir / "clips"))
        video_part = clip.get("video_part", "")
        part_offset_seconds = part_offset_map.get(video_part, 0.0)
        source_video_path = part_video_map.get(video_part) or default_source_video_path
        source_subtitle_path = part_subtitle_map.get(video_part)
        part_duration_seconds = _derive_part_duration_seconds(source_subtitle_path)
        raw_clip_path = str((clip_output_dir / raw_filename).resolve()) if raw_filename else None
        subtitle_filename = clip.get("subtitle_filename")
        whisper_subtitle_filename = clip.get("whisper_subtitle_filename")
        translated_subtitle_filename = clip.get("translated_subtitle_filename")
        title_text = clip.get("title", f"Clip {rank}")
        start_time, end_time = _extract_time_range_parts(clip.get("time_range"))
        original_start, original_end = _extract_time_range_parts(clip.get("original_time_range"))
        absolute_start_time = format_seconds_as_timecode(parse_timecode_to_seconds(start_time) + part_offset_seconds)
        absolute_end_time = format_seconds_as_timecode(parse_timecode_to_seconds(end_time) + part_offset_seconds)
        absolute_original_start_time = format_seconds_as_timecode(parse_timecode_to_seconds(original_start) + part_offset_seconds)
        absolute_original_end_time = format_seconds_as_timecode(parse_timecode_to_seconds(original_end) + part_offset_seconds)
        clip_id = new_clip_id(
            project_id,
            rank=rank,
            video_part=video_part,
            original_time_range=clip.get("original_time_range", ""),
            raw_filename=raw_filename,
        )
        existing_clip = existing_by_clip_id.get(clip_id)

        subtitle_sidecars = {}
        if subtitle_filename:
            subtitle_sidecars["original"] = str((clip_output_dir / subtitle_filename).resolve())
        if whisper_subtitle_filename:
            subtitle_sidecars["whisper"] = str((clip_output_dir / whisper_subtitle_filename).resolve())
        if translated_subtitle_filename:
            subtitle_sidecars["translated"] = str((clip_output_dir / translated_subtitle_filename).resolve())
        subtitle_sidecars["active"] = subtitle_sidecars.get("whisper") or subtitle_sidecars.get("original")
        if existing_clip:
            subtitle_sidecars = {**existing_clip.asset_registry.subtitle_sidecars, **subtitle_sidecars}

        current_composed = None
        if post_processing.get("output_dir") and raw_filename:
            post_path = Path(post_processing["output_dir"]) / raw_filename
            if post_path.exists():
                current_composed = str(post_path.resolve())
        if existing_clip and not current_composed:
            current_composed = existing_clip.asset_registry.current_composed_clip

        horizontal_cover = covers_by_rank.get(rank, {}).get("horizontal_cover")
        vertical_cover = covers_by_rank.get(rank, {}).get("vertical_cover")
        if existing_clip:
            horizontal_cover = horizontal_cover or existing_clip.asset_registry.horizontal_cover
            vertical_cover = vertical_cover or existing_clip.asset_registry.vertical_cover

        asset_registry = EditorAssetRegistry(
            raw_clip=raw_clip_path or (existing_clip.asset_registry.raw_clip if existing_clip else None),
            current_composed_clip=current_composed or raw_clip_path or (existing_clip.asset_registry.current_composed_clip if existing_clip else None),
            subtitle_sidecars=subtitle_sidecars,
            horizontal_cover=horizontal_cover,
            vertical_cover=vertical_cover,
        )

        title_recipe = TitleRecipe(
            text=existing_clip.title_recipe.text if existing_clip else title_text,
            style=existing_clip.title_recipe.style if existing_clip else title_style,
            font_size=existing_clip.title_recipe.font_size if existing_clip else title_font_size,
        )
        subtitle_recipe = SubtitleRecipe(
            override_text=existing_clip.subtitle_recipe.override_text if existing_clip else None,
            translation=subtitle_translation if subtitle_translation is not None else (existing_clip.subtitle_recipe.translation if existing_clip else None),
            style_preset=subtitle_style_preset if subtitle_style_preset is not None else (existing_clip.subtitle_recipe.style_preset if existing_clip else "default"),
            style_font_size=subtitle_style_font_size if subtitle_style_font_size is not None else (existing_clip.subtitle_recipe.style_font_size if existing_clip else "medium"),
            style_vertical_position=subtitle_style_vertical_position if subtitle_style_vertical_position is not None else (existing_clip.subtitle_recipe.style_vertical_position if existing_clip else "bottom"),
            style_bilingual_layout=subtitle_style_bilingual_layout if subtitle_style_bilingual_layout is not None else (existing_clip.subtitle_recipe.style_bilingual_layout if existing_clip else "auto"),
            style_background_style=subtitle_style_background_style if subtitle_style_background_style is not None else (existing_clip.subtitle_recipe.style_background_style if existing_clip else "none"),
        )
        cover_recipe = CoverRecipe(
            text=existing_clip.cover_recipe.text if existing_clip else title_text,
            text_location=existing_clip.cover_recipe.text_location if existing_clip else cover_text_location,
            fill_color=existing_clip.cover_recipe.fill_color if existing_clip else _normalize_path(cover_fill_color),
            outline_color=existing_clip.cover_recipe.outline_color if existing_clip else _normalize_path(cover_outline_color),
        )
        recovery = existing_clip.recovery if existing_clip else EditorRecoveryState()
        if not recovery.last_good_assets:
            recovery.last_good_assets = _snapshot_asset_registry(asset_registry)

        clips.append(
            EditorClip(
                clip_id=clip_id,
                rank=rank,
                title=existing_clip.title if existing_clip else title_text,
                video_part=video_part,
                source_video_path=source_video_path,
                source_video_duration=source_video_duration,
                part_offset_seconds=part_offset_seconds,
                part_duration_seconds=existing_clip.part_duration_seconds if existing_clip and existing_clip.part_duration_seconds is not None else part_duration_seconds,
                start_time=existing_clip.start_time if existing_clip else start_time,
                end_time=existing_clip.end_time if existing_clip else end_time,
                absolute_start_time=existing_clip.absolute_start_time if existing_clip else absolute_start_time,
                absolute_end_time=existing_clip.absolute_end_time if existing_clip else absolute_end_time,
                original_start_time=existing_clip.original_start_time if existing_clip else original_start,
                original_end_time=existing_clip.original_end_time if existing_clip else original_end,
                duration=existing_clip.duration if existing_clip and existing_clip.duration is not None else clip.get("duration"),
                time_range=existing_clip.time_range if existing_clip else clip.get("time_range", ""),
                original_time_range=existing_clip.original_time_range if existing_clip else clip.get("original_time_range", ""),
                absolute_time_range=existing_clip.absolute_time_range if existing_clip else f"{absolute_start_time} - {absolute_end_time}",
                asset_registry=asset_registry,
                title_recipe=title_recipe,
                subtitle_recipe=subtitle_recipe,
                cover_recipe=cover_recipe,
                recovery=recovery,
                metadata={
                    **(existing_clip.metadata if existing_clip else {}),
                    "engagement_level": clip.get("engagement_level"),
                    "why_engaging": clip.get("why_engaging"),
                    "normalization_details": clip.get("normalization_details", {}),
                    "source_clip_filename": raw_filename,
                    "source_subtitle_path": source_subtitle_path,
                    "title_overlay_enabled": bool(post_processing.get("title_overlay_enabled", post_processing.get("title_style"))),
                    "absolute_original_time_range": f"{absolute_original_start_time} - {absolute_original_end_time}",
                },
                speed=existing_clip.speed if existing_clip else 1.0,
                updated_at=utc_now_iso(),
            )
        )

    return EditorManifest(
        project_id=project_id,
        schema_version=EDITOR_MANIFEST_VERSION,
        source_video_title=source_title,
        source_video_path=default_source_video_path,
        source_video_duration=source_video_duration,
        project_root=str(video_root_dir.resolve()),
        created_at=created_at,
        updated_at=utc_now_iso(),
        clips=clips,
        metadata={
            **(existing_manifest.metadata if existing_manifest else {}),
            "safe_video_name": FileStringUtils.sanitize_filename(source_title),
            "manifest_filename": MANIFEST_FILENAME,
        },
    )


def _job_payload_from_sources(job_id: str, job_manager: Any | None, jobs_dir: str | Path) -> Dict[str, Any] | None:
    live_payload = None
    if job_manager is not None:
        live_job = job_manager.get_job(job_id)
        if live_job is not None:
            live_payload = live_job.to_dict()
    job_path = Path(jobs_dir) / f"{job_id}.json"
    disk_payload = json.loads(job_path.read_text(encoding="utf-8")) if job_path.exists() else None

    if live_payload is None:
        return disk_payload
    if disk_payload is None:
        return live_payload

    terminal_statuses = {"completed", "failed", "cancelled"}
    live_status = str(live_payload.get("status") or "")
    disk_status = str(disk_payload.get("status") or "")

    # Favor the source that has already reached a terminal state when the two
    # views disagree. This prevents stale in-memory jobs from keeping a clip in
    # "pending" after the persisted job record has already completed.
    if disk_status in terminal_statuses and live_status not in terminal_statuses:
        return disk_payload
    if live_status in terminal_statuses and disk_status not in terminal_statuses:
        return live_payload
    if live_status != disk_status:
        return live_payload

    return disk_payload


def reconcile_manifest(
    manifest: EditorManifest,
    *,
    job_manager: Any | None = None,
    jobs_dir: str | Path = "jobs",
) -> bool:
    changed = False
    now = utc_now_iso()

    for clip in manifest.clips:
        recovery = clip.recovery
        if not recovery.pending_job_id:
            recovery.last_reconciled_at = now
            continue

        job_payload = _job_payload_from_sources(recovery.pending_job_id, job_manager, jobs_dir)
        if job_payload is None:
            recovery.pending_job_id = None
            recovery.pending_operation = None
            recovery.pending_assets = {}
            recovery.dirty = True
            recovery.recovery_state = "stale_pending"
            recovery.last_error = "Pending editor job record is missing; retry or reconcile manually."
            recovery.last_reconciled_at = now
            clip.updated_at = now
            changed = True
            continue

        status = job_payload.get("status")
        current_step = job_payload.get("current_step") or ""
        error_text = job_payload.get("error")

        if status == "completed":
            _apply_asset_updates(clip.asset_registry, recovery.pending_assets)
            recovery.pending_job_id = None
            recovery.pending_operation = None
            recovery.pending_assets = {}
            recovery.last_good_assets = _snapshot_asset_registry(clip.asset_registry)
            recovery.last_error = None
            recovery.dirty = recovery.cover_dirty
            recovery.recovery_state = "cover_dirty" if recovery.cover_dirty else "clean"
            changed = True
        elif status in {"failed", "cancelled"}:
            _apply_asset_updates(clip.asset_registry, recovery.last_good_assets)
            recovery.pending_job_id = None
            recovery.pending_operation = None
            recovery.pending_assets = {}
            recovery.dirty = True
            recovery.recovery_state = status
            recovery.last_error = error_text or f"Editor job {status}."
            changed = True
        elif status == "pending" and current_step == "Interrupted - ready to restart":
            recovery.dirty = True
            recovery.recovery_state = "recoverable"
            recovery.last_error = current_step
            changed = True
        elif status in {"pending", "processing"}:
            recovery.dirty = True
            recovery.recovery_state = "pending"
            changed = True

        recovery.last_reconciled_at = now
        clip.updated_at = now

    if changed:
        manifest.updated_at = now
    return changed


def upsert_manifest(
    *,
    video_root_dir: Path,
    result: Any,
    title_style: str,
    title_font_size: int,
    subtitle_translation: Optional[str],
    subtitle_style_preset: str,
    subtitle_style_font_size: str,
    subtitle_style_vertical_position: str,
    subtitle_style_bilingual_layout: str,
    subtitle_style_background_style: str,
    cover_text_location: str,
    cover_fill_color: Any,
    cover_outline_color: Any,
) -> Path:
    manifest_path = manifest_path_for_project_root(video_root_dir)
    existing_manifest = load_manifest(manifest_path) if manifest_path.exists() else None
    manifest = build_manifest(
        video_root_dir=video_root_dir,
        result=result,
        title_style=title_style,
        title_font_size=title_font_size,
        subtitle_translation=subtitle_translation,
        subtitle_style_preset=subtitle_style_preset,
        subtitle_style_font_size=subtitle_style_font_size,
        subtitle_style_vertical_position=subtitle_style_vertical_position,
        subtitle_style_bilingual_layout=subtitle_style_bilingual_layout,
        subtitle_style_background_style=subtitle_style_background_style,
        cover_text_location=cover_text_location,
        cover_fill_color=cover_fill_color,
        cover_outline_color=cover_outline_color,
        existing_manifest=existing_manifest,
        existing_project_id=(existing_manifest.project_id if existing_manifest else None),
    )
    path = save_manifest(manifest, manifest_path)
    logger.info("🧭 Editor manifest saved: %s", path)
    return path
