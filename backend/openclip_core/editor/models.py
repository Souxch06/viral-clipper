from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid


EDITOR_MANIFEST_VERSION = 1
_PROJECT_NAMESPACE = uuid.UUID("e3fa2555-0f4d-49a9-9728-95aabf299b7a")
_CLIP_NAMESPACE = uuid.UUID("6d0f0f60-6712-4d29-bb17-11d8aa57888c")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_project_id(project_root: str | Path | None = None) -> str:
    if project_root is None:
        return f"proj_{uuid.uuid4().hex[:12]}"
    return str(uuid.uuid5(_PROJECT_NAMESPACE, str(Path(project_root).resolve())))


def new_clip_id(
    project_id: str,
    *,
    rank: int,
    video_part: str = "",
    original_time_range: str = "",
    raw_filename: str = "",
) -> str:
    seed = f"{project_id}:{rank}:{video_part}:{original_time_range}:{raw_filename}"
    return str(uuid.uuid5(_CLIP_NAMESPACE, seed))


@dataclass
class EditorRecoveryState:
    dirty: bool = False
    cover_dirty: bool = False
    pending_job_id: Optional[str] = None
    pending_operation: Optional[str] = None
    pending_assets: Dict[str, Any] = field(default_factory=dict)
    last_good_assets: Dict[str, Any] = field(default_factory=dict)
    last_error: Optional[str] = None
    recovery_state: str = "clean"
    last_reconciled_at: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "EditorRecoveryState":
        payload = payload or {}
        return cls(
            dirty=bool(payload.get("dirty", False)),
            cover_dirty=bool(payload.get("cover_dirty", False)),
            pending_job_id=payload.get("pending_job_id"),
            pending_operation=payload.get("pending_operation"),
            pending_assets=dict(payload.get("pending_assets") or {}),
            last_good_assets=dict(payload.get("last_good_assets") or {}),
            last_error=payload.get("last_error"),
            recovery_state=payload.get("recovery_state", "clean"),
            last_reconciled_at=payload.get("last_reconciled_at"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dirty": self.dirty,
            "cover_dirty": self.cover_dirty,
            "pending_job_id": self.pending_job_id,
            "pending_operation": self.pending_operation,
            "pending_assets": dict(self.pending_assets),
            "last_good_assets": dict(self.last_good_assets),
            "last_error": self.last_error,
            "recovery_state": self.recovery_state,
            "last_reconciled_at": self.last_reconciled_at,
        }


@dataclass
class EditorAssetRegistry:
    raw_clip: Optional[str] = None
    current_composed_clip: Optional[str] = None
    subtitle_sidecars: Dict[str, str] = field(default_factory=dict)
    horizontal_cover: Optional[str] = None
    vertical_cover: Optional[str] = None

    @property
    def subtitle_original(self) -> Optional[str]:
        return self.subtitle_sidecars.get("original")

    @property
    def subtitle_whisper(self) -> Optional[str]:
        return self.subtitle_sidecars.get("whisper")

    @property
    def subtitle_translated(self) -> Optional[str]:
        return self.subtitle_sidecars.get("translated")

    @property
    def subtitle_active(self) -> Optional[str]:
        return self.subtitle_sidecars.get("active") or self.subtitle_whisper or self.subtitle_original

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "EditorAssetRegistry":
        payload = payload or {}
        subtitle_sidecars = dict(payload.get("subtitle_sidecars") or {})
        if not subtitle_sidecars:
            if payload.get("subtitle_original"):
                subtitle_sidecars["original"] = payload["subtitle_original"]
            if payload.get("subtitle_whisper"):
                subtitle_sidecars["whisper"] = payload["subtitle_whisper"]
            if payload.get("subtitle_translated"):
                subtitle_sidecars["translated"] = payload["subtitle_translated"]
            if payload.get("subtitle_active"):
                subtitle_sidecars["active"] = payload["subtitle_active"]
        return cls(
            raw_clip=payload.get("raw_clip"),
            current_composed_clip=payload.get("current_composed_clip"),
            subtitle_sidecars=subtitle_sidecars,
            horizontal_cover=payload.get("horizontal_cover"),
            vertical_cover=payload.get("vertical_cover"),
        )

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "raw_clip": self.raw_clip,
            "current_composed_clip": self.current_composed_clip,
            "subtitle_sidecars": dict(self.subtitle_sidecars),
            "horizontal_cover": self.horizontal_cover,
            "vertical_cover": self.vertical_cover,
        }
        # Keep legacy flat fields for older readers.
        data["subtitle_original"] = self.subtitle_original
        data["subtitle_whisper"] = self.subtitle_whisper
        data["subtitle_translated"] = self.subtitle_translated
        data["subtitle_active"] = self.subtitle_active
        return data


@dataclass
class TitleRecipe:
    text: str
    style: str
    font_size: int

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None, *, fallback_text: str) -> "TitleRecipe":
        payload = payload or {}
        return cls(
            text=payload.get("text", fallback_text),
            style=payload.get("style", "default"),
            font_size=int(payload.get("font_size", 40)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "style": self.style,
            "font_size": self.font_size,
        }


@dataclass
class SubtitleSegment:
    start_time: str
    end_time: str
    text: str

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "SubtitleSegment":
        payload = payload or {}
        return cls(
            start_time=payload.get("start_time", "00:00:00,000"),
            end_time=payload.get("end_time", "00:00:00,500"),
            text=payload.get("text", ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
        }


@dataclass
class SubtitleRecipe:
    override_text: Optional[str] = None
    override_segments: List[SubtitleSegment] = field(default_factory=list)
    translation: Optional[str] = None
    style_preset: str = "default"
    style_font_size: str = "medium"
    style_vertical_position: str = "bottom"
    style_bilingual_layout: str = "auto"
    style_background_style: str = "none"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None) -> "SubtitleRecipe":
        payload = payload or {}
        return cls(
            override_text=payload.get("override_text"),
            override_segments=[SubtitleSegment.from_dict(item) for item in payload.get("override_segments", [])],
            translation=payload.get("translation"),
            style_preset=payload.get("style_preset", "default"),
            style_font_size=payload.get("style_font_size", "medium"),
            style_vertical_position=payload.get("style_vertical_position", "bottom"),
            style_bilingual_layout=payload.get("style_bilingual_layout", "auto"),
            style_background_style=payload.get("style_background_style", "none"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "override_text": self.override_text,
            "override_segments": [segment.to_dict() for segment in self.override_segments],
            "translation": self.translation,
            "style_preset": self.style_preset,
            "style_font_size": self.style_font_size,
            "style_vertical_position": self.style_vertical_position,
            "style_bilingual_layout": self.style_bilingual_layout,
            "style_background_style": self.style_background_style,
        }

    @property
    def has_override(self) -> bool:
        if self.override_segments:
            return True
        return bool((self.override_text or "").strip())


@dataclass
class CoverRecipe:
    text: str
    text_location: str = "center"
    fill_color: Any = "yellow"
    outline_color: Any = "black"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any] | None, *, fallback_text: str) -> "CoverRecipe":
        payload = payload or {}
        return cls(
            text=payload.get("text", fallback_text),
            text_location=payload.get("text_location", "center"),
            fill_color=payload.get("fill_color", "yellow"),
            outline_color=payload.get("outline_color", "black"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "text_location": self.text_location,
            "fill_color": self.fill_color,
            "outline_color": self.outline_color,
        }


@dataclass
class EditorClip:
    clip_id: str
    rank: int
    title: str
    video_part: str
    source_video_path: Optional[str]
    source_video_duration: Optional[float]
    part_offset_seconds: float
    part_duration_seconds: Optional[float]
    start_time: str
    end_time: str
    absolute_start_time: str
    absolute_end_time: str
    original_start_time: str
    original_end_time: str
    duration: Optional[float]
    time_range: str
    original_time_range: str
    absolute_time_range: str
    asset_registry: EditorAssetRegistry
    title_recipe: TitleRecipe
    subtitle_recipe: SubtitleRecipe = field(default_factory=SubtitleRecipe)
    cover_recipe: CoverRecipe = field(default_factory=lambda: CoverRecipe(text=""))
    recovery: EditorRecoveryState = field(default_factory=EditorRecoveryState)
    metadata: Dict[str, Any] = field(default_factory=dict)
    speed: float = 1.0
    updated_at: str = field(default_factory=utc_now_iso)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EditorClip":
        title = payload.get("title", "Untitled")
        return cls(
            clip_id=payload["clip_id"],
            rank=int(payload.get("rank", 0)),
            title=title,
            video_part=payload.get("video_part", ""),
            source_video_path=payload.get("source_video_path"),
            source_video_duration=payload.get("source_video_duration"),
            part_offset_seconds=float(payload.get("part_offset_seconds", 0.0)),
            part_duration_seconds=payload.get("part_duration_seconds"),
            start_time=payload.get("start_time", "00:00:00"),
            end_time=payload.get("end_time", "00:00:00"),
            absolute_start_time=payload.get("absolute_start_time", payload.get("start_time", "00:00:00")),
            absolute_end_time=payload.get("absolute_end_time", payload.get("end_time", "00:00:00")),
            original_start_time=payload.get("original_start_time", payload.get("start_time", "00:00:00")),
            original_end_time=payload.get("original_end_time", payload.get("end_time", "00:00:00")),
            duration=payload.get("duration"),
            speed=float(payload.get("speed", 1.0) or 1.0),
            time_range=payload.get("time_range", ""),
            original_time_range=payload.get("original_time_range", ""),
            absolute_time_range=payload.get("absolute_time_range", payload.get("time_range", "")),
            asset_registry=EditorAssetRegistry.from_dict(payload.get("asset_registry")),
            title_recipe=TitleRecipe.from_dict(payload.get("title_recipe"), fallback_text=title),
            subtitle_recipe=SubtitleRecipe.from_dict(payload.get("subtitle_recipe")),
            cover_recipe=CoverRecipe.from_dict(payload.get("cover_recipe"), fallback_text=title),
            recovery=EditorRecoveryState.from_dict(payload.get("recovery")),
            metadata=dict(payload.get("metadata") or {}),
            updated_at=payload.get("updated_at", utc_now_iso()),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clip_id": self.clip_id,
            "rank": self.rank,
            "title": self.title,
            "video_part": self.video_part,
            "source_video_path": self.source_video_path,
            "source_video_duration": self.source_video_duration,
            "part_offset_seconds": self.part_offset_seconds,
            "part_duration_seconds": self.part_duration_seconds,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "absolute_start_time": self.absolute_start_time,
            "absolute_end_time": self.absolute_end_time,
            "original_start_time": self.original_start_time,
            "original_end_time": self.original_end_time,
            "duration": self.duration,
            "speed": self.speed,
            "time_range": self.time_range,
            "original_time_range": self.original_time_range,
            "absolute_time_range": self.absolute_time_range,
            "asset_registry": self.asset_registry.to_dict(),
            "title_recipe": self.title_recipe.to_dict(),
            "subtitle_recipe": self.subtitle_recipe.to_dict(),
            "cover_recipe": self.cover_recipe.to_dict(),
            "recovery": self.recovery.to_dict(),
            "metadata": dict(self.metadata),
            "updated_at": self.updated_at,
        }

    def snapshot_assets(self) -> Dict[str, Any]:
        return self.asset_registry.to_dict()


@dataclass
class EditorManifest:
    project_id: str
    schema_version: int
    source_video_title: str
    source_video_path: Optional[str]
    source_video_duration: Optional[float]
    project_root: str
    created_at: str
    updated_at: str
    clips: List[EditorClip]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "schema_version": self.schema_version,
            "source_video_title": self.source_video_title,
            "source_video_path": self.source_video_path,
            "source_video_duration": self.source_video_duration,
            "project_root": self.project_root,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "clips": [clip.to_dict() for clip in self.clips],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EditorManifest":
        return cls(
            project_id=data["project_id"],
            schema_version=int(data.get("schema_version", EDITOR_MANIFEST_VERSION)),
            source_video_title=data.get("source_video_title", "video"),
            source_video_path=data.get("source_video_path"),
            source_video_duration=data.get("source_video_duration"),
            project_root=data.get("project_root", ""),
            created_at=data.get("created_at", utc_now_iso()),
            updated_at=data.get("updated_at", utc_now_iso()),
            clips=[EditorClip.from_dict(clip_data) for clip_data in data.get("clips", [])],
            metadata=dict(data.get("metadata") or {}),
        )

    def clip_by_id(self, clip_id: str) -> EditorClip:
        for clip in self.clips:
            if clip.clip_id == clip_id:
                return clip
        raise KeyError(f"Unknown clip_id: {clip_id}")

    @property
    def manifest_path(self) -> Path:
        return Path(self.project_root) / "editor_project.json"
