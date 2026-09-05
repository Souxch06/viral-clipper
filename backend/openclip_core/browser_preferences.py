from __future__ import annotations

import base64
import copy
import json
from typing import Any, Mapping

from openclip_core.browser_session import normalize_input_type

PREFERENCES_COOKIE_NAME = "openclip_sidebar_prefs"
PREFERENCES_SCHEMA_VERSION = 1
PREFERENCES_MAX_BYTES = 8192
PREFERENCES_HYDRATED_FLAG = "preferences_hydrated"

PERSISTED_TOP_LEVEL_FIELDS = {
    "ui_language",
    "input_type",
    "llm_provider",
    "llm_provider_settings",
    "language",
    "use_background",
    "force_whisper",
    "generate_clips",
    "max_clips",
    "add_titles",
    "burn_subtitles",
    "subtitle_translation",
    "subtitle_style_preset",
    "subtitle_style_font_size",
    "subtitle_style_vertical_position",
    "subtitle_style_background_style",
    "generate_cover",
    "cookie_mode",
    "cookie_browser",
    "mode",
    "agentic_analysis",
}

PERSISTED_LLM_PROVIDER_FIELDS = {"model", "base_url"}

SAFE_ENUM_FIELDS = {
    "ui_language": {"zh", "en"},
    "input_type": {"url", "upload", "server_path"},
    "language": {"zh", "en", "vi"},
    "cookie_mode": {"none", "browser", "file"},
    "cookie_browser": {"chrome", "firefox", "edge", "safari"},
    "subtitle_style_preset": {"default", "clean", "high_contrast", "stream"},
    "subtitle_style_font_size": {"small", "medium", "large"},
    "subtitle_style_vertical_position": {"bottom", "lower_middle", "middle"},
    "subtitle_style_background_style": {"none", "light_box", "solid_box"},
    "mode": {"engaging_moments"},
}

BOOLEAN_FIELDS = {
    "use_background",
    "force_whisper",
    "generate_clips",
    "add_titles",
    "burn_subtitles",
    "generate_cover",
    "agentic_analysis",
}

INTEGER_FIELDS = {
    "max_clips",
}


def _sanitize_preference_value(key: str, value: Any, default_data: Mapping[str, Any]) -> Any:
    if key == "input_type":
        normalized = normalize_input_type(value)
        return normalized if normalized in SAFE_ENUM_FIELDS["input_type"] else copy.deepcopy(default_data.get(key))
    if key == "llm_provider":
        provider_settings = default_data.get("llm_provider_settings") or {}
        return copy.deepcopy(value) if value in provider_settings else copy.deepcopy(default_data.get(key))
    if key == "subtitle_translation":
        allowed = {None, "Simplified Chinese", "English"}
        return copy.deepcopy(value) if value in allowed else copy.deepcopy(default_data.get(key))
    if key in SAFE_ENUM_FIELDS:
        return copy.deepcopy(value) if value in SAFE_ENUM_FIELDS[key] else copy.deepcopy(default_data.get(key))
    if key in BOOLEAN_FIELDS:
        return bool(value) if isinstance(value, bool) else copy.deepcopy(default_data.get(key))
    if key in INTEGER_FIELDS:
        return int(value) if isinstance(value, int) and value >= 0 else copy.deepcopy(default_data.get(key))
    return copy.deepcopy(value)

EXCLUDED_FIELDS = {
    "api_key",
    "video_source",
    "cookies_file",
    "custom_prompt_file",
    "custom_prompt_text",
    "speaker_references_dir",
    "processing_result",
    "output_dir",
    "user_intent",
}


def build_preferences_payload(browser_data: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "version": PREFERENCES_SCHEMA_VERSION,
        "prefs": {},
    }
    prefs = payload["prefs"]

    for key in PERSISTED_TOP_LEVEL_FIELDS:
        if key == "llm_provider_settings":
            provider_settings = browser_data.get("llm_provider_settings") or {}
            if isinstance(provider_settings, Mapping):
                sanitized: dict[str, dict[str, Any]] = {}
                for provider, values in provider_settings.items():
                    if not isinstance(values, Mapping):
                        continue
                    sanitized[str(provider)] = {
                        field: copy.deepcopy(values.get(field, ""))
                        for field in PERSISTED_LLM_PROVIDER_FIELDS
                    }
                prefs[key] = sanitized
            continue
        prefs[key] = copy.deepcopy(browser_data.get(key))

    return payload


def serialize_preferences_payload(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(raw) > PREFERENCES_MAX_BYTES:
        raise ValueError("Preference payload too large")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def deserialize_preferences_payload(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        padded = raw + ("=" * (-len(raw) % 4))
        decoded = base64.urlsafe_b64decode(padded.encode("ascii"))
    except Exception:
        return None
    if len(decoded) > PREFERENCES_MAX_BYTES:
        return None
    try:
        payload = json.loads(decoded.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    if payload.get("version") != PREFERENCES_SCHEMA_VERSION:
        return None
    prefs = payload.get("prefs")
    if not isinstance(prefs, Mapping):
        return None
    return {"version": payload["version"], "prefs": dict(prefs)}


def merge_browser_preferences(default_data: Mapping[str, Any], browser_data: Mapping[str, Any], payload: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = copy.deepcopy(dict(browser_data))
    if not payload:
        return merged

    prefs = payload.get("prefs")
    if not isinstance(prefs, Mapping):
        return merged

    for key in PERSISTED_TOP_LEVEL_FIELDS:
        if key not in prefs:
            continue
        if key == "llm_provider_settings":
            merged_settings = copy.deepcopy(default_data.get("llm_provider_settings") or {})
            raw_settings = prefs.get(key) or {}
            if isinstance(raw_settings, Mapping):
                for provider, values in raw_settings.items():
                    provider_key = str(provider)
                    if provider_key not in merged_settings or not isinstance(values, Mapping):
                        continue
                    existing = merged_settings.setdefault(provider_key, {})
                    for field in PERSISTED_LLM_PROVIDER_FIELDS:
                        raw_value = values.get(field, "")
                        existing[field] = copy.deepcopy(raw_value) if isinstance(raw_value, str) else ""
            merged[key] = merged_settings
            continue
        merged[key] = _sanitize_preference_value(key, prefs[key], default_data)

    merged["input_type"] = normalize_input_type(merged.get("input_type"))
    for key in EXCLUDED_FIELDS:
        if key in default_data:
            merged[key] = copy.deepcopy(default_data[key])
    return merged
