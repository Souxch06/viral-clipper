from __future__ import annotations

import copy
from typing import Any, Mapping

INPUT_TYPE_URL = "url"
INPUT_TYPE_UPLOAD = "upload"
INPUT_TYPE_SERVER_PATH = "server_path"

LEGACY_INPUT_TYPE_MAP = {
    "Video URL": INPUT_TYPE_URL,
    "Local File": INPUT_TYPE_SERVER_PATH,
    INPUT_TYPE_URL: INPUT_TYPE_URL,
    INPUT_TYPE_UPLOAD: INPUT_TYPE_UPLOAD,
    INPUT_TYPE_SERVER_PATH: INPUT_TYPE_SERVER_PATH,
}


def normalize_input_type(value: str | None) -> str:
    return LEGACY_INPUT_TYPE_MAP.get(value or "", INPUT_TYPE_URL)


def reset_browser_state(default_data: Mapping[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(dict(default_data))
    data["input_type"] = normalize_input_type(data.get("input_type"))
    return data
