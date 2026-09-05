#!/usr/bin/env python3
"""
Shared download error helpers.
"""

from typing import Optional


def enrich_download_error_message(
    source: str,
    error_text: str,
    platform: str,
    has_cookie_auth: bool = False,
) -> str:
    """Build a user-facing download error message with targeted guidance."""
    error_msg = f"Processing failed: {error_text}"

    if platform != 'youtube':
        return error_msg

    lowered = error_text.lower()
    auth_markers = [
        "sign in to confirm you're not a bot",
        "sign in to confirm you’re not a bot",
        "use --cookies-from-browser or --cookies",
        "login_required",
    ]
    if not any(marker in lowered for marker in auth_markers):
        return error_msg

    if has_cookie_auth:
        return error_msg

    return (
        f"{error_msg}\n"
        "TIPS FROM OPENCLIP: This YouTube video likely needs cookies. "
        "Use Browser cookies or Cookies file in Streamlit, "
        "or pass --browser <browser> / --cookies <cookies.txt> in CLI."
    )
