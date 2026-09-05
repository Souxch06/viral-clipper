from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)

RUNTIME_DIR = Path('.omx/runtime')
RUNTIME_FILE = RUNTIME_DIR / 'editor-service.json'
LOCK_FILE = RUNTIME_DIR / 'editor-service.lock'
HEALTH_PATH = '/healthz'
EDITOR_BASE_URL_ENV = 'OPENCLIP_EDITOR_BASE_URL'
EDITOR_HOST_ENV = 'OPENCLIP_EDITOR_HOST'
EDITOR_PORT_ENV = 'OPENCLIP_EDITOR_PORT'


def _ensure_runtime_dir() -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)


@contextlib.contextmanager
def _runtime_lock():
    _ensure_runtime_dir()
    with LOCK_FILE.open('a+', encoding='utf-8') as handle:
        try:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass


def _is_process_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _pick_free_port(host: str = '127.0.0.1') -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _load_runtime_record() -> dict:
    if not RUNTIME_FILE.exists():
        return {}
    try:
        return json.loads(RUNTIME_FILE.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _save_runtime_record(record: dict) -> None:
    _ensure_runtime_dir()
    RUNTIME_FILE.write_text(json.dumps(record, indent=2), encoding='utf-8')


def _health_url(host: str, port: int) -> str:
    return f'http://{host}:{port}{HEALTH_PATH}'


def _health_check_host(bind_host: str) -> str:
    # 0.0.0.0 is a bind address, not a reliable client destination.
    if bind_host in {'0.0.0.0', '::'}:
        return '127.0.0.1'
    return bind_host


def _healthy(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        response = requests.get(_health_url(host, port), timeout=timeout)
        return response.ok
    except Exception:
        return False


def _normalized_path(value: str | Path) -> str:
    return str(Path(value).resolve())


def _parse_editor_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise ValueError(f'{EDITOR_PORT_ENV} must be an integer') from exc
    if port <= 0 or port > 65535:
        raise ValueError(f'{EDITOR_PORT_ENV} must be between 1 and 65535')
    return port


def _configured_editor_base_url() -> str | None:
    base_url = os.environ.get(EDITOR_BASE_URL_ENV, '').strip()
    if not base_url:
        return None

    parsed = urlparse(base_url)
    if parsed.scheme not in {'http', 'https'} or not parsed.netloc:
        raise ValueError(f'{EDITOR_BASE_URL_ENV} must be an http(s) URL with a host')
    return base_url.rstrip('/')


def _editor_runtime_config(host: str) -> tuple[str, int | None, str | None]:
    base_url = _configured_editor_base_url()
    env_host = os.environ.get(EDITOR_HOST_ENV, '').strip()
    bind_host = env_host or ('0.0.0.0' if base_url else host)

    env_port = os.environ.get(EDITOR_PORT_ENV, '').strip()
    if env_port:
        return bind_host, _parse_editor_port(env_port), base_url

    if base_url:
        parsed = urlparse(base_url)
        try:
            base_url_port = parsed.port
        except ValueError as exc:
            raise ValueError(f'{EDITOR_BASE_URL_ENV} must include a valid port when a port is specified') from exc
        if base_url_port:
            return bind_host, base_url_port, base_url

    return bind_host, None, base_url


def _project_url(project_id: str, *, base_url: str | None, host: str, port: int) -> str:
    if base_url:
        return f'{base_url}/projects/{project_id}'
    return f'http://{host}:{port}/projects/{project_id}'


def ensure_editor_service(
    project_id: str,
    *,
    projects_root: str | Path = 'processed_videos',
    jobs_dir: str | Path = 'jobs',
    host: str = '127.0.0.1',
    open_browser: bool = False,
) -> str:
    normalized_projects_root = _normalized_path(projects_root)
    normalized_jobs_dir = _normalized_path(jobs_dir)
    bind_host, configured_port, public_base_url = _editor_runtime_config(host)
    health_host = _health_check_host(bind_host)

    with _runtime_lock():
        record = _load_runtime_record()
        port = int(record.get('port') or 0)
        same_port = configured_port is None or port == configured_port
        same_runtime = (
            record.get('projects_root') == normalized_projects_root
            and record.get('jobs_dir') == normalized_jobs_dir
            and record.get('host') == bind_host
            and same_port
        )
        if port and same_runtime and _healthy(health_host, port):
            url = _project_url(project_id, base_url=public_base_url, host=bind_host, port=port)
            if open_browser:
                webbrowser.open_new_tab(url)
            return url

        last_error = None
        for _ in range(2):
            port = configured_port or _pick_free_port(bind_host)
            uv_binary = shutil.which('uv')
            if uv_binary:
                cmd = [
                    uv_binary,
                    'run',
                    'python',
                    '-m',
                    'editor_runtime',
                    '--host',
                    bind_host,
                    '--port',
                    str(port),
                    '--projects-root',
                    normalized_projects_root,
                    '--jobs-dir',
                    normalized_jobs_dir,
                ]
            else:
                cmd = [
                    sys.executable,
                    '-m',
                    'editor_runtime',
                    '--host',
                    bind_host,
                    '--port',
                    str(port),
                    '--projects-root',
                    normalized_projects_root,
                    '--jobs-dir',
                    normalized_jobs_dir,
                ]
            process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            record = {
                'pid': process.pid,
                'port': port,
                'host': bind_host,
                'started_at': time.time(),
                'projects_root': normalized_projects_root,
                'jobs_dir': normalized_jobs_dir,
            }
            _save_runtime_record(record)
            for _ in range(50):
                if _healthy(health_host, port):
                    url = _project_url(project_id, base_url=public_base_url, host=bind_host, port=port)
                    if open_browser:
                        webbrowser.open_new_tab(url)
                    return url
                if process.poll() is not None:
                    break
                time.sleep(0.1)
            last_error = f'editor service failed to start on port {port}'
        raise RuntimeError(last_error or 'editor service failed to start')
