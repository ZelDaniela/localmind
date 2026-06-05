"""Security utilities: API key auth, path validation, input validation."""

from __future__ import annotations

import os
import re
import secrets
from pathlib import Path

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from localmind.config import Config

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

_BLOCKED_ROOTS: tuple[str, ...] = (
    "/etc",
    "/sys",
    "/proc",
    "/dev",
    "/root",
    "/boot",
    "/run",
    "/snap",
    "/lost+found",
)

# Project names: alphanumeric, dashes, underscores, max 64 chars
_PROJECT_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")

# Max content per memory entry: 1 MB
_MAX_CONTENT_BYTES = 1 * 1024 * 1024


def generate_api_key() -> str:
    """Generate a cryptographically secure API key."""
    return secrets.token_urlsafe(32)


def get_api_key(
    api_key_header: str | None = Security(API_KEY_HEADER),
) -> str | None:
    """FastAPI dependency: validate API key when auth is enabled.

    Fails CLOSED — if auth is enabled but no key is configured, denies all requests.
    """
    config = Config.load()

    if not config.security.api_key_enabled:
        return None

    valid_key = config.security.api_key or os.environ.get("LOCALMIND_API_KEY")

    if not valid_key:
        # Misconfigured: auth ON but no key → fail closed
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "API key auth is enabled but no key is configured. "
                "Run 'localmind keygen' and add it to config.yaml."
            ),
        )

    if api_key_header and secrets.compare_digest(api_key_header, valid_key):
        return api_key_header

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid or missing API key. Pass it via the X-API-Key header.",
    )


def validate_path_safety(path_str: str) -> Path:
    """Resolve and validate a filesystem path.

    Returns the resolved Path on success.
    Raises HTTPException 400 for malformed paths, 403 for blocked directories.
    """
    if not path_str or not path_str.strip():
        raise HTTPException(status_code=400, detail="Path cannot be empty.")

    if "\x00" in path_str:
        raise HTTPException(status_code=400, detail="Path contains invalid characters.")

    try:
        path = Path(path_str).expanduser().resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Malformed path.")

    for blocked in _BLOCKED_ROOTS:
        if str(path) == blocked or str(path).startswith(blocked + "/"):
            raise HTTPException(
                status_code=403,
                detail=f"Access to system directory '{blocked}' is not allowed.",
            )

    return path


def validate_project_name(project: str | None) -> str | None:
    """Ensure project name is a safe alphanumeric identifier."""
    if project is None:
        return None
    if not _PROJECT_RE.match(project):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid project name. Use only letters, numbers, "
                "dashes and underscores (max 64 chars)."
            ),
        )
    return project


def validate_content_size(content: str) -> str:
    """Reject content exceeding the per-entry size limit."""
    if len(content.encode("utf-8")) > _MAX_CONTENT_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Content exceeds the {_MAX_CONTENT_BYTES // 1024} KB limit.",
        )
    return content
