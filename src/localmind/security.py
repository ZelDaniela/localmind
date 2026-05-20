"""Security utilities: API key auth and rate limiting."""

import os
import secrets
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from localmind.config import Config

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def generate_api_key() -> str:
    """Generate a cryptographically secure API key."""
    return secrets.token_urlsafe(32)


def get_api_key(
    api_key_header: Optional[str] = Security(API_KEY_HEADER),
) -> Optional[str]:
    """Dependency: validate API key if security is enabled."""
    config = Config.load()

    if not config.security.api_key_enabled:
        return None

    # Also accept key from environment variable
    valid_key = config.security.api_key or os.environ.get("LOCALMIND_API_KEY")

    if not valid_key:
        return None

    if api_key_header and secrets.compare_digest(api_key_header, valid_key):
        return api_key_header

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid or missing API key. Pass it via X-API-Key header.",
    )


def validate_path_safety(path_str: str) -> None:
    """Ensure a path does not escape allowed boundaries."""
    from pathlib import Path

    try:
        path = Path(path_str).resolve()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path.")

    # Block system directories
    blocked = ["/etc", "/sys", "/proc", "/dev", "/root", "/boot"]
    for blocked_dir in blocked:
        if str(path).startswith(blocked_dir):
            raise HTTPException(
                status_code=403,
                detail=f"Access to {blocked_dir} is not allowed.",
            )
