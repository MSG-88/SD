from __future__ import annotations

import os
import sys
from typing import Optional


def get_hf_token() -> Optional[str]:
    """Return the first available HF token from common environment variable names.

    Common names: HUGGINGFACEHUB_API_TOKEN, HF_TOKEN, HUGGINGFACE_TOKEN
    """
    for name in (
        "HUGGINGFACEHUB_API_TOKEN",
        "HF_TOKEN",
        "HUGGINGFACE_TOKEN",
        "huggingface_token",
    ):
        val = os.getenv(name)
        if val:
            return val
    return None


def ensure_hf_login() -> Optional[str]:
    """If a token is available, attempt to log it into the huggingface cache.

    Returns the token if found (regardless of whether login succeeded).
    This function is safe to call multiple times.
    """
    token = get_hf_token()
    if not token:
        return None

    try:
        from huggingface_hub import login
    except Exception:
        # huggingface_hub is optional in some environments; just warn and return token
        print(
            "Warning: huggingface_hub not installed; token is available via env but cannot be logged.",
            file=sys.stderr,
        )
        return token

    try:
        # Do not add to git credential helper
        login(token=token, add_to_git_credential=False)
    except Exception as exc:  # pragma: no cover - best-effort login
        print(f"Warning: huggingface_hub.login failed: {exc}", file=sys.stderr)

    return token
