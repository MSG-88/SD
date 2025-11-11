from __future__ import annotations

import logging
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
            logging.info(f"Hugging Face token found in environment variable: {name}")
            return val
    logging.warning("No Hugging Face token found in any known environment variable.")
    return None


def ensure_hf_login() -> Optional[str]:
    """If a token is available, attempt to log it into the huggingface cache.

    Returns the token if found (regardless of whether login succeeded).
    This function is safe to call multiple times.
    """
    token = get_hf_token()
    if not token:
        logging.error("No Hugging Face token found. Please set HUGGINGFACEHUB_API_TOKEN or similar.")
        return None
    if not token.startswith("hf_"):
        logging.warning("The Hugging Face token found does not start with 'hf_'. It may be invalid.")
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
        logging.error(f"huggingface_hub.login failed: {exc}")
    return token
