#!/usr/bin/env python3
"""Utility helpers for WhisperX Hugging Face interactions."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Any


def _import_hf_error() -> Any:
    """Return HfHubHTTPError class with backwards compatibility."""
    try:
        from huggingface_hub import HfHubHTTPError  # type: ignore
        return HfHubHTTPError
    except Exception:
        try:
            from huggingface_hub.utils import HfHubHTTPError  # type: ignore
            return HfHubHTTPError
        except Exception:
            class _FallbackHttpError(Exception):
                """Fallback error when huggingface_hub is outdated."""
                pass

            return _FallbackHttpError


def _validate_token(token: str) -> Dict[str, Any]:
    """Validate token and optionally check diarization model access."""
    result: Dict[str, Any] = {"ok": False}
    try:
        from huggingface_hub import HfApi  # type: ignore
    except Exception as exc:
        result["error"] = repr(exc)
        return result

    try:
        api = HfApi()
        who = api.whoami(token=token)
        result["ok"] = True
        result["user"] = who.get("name") or who.get("email") or who.get("id")

        HfHubHTTPError = _import_hf_error()
        try:
            api.model_info("pyannote/speaker-diarization-community-1", token=token)
            result["model_access"] = True
        except HfHubHTTPError as err:  # type: ignore
            result["model_access"] = False
            response = getattr(err, "response", None)
            status = getattr(response, "status_code", None)
            if status is not None:
                result["model_error"] = status
            result["model_message"] = str(err)
        except Exception as exc:
            result["model_access"] = False
            result["model_message"] = repr(exc)
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def _prefetch_model(token: str) -> Dict[str, Any]:
    """Download diarization model snapshot into local cache."""
    result: Dict[str, Any] = {"ok": False}
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        result["error"] = repr(exc)
        return result

    try:
        cache_dir = os.environ.get("HF_HOME")
        path = snapshot_download(
            "pyannote/speaker-diarization-community-1",
            token=token or None,
            cache_dir=cache_dir,
        )
        result["ok"] = True
        result["path"] = path
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WhisperX Hugging Face helper utilities."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate Hugging Face token.")
    validate_parser.add_argument("--token", required=True, help="Hugging Face access token.")

    prefetch_parser = subparsers.add_parser("prefetch", help="Prefetch diarization model snapshot.")
    prefetch_parser.add_argument("--token", required=True, help="Hugging Face access token.")

    args = parser.parse_args()

    if args.command == "validate":
        result = _validate_token(args.token)
    elif args.command == "prefetch":
        result = _prefetch_model(args.token)
    else:
        parser.error(f"Unknown command: {args.command}")
        return

    sys.stdout.write(json.dumps(result))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
