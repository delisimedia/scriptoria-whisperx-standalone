"""
Patch SpeechBrain for compatibility with torchaudio 2.9+ and Windows.

Two patches are applied:

1. importutils.py — Windows path-separator bug in the lazy importer.
   SpeechBrain checks whether a frame's filename ends with "/inspect.py"
   to detect the stdlib inspect module. On Windows the path uses
   backslashes, so the check always fails and causes infinite
   AttributeError recursion. This patch normalises the filename first.

   Handles two forms across SpeechBrain versions:
     Old (0.5.x, single-line):
       if importer_frame is not None and importer_frame.filename.endswith("/inspect.py"):
           raise AttributeError()
     New (1.0.x, multi-line):
       if importer_frame is not None and importer_frame.filename.endswith(
           "/inspect.py"
       ):
           raise AttributeError()

2. torch_audio_backend.py — torchaudio 2.9+ removed list_audio_backends()
   and set_audio_backend(). SpeechBrain 1.0.x calls these unconditionally.
   This replaces the file with a version that guards each call with hasattr(),
   matching what SpeechBrain's develop branch already does.

Usage:
    python patch_speechbrain.py <path_to_whisperx_portable>
"""

import re
import sys
from pathlib import Path


# ── Fixed torch_audio_backend.py (matches SpeechBrain develop branch) ─────────
FIXED_TORCH_AUDIO_BACKEND = '''\
"""Library for checking the torchaudio backend.

Authors
-------
 * Mirco Ravanelli 2021
 * Adel Moumen 2025
"""

import platform
from typing import Optional, Tuple

import torchaudio

from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


def try_parse_torchaudio_major_version() -> Optional[Tuple[int, int]]:
    """Tries parsing the torchaudio major/minor version.

    Returns
    -------
    (major, minor) tuple or None if parsing fails.
    """
    if not hasattr(torchaudio, "__version__"):
        return None

    parts = torchaudio.__version__.split(".")
    if len(parts) < 2:
        return None

    try:
        return int(parts[0]), int(parts[1])
    except Exception:
        return None


def check_torchaudio_backend():
    """Checks the torchaudio backend and sets it to soundfile on Windows
    when using an old torchaudio version."""
    result = try_parse_torchaudio_major_version()

    if result is None:
        logger.warning(
            "Failed to detect torchaudio major version; unsure how to check "
            "your setup. We recommend keeping torchaudio up-to-date."
        )
        return

    torchaudio_major, torchaudio_minor = result

    if torchaudio_major >= 2 and torchaudio_minor >= 1:
        # torchaudio 2.9+ removed list_audio_backends(); audio loading is
        # handled by torchcodec. Guard with hasattr so both old and new
        # torchaudio 2.x releases work.
        if hasattr(torchaudio, "list_audio_backends"):
            available_backends = torchaudio.list_audio_backends()
            if len(available_backends) == 0:
                logger.warning(
                    "SpeechBrain could not find any working torchaudio backend. "
                    "Audio files may fail to load."
                )
        else:
            # torchaudio 2.9+ — backend selection handled by torchcodec
            logger.debug(
                "torchaudio 2.9+ detected — audio backend checking skipped "
                "(handled by torchcodec)."
            )
    else:
        logger.warning(
            "This version of torchaudio is old. SpeechBrain no longer tries "
            "using the torchaudio global backend mechanism in recipes, so if "
            "you encounter issues, update torchaudio to >=2.1.0."
        )
        if platform.system() == "Windows" and hasattr(torchaudio, "set_audio_backend"):
            logger.warning(
                "Switched audio backend to \\"soundfile\\" because you are "
                "running Windows with an old torchaudio version."
            )
            torchaudio.set_audio_backend("soundfile")


def validate_backend(backend):
    """Validates the specified audio backend.

    Parameters
    ----------
    backend : str or None
        Must be one of [None, 'ffmpeg', 'sox', 'soundfile'].

    Raises
    ------
    ValueError
        If *backend* is not an allowed value.
    """
    allowed_backends = [None, "ffmpeg", "sox", "soundfile"]
    if backend not in allowed_backends:
        if hasattr(torchaudio, "list_audio_backends"):
            available_backends_msg = (
                f"Available backends on your system: {torchaudio.list_audio_backends()}"
            )
        else:
            available_backends_msg = "Using torchaudio 2.9+ with torchcodec."
        raise ValueError(
            f"backend must be one of {allowed_backends}. {available_backends_msg}"
        )
'''


def patch_importutils(site_packages: Path) -> None:
    """Patch importutils.py for the Windows backslash inspect-frame bug."""
    target = site_packages / "speechbrain" / "utils" / "importutils.py"

    if not target.exists():
        print(f"importutils.py not found at {target} — skipping.")
        return

    src = target.read_text(encoding="utf-8")

    if 'replace("\\\\", "/")' in src:
        print("importutils.py patch already applied — nothing to do.")
        return

    replacement = (
        r'\1if importer_frame is not None:\n'
        r'\1    _fname = importer_frame.filename.replace("\\\\", "/")\n'
        r'\1    if _fname.endswith("/inspect.py"):\n'
        r'\1        raise AttributeError()'
    )

    # Pattern A: old single-line form (SpeechBrain 0.5.x)
    pattern_a = (
        r'(?m)^( +)if importer_frame is not None'
        r' and importer_frame\.filename\.endswith\("/inspect\.py"\):\r?\n'
        r'\1    raise AttributeError\(\)'
    )
    # Pattern B: new multi-line form (SpeechBrain 1.0.x)
    pattern_b = (
        r'(?m)^( +)if importer_frame is not None'
        r' and importer_frame\.filename\.endswith\(\r?\n'
        r'\1    "/inspect\.py"\r?\n'
        r'\1\):\r?\n'
        r'\1    raise AttributeError\(\)'
    )

    new_src, count = re.subn(pattern_a, replacement, src)
    if count == 0:
        new_src, count = re.subn(pattern_b, replacement, src)

    if count == 0:
        print("WARNING: importutils.py patch pattern not found — may already be fixed upstream.")
        return

    target.write_text(new_src, encoding="utf-8")
    print(f"importutils.py patched ({count} substitution(s)).")


def patch_torch_audio_backend(site_packages: Path) -> None:
    """Replace torch_audio_backend.py with torchaudio-2.9-compatible version."""
    target = site_packages / "speechbrain" / "utils" / "torch_audio_backend.py"

    if not target.exists():
        print(f"torch_audio_backend.py not found at {target} — skipping.")
        return

    # Already patched if it contains our sentinel comment
    if "torchaudio 2.9+" in target.read_text(encoding="utf-8"):
        print("torch_audio_backend.py already patched — nothing to do.")
        return

    target.write_text(FIXED_TORCH_AUDIO_BACKEND, encoding="utf-8")
    print(f"torch_audio_backend.py replaced with torchaudio-2.9-compatible version.")


# ── Main ───────────────────────────────────────────────────────────────────────

portable_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("whisperx_portable")
site_packages = portable_dir / "Lib" / "site-packages"

if not site_packages.exists():
    print(f"site-packages not found at {site_packages} — is the portable dir correct?")
    sys.exit(1)

patch_importutils(site_packages)
patch_torch_audio_backend(site_packages)
print("SpeechBrain patching complete.")
