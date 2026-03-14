"""
WhisperX Portable Package Installer

This module replaces the complex Python installation process with a simple
download-and-extract approach using pre-built portable packages.

Usage:
    Replace the existing setup_embedded_python() and related functions in
    generate_captions.py with download_and_extract_whisperx().
"""

import os
import sys
import subprocess
import urllib.request
import urllib.error
import zipfile
import hashlib
import shutil
import configparser


# ============================================================================
# Configuration
# ============================================================================

WHISPERX_VERSION = "3.7.4"
PACKAGE_VERSION = f"v{WHISPERX_VERSION}"

# GitHub release URLs
GITHUB_REPO = "delisimedia/scriptoria-whisperx-standalone"
GITHUB_RELEASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/whisperx-{PACKAGE_VERSION}"

# Package files
CUDA_PACKAGE = f"whisperx_portable_win64_cuda128_{PACKAGE_VERSION}.zip"
CPU_PACKAGE = f"whisperx_portable_win64_cpu_{PACKAGE_VERSION}.zip"

CUDA_URL = f"{GITHUB_RELEASE_URL}/{CUDA_PACKAGE}"
CPU_URL = f"{GITHUB_RELEASE_URL}/{CPU_PACKAGE}"

# CUDA release is split into multiple 2GB parts to fit hosting limits.
CUDA_PACKAGE_PART_SUFFIXES = ["aa", "ab"]
CUDA_PACKAGE_PARTS = [f"{CUDA_PACKAGE}.part{suffix}" for suffix in CUDA_PACKAGE_PART_SUFFIXES]
CPU_PACKAGE_PARTS = []

# Expected package sizes (for progress display)
CUDA_SIZE_MB = 3500  # ~3.5 GB split across parts
CPU_SIZE_MB = 800    # ~800 MB


# ============================================================================
# Helper Functions
# ============================================================================

def has_nvidia_gpu():
    """
    Detect if the system has an NVIDIA GPU with CUDA support.

    Returns:
        bool: True if NVIDIA GPU detected, False otherwise
    """
    try:
        # Try nvidia-smi command
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        )
        return result.returncode == 0 and result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return False


def get_subprocess_startup_info():
    """Get subprocess startup info to hide console windows on Windows."""
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        return startupinfo
    return None


def get_subprocess_creation_flags():
    """Get subprocess creation flags to hide console windows on Windows."""
    if sys.platform == "win32":
        return subprocess.CREATE_NO_WINDOW
    return 0


def repair_pyvenv_config(env_dir, python_exe, emit):
    """
    Rewrite pyvenv.cfg so the portable runtime points to the extracted location
    rather than the GitHub Actions host path.
    """
    cfg_path = os.path.join(env_dir, "pyvenv.cfg")
    if not os.path.exists(cfg_path):
        return

    try:
        with open(cfg_path, "r", encoding="utf-8") as cfg_file:
            content = cfg_file.read()

        parser = configparser.ConfigParser()
        parser.read_string("[pyvenv]\n" + content)

        if parser.has_section("pyvenv"):
            parser.set("pyvenv", "home", os.path.normpath(env_dir))
            parser.set("pyvenv", "executable", os.path.normpath(python_exe))
            parser.set(
                "pyvenv",
                "command",
                f"{os.path.normpath(python_exe)} -m venv {os.path.normpath(env_dir)}"
            )

            with open(cfg_path, "w", encoding="utf-8") as cfg_file:
                for key, value in parser.items("pyvenv"):
                    cfg_file.write(f"{key} = {value}\n")
    except Exception as exc:
        emit(f"⚠ Could not repair pyvenv.cfg: {exc}\n")




# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """Test the installer."""

    def progress(msg):
        print(msg, end='')

    print("Testing WhisperX Portable Installer\n")
    print("This will download and install WhisperX to ./test_whisperx_env/\n")

    test_dir = os.path.join(os.getcwd(), "test_whisperx_env")

    python_exe = download_and_extract_whisperx(
        install_dir=test_dir,
        use_cuda=None,  # Auto-detect
        progress_callback=progress,
        cancel_check=None
    )

    if python_exe:
        print(f"\n✓ Success! Python executable: {python_exe}")
    else:
        print(f"\n✗ Installation failed")
