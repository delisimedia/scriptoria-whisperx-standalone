#!/usr/bin/env python3
"""
Generate Captions Module for Scriptoria
Integrates faster-whisper caption generation functionality
Based on faster-whisper-xxl-gui.py architecture
"""

import sys
import os
import json
import platform
import shutil
import requests
import urllib.request
import urllib.error
import configparser
import re
import threading
import logging
import subprocess
import tempfile
import copy
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from PyQt6.QtWidgets import (
    QApplication, QProgressBar, QGridLayout, QDialog, QWidget,
    QHBoxLayout, QVBoxLayout, QLabel, QComboBox,
    QGroupBox, QFormLayout, QLineEdit, QPushButton, QCheckBox,
    QTextEdit, QTextBrowser, QDoubleSpinBox, QSpinBox, QScrollArea, QMessageBox,
    QProgressDialog, QFileDialog, QCompleter, QTabWidget, QSizePolicy, QRadioButton, QProgressBar
)
from PyQt6.QtCore import pyqtSignal, QThread, Qt, QTimer, QProcess, QProcessEnvironment, QByteArray, QUrl, QObject, QEvent
from PyQt6.QtGui import QIcon, QPalette, QColor, QTextCursor, QFont, QDragEnterEvent, QDropEvent, QPainter, QPen, QPixmap, QCursor, QDesktopServices


# ============================================================================
# WhisperX Version Requirements
# ============================================================================
# These values should be updated when WhisperX updates its dependencies.
# Check the latest requirements at:
# https://github.com/m-bain/whisperX/blob/main/pyproject.toml
#
# Last updated: 2025-01-25
# Based on WhisperX pyproject.toml requirements:
# - torch~=2.8.0
# - torchaudio~=2.8.0
# - CUDA 12.8 (cu128) via https://download.pytorch.org/whl/cu128
# ============================================================================

WHISPERX_VERSION = "3.8.2"  # WhisperX package version
WHISPERX_PYTORCH_VERSION = "2.8.0"
WHISPERX_TORCHAUDIO_VERSION = "2.8.0"
WHISPERX_CUDA_VERSION = "12.8"
WHISPERX_CUDA_SHORT = "cu128"
WHISPERX_MIN_DRIVER_VERSION = 520  # Minimum NVIDIA driver for CUDA 12.x support

# Python Full Installation Configuration (for .exe installations)
# DEPRECATED: Replaced by portable package approach (see below)
# Kept for reference and potential fallback
PYTHON_VERSION = "3.13.3"  # Python 3.13.3
PYTHON_INSTALLER_URL = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-amd64.exe"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"  # Official pip installer script (for legacy embedded support)

# ============================================================================
# WhisperX Portable Package Configuration (NEW APPROACH)
# ============================================================================
# Pre-built packages eliminate Python installation issues and provide faster setup
# Packages are built via GitHub Actions and hosted on GitHub Releases
#
# To build packages: See whisperx_portable_builder/QUICKSTART.md
# To update: Change GITHUB_REPO to your actual GitHub username/repository
# ============================================================================

PACKAGE_VERSION = f"v{WHISPERX_VERSION}"

# GitHub repository for pre-built packages
# IMPORTANT: Update this to your actual GitHub username/repo after building packages!
GITHUB_REPO = "delisimedia/scriptoria-whisperx-standalone"
GITHUB_RELEASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/whisperx-{PACKAGE_VERSION}"

# Package files
CUDA_PACKAGE = f"whisperx_portable_win64_cuda128_{PACKAGE_VERSION}.zip"
CPU_PACKAGE = f"whisperx_portable_win64_cpu_{PACKAGE_VERSION}.zip"

CUDA_DOWNLOAD_URL = f"{GITHUB_RELEASE_URL}/{CUDA_PACKAGE}"
CPU_DOWNLOAD_URL = f"{GITHUB_RELEASE_URL}/{CPU_PACKAGE}"

# CUDA package is split into multiple parts to satisfy hosting limits.
CUDA_PACKAGE_PART_SUFFIXES = ["aa", "ab"]
CUDA_PACKAGE_PARTS = [f"{CUDA_PACKAGE}.part{suffix}" for suffix in CUDA_PACKAGE_PART_SUFFIXES]
CPU_PACKAGE_PARTS: List[str] = []

# Expected package sizes (for progress display and user information)
CUDA_PACKAGE_SIZE_MB = 3500  # ~3.5 GB split across parts
CPU_PACKAGE_SIZE_MB = 800    # ~800 MB

# ============================================================================


# ============================================================================
# Debug: Window Creation Event Filter
# ============================================================================
# This filter catches all window show events to help debug the startup flash
class WindowShowEventFilter(QObject):
    """Event filter to catch and log all window show events during startup."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.enabled = True

    def eventFilter(self, obj, event):
        if self.enabled and event.type() == QEvent.Type.Show:
            # Only log top-level widgets (windows)
            if isinstance(obj, QWidget) and obj.isWindow():
                import traceback
                print(f"\n{'='*60}")
                print(f"[WINDOW SHOW EVENT] Widget: {obj.__class__.__name__}")
                print(f"  Object name: {obj.objectName() or '(unnamed)'}")
                print(f"  Window title: {obj.windowTitle() or '(no title)'}")
                print(f"  Visible: {obj.isVisible()}")
                print(f"  Parent: {obj.parent().__class__.__name__ if obj.parent() else 'None'}")
                print(f"  Window flags: {obj.windowFlags()}")
                print(f"Stack trace:")
                for line in traceback.format_stack()[:-1]:
                    print(line.rstrip())
                print(f"{'='*60}\n")
        return False  # Don't block the event

# Global instance - will be installed in GenerateCaptionsWidget.__init__
_window_show_filter = None


def get_clean_environment():
    """
    Get a clean environment for subprocess calls, especially for venv Python.

    When running as .exe, PyInstaller sets environment variables that can interfere
    with the venv's Python interpreter. This creates a minimal clean environment
    that allows the venv Python to run independently.
    """
    import os

    env = os.environ.copy()

    # Remove PyInstaller-specific environment variables that can cause conflicts
    # These variables make Python look for modules in the wrong location
    pyinstaller_vars = [
        'PYTHONHOME',      # Points to PyInstaller's bundled Python
        'PYTHONPATH',      # Can include PyInstaller's site-packages
        'PYTHONEXECUTABLE', # Points to the .exe instead of python.exe
    ]

    for var in pyinstaller_vars:
        env.pop(var, None)

    # Keep _MEIPASS for reference but don't use it in PYTHONPATH
    # The venv Python should use its own environment, not the bundled one

    return env


def get_subprocess_startup_info():
    """
    Get subprocess startup info for Windows to hide console windows.

    When running as .exe on Windows, subprocess calls can create visible
    console windows. This returns the appropriate startupinfo to prevent that.
    """
    if sys.platform == "win32":
        import subprocess
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        return startupinfo
    return None


def get_subprocess_creation_flags():
    """Get subprocess creation flags for Windows to prevent console windows."""
    if sys.platform == "win32":
        import subprocess
        # CREATE_NO_WINDOW flag prevents console window from appearing
        return subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0x08000000
    return 0


def download_python_installer(target_dir, progress_callback=None, cancel_check=None):
    """
    Download Python full installer to the target directory.

    Args:
        target_dir: Directory where the Python installer will be downloaded
        progress_callback: Optional function to call with progress messages
        cancel_check: Optional function that returns True if cancelled

    Returns:
        Path to the downloaded installer, or None if download failed/cancelled
    """
    import urllib.request
    import urllib.error

    def emit(msg):
        if progress_callback:
            progress_callback(msg)

    def is_cancelled():
        return cancel_check() if cancel_check else False

    os.makedirs(target_dir, exist_ok=True)
    installer_path = os.path.join(target_dir, f"python-{PYTHON_VERSION}-amd64.exe")

    # Check if already downloaded
    if os.path.exists(installer_path):
        emit(f"✓ Python installer already downloaded\n")
        return installer_path

    emit(f"Downloading Python {PYTHON_VERSION} full installer (~25MB)...\n")
    emit(f"From: {PYTHON_INSTALLER_URL}\n")

    try:
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            if is_cancelled():
                raise Exception("Download cancelled by user")
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                if block_num % 50 == 0:  # Report every ~50 blocks
                    emit(f"  Downloading... {percent}%\n")

        urllib.request.urlretrieve(PYTHON_INSTALLER_URL, installer_path, reporthook=report_progress)

        if is_cancelled():
            if os.path.exists(installer_path):
                os.remove(installer_path)
            return None

        emit(f"✓ Download complete: {installer_path}\n\n")
        return installer_path

    except urllib.error.URLError as e:
        emit(f"✗ Download failed: {str(e)}\n")
        if os.path.exists(installer_path):
            os.remove(installer_path)
        return None
    except Exception as e:
        emit(f"✗ Unexpected error during download: {str(e)}\n")
        if os.path.exists(installer_path):
            os.remove(installer_path)
        return None


def download_and_extract_whisperx(install_dir, use_cuda=True, progress_callback=None, cancel_check=None, target_version=None):
    """
    Download and extract pre-built WhisperX portable package.

    This is the NEW installation method that replaces setup_embedded_python().
    It downloads a pre-built environment instead of installing Python from scratch.

    Args:
        install_dir: Directory to extract WhisperX environment to
        use_cuda: True for CUDA version, False for CPU version
        progress_callback: Optional function(str) to call with progress messages
        cancel_check: Optional function() that returns True if cancelled
        target_version: Optional version string to download (e.g. "3.8.2").
            Defaults to the module-level WHISPERX_VERSION constant.

    Returns:
        str: Path to python.exe in the extracted environment, or None if failed
    """
    import zipfile

    def emit(msg):
        """Send progress message to callback."""
        if progress_callback:
            progress_callback(msg)

    def is_cancelled():
        """Check if operation was cancelled."""
        return cancel_check() if cancel_check else False

    # Allow version override for upgrades; default to the module constant
    if target_version:
        _pkg_ver = f"v{target_version}"
        _release_url = f"https://github.com/{GITHUB_REPO}/releases/download/whisperx-{_pkg_ver}"
        _cuda_package = f"whisperx_portable_win64_cuda128_{_pkg_ver}.zip"
        _cpu_package = f"whisperx_portable_win64_cpu_{_pkg_ver}.zip"
        _cuda_parts = [f"{_cuda_package}.part{s}" for s in CUDA_PACKAGE_PART_SUFFIXES]
    else:
        _release_url = GITHUB_RELEASE_URL
        _cuda_package = CUDA_PACKAGE
        _cpu_package = CPU_PACKAGE
        _cuda_parts = CUDA_PACKAGE_PARTS

    # Select appropriate package based on CUDA preference
    package_url = f"{_release_url}/{_cuda_package}" if use_cuda else f"{_release_url}/{_cpu_package}"
    package_name = _cuda_package if use_cuda else _cpu_package
    expected_size_mb = CUDA_PACKAGE_SIZE_MB if use_cuda else CPU_PACKAGE_SIZE_MB
    package_type = "CUDA" if use_cuda else "CPU"

    emit(f"\n{'='*60}\n")
    emit(f"WhisperX Portable Package Installer\n")
    emit(f"{'='*60}\n\n")
    emit(f"Package type: {package_type}\n")
    emit(f"Package file: {package_name}\n")
    emit(f"Expected size: ~{expected_size_mb} MB\n")
    emit(f"Download URL: {package_url}\n")
    emit(f"Install directory: {install_dir}\n\n")

    # Create install directory
    os.makedirs(install_dir, exist_ok=True)

    # Download path
    download_path = os.path.join(install_dir, package_name)
    python_exe = os.path.join(install_dir, "Scripts", "python.exe")

    # Check if already installed and working
    if os.path.exists(python_exe):
        emit("✓ WhisperX environment already exists\n")
        emit("  Verifying installation...\n")

        try:
            result = subprocess.run(
                [python_exe, "-c", "import whisperx; print(whisperx.__version__)"],
                capture_output=True,
                text=True,
                timeout=10,
                startupinfo=get_subprocess_startup_info(),
                creationflags=get_subprocess_creation_flags()
            )

            if result.returncode == 0:
                version = result.stdout.strip()
                emit(f"✓ WhisperX {version} is ready to use\n")
                return python_exe
            else:
                emit("⚠ Existing installation appears broken, will reinstall...\n")
        except Exception as e:
            emit(f"⚠ Verification failed: {e}\n")
            emit("  Will reinstall...\n")

    part_names = _cuda_parts if use_cuda else CPU_PACKAGE_PARTS

    class DownloadCancelled(Exception):
        """Raised when the user cancels the download."""
        pass

    def download_file(url: str, dest_path: str, label: str):
        """Stream download a file with progress updates."""
        emit(f"  Downloading {label}\n")
        request = urllib.request.Request(url, headers={"User-Agent": "Scriptoria WhisperX Installer"})

        try:
            with urllib.request.urlopen(request) as response, open(dest_path, "wb") as out_file:
                total_size_header = response.getheader("Content-Length")
                total_size = int(total_size_header) if total_size_header else None
                downloaded = 0
                chunk_size = 1024 * 1024
                last_percent = -2
                last_mb_report = -100.0

                while True:
                    if is_cancelled():
                        raise DownloadCancelled()

                    chunk = response.read(chunk_size)
                    if not chunk:
                        break

                    out_file.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        percent = (downloaded * 100) // total_size
                        if percent >= last_percent + 2 or percent >= 100:
                            mb_downloaded = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            emit(f"    {mb_downloaded:.1f} MB / {mb_total:.1f} MB ({percent}%)\n")
                            last_percent = percent
                    else:
                        mb_downloaded = downloaded / (1024 * 1024)
                        if mb_downloaded - last_mb_report >= 100:
                            emit(f"    Downloaded {mb_downloaded:.1f} MB\n")
                            last_mb_report = mb_downloaded

        except DownloadCancelled:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            raise

        except urllib.error.HTTPError:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            raise

        except urllib.error.URLError:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            raise

        except Exception:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            raise

    # Download package
    emit(f"\n{'='*60}\n")
    emit(f"Downloading WhisperX portable package ({package_type})\n")
    emit(f"{'='*60}\n\n")
    emit(f"This may take 5-15 minutes depending on your internet speed...\n\n")

    download_url_for_error = package_url
    part_paths: List[str] = []

    try:
        if part_names:
            emit(f"Package split into {len(part_names)} parts to satisfy hosting limits.\n\n")

            for idx, part_name in enumerate(part_names, start=1):
                part_url = f"{GITHUB_RELEASE_URL}/{part_name}"
                download_url_for_error = part_url
                part_path = os.path.join(install_dir, part_name)
                download_file(part_url, part_path, f"part {idx}/{len(part_names)} ({part_name})")
                part_paths.append(part_path)

            if is_cancelled():
                raise DownloadCancelled()

            emit("\nCombining downloaded parts...\n")
            with open(download_path, "wb") as combined:
                for part_path in part_paths:
                    with open(part_path, "rb") as src:
                        shutil.copyfileobj(src, combined, 1024 * 1024)

            emit("✓ Combined parts into single archive\n\n")

            for part_path in part_paths:
                try:
                    os.remove(part_path)
                except OSError:
                    pass
        else:
            download_url_for_error = package_url
            download_file(package_url, download_path, package_name)

        emit(f"✓ Download complete: {download_path}\n\n")

    except urllib.error.HTTPError as e:
        emit(f"\n✗ Download failed: HTTP {e.code}\n")
        emit(f"  URL: {download_url_for_error}\n")
        emit(f"  Error: {e.reason}\n\n")
        emit("  Please check:\n")
        emit("  1. The package has been built and uploaded to GitHub Releases\n")
        emit(f"  2. The GITHUB_REPO setting is correct (currently: {GITHUB_REPO})\n")
        emit("  3. Your internet connection is working\n")
        emit(f"\n  To build packages, see: whisperx_portable_builder/QUICKSTART.md\n")
        return None

    except urllib.error.URLError as e:
        emit(f"\n✗ Download failed: {e.reason}\n")
        emit(f"  URL: {download_url_for_error}\n")
        emit("  Please check your internet connection and try again.\n")
        return None

    except DownloadCancelled:
        if os.path.exists(download_path):
            os.remove(download_path)
        for part_path in part_paths:
            if os.path.exists(part_path):
                os.remove(part_path)
        emit("\n✗ Download cancelled by user\n")
        return None

    except Exception as e:
        emit(f"\n✗ Unexpected error during download: {str(e)}\n")
        if os.path.exists(download_path):
            os.remove(download_path)
        return None

    # Extract package
    emit(f"\n{'='*60}\n")
    emit(f"Extracting WhisperX environment\n")
    emit(f"{'='*60}\n\n")
    emit(f"This may take 2-5 minutes...\n\n")

    try:
        with zipfile.ZipFile(download_path, 'r') as zip_ref:
            total_files = len(zip_ref.namelist())
            extracted = 0

            for file_info in zip_ref.infolist():
                if is_cancelled():
                    emit("\n✗ Extraction cancelled by user\n")
                    return None

                zip_ref.extract(file_info, install_dir)
                extracted += 1

                # Report every 5%
                if extracted % max(1, total_files // 20) == 0:
                    percent = (extracted * 100) // total_files
                    emit(f"  Extracting... {percent}%\n")

        emit(f"\n✓ Extraction complete\n\n")

    except zipfile.BadZipFile:
        emit("\n✗ Error: Downloaded file is not a valid zip archive\n")
        emit("  The download may have been corrupted. Please try again.\n")
        if os.path.exists(download_path):
            os.remove(download_path)
        return None

    except Exception as e:
        emit(f"\n✗ Extraction failed: {str(e)}\n")
        return None

    # Clean up downloaded zip
    try:
        os.remove(download_path)
        emit("✓ Cleaned up installation files\n\n")
    except Exception:
        pass  # Not critical if cleanup fails

    repair_pyvenv_config(install_dir, python_exe, emit)

    # Verify installation
    emit(f"\n{'='*60}\n")
    emit(f"Verifying installation\n")
    emit(f"{'='*60}\n\n")

    if not os.path.exists(python_exe):
        emit(f"✗ Error: python.exe not found at expected location\n")
        emit(f"  Expected: {python_exe}\n")
        return None

    try:
        # Test Python
        result = subprocess.run(
            [python_exe, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            startupinfo=get_subprocess_startup_info(),
            creationflags=get_subprocess_creation_flags()
        )

        if result.returncode == 0:
            emit(f"✓ Python: {result.stdout.strip()}\n")
        else:
            emit(f"⚠ Python test failed\n")

        # Test PyTorch
        if use_cuda:
            result = subprocess.run(
                [python_exe, "-c", "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"],
                capture_output=True,
                text=True,
                timeout=30,
                startupinfo=get_subprocess_startup_info(),
                creationflags=get_subprocess_creation_flags()
            )
        else:
            result = subprocess.run(
                [python_exe, "-c", "import torch; print(f'PyTorch {torch.__version__} (CPU)')"],
                capture_output=True,
                text=True,
                timeout=30,
                startupinfo=get_subprocess_startup_info(),
                creationflags=get_subprocess_creation_flags()
            )

        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                emit(f"✓ {line}\n")

    except Exception as e:
        emit(f"⚠ Verification tests failed: {e}\n")

    emit(f"\n{'='*60}\n")
    emit(f"Installation Complete!\n")
    emit(f"{'='*60}\n\n")
    emit(f"WhisperX is ready to use.\n")

    return python_exe


def setup_embedded_python(whisperx_dir, progress_callback=None, cancel_check=None):
    """
    Download and install full Python, then create a virtual environment for WhisperX.

    Args:
        whisperx_dir: Base directory for WhisperX installation
        progress_callback: Optional function to call with progress messages
        cancel_check: Optional function that returns True if cancelled

    Returns:
        Path to python.exe in the virtual environment, or None if setup failed/cancelled
    """
    import subprocess

    def emit(msg):
        if progress_callback:
            progress_callback(msg)

    def is_cancelled():
        return cancel_check() if cancel_check else False

    # New structure: full Python installation + venv
    python_install_dir = os.path.join(whisperx_dir, "python_full")
    venv_dir = os.path.join(whisperx_dir, "venv")
    python_exe = os.path.join(python_install_dir, "python.exe")
    venv_python_exe = os.path.join(venv_dir, "Scripts", "python.exe")

    # Check for legacy embedded Python and offer migration
    legacy_embedded = os.path.join(whisperx_dir, "python_embedded")
    if os.path.exists(legacy_embedded):
        emit(f"\n{'='*60}\n")
        emit(f"⚠ Legacy embedded Python detected at: {legacy_embedded}\n")
        emit(f"  Upgrading to full Python installation for better compatibility\n")
        emit(f"{'='*60}\n\n")

    # Check if venv already exists and is valid
    if os.path.exists(venv_python_exe):
        # Verify the venv is functional
        try:
            result = subprocess.run(
                [venv_python_exe, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                env=get_clean_environment(),
                startupinfo=get_subprocess_startup_info(),
                creationflags=get_subprocess_creation_flags()
            )
            if result.returncode == 0:
                emit(f"✓ Python virtual environment already set up at: {venv_dir}\n")
                return venv_python_exe
        except Exception:
            emit(f"⚠ Existing venv appears broken, will recreate...\n")

    emit(f"\n{'='*60}\n")
    emit(f"Setting up full Python {PYTHON_VERSION} for WhisperX installation\n")
    emit(f"{'='*60}\n\n")

    if is_cancelled():
        emit("Installation cancelled by user\n")
        return None

    # Download the Python installer
    installer_path = download_python_installer(whisperx_dir, progress_callback, cancel_check)
    if not installer_path:
        return None

    if is_cancelled():
        emit("Installation cancelled by user\n")
        return None

    # Install Python silently
    emit(f"Installing Python {PYTHON_VERSION} to: {python_install_dir}...\n")
    emit(f"  (This may take 1-2 minutes)\n")
    try:
        # Silent install with: InstallAllUsers=0, PrependPath=0, Include_test=0
        install_cmd = [
            installer_path,
            "/quiet",                    # Silent installation
            "InstallAllUsers=0",         # Install for current user only
            f"TargetDir={python_install_dir}",  # Install to specific directory
            "PrependPath=0",             # Don't add to system PATH
            "Include_test=0",            # Don't include test suite
            "Include_doc=0",             # Don't include documentation
            "Include_tcltk=0",           # Don't include Tcl/Tk (not needed for WhisperX)
            "Include_launcher=0"         # Don't include py launcher
        ]

        result = subprocess.run(
            install_cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
            env=get_clean_environment(),
            startupinfo=get_subprocess_startup_info(),
            creationflags=get_subprocess_creation_flags()
        )

        if result.returncode != 0:
            emit(f"✗ Python installation failed with code {result.returncode}\n")
            if result.stderr:
                emit(f"  Error: {result.stderr}\n")
            return None

        emit(f"✓ Python installed successfully\n\n")

        # Clean up installer
        try:
            os.remove(installer_path)
            emit(f"✓ Cleaned up installation files\n\n")
        except:
            pass  # Not critical if cleanup fails

    except subprocess.TimeoutExpired:
        emit(f"✗ Python installation timed out\n")
        return None
    except Exception as e:
        emit(f"✗ Failed to install Python: {str(e)}\n")
        return None

    if is_cancelled():
        emit("Installation cancelled by user\n")
        return None

    # Verify python.exe exists
    if not os.path.exists(python_exe):
        emit(f"✗ Error: python.exe not found after installation at: {python_exe}\n")
        return None

    # Create virtual environment
    emit(f"Creating virtual environment at: {venv_dir}...\n")
    try:
        result = subprocess.run(
            [python_exe, "-m", "venv", venv_dir],
            capture_output=True,
            text=True,
            timeout=120,
            env=get_clean_environment(),
            startupinfo=get_subprocess_startup_info(),
            creationflags=get_subprocess_creation_flags()
        )

        if result.returncode != 0:
            emit(f"✗ Virtual environment creation failed\n")
            if result.stderr:
                emit(f"  Error: {result.stderr}\n")
            return None

        emit(f"✓ Virtual environment created successfully\n\n")

    except Exception as e:
        emit(f"✗ Failed to create virtual environment: {str(e)}\n")
        return None

    # Verify venv python.exe exists
    if not os.path.exists(venv_python_exe):
        emit(f"✗ Error: venv python.exe not found at: {venv_python_exe}\n")
        return None

    emit(f"✓ Full Python setup complete!\n")
    emit(f"  Python installation: {python_install_dir}\n")
    emit(f"  Virtual environment: {venv_dir}\n\n")

    return venv_python_exe


def repair_pyvenv_config(env_dir: str, python_exe: str, emit=None) -> None:
    """
    Rewrite pyvenv.cfg so the portable runtime references its local installation
    instead of the GitHub Actions host path used during packaging.
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
        message = f"Could not repair pyvenv.cfg: {exc}"
        logging.warning(message)
        if emit:
            emit(f"⚠ {message}\n")



    


def install_pip_in_embedded_python(python_exe, progress_callback=None):
    """
    Install pip into the Python installation (if needed).

    For new venv-based installations, pip is already included.
    For legacy embedded Python, downloads and installs pip.

    Args:
        python_exe: Path to python.exe (either venv or embedded)
        progress_callback: Optional function to call with progress messages

    Returns:
        True if successful or pip already exists, False otherwise
    """
    import subprocess
    import urllib.request

    def emit(msg):
        if progress_callback:
            progress_callback(msg)

    # Check if pip is already available (e.g., in venv installations)
    try:
        result = subprocess.run(
            [python_exe, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            env=get_clean_environment(),
            startupinfo=get_subprocess_startup_info(),
            creationflags=get_subprocess_creation_flags()
        )
        if result.returncode == 0:
            emit(f"✓ Pip is already installed\n\n")
            return True
    except Exception:
        pass  # Pip not available, continue with installation

    emit(f"Installing pip into Python environment...\n")

    # Download get-pip.py
    python_dir = os.path.dirname(python_exe)
    get_pip_path = os.path.join(python_dir, "get-pip.py")

    try:
        emit(f"  Downloading get-pip.py...\n")
        urllib.request.urlretrieve(GET_PIP_URL, get_pip_path)
        emit(f"  ✓ Downloaded get-pip.py\n")

        # Run get-pip.py
        emit(f"  Installing pip (this may take a minute)...\n")
        result = subprocess.run(
            [python_exe, get_pip_path, "--no-warn-script-location"],
            capture_output=True,
            text=True,
            timeout=120,
            env=get_clean_environment(),
            startupinfo=get_subprocess_startup_info(),
            creationflags=get_subprocess_creation_flags()
        )

        if result.returncode == 0:
            emit(f"✓ Pip installed successfully\n\n")

            # Clean up get-pip.py
            try:
                os.remove(get_pip_path)
            except:
                pass

            return True
        else:
            emit(f"✗ Pip installation failed\n")
            emit(f"  Return code: {result.returncode}\n")
            emit(f"  Stderr: {result.stderr}\n\n")
            return False

    except Exception as e:
        emit(f"✗ Error installing pip: {str(e)}\n\n")
        return False


def get_app_directory():
    """Get the application's base directory consistently, whether running from source or as executable"""
    try:
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        else:
            # For Scriptoria, return the backend directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            return os.path.dirname(script_dir)  # Go up from data/ to backend/
    except Exception as e:
        logging.warning(f"Could not determine app directory: {e}, using current working directory")
        return os.getcwd()


def truncate_filename(filename, max_length=50):
    """Truncate a filename to max_length characters, adding ellipsis if needed."""
    if not filename or len(filename) <= max_length:
        return filename
    # Keep the extension visible
    name, ext = os.path.splitext(filename)
    if ext:
        # Reserve space for ellipsis (3) and extension
        available = max_length - len(ext) - 3
        if available > 0:
            return name[:available] + "..." + ext
    # Fallback: just truncate with ellipsis
    return filename[:max_length - 3] + "..."


def get_hf_cache_directory():
    """Return a stable location for Hugging Face caches within the Scriptoria folder."""
    try:
        cache_dir = os.path.join(get_app_directory(), "huggingface_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir
    except Exception:
        fallback = os.path.join(os.getcwd(), "huggingface_cache")
        try:
            os.makedirs(fallback, exist_ok=True)
        except Exception:
            pass
        return fallback


def detect_cuda_version():
    """
    Validate that the NVIDIA driver supports WhisperX's required CUDA version.
    Returns a tuple: (cuda_version_string, cuda_short_name, warning_message)

    WhisperX requires CUDA 12.8, which needs NVIDIA driver >= 520.
    If driver is too old, returns a warning message (third tuple element).
    """
    try:
        import subprocess

        # Try to get NVIDIA driver version using nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0 and result.stdout.strip():
                driver_version = result.stdout.strip().split('\n')[0]
                # Extract major version
                driver_major = int(driver_version.split('.')[0])

                # Check if driver supports CUDA 12.x (minimum driver 520)
                # https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
                if driver_major < WHISPERX_MIN_DRIVER_VERSION:
                    warning = (
                        f"Warning: Your NVIDIA driver (version {driver_version}) may be too old.\n"
                        f"WhisperX requires CUDA {WHISPERX_CUDA_VERSION}, which needs driver >= {WHISPERX_MIN_DRIVER_VERSION}.\n"
                        f"Please update your NVIDIA driver before installing WhisperX."
                    )
                    return (WHISPERX_CUDA_VERSION, WHISPERX_CUDA_SHORT, warning)

                # Driver is compatible
                return (WHISPERX_CUDA_VERSION, WHISPERX_CUDA_SHORT, None)

        except FileNotFoundError:
            # nvidia-smi not found, might not have NVIDIA GPU
            # Still return CUDA version for installation, but with warning
            warning = (
                "Warning: nvidia-smi not found. Cannot verify driver compatibility.\n"
                f"WhisperX requires CUDA {WHISPERX_CUDA_VERSION} and NVIDIA driver >= {WHISPERX_MIN_DRIVER_VERSION}."
            )
            return (WHISPERX_CUDA_VERSION, WHISPERX_CUDA_SHORT, warning)
        except Exception as e:
            # Error during detection
            warning = f"Warning: Could not detect NVIDIA driver version: {str(e)}"
            return (WHISPERX_CUDA_VERSION, WHISPERX_CUDA_SHORT, warning)

        # Fallback: return required CUDA version
        return (WHISPERX_CUDA_VERSION, WHISPERX_CUDA_SHORT, None)

    except Exception:
        # Ultimate fallback
        return (WHISPERX_CUDA_VERSION, WHISPERX_CUDA_SHORT, None)


def get_settings_directory():
    """Get a writable, persistent directory for settings"""
    try:
        if sys.platform == "win32":
            appdata = os.environ.get('APPDATA')
            if appdata:
                settings_dir = os.path.join(appdata, "Scriptoria", "generate_captions")
            else:
                settings_dir = os.path.join(os.path.expanduser("~"), ".scriptoria", "generate_captions")
        else:
            settings_dir = os.path.join(os.path.expanduser("~"), ".scriptoria", "generate_captions")

        os.makedirs(settings_dir, exist_ok=True)
        return settings_dir
    except Exception as e:
        logging.warning(f"Could not create settings directory: {e}, using app directory")
        return get_app_directory()


def get_portable_settings_directory():
    """Get settings directory that stays with the application (portable)"""
    try:
        if getattr(sys, 'frozen', False):
            return os.path.dirname(sys.executable)
        else:
            return os.path.dirname(os.path.abspath(__file__))
    except Exception as e:
        logging.warning(f"Could not determine portable settings directory: {e}")
        return os.getcwd()


def create_always_on_top_message_box(parent, icon, title, text, buttons=None, default_button=None):
    """Create a message box that always stays on top"""
    msg = QMessageBox(parent)
    msg.setIcon(icon)
    msg.setWindowTitle(title)
    msg.setText(text)

    try:
        msg.setWindowFlags(msg.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
    except Exception as e:
        logging.warning(f"Could not set always-on-top flag: {e}")

    if buttons:
        msg.setStandardButtons(buttons)
    if default_button:
        msg.setDefaultButton(default_button)

    msg.activateWindow()
    msg.raise_()
    return msg.exec()


def show_setup_question(parent, title, text, buttons, default_button):
    """Show setup question dialog"""
    return create_always_on_top_message_box(parent, QMessageBox.Icon.Question, title, text, buttons, default_button)


def show_setup_warning(parent, title, text):
    """Show setup warning dialog"""
    return create_always_on_top_message_box(parent, QMessageBox.Icon.Warning, title, text, QMessageBox.StandardButton.Ok)


def show_setup_critical(parent, title, text):
    """Show setup critical dialog"""
    return create_always_on_top_message_box(parent, QMessageBox.Icon.Critical, title, text, QMessageBox.StandardButton.Ok)


def show_setup_information(parent, title, text):
    """Show setup information dialog"""
    return create_always_on_top_message_box(parent, QMessageBox.Icon.Information, title, text, QMessageBox.StandardButton.Ok)


def get_execution_environment():
    """Detect execution environment"""
    try:
        if getattr(sys, 'frozen', False):
            return "exe_with_python" if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else "exe_no_python"
        else:
            return "source"
    except Exception:
        return "unknown"


def try_pytorch_detection():
    """Try PyTorch detection with detailed diagnostics"""
    try:
        import torch

        # Check if CUDA is available
        cuda_available = torch.cuda.is_available()

        if cuda_available:
            device_count = torch.cuda.device_count()
            if device_count > 0:
                gpu_name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                memory_gb = props.total_memory / (1024**3)
                cuda_version = torch.version.cuda

                return {
                    "success": True,
                    "gpu_info": {
                        "has_cuda": True,
                        "gpu_memory_gb": memory_gb,
                        "gpu_name": gpu_name,
                        "cuda_version": cuda_version,
                        "recommended_device": "cuda",
                        "detection_method": "pytorch"
                    }
                }

        return {
            "success": False,
            "error": "PyTorch available but no CUDA support detected",
            "gpu_info": {"has_cuda": False, "detection_method": "pytorch"}
        }
    except ImportError:
        return {
            "success": False,
            "error": "PyTorch not available",
            "gpu_info": {"has_cuda": False, "detection_method": "pytorch"}
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"PyTorch detection failed: {str(e)}",
            "gpu_info": {"has_cuda": False, "detection_method": "pytorch"}
        }


def try_nvml_detection():
    """Try NVML detection"""
    try:
        import pynvml
        pynvml.nvmlInit()

        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_gb = mem_info.total / (1024**3)

            return {
                "success": True,
                "gpu_info": {
                    "has_cuda": True,
                    "gpu_memory_gb": memory_gb,
                    "gpu_name": name,
                    "recommended_device": "cuda",
                    "detection_method": "nvml"
                }
            }

        return {"success": False, "error": "No NVIDIA devices found", "gpu_info": {"has_cuda": False}}
    except ImportError:
        return {"success": False, "error": "pynvml not available", "gpu_info": {"has_cuda": False}}
    except Exception as e:
        return {"success": False, "error": f"NVML detection failed: {str(e)}", "gpu_info": {"has_cuda": False}}


def try_nvidia_smi_detection():
    """Try nvidia-smi detection"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, timeout=10)

        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.strip().split('\n')
            if lines:
                parts = lines[0].split(', ')
                if len(parts) >= 2:
                    gpu_name = parts[0].strip()
                    memory_mb = int(parts[1].strip())
                    memory_gb = memory_mb / 1024

                    return {
                        "success": True,
                        "gpu_info": {
                            "has_cuda": True,
                            "gpu_memory_gb": memory_gb,
                            "gpu_name": gpu_name,
                            "recommended_device": "cuda",
                            "detection_method": "nvidia_smi"
                        }
                    }

        return {"success": False, "error": "nvidia-smi returned no valid data", "gpu_info": {"has_cuda": False}}
    except FileNotFoundError:
        return {"success": False, "error": "nvidia-smi not found", "gpu_info": {"has_cuda": False}}
    except Exception as e:
        return {"success": False, "error": f"nvidia-smi detection failed: {str(e)}", "gpu_info": {"has_cuda": False}}


def try_platform_detection():
    """Try platform-specific GPU detection"""
    try:
        if sys.platform == "win32":
            import subprocess
            result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                                  capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and 'nvidia' in result.stdout.lower():
                # Basic NVIDIA detection on Windows
                return {
                    "success": True,
                    "gpu_info": {
                        "has_cuda": True,
                        "gpu_memory_gb": 0,  # Unknown
                        "gpu_name": "NVIDIA GPU (detected via wmic)",
                        "recommended_device": "cuda",
                        "detection_method": "wmic"
                    }
                }

        return {"success": False, "error": "No platform detection available", "gpu_info": {"has_cuda": False}}
    except Exception as e:
        return {"success": False, "error": f"Platform detection failed: {str(e)}", "gpu_info": {"has_cuda": False}}


def detect_gpu_info():
    """Comprehensive GPU detection"""
    all_detected_gpus = []
    detection_details = []

    # Try PyTorch first (most reliable)
    pytorch_result = try_pytorch_detection()
    if pytorch_result["success"]:
        logging.info(f"NVIDIA GPU detected via PyTorch: {pytorch_result['gpu_info']['gpu_name']}")
        all_detected_gpus.append(pytorch_result["gpu_info"])
    else:
        detection_details.append(f"PyTorch: {pytorch_result['error']}")
        logging.info(f"PyTorch detection failed: {pytorch_result['error']}")

    # Try NVML if PyTorch failed
    if not all_detected_gpus:
        nvml_result = try_nvml_detection()
        if nvml_result["success"]:
            logging.info(f"NVIDIA GPU detected via NVML: {nvml_result['gpu_info']['gpu_name']}")
            all_detected_gpus.append(nvml_result["gpu_info"])
        else:
            detection_details.append(f"NVML: {nvml_result['error']}")

    # Try nvidia-smi if others failed
    if not all_detected_gpus:
        nvidia_smi_result = try_nvidia_smi_detection()
        if nvidia_smi_result["success"]:
            logging.info(f"NVIDIA GPU detected via nvidia-smi: {nvidia_smi_result['gpu_info']['gpu_name']}")
            all_detected_gpus.append(nvidia_smi_result["gpu_info"])
        else:
            detection_details.append(f"nvidia-smi: {nvidia_smi_result['error']}")

    # Try platform detection as last resort
    if not all_detected_gpus:
        platform_result = try_platform_detection()
        if platform_result["success"]:
            logging.info(f"GPU detected via platform: {platform_result['gpu_info']['gpu_name']}")
            all_detected_gpus.append(platform_result["gpu_info"])
        else:
            detection_details.append(f"Platform: {platform_result['error']}")

    # Return best result or fallback
    if all_detected_gpus:
        best_gpu = all_detected_gpus[0]  # Use first (best) detection
        return {
            "has_cuda": best_gpu.get("has_cuda", False),
            "gpu_memory_gb": best_gpu.get("gpu_memory_gb", 0),
            "gpu_name": best_gpu.get("gpu_name", "Unknown"),
            "cuda_version": best_gpu.get("cuda_version", "Unknown"),
            "detection_method": best_gpu.get("detection_method", "unknown"),
            "recommended_device": best_gpu.get("recommended_device", "cpu"),
            "all_detected_gpus": all_detected_gpus,
            "detection_details": detection_details
        }
    else:
        return {
            "has_cuda": False,
            "gpu_memory_gb": 0,
            "gpu_name": "No NVIDIA GPU detected",
            "cuda_version": "N/A",
            "detection_method": "none",
            "recommended_device": "cpu",
            "all_detected_gpus": [],
            "detection_details": detection_details
        }


def detect_system_info():
    """Detect system information"""
    system_info = {}

    try:
        # CPU information
        system_info["cpu_count"] = os.cpu_count()
        system_info["platform"] = platform.system()
        system_info["platform_version"] = platform.release()

        # Memory information
        try:
            import psutil
            memory = psutil.virtual_memory()
            system_info["ram_gb"] = memory.total / (1024**3)
        except ImportError:
            # Fallback method
            try:
                if sys.platform == "linux":
                    with open('/proc/meminfo', 'r') as f:
                        meminfo = f.read()
                        match = re.search(r'MemTotal:\s+(\d+)', meminfo)
                        if match:
                            system_info["ram_gb"] = int(match.group(1)) / (1024 * 1024)
            except:
                system_info["ram_gb"] = 0
    except Exception as e:
        logging.warning(f"Error detecting system info: {e}")

    return system_info


def detect_hardware_capabilities():
    """Comprehensive hardware detection"""
    gpu_info = detect_gpu_info()
    system_info = detect_system_info()

    import time
    return {
        **gpu_info,
        **system_info,
        "detection_timestamp": time.time()
    }


def get_recommended_settings(hardware_info):
    """Generate optimal settings based on hardware"""
    recommendations = {}

    # Device Selection
    if hardware_info["has_cuda"] and hardware_info["gpu_memory_gb"] >= 4:
        recommendations["device"] = "cuda"
    else:
        recommendations["device"] = "cpu"

    # Compute Type Based on Hardware
    if hardware_info["has_cuda"]:
        if hardware_info["gpu_memory_gb"] >= 8:
            recommendations["compute_type"] = "float16"  # Best quality
        elif hardware_info["gpu_memory_gb"] >= 4:
            recommendations["compute_type"] = "int8_float16"  # Good balance
        else:
            recommendations["compute_type"] = "int8"  # Memory limited
    else:
        # CPU optimizations
        if hardware_info.get("ram_gb", 0) >= 16:
            recommendations["compute_type"] = "int8"  # Faster on CPU
        else:
            recommendations["compute_type"] = "int8"  # Memory conservative

    # Model Recommendation (Faster-Whisper default)
    # Prefer distil-large-v3.5 on CUDA for best speed/quality balance on modern GPUs (e.g., RTX 5090)
    if hardware_info.get("has_cuda"):
        recommendations["model"] = "distil-large-v3.5"
    else:
        # CPU fallback: large-v2 is too heavy; prefer smaller models if needed
        if hardware_info.get("ram_gb", 0) >= 8:
            recommendations["model"] = "small"
        else:
            recommendations["model"] = "tiny"

    # VAD Method Recommendation
    # For RTX 50 series, prefer ONNX-based pyannote VAD for compatibility; otherwise default to Silero V4
    gpu_name = (hardware_info.get("gpu_name") or "").lower()
    if hardware_info.get("has_cuda") and any(tag in gpu_name for tag in ["rtx 50", "rtx 51", "rtx 52"]):
        recommendations["vad_method"] = "pyannote_onnx_v3"
    else:
        recommendations["vad_method"] = "silero_v4"

    # Speaker Diarization Method Based on Hardware
    if hardware_info["has_cuda"] and hardware_info["gpu_memory_gb"] >= 4:
        # Check for RTX 50 series - they have sm_120 compatibility issues with bundled PyTorch
        gpu_name = hardware_info.get("gpu_name", "").lower()
        if "rtx 50" in gpu_name or "rtx 51" in gpu_name or "rtx 52" in gpu_name:
            recommendations["diarization_method"] = "reverb_v1"  # RTX 50 series fallback (no ONNX diarization available)
        else:
            recommendations["diarization_method"] = "pyannote_v3.1"  # Fastest with older CUDA
    else:
        recommendations["diarization_method"] = "pyannote_v3.0"  # CPU optimized

    return recommendations


class CleanerInputOverlayDialog(QDialog):
    """
    Modal overlay dialog that appears directly over the Step 1 input textbox.
    Makes it visually clear that the action applies to that specific area.
    """

    def __init__(self, target_widget, parent=None):
        super().__init__(parent)
        self.target_widget = target_widget
        self.result_action = None

        # Frameless window with dark overlay effect
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Dialog
        )
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        self._setup_ui()
        self._position_over_target()

    def _setup_ui(self):
        """Create the overlay UI with semi-transparent background and centered card."""
        # Main layout fills the entire overlay
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Semi-transparent overlay background
        self.overlay_bg = QWidget(self)
        self.overlay_bg.setStyleSheet("background-color: rgba(0, 0, 0, 0.5);")
        main_layout.addWidget(self.overlay_bg)

        # Card container centered in the overlay
        card_layout = QVBoxLayout(self.overlay_bg)
        card_layout.setContentsMargins(20, 20, 20, 20)

        # Spacer to center vertically
        card_layout.addStretch(1)

        # The actual dialog card
        card = QWidget()
        card.setStyleSheet("""
            QWidget {
                background-color: white;
                border-radius: 12px;
                border: 2px solid #1976D2;
            }
        """)
        card.setMaximumWidth(420)
        card.setMinimumWidth(320)

        card_inner_layout = QVBoxLayout(card)
        card_inner_layout.setContentsMargins(24, 20, 24, 20)
        card_inner_layout.setSpacing(16)

        # Header with icon-like indicator
        header_layout = QHBoxLayout()
        header_layout.setSpacing(10)

        # Visual indicator pointing to Step 1
        indicator_label = QLabel("⬇")
        indicator_label.setStyleSheet("""
            QLabel {
                color: #1976D2;
                font-size: 20px;
                font-weight: bold;
                background: transparent;
                border: none;
            }
        """)
        header_layout.addWidget(indicator_label)

        title_label = QLabel("Step 1 Input")
        title_label.setStyleSheet("""
            QLabel {
                color: #1976D2;
                font-size: 15px;
                font-weight: bold;
                background: transparent;
                border: none;
            }
        """)
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        card_inner_layout.addLayout(header_layout)

        # Message
        message_label = QLabel(
            "This textbox already contains caption input.\n\n"
            "What would you like to do with the new content?"
        )
        message_label.setWordWrap(True)
        message_label.setStyleSheet("""
            QLabel {
                color: #333333;
                font-size: 13px;
                background: transparent;
                border: none;
                padding: 0px;
            }
        """)
        card_inner_layout.addWidget(message_label)

        # Button container
        button_layout = QVBoxLayout()
        button_layout.setSpacing(8)

        # Append button (primary action)
        self.append_btn = QPushButton("Append to Existing")
        self.append_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976D2;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        self.append_btn.clicked.connect(lambda: self._set_result("append"))
        button_layout.addWidget(self.append_btn)

        # Overwrite button (destructive action)
        self.overwrite_btn = QPushButton("Clear and Overwrite")
        self.overwrite_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:pressed {
                background-color: #bd2130;
            }
        """)
        self.overwrite_btn.clicked.connect(lambda: self._set_result("overwrite"))
        button_layout.addWidget(self.overwrite_btn)

        # Cancel button (subtle)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #666666;
                border: 1px solid #cccccc;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #f5f5f5;
                border-color: #999999;
            }
            QPushButton:pressed {
                background-color: #e0e0e0;
            }
        """)
        self.cancel_btn.clicked.connect(lambda: self._set_result(None))
        button_layout.addWidget(self.cancel_btn)

        card_inner_layout.addLayout(button_layout)

        # Center the card horizontally
        card_container = QHBoxLayout()
        card_container.addStretch(1)
        card_container.addWidget(card)
        card_container.addStretch(1)
        card_layout.addLayout(card_container)

        # Spacer to center vertically
        card_layout.addStretch(1)

    def _position_over_target(self):
        """Position the dialog directly over the target widget."""
        if self.target_widget:
            # Get the global position and size of the target widget
            target_global_pos = self.target_widget.mapToGlobal(self.target_widget.rect().topLeft())
            target_size = self.target_widget.size()

            # Set dialog geometry to match target widget exactly
            self.setGeometry(
                target_global_pos.x(),
                target_global_pos.y(),
                target_size.width(),
                target_size.height()
            )

    def _set_result(self, action):
        """Set the result and close the dialog."""
        self.result_action = action
        self.accept() if action else self.reject()

    def keyPressEvent(self, event):
        """Handle escape key to cancel."""
        if event.key() == Qt.Key.Key_Escape:
            self._set_result(None)
        else:
            super().keyPressEvent(event)


class OverlayHistoryButton(QPushButton):
    """
    Custom overlay button that floats in the bottom-right corner of its parent.
    Uses paintEvent for custom semi-transparent overlay appearance.
    """

    def __init__(self, text="Full Generation History", parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(165, 30)

        # Enable transparency for overlay effect
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.raise_()

    def paintEvent(self, event):
        """Custom paint for semi-transparent overlay appearance."""
        from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Determine colors based on state - with transparency (alpha)
        if not self.isEnabled():
            bg_color = QColor(200, 200, 200, 180)  # Light gray, semi-transparent
            text_color = QColor(120, 120, 120, 200)
            border_color = QColor(180, 180, 180, 200)
        elif self.underMouse():
            bg_color = QColor(21, 101, 192, 230)  # Darker blue on hover
            text_color = QColor(255, 255, 255, 255)
            border_color = QColor(13, 71, 161, 240)
        else:
            bg_color = QColor(25, 118, 210, 200)  # Blue, semi-transparent
            text_color = QColor(255, 255, 255, 240)
            border_color = QColor(21, 101, 192, 210)

        # Draw rounded rectangle background
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setPen(QPen(border_color, 1))
        painter.setBrush(QBrush(bg_color))
        painter.drawRoundedRect(rect, 8, 8)

        # Draw text
        painter.setPen(text_color)
        font = QFont()
        font.setPointSize(9)
        font.setBold(False)
        painter.setFont(font)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.text())

        painter.end()

    def reposition(self):
        """Reposition the button to the bottom-right corner of its parent."""
        if self.parent():
            parent_rect = self.parent().rect()
            margin = 8
            x = parent_rect.width() - self.width() - margin
            y = parent_rect.height() - self.height() - margin
            self.move(x, y)


class DownloadProgressDialog(QDialog):
    """Progress dialog for downloading dependencies"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Downloading Dependencies")
        self.setModal(True)
        self.setFixedSize(400, 150)

        layout = QVBoxLayout(self)

        self.status_label = QLabel("Preparing download...")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)

        self.cancelled = False

    def update_progress(self, value, total, text):
        """Update progress display"""
        if not self.cancelled:
            if total > 0:
                self.progress_bar.setValue(int((value / total) * 100))
            self.status_label.setText(text)

    def reject(self):
        """Handle cancel"""
        self.cancelled = True
        super().reject()


class DependencyDownloader(QThread):
    """Thread for downloading faster-whisper-xxl executable"""

    progress = pyqtSignal(int, int, str)  # value, total, text
    finished = pyqtSignal(bool, str)  # success, message

    def __init__(self, url, files_to_check, bin_dir):
        super().__init__()
        self.url = url
        self.files_to_check = files_to_check
        self.bin_dir = bin_dir
        self.cancelled = False

    def run(self):
        """Download and extract dependencies"""
        try:
            self.progress.emit(0, 100, "Starting download...")

            # Create bin directory
            os.makedirs(self.bin_dir, exist_ok=True)

            # Download file
            response = requests.get(self.url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.7z') as temp_file:
                temp_path = temp_file.name
                downloaded = 0

                for chunk in response.iter_content(chunk_size=8192):
                    if self.cancelled:
                        return

                    temp_file.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        progress_percent = (downloaded / total_size) * 70  # 70% for download
                        self.progress.emit(int(progress_percent), 100, f"Downloading... {downloaded / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB")

            if self.cancelled:
                return

            self.progress.emit(70, 100, "Extracting files...")

            # Extract using system 7z command (same method as faster-whisper-xxl-gui.py)
            try:
                # Find 7z executable (matches original implementation)
                sevenzip_executable = shutil.which('7z')
                if not sevenzip_executable and sys.platform == "win32":
                    prog_files = os.environ.get("ProgramFiles", "C:\\Program Files")
                    prog_files_x86 = os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")
                    possible_paths = [
                        os.path.join(prog_files, "7-Zip", "7z.exe"),
                        os.path.join(prog_files_x86, "7-Zip", "7z.exe")
                    ]
                    for path in possible_paths:
                        if os.path.exists(path):
                            sevenzip_executable = path
                            break

                if not sevenzip_executable:
                    error_msg = ("7-Zip/p7zip executable not found. Please install it and ensure it's in your system's PATH.\n\n"
                                "Windows: Download from https://www.7-zip.org/\n"
                                "Linux: sudo apt install p7zip-full\n"
                                "Mac: brew install p7zip")
                    self.finished.emit(False, error_msg)
                    return

                self.progress.emit(75, 100, "Extracting with 7-Zip...")

                # Use same command format as original
                command = [sevenzip_executable, 'x', temp_path, f'-o{self.bin_dir}', '-y']
                result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300)

                if result.returncode != 0:
                    self.finished.emit(False, f"7-Zip extraction failed with code {result.returncode}.\nError: {result.stderr or result.stdout}")
                    return

                self.progress.emit(80, 100, "Moving files...")

                # Find source directory and move files (matches reference implementation)
                with tempfile.TemporaryDirectory(prefix="whisper_extract_") as extract_dir:
                    # Extract to temp directory first
                    command = [sevenzip_executable, 'x', temp_path, f'-o{extract_dir}', '-y']
                    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300)

                    if result.returncode != 0:
                        self.finished.emit(False, f"7-Zip extraction failed with code {result.returncode}.\nError: {result.stderr or result.stdout}")
                        return

                    # Find source directory (matches reference logic exactly)
                    extracted_items = os.listdir(extract_dir)
                    logging.info(f"Items in temp_extract: {extracted_items}")

                    source_dir = None
                    for item in extracted_items:
                        path = os.path.join(extract_dir, item)
                        if os.path.isdir(path):
                            source_dir = path
                            break

                    if not source_dir:
                        # Check if files are directly in extract_dir
                        if any(f in extracted_items for f in self.files_to_check):
                            source_dir = extract_dir
                        else:
                            self.finished.emit(False, f"Extraction failed: Could not find required files in {extract_dir}.")
                            return

                    logging.info(f"Source directory for moving files: {source_dir}")

                    # Ensure destination directory exists
                    os.makedirs(self.bin_dir, exist_ok=True)

                    # Move all files from source to destination
                    for item_name in os.listdir(source_dir):
                        source_path = os.path.join(source_dir, item_name)
                        dest_path = os.path.join(self.bin_dir, item_name)

                        # Remove existing file/dir at destination
                        if os.path.isdir(dest_path):
                            shutil.rmtree(dest_path)
                        elif os.path.exists(dest_path):
                            os.remove(dest_path)

                        # Move the file
                        shutil.move(source_path, dest_path)

            except Exception as e:
                self.finished.emit(False, f"Extraction failed: {str(e)}")
                return

            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass

            self.progress.emit(95, 100, "Verifying files...")

            # Verify files exist (matches reference)
            missing_files = []
            for file_name in self.files_to_check:
                file_path = os.path.join(self.bin_dir, file_name)
                if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                    missing_files.append(file_name)

            if missing_files:
                self.finished.emit(False, f"Verification failed: Missing or empty files after extraction: {', '.join(missing_files)}")
            else:
                self.progress.emit(100, 100, "Download complete!")
                self.finished.emit(True, "Dependencies downloaded successfully")

        except Exception as e:
            self.finished.emit(False, f"Download failed: {str(e)}")

    def cancel(self):
        """Cancel download"""
        self.cancelled = True


class HardwareOptimizationDialog(QDialog):
    """Dialog for showing hardware detection results and optimization recommendations (matches reference)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Hardware Optimization")
        self.setModal(True)
        self.setMinimumSize(500, 400)

        # Detect hardware
        self.hardware_info = detect_hardware_capabilities()
        self.recommendations = get_recommended_settings(self.hardware_info)
        self.user_accepted = False

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Title
        title_label = QLabel("Hardware Optimization")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title_label)

        # Description
        desc_label = QLabel("Your system has been analyzed. Below are the detected hardware specifications and recommended settings for optimal performance.")
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("margin-bottom: 15px;")
        layout.addWidget(desc_label)

        # Hardware info section
        hw_group = QGroupBox("Detected Hardware")
        hw_layout = QFormLayout(hw_group)

        # System info
        hw_layout.addRow("System:", QLabel(f"{self.hardware_info.get('platform', 'Unknown')} {self.hardware_info.get('platform_version', '')}"))
        hw_layout.addRow("CPU Cores:", QLabel(str(self.hardware_info.get('cpu_count', 'Unknown'))))
        hw_layout.addRow("RAM:", QLabel(f"{self.hardware_info.get('ram_gb', 0):.1f} GB"))

        # GPU info
        if self.hardware_info.get("has_cuda"):
            hw_layout.addRow("GPU:", QLabel(self.hardware_info.get("gpu_name", "Unknown NVIDIA GPU")))
            hw_layout.addRow("GPU Memory:", QLabel(f"{self.hardware_info.get('gpu_memory_gb', 0):.1f} GB"))
            hw_layout.addRow("CUDA Version:", QLabel(self.hardware_info.get("cuda_version", "Unknown")))
        else:
            hw_layout.addRow("GPU:", QLabel("No CUDA-capable GPU detected"))

        layout.addWidget(hw_group)

        # Recommendations section
        rec_group = QGroupBox("Recommended Settings")
        rec_layout = QFormLayout(rec_group)

        # Device
        device_text = self.recommendations["device"].upper()
        if self.recommendations["device"] == "cuda":
            device_text += " (GPU acceleration enabled)"
        rec_layout.addRow("Device:", QLabel(device_text))

        # Model
        model_text = self.recommendations["model"]
        rec_layout.addRow("Model:", QLabel(model_text))

        # Compute type
        compute_text = self.recommendations["compute_type"]
        rec_layout.addRow("Compute Type:", QLabel(compute_text))

        # VAD method
        vad_text = self.recommendations["vad_method"]
        if "pyannote" in vad_text:
            vad_text += " (High accuracy with CUDA)"
        else:
            vad_text += " (CPU-optimized)"
        rec_layout.addRow("VAD Method:", QLabel(vad_text))

        layout.addWidget(rec_group)

        # Buttons
        button_layout = QHBoxLayout()

        accept_btn = QPushButton("Apply Recommendations")
        accept_btn.clicked.connect(self.accept_recommendations)
        accept_btn.setStyleSheet("""
            QPushButton {
                background: #28a745;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #218838;
            }
        """)

        decline_btn = QPushButton("Keep Current Settings")
        decline_btn.clicked.connect(self.reject)

        button_layout.addWidget(decline_btn)
        button_layout.addWidget(accept_btn)
        layout.addLayout(button_layout)

    def accept_recommendations(self):
        self.user_accepted = True
        self.accept()


class PythonRuntimeDownloader(QThread):
    """Download and prepare a private Python 3.12 runtime (Windows-focused)."""

    progress = pyqtSignal(int, int, str)  # value, total, text
    finished = pyqtSignal(bool, str, str)  # success, message, python_exe_path

    def __init__(self, target_dir):
        super().__init__()
        self.target_dir = target_dir
        self.cancelled = False

    def run(self):
        try:
            if sys.platform != 'win32':
                self.finished.emit(False, "Auto Python runtime download is only implemented for Windows.", "")
                return

            import zipfile
            os.makedirs(self.target_dir, exist_ok=True)

            # URLs for Python 3.12 runtime and get-pip
            py_url = "https://www.python.org/ftp/python/3.12.7/python-3.12.7-embed-amd64.zip"
            get_pip_url = "https://bootstrap.pypa.io/get-pip.py"

            # 1) Download embeddable zip
            self.progress.emit(0, 100, "Downloading Python 3.12 runtime (embed)...")
            embed_zip = os.path.join(self.target_dir, "python312_embed.zip")
            if not self._download(py_url, embed_zip, 0, 60):
                return

            if self.cancelled:
                return

            # 2) Extract
            self.progress.emit(60, 100, "Extracting Python runtime...")
            with zipfile.ZipFile(embed_zip, 'r') as zf:
                zf.extractall(self.target_dir)

            # 3) Enable site in embedded by ensuring 'import site' (uncomment if present) and '.' path
            self.progress.emit(70, 100, "Configuring Python runtime...")
            pth_files = [f for f in os.listdir(self.target_dir) if f.endswith('._pth')]
            for pth in pth_files:
                pth_path = os.path.join(self.target_dir, pth)
                try:
                    with open(pth_path, 'r', encoding='utf-8') as fh:
                        lines = fh.read().splitlines()
                except Exception:
                    lines = []

                found_import_site = False
                found_dot = False
                new_lines = []
                for ln in lines:
                    s = ln.strip()
                    if s.startswith('#') and 'import site' in s:
                        new_lines.append('import site')
                        found_import_site = True
                    else:
                        if s == 'import site':
                            found_import_site = True
                        if s == '.':
                            found_dot = True
                        new_lines.append(ln)
                if not found_dot:
                    new_lines.append('.')
                if not found_import_site:
                    new_lines.append('import site')
                try:
                    with open(pth_path, 'w', encoding='utf-8') as fh:
                        fh.write('\n'.join(new_lines) + '\n')
                except Exception:
                    pass

            # Ensure Lib\site-packages and Scripts exist
            try:
                os.makedirs(os.path.join(self.target_dir, 'Lib', 'site-packages'), exist_ok=True)
                os.makedirs(os.path.join(self.target_dir, 'Scripts'), exist_ok=True)
            except Exception:
                pass

            # Important: disable strict path isolation by removing the *._pth file entirely
            # so that build sdists (e.g., docopt) can import modules from their build directory.
            try:
                for pth in pth_files:
                    pth_path = os.path.join(self.target_dir, pth)
                    if os.path.exists(pth_path):
                        os.remove(pth_path)
            except Exception:
                pass

            # 4) Download get-pip.py
            self.progress.emit(75, 100, "Downloading pip bootstrap...")
            get_pip_path = os.path.join(self.target_dir, 'get-pip.py')
            if not self._download(get_pip_url, get_pip_path, 75, 85):
                return

            if self.cancelled:
                return

            # 5) Run get-pip.py
            self.progress.emit(85, 100, "Installing pip into runtime...")
            python_exe = os.path.join(self.target_dir, 'python.exe')
            if not os.path.exists(python_exe):
                python_exe = os.path.join(self.target_dir, 'pythonw.exe') if os.path.exists(os.path.join(self.target_dir, 'pythonw.exe')) else python_exe

            import subprocess
            proc = subprocess.run([python_exe, get_pip_path, '--no-warn-script-location'], capture_output=True, text=True)
            if proc.returncode != 0:
                self.finished.emit(False, f"pip bootstrap failed: {proc.stderr or proc.stdout}", "")
                return

            # 6) Verify pip
            self.progress.emit(95, 100, "Verifying pip...")
            proc2 = subprocess.run([python_exe, '-m', 'pip', '--version'], capture_output=True, text=True)
            if proc2.returncode != 0:
                self.finished.emit(False, f"pip verification failed: {proc2.stderr or proc2.stdout}", "")
                return

            self.progress.emit(100, 100, "Python runtime ready")
            self.finished.emit(True, "Python 3.12 runtime prepared.", python_exe)
        except Exception as e:
            self.finished.emit(False, f"Runtime setup error: {e}", "")

    def _download(self, url, dest, start_pct, end_pct):
        try:
            import requests
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total = int(r.headers.get('content-length') or 0)
                done = 0
                with open(dest, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if self.cancelled:
                            return False
                        if not chunk:
                            continue
                        f.write(chunk)
                        done += len(chunk)
                        pct = start_pct
                        if total:
                            pct = start_pct + int((done / total) * (end_pct - start_pct))
                        self.progress.emit(pct, 100, f"Downloading... {done//(1024*1024)}MB")
            return True
        except Exception as e:
            self.finished.emit(False, f"Download failed: {e}", "")
            return False

class FileDropArea(QWidget):
    """Modern drag-and-drop file input area"""

    file_dropped = pyqtSignal(str)  # Single file (for backward compatibility)
    files_dropped = pyqtSignal(list)  # Multiple files (for batch queue)

    def __init__(self, parent=None):
        super().__init__(parent)  # Pass parent to prevent top-level window flash
        self.setAcceptDrops(True)
        self.setMinimumHeight(70)
        self.setMaximumHeight(70)
        self.setupUI()

    def setupUI(self):
        """Setup the drop area UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)

        # Icon and text
        icon_label = QLabel("📁")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet("font-size: 24px;")

        text_label = QLabel("Drop audio/video file(s) here or click to browse")
        text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text_label.setStyleSheet("color: #666; font-size: 12px;")

        layout.addWidget(icon_label)
        layout.addWidget(text_label)

        # Style the drop area
        self.setStyleSheet("""
            FileDropArea {
                border: 2px dashed #ccc;
                border-radius: 8px;
                background-color: #f8f9fa;
            }
            FileDropArea:hover {
                border-color: #0078d4;
                background-color: #e3f2fd;
            }
        """)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            self.setStyleSheet("""
                FileDropArea {
                    border: 2px dashed #0078d4;
                    background-color: #e3f2fd;
                    border-radius: 8px;
                }
            """)
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            FileDropArea {
                border: 2px dashed #ccc;
                border-radius: 8px;
                background-color: #f8f9fa;
            }
            FileDropArea:hover {
                border-color: #0078d4;
                background-color: #e3f2fd;
            }
        """)

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            # Filter for media files
            media_extensions = {'.mp3', '.wav', '.mp4', '.avi', '.mov', '.mkv', '.m4a', '.aac', '.flac', '.ogg', '.webm'}
            valid_files = [f for f in files if os.path.splitext(f.lower())[1] in media_extensions]

            if valid_files:
                if len(valid_files) == 1:
                    # Single file - emit both signals for compatibility
                    self.file_dropped.emit(valid_files[0])
                    self.files_dropped.emit(valid_files)
                else:
                    # Multiple files - only emit files_dropped for batch processing
                    self.files_dropped.emit(valid_files)
        self.dragLeaveEvent(event)

    def mousePressEvent(self, event):
        # Handle click to browse - support multiple file selection
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio/Video File(s) - Multiple files will be added to batch queue", "",
            "Media Files (*.mp3 *.wav *.mp4 *.avi *.mov *.mkv *.m4a *.aac *.flac *.ogg *.webm);;All Files (*)"
        )
        if file_paths:
            if len(file_paths) == 1:
                # Single file - emit both signals for compatibility
                self.file_dropped.emit(file_paths[0])
                self.files_dropped.emit(file_paths)
            else:
                # Multiple files - only emit files_dropped for batch processing
                self.files_dropped.emit(file_paths)


class SplitColorProgressBar(QWidget):
    """
    Custom progress bar with split text color - white over the filled (blue) portion,
    black over the unfilled portion.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0
        self._maximum = 100
        self._format = "%p%"
        self._visible = True

        # Main layout
        self.setFixedHeight(20)

        # Create the actual progress bar (with transparent text)
        self._progress_bar = QProgressBar(self)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 4px;
                background-color: #f0f0f0;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #007acc;
                border-radius: 3px;
            }
        """)

        # Black text label (shown over unfilled area)
        self._black_label = QLabel(self)
        self._black_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._black_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-weight: bold;
                color: #333;
                background: transparent;
            }
        """)

        # White text label (shown over filled area)
        self._white_label = QLabel(self)
        self._white_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._white_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-weight: bold;
                color: #fff;
                background: transparent;
            }
        """)

        self._update_geometry()
        self._update_text()

    def _update_geometry(self):
        """Update the geometry of child widgets to match parent size"""
        rect = self.rect()
        self._progress_bar.setGeometry(rect)
        self._black_label.setGeometry(rect)
        self._white_label.setGeometry(rect)

        # Calculate clip region for white label based on fill percentage
        if self._maximum > 0:
            fill_width = int((self._value / self._maximum) * rect.width())
        else:
            fill_width = 0

        # Handle edge cases where masks don't work properly
        if fill_width <= 0:
            # No fill - show only black text, hide white completely
            self._white_label.hide()
            self._black_label.clearMask()
            self._black_label.show()
        elif fill_width >= rect.width():
            # Fully filled - show only white text, hide black completely
            self._black_label.hide()
            self._white_label.clearMask()
            self._white_label.show()
        else:
            # Partial fill - use masks for both labels
            self._black_label.show()
            self._white_label.show()

            from PyQt6.QtGui import QRegion
            filled_region = QRegion(0, 0, fill_width, rect.height())
            unfilled_region = QRegion(fill_width, 0, rect.width() - fill_width, rect.height())

            # Apply masks - white text only visible over filled area, black over unfilled
            self._white_label.setMask(filled_region)
            self._black_label.setMask(unfilled_region)

    def _update_text(self):
        """Update the text on both labels"""
        if self._maximum > 0:
            percent = int((self._value / self._maximum) * 100)
        else:
            percent = 0

        text = self._format.replace("%p", str(percent)).replace("%v", str(self._value)).replace("%m", str(self._maximum))
        self._black_label.setText(text)
        self._white_label.setText(text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_geometry()

    def setValue(self, value):
        self._value = value
        self._progress_bar.setValue(value)
        self._update_text()
        self._update_geometry()

    def value(self):
        return self._value

    def setMaximum(self, maximum):
        self._maximum = maximum
        self._progress_bar.setMaximum(maximum)
        self._update_text()
        self._update_geometry()

    def maximum(self):
        return self._maximum

    def setFormat(self, format_str):
        self._format = format_str
        self._update_text()

    def format(self):
        return self._format

    def setVisible(self, visible):
        super().setVisible(visible)
        self._visible = visible

    def setStyleSheet(self, style):
        # Ignore external stylesheet calls - we manage our own styles
        pass


class ProgressPanel(QWidget):
    """
    Clean progress panel with animated indicators for transcription status.
    Replaces raw console output with user-friendly progress display.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

        # State tracking
        self.current_stage = None
        self.stages = [
            ("loading", "Loading model"),
            ("transcribing", "Transcribing audio"),
            ("aligning", "Aligning timestamps"),
            ("diarizing", "Identifying speakers"),
            ("saving", "Saving output")
        ]
        self.stage_status = {}  # 'pending', 'active', 'complete'
        self.reset_stages()

        # Animation state
        self.dot_count = 0
        self.pulse_opacity = 1.0
        self.pulse_direction = -1  # -1 = fading, 1 = brightening
        self.elapsed_seconds = 0
        self.is_running = False
        self.current_file = ""
        self.progress_percent = 0
        self.console_visible = False

        # Audio duration for timestamp-based progress (in seconds)
        self.audio_duration = None

        # Raw output buffer
        self.raw_output = ""

        self._setup_ui()
        self._setup_timers()

    def _setup_ui(self):
        """Setup the progress panel UI"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(8)

        # Horizontal container for both sections
        self.top_container = QWidget(self)  # Parent to prevent top-level window flash
        self.top_container.setMaximumHeight(320)  # Constrain height to prevent expansion
        top_layout = QHBoxLayout(self.top_container)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)

        # LEFT SIDE: Progress section
        self.progress_group = QGroupBox("Processing Status")
        self.progress_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
        progress_layout = QVBoxLayout(self.progress_group)
        progress_layout.setSpacing(12)
        progress_layout.setContentsMargins(15, 20, 15, 15)

        # File name label
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("""
            QLabel {
                font-size: 13px;
                color: #333;
                padding: 8px 12px;
                background: #f0f4f8;
                border-radius: 6px;
                font-weight: 500;
            }
        """)
        self.file_label.setWordWrap(True)
        progress_layout.addWidget(self.file_label)

        # Current stage with animated indicator
        self.stage_container = QWidget(self)  # Parent to prevent top-level window flash
        stage_layout = QHBoxLayout(self.stage_container)
        stage_layout.setContentsMargins(0, 5, 0, 5)
        stage_layout.setSpacing(10)

        # Breathing pulse indicator
        self.pulse_indicator = QLabel("●")
        self.pulse_indicator.setFixedWidth(20)
        self.pulse_indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pulse_indicator.setStyleSheet("font-size: 16px; color: #2196F3;")
        stage_layout.addWidget(self.pulse_indicator)

        # Stage text with animated dots
        self.stage_label = QLabel("Ready to start")
        self.stage_label.setStyleSheet("""
            QLabel {
                font-size: 15px;
                font-weight: 600;
                color: #1976D2;
            }
        """)
        stage_layout.addWidget(self.stage_label)
        stage_layout.addStretch()

        progress_layout.addWidget(self.stage_container)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ddd;
                border-radius: 6px;
                background: #f5f5f5;
                height: 22px;
                text-align: center;
                font-weight: 500;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #42A5F5, stop:1 #1E88E5);
                border-radius: 5px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)

        # Elapsed time
        self.elapsed_label = QLabel("⏱ Elapsed: 0:00")
        self.elapsed_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #666;
                padding-top: 3px;
            }
        """)
        progress_layout.addWidget(self.elapsed_label)

        # Stage checklist
        self.checklist_widget = QWidget(self)  # Parent to prevent top-level window flash
        checklist_layout = QVBoxLayout(self.checklist_widget)
        checklist_layout.setContentsMargins(5, 10, 5, 10)  # Extra bottom margin for descenders
        checklist_layout.setSpacing(2)  # Tighter spacing, rely on label height

        self.stage_labels = {}
        for stage_id, stage_name in self.stages:
            stage_row = QHBoxLayout()
            stage_row.setSpacing(8)

            # Status icon
            icon_label = QLabel("○")
            icon_label.setFixedWidth(18)
            icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            icon_label.setStyleSheet("font-size: 14px; color: #ccc;")
            icon_label.setObjectName(f"icon_{stage_id}")
            stage_row.addWidget(icon_label)

            # Stage name - use min-height to ensure descenders aren't clipped
            name_label = QLabel(stage_name)
            name_label.setMinimumHeight(20)  # Ensure enough height for descenders
            name_label.setStyleSheet("font-size: 13px; color: #888;")
            name_label.setObjectName(f"name_{stage_id}")
            stage_row.addWidget(name_label)
            stage_row.addStretch()

            self.stage_labels[stage_id] = (icon_label, name_label)
            checklist_layout.addLayout(stage_row)

        progress_layout.addWidget(self.checklist_widget)

        top_layout.addWidget(self.progress_group)

        # RIGHT SIDE: Static Transcripts section
        self.transcripts_group = QGroupBox("Static Transcripts for Premiere Pro")
        self.transcripts_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
        transcripts_layout = QVBoxLayout(self.transcripts_group)
        transcripts_layout.setSpacing(10)
        transcripts_layout.setContentsMargins(15, 20, 15, 15)

        # Instructions
        instructions = QLabel(
            'In Adobe Premiere, create individual sequence(s) with video/audio. '
            'In the "Text" panel action menu, select "Import static transcript" '
            'and upload file(s) below to their respective sequences to sync text with Premiere.'
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #555;
                line-height: 1.4;
                padding: 8px;
                background: #f8f9fa;
                border-radius: 4px;
            }
        """)
        transcripts_layout.addWidget(instructions)

        # Scrollable file list
        self.transcripts_scroll = QScrollArea(self)  # Parent to prevent top-level window flash
        self.transcripts_scroll.setWidgetResizable(True)
        self.transcripts_scroll.setStyleSheet("""
            QScrollArea {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                background: white;
            }
        """)

        self.transcripts_list_widget = QWidget(self)  # Parent to prevent top-level window flash
        self.transcripts_list_layout = QVBoxLayout(self.transcripts_list_widget)
        self.transcripts_list_layout.setContentsMargins(8, 8, 8, 8)
        self.transcripts_list_layout.setSpacing(4)
        self.transcripts_list_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Placeholder label for empty state
        self.no_transcripts_label = QLabel("No static transcripts generated yet.")
        self.no_transcripts_label.setStyleSheet("color: #999; font-size: 12px; padding: 10px;")
        self.no_transcripts_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.transcripts_list_layout.addWidget(self.no_transcripts_label)

        self.transcripts_scroll.setWidget(self.transcripts_list_widget)
        transcripts_layout.addWidget(self.transcripts_scroll)

        # Add overlay history button to bottom-right of transcripts_group
        self.history_btn = OverlayHistoryButton("Full Generation History", self.transcripts_group)
        # Connect to parent's show_session_history_dialog method
        self.history_btn.clicked.connect(lambda: self.parent.show_session_history_dialog() if self.parent else None)
        self.history_btn.reposition()

        # Install resize event filter to reposition button when group resizes
        class TranscriptsResizeFilter(QObject):
            def __init__(self, button, parent=None):
                super().__init__(parent)
                self.button = button

            def eventFilter(self, obj, event):
                if event.type() == QEvent.Type.Resize:
                    self.button.reposition()
                return False

        self._transcripts_resize_filter = TranscriptsResizeFilter(self.history_btn, self)
        self.transcripts_group.installEventFilter(self._transcripts_resize_filter)

        top_layout.addWidget(self.transcripts_group)

        # Add top container to main layout
        main_layout.addWidget(self.top_container)

        # Toggle button for console
        self.toggle_btn = QPushButton("▶ Show Console Output")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px 12px;
                font-size: 12px;
                color: #666;
                text-align: left;
            }
            QPushButton:hover {
                background: #f5f5f5;
                border-color: #ccc;
            }
            QPushButton:checked {
                background: #f0f4f8;
            }
        """)
        self.toggle_btn.clicked.connect(self._toggle_console)
        main_layout.addWidget(self.toggle_btn)

        # Console output (hidden by default)
        self.console_group = QGroupBox("Console Output")
        self.console_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
        """)
        console_layout = QVBoxLayout(self.console_group)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        font = QFont("Courier New" if sys.platform == "win32" else "Monospace")
        font.setPointSize(10)
        self.output_text.setFont(font)
        self.output_text.setStyleSheet("""
            QTextEdit {
                background: #f8f9fa;
                color: #333;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        self.output_text.setMinimumHeight(200)
        console_layout.addWidget(self.output_text)

        self.console_group.setVisible(False)
        main_layout.addWidget(self.console_group, 1)

        # Add stretch at end to push content up when console is hidden
        self.bottom_stretch = main_layout.addStretch()

    def _setup_timers(self):
        """Setup animation timers"""
        # Dot animation timer (cycles every 500ms)
        self.dot_timer = QTimer(self)
        self.dot_timer.timeout.connect(self._animate_dots)
        self.dot_timer.setInterval(400)

        # Pulse animation timer (smooth breathing effect)
        self.pulse_timer = QTimer(self)
        self.pulse_timer.timeout.connect(self._animate_pulse)
        self.pulse_timer.setInterval(50)

        # Elapsed time timer
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.timeout.connect(self._update_elapsed)
        self.elapsed_timer.setInterval(1000)

    def _animate_dots(self):
        """Animate the dots after the current stage text"""
        if not self.is_running or not self.current_stage:
            return

        self.dot_count = (self.dot_count + 1) % 4
        dots = "." * self.dot_count

        # Get current stage name
        stage_name = None
        for sid, sname in self.stages:
            if sid == self.current_stage:
                stage_name = sname
                break

        if stage_name:
            self.stage_label.setText(f"{stage_name}{dots}")

    def _animate_pulse(self):
        """Animate the breathing pulse indicator"""
        if not self.is_running:
            return

        # Adjust opacity for breathing effect
        self.pulse_opacity += self.pulse_direction * 0.04

        if self.pulse_opacity <= 0.3:
            self.pulse_opacity = 0.3
            self.pulse_direction = 1
        elif self.pulse_opacity >= 1.0:
            self.pulse_opacity = 1.0
            self.pulse_direction = -1

        # Apply opacity to pulse indicator color
        alpha = int(self.pulse_opacity * 255)
        self.pulse_indicator.setStyleSheet(f"font-size: 16px; color: rgba(33, 150, 243, {alpha});")

    def _update_elapsed(self):
        """Update the elapsed time display"""
        if not self.is_running:
            return

        self.elapsed_seconds += 1
        minutes = self.elapsed_seconds // 60
        seconds = self.elapsed_seconds % 60
        self.elapsed_label.setText(f"⏱ Elapsed: {minutes}:{seconds:02d}")

    def _toggle_console(self):
        """Toggle console visibility"""
        self.console_visible = self.toggle_btn.isChecked()
        self.console_group.setVisible(self.console_visible)

        if self.console_visible:
            self.toggle_btn.setText("▼ Hide Console Output")
        else:
            self.toggle_btn.setText("▶ Show Console Output")

    def reset_stages(self):
        """Reset all stages to pending"""
        for stage_id, _ in self.stages:
            self.stage_status[stage_id] = 'pending'
        self.current_stage = None
        # Reset timestamp-based progress tracking
        self._last_timestamp_progress = 0

    def set_audio_duration(self, duration_seconds):
        """Set the audio duration for timestamp-based progress calculation"""
        self.audio_duration = duration_seconds

    def configure_stages(self, show_diarization=True):
        """Configure which stages are visible based on settings"""
        self.show_diarization = show_diarization

        # Show/hide the diarization row in the checklist
        if 'diarizing' in self.stage_labels:
            icon_label, name_label = self.stage_labels['diarizing']
            # Find the parent layout row and hide/show it
            icon_label.setVisible(show_diarization)
            name_label.setVisible(show_diarization)

    def start_processing(self, filename="", show_diarization=True):
        """Start processing - reset state and start animations"""
        self.is_running = True
        self.elapsed_seconds = 0
        self.progress_percent = 0
        self.dot_count = 0
        self.pulse_opacity = 1.0
        self.raw_output = ""

        # Configure visible stages
        self.configure_stages(show_diarization=show_diarization)

        # Reset stages
        self.reset_stages()
        self._update_stage_display()

        # Set file name
        if filename:
            display_name = os.path.basename(filename)
            self.file_label.setText(f"📁 {display_name}")
            self.current_file = filename
        else:
            self.file_label.setText("Processing...")

        # Set initial stage
        self.set_stage("loading")

        # Reset progress
        self.progress_bar.setValue(0)
        self.elapsed_label.setText("⏱ Elapsed: 0:00")

        # Start timers
        self.dot_timer.start()
        self.pulse_timer.start()
        self.elapsed_timer.start()

        # Clear console
        self.output_text.clear()

    def stop_processing(self, success=True):
        """Stop processing - stop animations"""
        self.is_running = False

        # Stop timers
        self.dot_timer.stop()
        self.pulse_timer.stop()
        self.elapsed_timer.stop()

        # Update display
        if success:
            self.stage_label.setText("Complete!")
            self.stage_label.setStyleSheet("""
                QLabel {
                    font-size: 15px;
                    font-weight: 600;
                    color: #4CAF50;
                }
            """)
            self.pulse_indicator.setStyleSheet("font-size: 16px; color: #4CAF50;")
            self.pulse_indicator.setText("✓")
            self.progress_bar.setValue(100)

            # Mark all stages as complete
            for stage_id, _ in self.stages:
                self.stage_status[stage_id] = 'complete'
            self._update_stage_display()
        else:
            self.stage_label.setText("Stopped")
            self.stage_label.setStyleSheet("""
                QLabel {
                    font-size: 15px;
                    font-weight: 600;
                    color: #f44336;
                }
            """)
            self.pulse_indicator.setStyleSheet("font-size: 16px; color: #f44336;")
            self.pulse_indicator.setText("✕")

    def set_stage(self, stage_id):
        """Set the current processing stage"""
        # Mark previous stages as complete
        for sid, _ in self.stages:
            if sid == stage_id:
                break
            self.stage_status[sid] = 'complete'

        # Set current stage as active
        self.current_stage = stage_id
        self.stage_status[stage_id] = 'active'
        self.dot_count = 0

        # Detect if using Faster-Whisper-XXL for appropriate progress values
        # Check both the active tab AND the engine selection
        is_faster_whisper = False
        if self.parent:
            # If on WhisperX tab, always use WhisperX weights
            if hasattr(self.parent, 'tab_widget') and hasattr(self.parent, 'whisperx_tab_index'):
                current_tab = self.parent.tab_widget.currentIndex()
                if current_tab == self.parent.whisperx_tab_index:
                    is_faster_whisper = False  # WhisperX tab = use WhisperX weights
                elif hasattr(self.parent, 'engine_faster_whisper'):
                    is_faster_whisper = self.parent.engine_faster_whisper.isChecked()
            elif hasattr(self.parent, 'engine_faster_whisper'):
                is_faster_whisper = self.parent.engine_faster_whisper.isChecked()

        # Update progress bar to start of this stage
        if is_faster_whisper:
            # Faster-Whisper-XXL: Transcribing takes 80%
            stage_progress = {
                'loading': 0,
                'transcribing': 10,
                'aligning': 90,
                'diarizing': 90,
                'saving': 90
            }
        else:
            # WhisperX: Full pipeline
            stage_progress = {
                'loading': 0,
                'transcribing': 5,
                'aligning': 65,
                'diarizing': 80,
                'saving': 95
            }
        if stage_id in stage_progress:
            self.set_progress(stage_progress[stage_id])

        # Update stage label
        stage_name = None
        for sid, sname in self.stages:
            if sid == stage_id:
                stage_name = sname
                break

        if stage_name:
            self.stage_label.setText(stage_name)
            self.stage_label.setStyleSheet("""
                QLabel {
                    font-size: 15px;
                    font-weight: 600;
                    color: #1976D2;
                }
            """)

        # Reset pulse indicator
        self.pulse_indicator.setText("●")
        self.pulse_indicator.setStyleSheet("font-size: 16px; color: rgba(33, 150, 243, 255);")

        self._update_stage_display()

    def _update_stage_display(self):
        """Update the stage checklist display"""
        for stage_id, (icon_label, name_label) in self.stage_labels.items():
            status = self.stage_status.get(stage_id, 'pending')

            if status == 'complete':
                icon_label.setText("✓")
                icon_label.setStyleSheet("font-size: 14px; color: #4CAF50; font-weight: bold;")
                name_label.setStyleSheet("font-size: 13px; color: #4CAF50;")
            elif status == 'active':
                icon_label.setText("●")
                icon_label.setStyleSheet("font-size: 14px; color: #2196F3;")
                name_label.setStyleSheet("font-size: 13px; color: #1976D2; font-weight: 600;")
            else:  # pending
                icon_label.setText("○")
                icon_label.setStyleSheet("font-size: 14px; color: #ccc;")
                name_label.setStyleSheet("font-size: 13px; color: #888;")

    def set_progress(self, percent):
        """Set the progress bar value and update text color for contrast"""
        self.progress_percent = max(0, min(100, percent))
        self.progress_bar.setValue(int(self.progress_percent))

        # Change text color to white when fill covers the text (around 50%)
        text_color = "#fff" if self.progress_percent >= 45 else "#333"
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #ddd;
                border-radius: 6px;
                background: #f5f5f5;
                height: 22px;
                text-align: center;
                font-weight: 500;
                color: {text_color};
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #42A5F5, stop:1 #1E88E5);
                border-radius: 5px;
            }}
        """)

    def append_output(self, text):
        """Append text to console output and parse for progress info"""
        self.raw_output += text

        # Append to console
        cursor = self.output_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.output_text.setTextCursor(cursor)
        self.output_text.ensureCursorVisible()

        # Parse output for stage detection
        self._parse_output_for_progress(text)

    def _parse_output_for_progress(self, text):
        """Parse output text to detect stages and progress based on actual WhisperX/Faster-Whisper output"""
        import re

        # Detect if using Faster-Whisper-XXL (no alignment/diarization stages)
        # Check both the active tab AND the engine selection
        is_faster_whisper = False
        if self.parent:
            # If on WhisperX tab, always use WhisperX weights
            if hasattr(self.parent, 'tab_widget') and hasattr(self.parent, 'whisperx_tab_index'):
                current_tab = self.parent.tab_widget.currentIndex()
                if current_tab == self.parent.whisperx_tab_index:
                    is_faster_whisper = False  # WhisperX tab = use WhisperX weights
                elif hasattr(self.parent, 'engine_faster_whisper'):
                    is_faster_whisper = self.parent.engine_faster_whisper.isChecked()
            elif hasattr(self.parent, 'engine_faster_whisper'):
                is_faster_whisper = self.parent.engine_faster_whisper.isChecked()

        if is_faster_whisper:
            # Faster-Whisper-XXL: Transcribing takes 80% (skips alignment/diarization)
            # Loading: 0-10%, Transcribing: 10-90%, Saving: 90-100%
            stage_weights = {
                'loading': (0, 10),
                'transcribing': (10, 90),
                'aligning': (90, 90),  # Not used, but keep for consistency
                'diarizing': (90, 90),  # Not used, but keep for consistency
                'saving': (90, 100)
            }
        else:
            # WhisperX: Full pipeline with alignment and diarization
            # Loading: 0-5%, Transcribing: 5-65%, Aligning: 65-80%, Diarizing: 80-95%, Saving: 95-100%
            stage_weights = {
                'loading': (0, 5),
                'transcribing': (5, 65),
                'aligning': (65, 80),
                'diarizing': (80, 95),
                'saving': (95, 100)
            }

        # Process LINE BY LINE to detect stage changes in sequence
        # This ensures we see each stage even when output is buffered together
        lines = text.split('\n')

        for line in lines:
            line_lower = line.lower().strip()
            if not line_lower:
                continue

            # BATCH MODE: Detect new file starting - reset progress for each file
            # Pattern: "[HH:MM:SS] Processing: filename.ext"
            batch_file_match = re.search(r'\[\d{2}:\d{2}:\d{2}\]\s*processing:\s*(.+)', line_lower)
            if batch_file_match:
                # New file in batch mode - reset progress and stage
                self.reset_stages()
                self.audio_duration = None  # Reset duration for new file
                self.set_stage('loading')
                # Update file label if we can extract the filename
                try:
                    filename = batch_file_match.group(1).strip()
                    if filename:
                        self.file_label.setText(filename)
                        # Try to get duration for this file if parent has the method
                        if self.parent and hasattr(self.parent, '_get_audio_duration'):
                            # Construct full path from filename if possible
                            if hasattr(self.parent, '_whisperx_batch_files'):
                                for file_path in self.parent._whisperx_batch_files:
                                    if os.path.basename(file_path).lower() == filename.lower():
                                        duration = self.parent._get_audio_duration(file_path)
                                        if duration:
                                            self.audio_duration = duration
                                        break
                except Exception:
                    pass
                continue  # Process next line

            # Check ALL patterns and set the appropriate stage
            # Use flags to track what was detected in this line
            detected_stage = None

            # SAVING stage patterns (highest priority - check first)
            saving_patterns = [
                'saved json:',
                'saved srt:',
                'transcription complete.',
                'files saved to',
                # Faster-Whisper-XXL patterns
                'results saved to:',
                'transcription results for',
                'saving results',
                'writing output'
            ]
            if any(p in line_lower for p in saving_patterns):
                detected_stage = 'saving'

            # DIARIZING stage patterns
            # Note: "loading diarization model" removed - it appears during initial setup, not per-file diarization
            if detected_stage is None:
                diarizing_patterns = [
                    'running diarization',
                    'diarization done'
                ]
                if any(p in line_lower for p in diarizing_patterns):
                    detected_stage = 'diarizing'

            # ALIGNING stage patterns
            if detected_stage is None:
                aligning_patterns = [
                    'aligning for word-level',
                    'forced alignment',
                    'alignment done'
                ]
                if any(p in line_lower for p in aligning_patterns):
                    detected_stage = 'aligning'

            # TRANSCRIBING stage patterns
            if detected_stage is None:
                transcribing_patterns = [
                    'transcribing...',
                    'detected language:',
                    # Faster-Whisper-XXL patterns
                    'starting sequential faster-whisper inference',
                    'starting batched faster-whisper inference',
                    'processing audio',
                    'processing segment',
                    'transcribing segment',
                    'audio duration:',
                    'vad filter removed',
                    'speech timestamps'
                ]
                if any(p in line_lower for p in transcribing_patterns):
                    detected_stage = 'transcribing'

            # LOADING stage patterns
            if detected_stage is None:
                loading_patterns = [
                    'loading whisperx model',
                    'loading faster-whisper',
                    'downloading model',
                    'model ready in',
                    'loading audio',
                    # Faster-Whisper-XXL patterns
                    'starting to process:',
                    'standalone faster-whisper-xxl',
                    'running on:',
                    'loading model',
                    'model loaded',
                    'initializing',
                    'compute type:',
                    'device:'
                ]
                if any(p in line_lower for p in loading_patterns):
                    detected_stage = 'loading'

            # Apply the detected stage if it's a valid forward transition
            if detected_stage:
                stage_order = ['loading', 'transcribing', 'aligning', 'diarizing', 'saving']
                current_idx = stage_order.index(self.current_stage) if self.current_stage in stage_order else -1
                detected_idx = stage_order.index(detected_stage)
                # Only transition forward (or to same stage if not yet set)
                if detected_idx >= current_idx:
                    if self.current_stage != detected_stage:
                        self.set_stage(detected_stage)
                continue

            # Progress percentage detection for "Progress: X%..." format from WhisperX
            progress_match = re.search(r'progress:\s*(\d+(?:\.\d+)?)\s*%', line_lower)
            if progress_match:
                try:
                    percent = float(progress_match.group(1))
                    # If we see progress output, we should be in transcribing stage
                    if self.current_stage in [None, 'loading']:
                        self.set_stage('transcribing')
                    # Map progress to overall progress within transcribing range
                    if self.current_stage == 'transcribing':
                        start, end = stage_weights['transcribing']
                        overall = start + (percent / 100.0) * (end - start)
                        self.set_progress(overall)
                except ValueError:
                    pass
                continue

            # Also handle generic percentage for other tools (e.g., "%|" from tqdm)
            tqdm_match = re.search(r'(\d+(?:\.\d+)?)\s*%\s*\|', line)
            if tqdm_match:
                try:
                    percent = float(tqdm_match.group(1))
                    if self.current_stage in stage_weights:
                        start, end = stage_weights[self.current_stage]
                        overall = start + (percent / 100.0) * (end - start)
                        self.set_progress(overall)
                except ValueError:
                    pass
                continue

            # Faster-Whisper-XXL: Detect audio duration from output
            # Pattern: "Audio duration: 10.01 minutes" or similar
            duration_match = re.search(r'audio duration:\s*([\d.]+)\s*(minutes?|seconds?|min|sec|s|m)?', line_lower)
            if duration_match:
                try:
                    duration_val = float(duration_match.group(1))
                    unit = duration_match.group(2) or 'seconds'
                    if unit.startswith('min') or unit == 'm':
                        self.audio_duration = duration_val * 60
                    else:
                        self.audio_duration = duration_val
                except (ValueError, TypeError):
                    pass
                continue

            # Faster-Whisper-XXL: Parse timestamp output for progress
            # Pattern: "[01:32.330 --> 01:35.710]  text..."
            timestamp_match = re.search(r'\[(\d{1,2}):(\d{2})\.(\d{3})\s*-->\s*(\d{1,2}):(\d{2})\.(\d{3})\]', line)
            if timestamp_match and self.current_stage == 'transcribing':
                try:
                    # Parse end timestamp (groups 4, 5, 6)
                    end_min = int(timestamp_match.group(4))
                    end_sec = int(timestamp_match.group(5))
                    end_ms = int(timestamp_match.group(6))
                    current_time = end_min * 60 + end_sec + end_ms / 1000.0

                    # If we have audio duration, calculate exact progress
                    if self.audio_duration and self.audio_duration > 0:
                        percent = min((current_time / self.audio_duration) * 100, 100)
                        start, end = stage_weights['transcribing']
                        overall = start + (percent / 100.0) * (end - start)
                        self.set_progress(overall)
                    else:
                        # No duration known - estimate based on timestamp growth
                        # Assume we're making progress, update the last known timestamp
                        if not hasattr(self, '_last_timestamp_progress'):
                            self._last_timestamp_progress = 0
                        # Slowly increment progress as we see more timestamps
                        new_progress = min(5 + (current_time / 60) * 5, 60)  # Cap at 60%
                        if new_progress > self._last_timestamp_progress:
                            self._last_timestamp_progress = new_progress
                            start, end = stage_weights['transcribing']
                            overall = start + (new_progress / 100.0) * (end - start)
                            self.set_progress(overall)
                except (ValueError, TypeError):
                    pass

    def reset_to_ready(self):
        """Reset panel to ready state"""
        self.is_running = False
        self.dot_timer.stop()
        self.pulse_timer.stop()
        self.elapsed_timer.stop()

        self.file_label.setText("No file selected")
        self.stage_label.setText("Ready to start")
        self.stage_label.setStyleSheet("""
            QLabel {
                font-size: 15px;
                font-weight: 600;
                color: #1976D2;
            }
        """)
        self.pulse_indicator.setText("●")
        self.pulse_indicator.setStyleSheet("font-size: 16px; color: #2196F3;")
        self.progress_bar.setValue(0)
        self.elapsed_label.setText("⏱ Elapsed: 0:00")

        self.reset_stages()
        self._update_stage_display()

    def refresh_transcripts_list(self, session_history: list):
        """Refresh the static transcripts list from session history"""
        # Clear existing items (except the placeholder)
        while self.transcripts_list_layout.count() > 0:
            item = self.transcripts_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Collect all Adobe JSON files from session history
        adobe_files = []
        for entry in session_history:
            files = entry.get("files") or []
            for file_info in files:
                file_path = file_info.get("path", "")
                file_label = file_info.get("label", "")
                # Check if it's an Adobe/Static Transcript file
                if file_path and ("adobe" in file_label.lower() or "static" in file_label.lower() or file_path.endswith("_adobe.json")):
                    if os.path.exists(file_path):
                        adobe_files.append({
                            "path": file_path,
                            "name": os.path.basename(file_path),
                            "timestamp": entry.get("timestamp", "")
                        })

        if not adobe_files:
            # Show placeholder
            self.no_transcripts_label = QLabel("No static transcripts generated yet.")
            self.no_transcripts_label.setStyleSheet("color: #999; font-size: 12px; padding: 10px;")
            self.no_transcripts_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.transcripts_list_layout.addWidget(self.no_transcripts_label)
        else:
            # Add file links
            for file_info in adobe_files:
                file_widget = self._create_transcript_link(file_info["path"], file_info["name"])
                self.transcripts_list_layout.addWidget(file_widget)

            # Add stretch at end
            self.transcripts_list_layout.addStretch()

    def _create_transcript_link(self, file_path: str, display_name: str):
        """Create a clickable link widget for a transcript file with hover effects"""
        link_widget = QWidget(self)  # Parent to prevent top-level window flash
        link_layout = QHBoxLayout(link_widget)
        link_layout.setContentsMargins(4, 2, 4, 2)
        link_layout.setSpacing(6)

        # File icon
        icon_label = QLabel("📄")
        icon_label.setFixedWidth(16)
        link_layout.addWidget(icon_label)

        # Clickable file name with hover support
        name_label = QLabel()
        name_label.setTextFormat(Qt.TextFormat.RichText)
        name_label.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse)
        name_label.setOpenExternalLinks(False)
        name_label.setStyleSheet("font-size: 12px;")
        name_label.setCursor(Qt.CursorShape.PointingHandCursor)
        name_label.setToolTip(f"Click to open file location:\n{file_path}")

        # Store data for hover effects
        name_label.setProperty("file_path", file_path)
        name_label.setProperty("display_name", display_name)
        name_label.setProperty("is_hovered", False)

        # Set initial non-hover state
        name_label.setText(f'<a href="file://{file_path}" style="color: #1976D2; text-decoration: none;">{display_name}</a>')

        # Install event filter for hover effects
        class LinkHoverFilter(QObject):
            def __init__(self, label, file_path, display_name, parent=None):
                super().__init__(parent)
                self.label = label
                self.file_path = file_path
                self.display_name = display_name

            def eventFilter(self, obj, event):
                if event.type() == QEvent.Type.Enter:
                    # Hover state - underlined and darker color
                    self.label.setText(
                        f'<a href="file://{self.file_path}" style="color: #0D47A1; text-decoration: underline;">{self.display_name}</a>'
                    )
                elif event.type() == QEvent.Type.Leave:
                    # Normal state
                    self.label.setText(
                        f'<a href="file://{self.file_path}" style="color: #1976D2; text-decoration: none;">{self.display_name}</a>'
                    )
                return False

        hover_filter = LinkHoverFilter(name_label, file_path, display_name, name_label)
        name_label.installEventFilter(hover_filter)
        # Keep reference to prevent garbage collection
        name_label.setProperty("hover_filter", hover_filter)

        name_label.linkActivated.connect(lambda _, p=file_path: self._open_file_location(p))
        link_layout.addWidget(name_label, 1)

        return link_widget

    def _open_file_location(self, file_path: str):
        """Open the folder containing the file and select it"""
        try:
            if sys.platform == "win32":
                # Windows: Use explorer with /select to highlight the file
                subprocess.run(["explorer", "/select,", os.path.normpath(file_path)], check=False)
            elif sys.platform == "darwin":
                # macOS: Use open with -R to reveal in Finder
                subprocess.run(["open", "-R", file_path], check=False)
            else:
                # Linux: Open the containing folder
                folder = os.path.dirname(file_path)
                subprocess.run(["xdg-open", folder], check=False)
        except Exception as e:
            print(f"Could not open file location: {e}")


class GenerateCaptionsWidget(QWidget):
    """Main widget for Generate Captions functionality"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

        # Install window show event filter to catch the flash
        global _window_show_filter
        if _window_show_filter is None:
            app = QApplication.instance()
            if app:
                _window_show_filter = WindowShowEventFilter()
                app.installEventFilter(_window_show_filter)
                print("[DEBUG] Installed WindowShowEventFilter to catch window show events")

        # Initialize variables
        self.executable_path = None
        self.executable_name = None
        self.bin_dir = os.path.join(get_app_directory(), "bin")
        self.process = None
        self.downloader = None
        self.stop_requested = False
        self.transcription_completed_successfully = False
        self.current_input_file = None
        self.whisperx_python = None  # Preferred Python interpreter for WhisperX (e.g., Python 3.12)
        self._whisperx_execution_env = None
        self._whisperx_execution_python = None
        self.output_buffer = ""
        self.last_line_was_overwrite = False
        self.last_srt_text = ""
        self._active_run_input_file = None
        self._active_run_output_dir = None
        self._last_completed_input_file = None
        self._last_completed_output_dir = None
        self.session_history: List[dict] = []
        self._cleaner_insert_mode: Optional[str] = None
        self._cleaner_insert_mode_once: bool = False

        # Batch processing variables
        self.batch_files = []
        self.batch_results = []
        self.batch_mode_active = False
        self._batch_index = 0

        # Settings storage
        self.settings = {}
        self.settings_file = os.path.join(get_settings_directory(), "generate_captions_settings.json")

        # Hardware info
        self.hardware_info = None

        print("[DEBUG] >>> init_ui()")
        self.init_ui()
        print("[DEBUG] <<< init_ui() complete")
        # Initialize model list for default engine (Faster-Whisper)
        print("[DEBUG] >>> on_engine_changed()")
        self.on_engine_changed()
        print("[DEBUG] <<< on_engine_changed() complete")
        # Avoid heavy dependency checks during startup
        self._suppress_dep_check = True
        print("[DEBUG] >>> load_settings()")
        self.load_settings()
        print("[DEBUG] <<< load_settings() complete")
        self._suppress_dep_check = False
        # Defer dependency checks to user action (tab open or engine toggle)
        print("[DEBUG] >>> _load_session_history_from_session()")
        self._load_session_history_from_session()
        print("[DEBUG] <<< _load_session_history_from_session() complete")
        print("[DEBUG] === GenerateCaptionsWidget __init__ COMPLETE ===")

    def _refresh_transcripts_panel(self):
        """Refresh the static transcripts panel from session history"""
        if hasattr(self, 'progress_panel') and self.progress_panel:
            self.progress_panel.refresh_transcripts_list(self.session_history)

    def init_ui(self):
        """Initialize the modern, clean user interface"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Add information label at the top spanning the full width
        info_label = QLabel("This tab handles audio to text generation, necessary as the first step to create a Scriptoria transcript. Scriptoria uses OpenAI's 'Whisper' for highly accurate captioning. You must upload the generated JSON file(s) into Premiere as static transcript(s) to link the text in Scriptoria with your video in Premiere.")
        info_label.setStyleSheet("""
            QLabel {
                background-color: #f0f7ff;
                border: 1px solid #d0e3f7;
                border-radius: 4px;
                color: #2e5075;
                font-size: 12px;
                padding: 6px 10px;
                margin-bottom: 0px;
            }
        """)
        info_label.setWordWrap(True)
        info_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        main_layout.addWidget(info_label)

        # Engine selection
        engine_group = QGroupBox("Transcription Engine")
        engine_layout = QHBoxLayout(engine_group)

        self.engine_faster_whisper = QRadioButton("Faster-Whisper-XXL (Simplest)")
        self.engine_faster_whisper.setChecked(True)
        self.engine_faster_whisper.toggled.connect(self.on_engine_changed)
        self.engine_faster_whisper.toggled.connect(self.save_settings_delayed)

        self.engine_whisperx = QRadioButton("WhisperX (More powerful, complex, large install)")
        self.engine_whisperx.toggled.connect(self.on_engine_changed)
        self.engine_whisperx.toggled.connect(self.save_settings_delayed)

        engine_layout.addWidget(self.engine_faster_whisper)
        engine_layout.addWidget(self.engine_whisperx)
        engine_layout.addStretch()

        # Hide engine selection to simplify UX (default: Faster-Whisper-XXL)
        engine_group.setVisible(False)
        # Add (hidden) so widgets have a parent and aren't deleted
        main_layout.addWidget(engine_group)

        # Create horizontal layout for left and right sides
        content_layout = QHBoxLayout()
        content_layout.setSpacing(10)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # LEFT SIDE: Tabbed interface for controls
        # IMPORTANT: Pass self as parent to prevent top-level window flash on Windows
        left_widget = QWidget(self)
        left_widget.setMaximumWidth(520)
        left_widget.setMinimumWidth(460)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Create tabs
        # IMPORTANT: Pass self as parent to prevent top-level window flash on Windows
        self.tab_widget = QTabWidget(self)
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ddd;
                border-radius: 4px;
                background: white;
            }
            QTabBar::tab {
                background: #f5f5f5;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: white;
                border-bottom: 2px solid #0078d4;
            }
            QTabBar::tab:hover {
                background: #e8e8e8;
            }
        """)

        # Dependencies Tab
        deps_tab = self.create_dependencies_tab()
        self.deps_tab_index = self.tab_widget.addTab(deps_tab, "Install")

        # Generate Tab
        print("[DEBUG] >>> Creating generate tab")
        generate_tab = self.create_generate_tab()
        self.generate_tab_index = self.tab_widget.addTab(generate_tab, "Faster-Whisper-XXL")
        print("[DEBUG] <<< Generate tab created")

        # WhisperX Tab
        print("[DEBUG] >>> Creating whisperx tab")
        whisperx_tab = self.create_whisperx_tab()
        self.whisperx_tab_index = self.tab_widget.addTab(whisperx_tab, "WhisperX")
        print("[DEBUG] <<< WhisperX tab created")

        # Defer WhisperX dependency checks to after main window is visible.
        # Running subprocess calls during startup can flash console windows on Windows
        # because PyTorch DLL loading briefly shows a window even with CREATE_NO_WINDOW.
        # Use QTimer.singleShot to defer this check until after splash screen closes.
        QTimer.singleShot(1000, self._deferred_whisperx_check)

        left_layout.addWidget(self.tab_widget)

        # Control buttons at bottom of left side
        button_layout = self.create_button_section()
        left_layout.addLayout(button_layout)

        # RIGHT SIDE: Console output
        right_widget = self.create_console_section()

        # Add to content layout
        content_layout.addWidget(left_widget)
        content_layout.addWidget(right_widget, 1)

        # Add content layout to main layout
        main_layout.addLayout(content_layout)

        # Auto-switch to appropriate tab at startup is now handled in _deferred_whisperx_check
        # to avoid subprocess calls during splash screen

    def create_dependencies_tab(self):
        """Create the Dependencies tab with clean layout"""
        tab = QWidget(self)  # Parent to prevent top-level window flash
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(20)

        # Status section
        status_group = QGroupBox("Install Faster-Whisper-XXL")
        status_layout = QVBoxLayout(status_group)

        # Simplified flow: default to Faster-Whisper-XXL; no engine selection needed
        self.deps_status_label = QLabel("Faster-Whisper-XXL: Download the dependency bundle if missing.")
        self.deps_status_label.setStyleSheet("font-size: 14px; padding: 8px;")
        status_layout.addWidget(self.deps_status_label)

        self.download_deps_btn = QPushButton("📥 Download Dependencies (1.4GB)")
        self.download_deps_btn.clicked.connect(self.download_dependencies)
        self.download_deps_btn.setVisible(False)
        self.download_deps_btn.setMinimumHeight(36)
        self.download_deps_btn.setStyleSheet("""
            QPushButton {
                background: #0078d4;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #106ebe;
            }
        """)
        status_layout.addWidget(self.download_deps_btn)

        # Since we default to Faster-Whisper, run a quick filesystem check to set status
        try:
            self._check_faster_whisper_dependencies()
        except Exception:
            pass

        # CUDA status and enable button (for WhisperX private runtime)
        self.cuda_status_label = QLabel("")
        self.cuda_status_label.setStyleSheet("color: #666; font-size: 12px; padding: 4px 8px;")
        self.cuda_status_label.setVisible(False)
        status_layout.addWidget(self.cuda_status_label)

        self.enable_cuda_btn = QPushButton("⚡ Enable GPU (CUDA) in WhisperX runtime")
        self.enable_cuda_btn.setVisible(False)
        self.enable_cuda_btn.setMinimumHeight(32)
        self.enable_cuda_btn.clicked.connect(self.enable_cuda_for_whisperx)
        status_layout.addWidget(self.enable_cuda_btn)

        # Manual CUDA status refresh (no auto-checks by default)
        self.refresh_cuda_btn = QPushButton("Refresh CUDA status")
        self.refresh_cuda_btn.setVisible(False)
        self.refresh_cuda_btn.setMinimumHeight(28)
        self.refresh_cuda_btn.clicked.connect(lambda: self.start_cuda_status_check())
        status_layout.addWidget(self.refresh_cuda_btn)

        self.cuda_diag_btn = QPushButton("🧪 Test CUDA runtime")
        self.cuda_diag_btn.setVisible(False)
        self.cuda_diag_btn.setMinimumHeight(28)
        self.cuda_diag_btn.clicked.connect(self.diagnose_cuda_runtime)
        status_layout.addWidget(self.cuda_diag_btn)

        # Hardware section
        hardware_group = QGroupBox("Hardware Detection")
        hardware_layout = QVBoxLayout(hardware_group)

        hw_button_layout = QHBoxLayout()
        self.detect_hardware_btn = QPushButton("🔍 Detect Hardware")
        self.detect_hardware_btn.clicked.connect(self.detect_hardware)
        self.detect_hardware_btn.setMinimumHeight(36)

        hw_button_layout.addWidget(self.detect_hardware_btn)
        hw_button_layout.addStretch()

        self.hardware_status_label = QLabel("Click 'Detect Hardware' to scan system capabilities")
        self.hardware_status_label.setStyleSheet("color: #666; font-size: 13px; padding: 8px;")
        self.hardware_status_label.setWordWrap(True)

        hardware_layout.addLayout(hw_button_layout)
        hardware_layout.addWidget(self.hardware_status_label)

        # WhisperX Installation Section
        whisperx_deps_group = QGroupBox("Install WhisperX")
        whisperx_deps_layout = QVBoxLayout(whisperx_deps_group)

        self.whisperx_deps_status = QLabel("Checking WhisperX installation...")
        self.whisperx_deps_status.setStyleSheet("color: #666; font-size: 12px;")
        whisperx_deps_layout.addWidget(self.whisperx_deps_status)

        # Install/Reinstall buttons
        install_buttons_layout = QHBoxLayout()

        self.whisperx_cuda_status_label = QLabel("CUDA status: not checked")
        self.whisperx_cuda_status_label.setStyleSheet("color: #666; font-size: 12px;")
        self.whisperx_cuda_status_label.setVisible(False)
        whisperx_deps_layout.addWidget(self.whisperx_cuda_status_label)

        self.whisperx_cuda_test_btn = QPushButton("Check CUDA")
        self.whisperx_cuda_test_btn.setVisible(False)
        self.whisperx_cuda_test_btn.setMinimumHeight(22)
        self.whisperx_cuda_test_btn.setMaximumWidth(100)
        self.whisperx_cuda_test_btn.setStyleSheet("font-size: 11px; padding: 4px 8px; border-radius: 3px;")
        self.whisperx_cuda_test_btn.clicked.connect(self.diagnose_cuda_runtime)
        whisperx_deps_layout.addWidget(self.whisperx_cuda_test_btn)

        self.whisperx_install_btn = QPushButton("📦 Install WhisperX")

        # Wrap the install function to catch early crashes
        def safe_install_whisperx():
            try:
                # Create immediate crash log BEFORE calling install
                import datetime
                crash_log_path = None
                try:
                    if getattr(sys, 'frozen', False):
                        log_dir = os.path.dirname(sys.executable)
                    else:
                        log_dir = get_app_directory()

                    crash_log_path = os.path.join(log_dir, "whisperx_BUTTON_CLICK_crash.log")
                    with open(crash_log_path, 'w', encoding='utf-8') as f:
                        f.write(f"Button clicked at: {datetime.datetime.now()}\n")
                        f.write(f"About to call install_whisperx_simple()\n")
                        f.write(f"sys.frozen: {getattr(sys, 'frozen', False)}\n")
                        f.write(f"sys.executable: {sys.executable}\n")
                except Exception as log_err:
                    print(f"Could not create button click log: {log_err}")

                # Now call the actual install function
                self.install_whisperx_simple()

            except Exception as e:
                import traceback
                error_msg = f"CRASH IN BUTTON HANDLER: {str(e)}\n{traceback.format_exc()}"
                print(error_msg)

                # Try to write crash to file
                if crash_log_path:
                    try:
                        with open(crash_log_path, 'a', encoding='utf-8') as f:
                            f.write(f"\nCRASH IN BUTTON HANDLER:\n")
                            f.write(error_msg)
                    except:
                        pass

                # Show error to user
                QMessageBox.critical(self, "Installation Error",
                    f"The installation crashed before it could start:\n\n{str(e)}\n\nCheck whisperx_BUTTON_CLICK_crash.log for details.")

        self.whisperx_install_btn.clicked.connect(safe_install_whisperx)
        self.whisperx_install_btn.setVisible(False)
        self.whisperx_install_btn.setMinimumHeight(36)
        self.whisperx_install_btn.setStyleSheet("""
            QPushButton {
                background: #0078d4;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #106ebe;
            }
        """)

        self.whisperx_update_btn = QPushButton("🔄 Update WhisperX")
        self.whisperx_update_btn.clicked.connect(self.update_whisperx)
        self.whisperx_update_btn.setVisible(False)
        self.whisperx_update_btn.setMinimumHeight(22)
        self.whisperx_update_btn.setMaximumWidth(140)
        self.whisperx_update_btn.setStyleSheet("font-size: 11px; padding: 4px 8px; border-radius: 3px;")

        self.whisperx_reinstall_btn = QPushButton("🔃 Reinstall WhisperX")
        self.whisperx_reinstall_btn.clicked.connect(self.reinstall_whisperx)
        self.whisperx_reinstall_btn.setVisible(False)
        self.whisperx_reinstall_btn.setMinimumHeight(22)
        self.whisperx_reinstall_btn.setMaximumWidth(140)
        self.whisperx_reinstall_btn.setStyleSheet("font-size: 11px; padding: 4px 8px; border-radius: 3px;")

        # Cancel button - only visible during installation
        self.whisperx_cancel_btn = QPushButton("❌ Cancel Installation")
        self.whisperx_cancel_btn.clicked.connect(self.cancel_whisperx_installation)
        self.whisperx_cancel_btn.setVisible(False)
        self.whisperx_cancel_btn.setMinimumHeight(36)
        self.whisperx_cancel_btn.setStyleSheet("""
            QPushButton {
                background: #d32f2f;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #b71c1c;
            }
        """)

        install_buttons_layout.addWidget(self.whisperx_install_btn)
        install_buttons_layout.addWidget(self.whisperx_cancel_btn)
        install_buttons_layout.addWidget(self.whisperx_update_btn)
        install_buttons_layout.addWidget(self.whisperx_reinstall_btn)
        install_buttons_layout.addWidget(self.whisperx_cuda_test_btn)
        install_buttons_layout.addStretch()

        whisperx_deps_layout.addLayout(install_buttons_layout)

        layout.addWidget(status_group)
        layout.addWidget(whisperx_deps_group)
        layout.addWidget(hardware_group)
        layout.addStretch()

        return tab

    def create_generate_tab(self):
        """Create the Generate tab with modern layout"""
        tab = QWidget(self)  # Parent to prevent top-level window flash
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # Info banner
        info_banner = QLabel("Standard Transcription Using Faster-Whisper-XXL")
        info_banner.setStyleSheet("""
            QLabel {
                background-color: #f0f7ff;
                border: 1px solid #d0e3f7;
                border-radius: 4px;
                color: #2e5075;
                font-size: 12px;
                padding: 8px 12px;
                margin-bottom: 10px;
            }
        """)
        info_banner.setWordWrap(True)
        layout.addWidget(info_banner)

        # File input with drag-drop
        input_group = QGroupBox("Input File")
        input_layout = QVBoxLayout(input_group)

        # Drag-drop area
        self.drop_area = FileDropArea(self)  # Pass parent to prevent top-level window flash
        self.drop_area.file_dropped.connect(self.on_file_dropped)
        self.drop_area.files_dropped.connect(self.on_files_dropped)
        input_layout.addWidget(self.drop_area)

        # Current file display
        self.current_file_label = QLabel("No file selected")
        self.current_file_label.setStyleSheet("color: #666; font-size: 12px; font-style: italic;")
        input_layout.addWidget(self.current_file_label)

        # Batch processing buttons
        batch_layout = QHBoxLayout()
        self.add_files_btn = QPushButton("📁 Add Files...")
        self.add_files_btn.clicked.connect(self.add_batch_files)
        self.add_files_btn.setMaximumWidth(120)

        self.clear_batch_btn = QPushButton("🗑 Clear")
        self.clear_batch_btn.clicked.connect(self.clear_batch_files)
        self.clear_batch_btn.setMaximumWidth(80)
        self.clear_batch_btn.setEnabled(False)

        batch_layout.addWidget(self.add_files_btn)
        batch_layout.addWidget(self.clear_batch_btn)
        batch_layout.addStretch()

        input_layout.addLayout(batch_layout)

        # Batch progress bar (hidden initially) - uses SplitColorProgressBar for white/black text
        self.batch_progress_bar = SplitColorProgressBar(self)  # Pass parent to prevent flash
        self.batch_progress_bar.setVisible(False)
        input_layout.addWidget(self.batch_progress_bar)

        # Transcription settings in compact layout with tabs
        settings_group = QGroupBox("Faster-Whisper-XXL Settings")
        settings_main_layout = QVBoxLayout(settings_group)
        settings_main_layout.setContentsMargins(8, 8, 8, 8)  # Reduce padding inside groupbox

        # Create tab widget for settings organization
        settings_tabs = QTabWidget(self)  # Parent to prevent top-level window flash
        settings_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #cfd6dc;
                border-radius: 4px;
                padding: 4px;
            }
            QTabBar::tab {
                padding: 6px 12px;
                margin-right: 2px;
            }
        """)

        # === MAIN TAB ===
        main_tab = QWidget(self)  # Parent to prevent top-level window flash
        main_tab_layout = QFormLayout(main_tab)
        main_tab_layout.setContentsMargins(6, 6, 6, 6)  # Reduce margins
        main_tab_layout.setHorizontalSpacing(8)
        main_tab_layout.setVerticalSpacing(4)
        main_tab_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)

        # Model
        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        # Will be populated based on selected engine
        self.model_combo.currentTextChanged.connect(self.save_settings_delayed)

        # Add help button for model trade-offs
        model_help_btn = QPushButton("?")
        model_help_btn.setToolTip("Model trade-offs: speed, accuracy, memory")
        model_help_btn.setFixedSize(22, 22)
        model_help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        model_help_btn.setStyleSheet("""
            QPushButton { border: 1px solid #cfd6dc; border-radius: 11px; background: #f4f7f9; color: #334; font-weight: bold; }
            QPushButton:hover { background: #eaf1f5; }
            QPushButton:pressed { background: #e0ebf2; }
        """)
        model_help_btn.clicked.connect(self.show_model_help_dialog)

        model_row.addWidget(model_help_btn)
        model_row.addWidget(QLabel("Model:"))
        model_row.addWidget(self.model_combo, 1)
        main_tab_layout.addRow(model_row)

        # Language
        language_row = QHBoxLayout()
        self.language_combo = QComboBox()
        languages = ['auto'] + ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'hi', 'ar', 'nl', 'sv', 'no', 'da', 'pl', 'tr', 'th', 'vi', 'id', 'ms']
        self.language_combo.addItems(languages)
        self.language_combo.setEditable(True)
        self.language_combo.currentTextChanged.connect(self.save_settings_delayed)

        language_row.addWidget(QLabel("Language:"))
        language_row.addWidget(self.language_combo, 1)
        main_tab_layout.addRow(language_row)

        # Task
        task_row = QHBoxLayout()
        self.task_combo = QComboBox()
        self.task_combo.addItems(['transcribe', 'translate'])
        self.task_combo.currentTextChanged.connect(self.save_settings_delayed)

        # Add help button for Task
        task_help_btn = QPushButton("?")
        task_help_btn.setToolTip("Task selection: transcribe vs translate")
        task_help_btn.setFixedSize(22, 22)
        task_help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        task_help_btn.setStyleSheet("""
            QPushButton { border: 1px solid #cfd6dc; border-radius: 11px; background: #f4f7f9; color: #334; font-weight: bold; }
            QPushButton:hover { background: #eaf1f5; }
            QPushButton:pressed { background: #e0ebf2; }
        """)
        task_help_btn.clicked.connect(self.show_task_help_dialog)

        task_row.addWidget(task_help_btn)
        task_row.addWidget(QLabel("Task:"))
        task_row.addWidget(self.task_combo, 1)
        main_tab_layout.addRow(task_row)

        # Device
        device_row = QHBoxLayout()
        self.device_combo = QComboBox()
        self.device_combo.addItems(['cpu', 'cuda'])
        self.device_combo.currentTextChanged.connect(self.save_settings_delayed)

        # Add help button for Device
        device_help_btn = QPushButton("?")
        device_help_btn.setToolTip("Device selection: CPU vs GPU trade-offs")
        device_help_btn.setFixedSize(22, 22)
        device_help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        device_help_btn.setStyleSheet("""
            QPushButton { border: 1px solid #cfd6dc; border-radius: 11px; background: #f4f7f9; color: #334; font-weight: bold; }
            QPushButton:hover { background: #eaf1f5; }
            QPushButton:pressed { background: #e0ebf2; }
        """)
        device_help_btn.clicked.connect(self.show_device_help_dialog)

        device_row.addWidget(device_help_btn)
        device_row.addWidget(QLabel("Device:"))
        device_row.addWidget(self.device_combo, 1)
        main_tab_layout.addRow(device_row)

        settings_tabs.addTab(main_tab, "Model")

        # === ADVANCED TAB ===
        advanced_tab = QWidget(self)  # Parent to prevent top-level window flash
        advanced_tab_layout = QFormLayout(advanced_tab)
        advanced_tab_layout.setContentsMargins(6, 6, 6, 6)  # Reduce margins
        advanced_tab_layout.setHorizontalSpacing(8)
        advanced_tab_layout.setVerticalSpacing(4)
        advanced_tab_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)

        # Compute Type
        compute_row = QHBoxLayout()
        self.compute_combo = QComboBox()
        self.compute_combo.addItems(['default', 'auto', 'int8', 'int8_float16', 'int8_float32', 'int16', 'float16', 'float32'])
        self.compute_combo.setCurrentText('float16')  # Default to float16
        self.compute_combo.currentTextChanged.connect(self.save_settings_delayed)

        # Add help button for Compute Type
        compute_help_btn = QPushButton("?")
        compute_help_btn.setToolTip("Compute type trade-offs: precision, speed, memory")
        compute_help_btn.setFixedSize(22, 22)
        compute_help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        compute_help_btn.setStyleSheet("""
            QPushButton { border: 1px solid #cfd6dc; border-radius: 11px; background: #f4f7f9; color: #334; font-weight: bold; }
            QPushButton:hover { background: #eaf1f5; }
            QPushButton:pressed { background: #e0ebf2; }
        """)
        compute_help_btn.clicked.connect(self.show_compute_help_dialog)

        compute_row.addWidget(compute_help_btn)
        compute_row.addWidget(QLabel("Compute:"))
        compute_row.addWidget(self.compute_combo, 1)
        advanced_tab_layout.addRow(compute_row)

        # VAD Method
        vad_row = QHBoxLayout()
        self.vad_method = QComboBox()
        # Default full set (remove deprecated Silero V4 options)
        self.vad_method.addItems(['silero_v4', 'pyannote_v3', 'pyannote_onnx_v3'])
        self.vad_method.currentTextChanged.connect(self.save_settings_delayed)

        # Add help button for VAD trade-offs
        vad_help_btn = QPushButton("?")
        vad_help_btn.setToolTip("VAD method trade-offs: speed, accuracy, compatibility")
        vad_help_btn.setFixedSize(22, 22)
        vad_help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        vad_help_btn.setStyleSheet("""
            QPushButton { border: 1px solid #cfd6dc; border-radius: 11px; background: #f4f7f9; color: #334; font-weight: bold; }
            QPushButton:hover { background: #eaf1f5; }
            QPushButton:pressed { background: #e0ebf2; }
        """)
        vad_help_btn.clicked.connect(self.show_vad_help_dialog)

        vad_row.addWidget(vad_help_btn)
        vad_row.addWidget(QLabel("VAD:"))
        vad_row.addWidget(self.vad_method, 1)
        advanced_tab_layout.addRow(vad_row)

        # VAD Filter checkbox
        self.vad_filter = QCheckBox("Enable Voice Activity Detection (VAD)")
        self.vad_filter.setChecked(True)
        self.vad_filter.stateChanged.connect(self.save_settings_delayed)
        advanced_tab_layout.addRow(self.vad_filter)

        settings_tabs.addTab(advanced_tab, "Advanced")

        # Set size policy to not expand vertically
        settings_tabs.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        settings_main_layout.addWidget(settings_tabs)

        # Ensure VAD options reflect the current engine selection
        try:
            if hasattr(self, 'engine_whisperx') and hasattr(self, 'engine_faster_whisper'):
                # Initial sync on first render
                self._refresh_vad_options()
        except Exception:
            pass

        # Speaker Diarization (disabled for now)
        diarization_group = QGroupBox("Speaker Diarization (disabled)")
        diarization_layout = QVBoxLayout(diarization_group)
        disabled_msg = QLabel("Speaker diarization is temporarily disabled while we focus on GPU transcription and VAD.")
        disabled_msg.setStyleSheet("color: #666; font-size: 12px;")
        diarization_layout.addWidget(disabled_msg)

        # Export options
        export_group = QGroupBox("Export Formats")
        export_group_layout = QVBoxLayout(export_group)
        export_desc = QLabel("Choose what to generate. At least one option must be selected.")
        export_desc.setStyleSheet("color: #666; font-size: 12px;")
        export_group_layout.addWidget(export_desc)

        export_layout = QHBoxLayout()

        # Default (Adobe Static Transcript + reference transcript)
        default_container = QVBoxLayout()
        self.export_adobe = QCheckBox("Default")
        self.export_adobe.setChecked(True)
        self.export_adobe.stateChanged.connect(self.save_settings_delayed)
        self.export_adobe.stateChanged.connect(lambda _=None: self._ensure_one_export_checked('adobe'))
        self.export_adobe.stateChanged.connect(self.update_formatting_visibility)
        default_container.addWidget(self.export_adobe)
        default_sub = QLabel("Create an Adobe Premiere Pro compatible static transcript with word-level accuracy, raw text transcription")
        default_sub.setStyleSheet("color: #6a6a6a; font-size: 11px;")
        default_sub.setWordWrap(True)
        default_container.addWidget(default_sub)

        # SRT option - always enabled (hidden from UI)
        srt_container = QVBoxLayout()
        self.export_srt = QCheckBox("Create SRT")
        self.export_srt.setChecked(True)  # Always generate SRT files
        self.export_srt.stateChanged.connect(self.save_settings_delayed)
        self.export_srt.stateChanged.connect(lambda _=None: self._ensure_one_export_checked('srt'))
        srt_container.addWidget(self.export_srt)
        srt_sub = QLabel("Will generate SRT files for use in subtitling.")
        srt_sub.setStyleSheet("color: #6a6a6a; font-size: 11px;")
        srt_sub.setWordWrap(True)
        srt_container.addWidget(srt_sub)

        export_layout.addLayout(default_container)
        export_layout.addSpacing(24)
        export_layout.addLayout(srt_container)
        export_layout.addStretch()
        export_group_layout.addLayout(export_layout)

        # Format Transcript options
        self.formatting_group = QGroupBox("Paragraph Formatting")
        formatting_group = self.formatting_group
        formatting_layout = QVBoxLayout(formatting_group)
        formatting_desc = QLabel("Adjust duration of gap to be considered a paragraph break")
        formatting_desc.setStyleSheet("color: #666; font-size: 12px;")
        formatting_layout.addWidget(formatting_desc)

        # Hide continuous segment option
        self.format_single_segment = QRadioButton("Continuous Segment")
        self.format_single_segment.setVisible(False)
        self.format_paragraph_form = QRadioButton("Paragraph Form 2.0")
        # Default: Paragraph Form is now the default
        self.format_paragraph_form.setChecked(True)
        self.format_paragraph_form.setVisible(False)  # Hide radio button, keep functionality
        self.format_single_segment.toggled.connect(self.save_settings_delayed)
        self.format_paragraph_form.toggled.connect(self.save_settings_delayed)

        # Don't add the radio button to layout since it's hidden
        # formatting_layout.addWidget(self.format_paragraph_form)

        # Gap threshold control (enabled only for Paragraph Form)
        gap_row = QHBoxLayout()
        gap_row.addSpacing(24)
        gap_row.addWidget(QLabel("Paragraph gap threshold (seconds):"))
        self.paragraph_gap_spin = QDoubleSpinBox()
        self.paragraph_gap_spin.setRange(0.5, 60.0)
        self.paragraph_gap_spin.setSingleStep(0.5)
        self.paragraph_gap_spin.setDecimals(1)
        self.paragraph_gap_spin.setValue(3.0)
        self.paragraph_gap_spin.setToolTip("Start a new paragraph when the pause between consecutive words exceeds this many seconds.")
        self.paragraph_gap_spin.valueChanged.connect(self.save_settings_delayed)
        gap_row.addWidget(self.paragraph_gap_spin)
        gap_row.addStretch()
        formatting_layout.addLayout(gap_row)

        # Don't add single segment radio button since it's hidden
        # formatting_layout.addWidget(self.format_single_segment)

        # Enable/disable threshold input depending on mode
        self.format_single_segment.toggled.connect(self.update_formatting_controls)
        self.format_paragraph_form.toggled.connect(self.update_formatting_controls)

        # Initialize enabled state
        self.update_formatting_controls()
        # NOTE: update_formatting_visibility() is called AFTER formatting_group is added to layout
        # to prevent window flash when setVisible() is called on a parentless widget

        # Output directory section
        output_group = QGroupBox("Output Directory")
        output_layout = QHBoxLayout(output_group)
        output_layout.setContentsMargins(15, 10, 15, 10)

        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("Default: Same directory as input file")
        self.output_dir.textChanged.connect(self.save_settings_delayed)

        browse_output_btn = QPushButton("Browse...")
        browse_output_btn.clicked.connect(self.browse_output_dir)
        browse_output_btn.setMaximumWidth(80)
        browse_output_btn.setStyleSheet("""
            QPushButton {
                background: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #e0e0e0;
            }
        """)

        output_layout.addWidget(self.output_dir)
        output_layout.addWidget(browse_output_btn)

        layout.addWidget(input_group)
        layout.addWidget(settings_group)
        # Hide diarization UI to reduce confusion
        diarization_group.setVisible(False)
        # Hide export formats - SRT is always generated
        export_group.setVisible(False)
        layout.addWidget(export_group)
        layout.addWidget(formatting_group)
        # Update visibility AFTER adding to layout to prevent window flash
        self.update_formatting_visibility()
        layout.addWidget(output_group)
        layout.addStretch()

        # Initialize Faster-Whisper defaults on first render (engine UI hidden)
        try:
            if hasattr(self, 'engine_faster_whisper'):
                # Populate model list
                self.model_combo.clear()
                self.model_combo.addItems(['distil-large-v3.5', 'large-v3-turbo', 'large-v3', 'large-v2'])
                self.model_combo.setCurrentText('distil-large-v3.5')
        except Exception:
            pass

        return tab

    def create_whisperx_tab(self):
        """Create the WhisperX tab with WhisperX-specific features"""
        tab = QWidget(self)  # Parent to prevent top-level window flash
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # WhisperX Info Banner
        info_banner = QLabel("WhisperX: Advanced transcription with speaker diarization support")
        info_banner.setStyleSheet("""
            QLabel {
                background-color: #e3f2fd;
                border: 1px solid #2196f3;
                border-radius: 4px;
                color: #0d47a1;
                font-size: 12px;
                padding: 8px 12px;
                margin-bottom: 10px;
            }
        """)
        info_banner.setWordWrap(True)
        layout.addWidget(info_banner)

        # Warning label for when WhisperX is not installed
        self.whisperx_not_installed_label = QLabel("⚠ WhisperX not installed")
        self.whisperx_not_installed_label.setStyleSheet("""
            QLabel {
                background-color: #ffebee;
                border: 1px solid #ef5350;
                border-radius: 4px;
                color: #c62828;
                font-size: 13px;
                font-weight: bold;
                padding: 8px 12px;
                margin-bottom: 10px;
            }
        """)
        self.whisperx_not_installed_label.setWordWrap(True)
        self.whisperx_not_installed_label.setVisible(False)  # Will be shown by check_whisperx_deps_simple()
        layout.addWidget(self.whisperx_not_installed_label)

        # File input with drag-drop (DUPLICATE of Generate tab - shares data but separate UI)
        input_group = QGroupBox("Input File")
        input_layout = QVBoxLayout(input_group)

        # Create a second drag-drop area for WhisperX tab
        self.whisperx_drop_area = FileDropArea(self)  # Pass parent to prevent top-level window flash
        self.whisperx_drop_area.file_dropped.connect(self.on_file_dropped)
        self.whisperx_drop_area.files_dropped.connect(self.on_files_dropped)
        input_layout.addWidget(self.whisperx_drop_area)

        # Current file display (separate label, same data)
        self.whisperx_current_file_label = QLabel("No file selected")
        self.whisperx_current_file_label.setStyleSheet("color: #666; font-size: 12px; font-style: italic;")
        input_layout.addWidget(self.whisperx_current_file_label)

        # Batch processing buttons
        whisperx_batch_layout = QHBoxLayout()
        self.whisperx_add_files_btn = QPushButton("📁 Add Files...")
        self.whisperx_add_files_btn.clicked.connect(self.add_batch_files)
        self.whisperx_add_files_btn.setMaximumWidth(120)

        self.whisperx_clear_batch_btn = QPushButton("🗑 Clear")
        self.whisperx_clear_batch_btn.clicked.connect(self.clear_batch_files)
        self.whisperx_clear_batch_btn.setMaximumWidth(80)
        self.whisperx_clear_batch_btn.setEnabled(False)

        whisperx_batch_layout.addWidget(self.whisperx_add_files_btn)
        whisperx_batch_layout.addWidget(self.whisperx_clear_batch_btn)
        whisperx_batch_layout.addStretch()

        input_layout.addLayout(whisperx_batch_layout)

        # Batch progress bar (separate widget, same data) - uses SplitColorProgressBar for white/black text
        self.whisperx_batch_progress_bar = SplitColorProgressBar(self)  # Pass parent to prevent flash
        self.whisperx_batch_progress_bar.setVisible(False)
        input_layout.addWidget(self.whisperx_batch_progress_bar)

        layout.addWidget(input_group)

        # Settings for WhisperX - with tabs
        settings_group = QGroupBox("WhisperX Settings")
        whisperx_settings_main_layout = QVBoxLayout(settings_group)
        whisperx_settings_main_layout.setContentsMargins(8, 8, 8, 8)  # Reduce padding inside groupbox

        # Create tab widget for settings organization
        whisperx_settings_tabs = QTabWidget(self)  # Parent to prevent top-level window flash
        whisperx_settings_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #cfd6dc;
                border-radius: 4px;
                padding: 4px;
            }
            QTabBar::tab {
                padding: 6px 12px;
                margin-right: 2px;
            }
        """)

        # === MAIN TAB ===
        whisperx_main_tab = QWidget(self)  # Parent to prevent top-level window flash
        whisperx_main_tab_layout = QFormLayout(whisperx_main_tab)
        whisperx_main_tab_layout.setContentsMargins(6, 6, 6, 6)  # Reduce margins
        whisperx_main_tab_layout.setHorizontalSpacing(8)
        whisperx_main_tab_layout.setVerticalSpacing(4)
        whisperx_main_tab_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)

        # Model
        whisperx_model_row = QHBoxLayout()

        whisperx_model_help_btn = QPushButton("?")
        whisperx_model_help_btn.setToolTip("Model trade-offs: speed, accuracy, memory")
        whisperx_model_help_btn.setFixedSize(22, 22)
        whisperx_model_help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        whisperx_model_help_btn.setStyleSheet("""
            QPushButton { border: 1px solid #cfd6dc; border-radius: 11px; background: #f4f7f9; color: #334; font-weight: bold; }
            QPushButton:hover { background: #eaf1f5; }
            QPushButton:pressed { background: #e0ebf2; }
        """)
        whisperx_model_help_btn.clicked.connect(self.show_model_help_dialog)

        self.whisperx_model_combo = QComboBox()
        self.whisperx_model_combo.addItems([
            'distil-large-v3.5',
            'large-v3-turbo',
            'large-v3',
            'large-v2'
        ])
        self.whisperx_model_combo.setCurrentText('distil-large-v3.5')
        self.whisperx_model_combo.currentTextChanged.connect(self.save_settings_delayed)

        whisperx_model_row.addWidget(whisperx_model_help_btn)
        whisperx_model_row.addWidget(QLabel("Model:"))
        whisperx_model_row.addWidget(self.whisperx_model_combo, 1)
        whisperx_main_tab_layout.addRow(whisperx_model_row)

        # Language
        whisperx_language_row = QHBoxLayout()
        self.whisperx_language_combo = QComboBox()
        languages = ['auto'] + ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'hi', 'ar', 'nl', 'sv', 'no', 'da', 'pl', 'tr', 'th', 'vi', 'id', 'ms']
        self.whisperx_language_combo.addItems(languages)
        self.whisperx_language_combo.setEditable(True)
        self.whisperx_language_combo.currentTextChanged.connect(self.save_settings_delayed)

        whisperx_language_row.addWidget(QLabel("Language:"))
        whisperx_language_row.addWidget(self.whisperx_language_combo, 1)
        whisperx_main_tab_layout.addRow(whisperx_language_row)

        # Task
        whisperx_task_row = QHBoxLayout()
        whisperx_task_help_btn = QPushButton("?")
        whisperx_task_help_btn.setToolTip("Task selection: transcribe vs translate")
        whisperx_task_help_btn.setFixedSize(22, 22)
        whisperx_task_help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        whisperx_task_help_btn.setStyleSheet("""
            QPushButton { border: 1px solid #cfd6dc; border-radius: 11px; background: #f4f7f9; color: #334; font-weight: bold; }
            QPushButton:hover { background: #eaf1f5; }
            QPushButton:pressed { background: #e0ebf2; }
        """)
        whisperx_task_help_btn.clicked.connect(self.show_task_help_dialog)

        self.whisperx_task_combo = QComboBox()
        self.whisperx_task_combo.addItems(['transcribe', 'translate'])
        self.whisperx_task_combo.currentTextChanged.connect(self.save_settings_delayed)

        whisperx_task_row.addWidget(whisperx_task_help_btn)
        whisperx_task_row.addWidget(QLabel("Task:"))
        whisperx_task_row.addWidget(self.whisperx_task_combo, 1)
        whisperx_main_tab_layout.addRow(whisperx_task_row)

        # Device
        whisperx_device_row = QHBoxLayout()
        whisperx_device_help_btn = QPushButton("?")
        whisperx_device_help_btn.setToolTip("Device selection: CPU vs GPU trade-offs")
        whisperx_device_help_btn.setFixedSize(22, 22)
        whisperx_device_help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        whisperx_device_help_btn.setStyleSheet("""
            QPushButton { border: 1px solid #cfd6dc; border-radius: 11px; background: #f4f7f9; color: #334; font-weight: bold; }
            QPushButton:hover { background: #eaf1f5; }
            QPushButton:pressed { background: #e0ebf2; }
        """)
        whisperx_device_help_btn.clicked.connect(self.show_device_help_dialog)

        self.whisperx_device_combo = QComboBox()
        self.whisperx_device_combo.addItems(['cuda', 'cpu'])
        self.whisperx_device_combo.currentTextChanged.connect(self.save_settings_delayed)

        whisperx_device_row.addWidget(whisperx_device_help_btn)
        whisperx_device_row.addWidget(QLabel("Device:"))
        whisperx_device_row.addWidget(self.whisperx_device_combo, 1)
        whisperx_main_tab_layout.addRow(whisperx_device_row)

        whisperx_settings_tabs.addTab(whisperx_main_tab, "Model")

        # === ADVANCED TAB ===
        whisperx_advanced_tab = QWidget(self)  # Parent to prevent top-level window flash
        whisperx_advanced_tab_layout = QFormLayout(whisperx_advanced_tab)
        whisperx_advanced_tab_layout.setContentsMargins(6, 6, 6, 6)  # Reduce margins
        whisperx_advanced_tab_layout.setHorizontalSpacing(8)
        whisperx_advanced_tab_layout.setVerticalSpacing(4)
        whisperx_advanced_tab_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)

        # Compute Type
        whisperx_compute_row = QHBoxLayout()
        whisperx_compute_help_btn = QPushButton("?")
        whisperx_compute_help_btn.setToolTip("Compute type trade-offs: precision, speed, memory")
        whisperx_compute_help_btn.setFixedSize(22, 22)
        whisperx_compute_help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        whisperx_compute_help_btn.setStyleSheet("""
            QPushButton { border: 1px solid #cfd6dc; border-radius: 11px; background: #f4f7f9; color: #334; font-weight: bold; }
            QPushButton:hover { background: #eaf1f5; }
            QPushButton:pressed { background: #e0ebf2; }
        """)
        whisperx_compute_help_btn.clicked.connect(self.show_compute_help_dialog)

        self.whisperx_compute_combo = QComboBox()
        self.whisperx_compute_combo.addItems(['float16', 'int8', 'float32'])
        self.whisperx_compute_combo.setCurrentText('float16')  # Default to float16
        self.whisperx_compute_combo.currentTextChanged.connect(self.save_settings_delayed)

        whisperx_compute_row.addWidget(whisperx_compute_help_btn)
        whisperx_compute_row.addWidget(QLabel("Compute Type:"))
        whisperx_compute_row.addWidget(self.whisperx_compute_combo, 1)
        whisperx_advanced_tab_layout.addRow(whisperx_compute_row)

        # VAD Method
        whisperx_vad_row = QHBoxLayout()
        whisperx_vad_help_btn = QPushButton("?")
        whisperx_vad_help_btn.setToolTip("VAD method trade-offs: speed, accuracy, compatibility")
        whisperx_vad_help_btn.setFixedSize(22, 22)
        whisperx_vad_help_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        whisperx_vad_help_btn.setStyleSheet("""
            QPushButton { border: 1px solid #cfd6dc; border-radius: 11px; background: #f4f7f9; color: #334; font-weight: bold; }
            QPushButton:hover { background: #eaf1f5; }
            QPushButton:pressed { background: #e0ebf2; }
        """)
        whisperx_vad_help_btn.clicked.connect(self.show_vad_help_dialog)

        self.whisperx_vad_method = QComboBox()
        self.whisperx_vad_method.addItems(['silero_v4', 'pyannote_v3', 'pyannote_onnx_v3'])
        self.whisperx_vad_method.currentTextChanged.connect(self.save_settings_delayed)

        whisperx_vad_row.addWidget(whisperx_vad_help_btn)
        whisperx_vad_row.addWidget(QLabel("VAD Method:"))
        whisperx_vad_row.addWidget(self.whisperx_vad_method, 1)
        whisperx_advanced_tab_layout.addRow(whisperx_vad_row)

        # VAD Filter checkbox
        self.whisperx_vad_filter = QCheckBox("Enable Voice Activity Detection (VAD)")
        self.whisperx_vad_filter.setChecked(True)
        self.whisperx_vad_filter.stateChanged.connect(self.save_settings_delayed)
        whisperx_advanced_tab_layout.addRow(self.whisperx_vad_filter)

        whisperx_settings_tabs.addTab(whisperx_advanced_tab, "Advanced")

        # Set size policy to not expand vertically
        whisperx_settings_tabs.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)

        whisperx_settings_main_layout.addWidget(whisperx_settings_tabs)
        layout.addWidget(settings_group)

        # Speaker Diarization (WhisperX-specific feature)
        diarization_group = QGroupBox("Speaker Diarization")
        diarization_layout = QGridLayout(diarization_group)
        diarization_layout.setColumnStretch(1, 1)  # Make second column expandable

        # Row 0: Enable Diarization checkbox (spans both columns)
        self.whisperx_enable_diarization = QCheckBox("Enable Speaker Diarization")
        self.whisperx_enable_diarization.setToolTip("Identify and label different speakers in the audio")
        self.whisperx_enable_diarization.stateChanged.connect(self.save_settings_delayed)
        diarization_layout.addWidget(self.whisperx_enable_diarization, 0, 0, 1, 2)

        # Row 1: HF Token
        diarization_layout.addWidget(QLabel("HF Token:"), 1, 0)
        self.whisperx_hf_token = QLineEdit()
        self.whisperx_hf_token.setPlaceholderText("Hugging Face token (required for diarization)")
        self.whisperx_hf_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.whisperx_hf_token.textChanged.connect(self.save_settings_delayed)
        diarization_layout.addWidget(self.whisperx_hf_token, 1, 1)

        # Row 2: Target speakers
        diarization_layout.addWidget(QLabel("Target speakers:"), 2, 0)
        self.whisperx_num_speakers = QComboBox()
        self.whisperx_num_speakers.setEditable(False)
        self.whisperx_num_speakers.addItem("Auto")
        for i in range(1, 11):
            self.whisperx_num_speakers.addItem(str(i))
        self.whisperx_num_speakers.setCurrentIndex(0)
        self.whisperx_num_speakers.setToolTip("Choose Auto or a specific speaker count")
        self.whisperx_num_speakers.currentIndexChanged.connect(self.save_settings_delayed)
        diarization_layout.addWidget(self.whisperx_num_speakers, 2, 1)

        # Row 3: Validate Token button
        self.whisperx_validate_token_btn = QPushButton("Validate Token")
        self.whisperx_validate_token_btn.setMaximumWidth(140)
        self.whisperx_validate_token_btn.clicked.connect(self.validate_hf_token)
        diarization_layout.addWidget(self.whisperx_validate_token_btn, 3, 0)

        # Row 4: Status label (spans both columns)
        self.whisperx_hf_status_label = QLabel("")
        self.whisperx_hf_status_label.setStyleSheet("color: #666; font-size: 11px;")
        self.whisperx_hf_status_label.setWordWrap(True)
        diarization_layout.addWidget(self.whisperx_hf_status_label, 4, 0, 1, 2)

        # Row 5: Note label (spans both columns)
        diar_note = QLabel(
            'Diarization requires <a href="https://huggingface.co/settings/tokens">HuggingFace token</a> '
            'to download required dependency. You must also, with a HuggingFace account, accept the user '
            'agreement for <a href="https://huggingface.co/pyannote/speaker-diarization-3.1">pyannote/speaker-diarization-3.1</a>. '
            'There are no costs involved.'
        )
        diar_note.setStyleSheet("color: #666; font-size: 11px; font-style: italic;")
        diar_note.setWordWrap(True)
        diar_note.setOpenExternalLinks(True)  # Enable clickable hyperlinks
        diarization_layout.addWidget(diar_note, 5, 0, 1, 2)

        layout.addWidget(diarization_group)

        # Export options (match Generate tab)
        export_group = QGroupBox("Export Formats")
        export_group_layout = QVBoxLayout(export_group)
        export_desc = QLabel("Choose what to generate. At least one option must be selected.")
        export_desc.setStyleSheet("color: #666; font-size: 12px;")
        export_group_layout.addWidget(export_desc)

        export_layout = QHBoxLayout()

        # Default (Adobe Static Transcript + reference transcript)
        default_container = QVBoxLayout()
        self.whisperx_export_adobe = QCheckBox("Default")
        self.whisperx_export_adobe.setChecked(True)
        self.whisperx_export_adobe.stateChanged.connect(self.save_settings_delayed)
        self.whisperx_export_adobe.stateChanged.connect(lambda _=None: self._ensure_one_whisperx_export_checked('adobe'))
        self.whisperx_export_adobe.stateChanged.connect(self.update_whisperx_formatting_visibility)
        default_container.addWidget(self.whisperx_export_adobe)
        default_sub = QLabel("Create an Adobe Premiere Pro compatible static transcript with word-level accuracy, raw text transcription")
        default_sub.setStyleSheet("color: #6a6a6a; font-size: 11px;")
        default_sub.setWordWrap(True)
        default_container.addWidget(default_sub)

        # SRT option - always enabled (hidden from UI)
        srt_container = QVBoxLayout()
        self.whisperx_export_srt = QCheckBox("Create SRT")
        self.whisperx_export_srt.setChecked(True)  # Always generate SRT files
        self.whisperx_export_srt.stateChanged.connect(self.save_settings_delayed)
        self.whisperx_export_srt.stateChanged.connect(lambda _=None: self._ensure_one_whisperx_export_checked('srt'))
        srt_container.addWidget(self.whisperx_export_srt)
        srt_sub = QLabel("Will generate SRT files for use in subtitling.")
        srt_sub.setStyleSheet("color: #6a6a6a; font-size: 11px;")
        srt_sub.setWordWrap(True)
        srt_container.addWidget(srt_sub)

        export_layout.addLayout(default_container)
        export_layout.addSpacing(24)
        export_layout.addLayout(srt_container)
        export_layout.addStretch()
        export_group_layout.addLayout(export_layout)

        # Hide export formats - SRT is always generated
        export_group.setVisible(False)
        layout.addWidget(export_group)

        # Format Transcript options (match Generate tab)
        self.whisperx_formatting_group = QGroupBox("Paragraph Formatting")
        formatting_group = self.whisperx_formatting_group
        formatting_layout = QVBoxLayout(formatting_group)
        formatting_desc = QLabel("Adjust duration of gap to be considered a paragraph break")
        formatting_desc.setStyleSheet("color: #666; font-size: 12px;")
        formatting_layout.addWidget(formatting_desc)

        # Hide continuous segment option
        self.whisperx_format_single_segment = QRadioButton("Continuous Segment")
        self.whisperx_format_single_segment.setVisible(False)
        self.whisperx_format_paragraph_form = QRadioButton("Paragraph Form 2.0")
        # Default: Paragraph Form is now the default
        self.whisperx_format_paragraph_form.setChecked(True)
        self.whisperx_format_paragraph_form.setVisible(False)  # Hide radio button, keep functionality
        self.whisperx_format_single_segment.toggled.connect(self.save_settings_delayed)
        self.whisperx_format_paragraph_form.toggled.connect(self.save_settings_delayed)

        # Don't add the radio button to layout since it's hidden
        # formatting_layout.addWidget(self.whisperx_format_paragraph_form)

        # Gap threshold control (enabled only for Paragraph Form)
        gap_row = QHBoxLayout()
        gap_row.addSpacing(24)
        gap_row.addWidget(QLabel("Paragraph gap threshold (seconds):"))
        self.whisperx_paragraph_gap_spin = QDoubleSpinBox()
        self.whisperx_paragraph_gap_spin.setRange(0.5, 60.0)
        self.whisperx_paragraph_gap_spin.setSingleStep(0.5)
        self.whisperx_paragraph_gap_spin.setDecimals(1)
        self.whisperx_paragraph_gap_spin.setValue(2.0)
        self.whisperx_paragraph_gap_spin.setToolTip("Start a new paragraph when the pause between consecutive words exceeds this many seconds.")
        self.whisperx_paragraph_gap_spin.valueChanged.connect(self.save_settings_delayed)
        gap_row.addWidget(self.whisperx_paragraph_gap_spin)
        gap_row.addStretch()
        formatting_layout.addLayout(gap_row)

        # Don't add single segment radio button since it's hidden
        # formatting_layout.addWidget(self.whisperx_format_single_segment)

        # Enable/disable threshold input depending on mode
        self.whisperx_format_single_segment.toggled.connect(self.update_whisperx_formatting_controls)
        self.whisperx_format_paragraph_form.toggled.connect(self.update_whisperx_formatting_controls)

        # Initialize enabled state
        self.update_whisperx_formatting_controls()
        # NOTE: update_whisperx_formatting_visibility() is called AFTER formatting_group is added to layout
        # to prevent window flash when setVisible() is called on a parentless widget

        layout.addWidget(formatting_group)
        # Update visibility AFTER adding to layout to prevent window flash
        self.update_whisperx_formatting_visibility()

        # Output directory
        output_group = QGroupBox("Output Directory")
        output_layout = QHBoxLayout(output_group)

        self.whisperx_output_dir = QLineEdit()
        self.whisperx_output_dir.setPlaceholderText("Default: Same directory as input file")
        self.whisperx_output_dir.textChanged.connect(self.save_settings_delayed)

        browse_output_btn = QPushButton("Browse...")
        browse_output_btn.clicked.connect(self.browse_whisperx_output_dir)
        browse_output_btn.setMaximumWidth(80)

        output_layout.addWidget(self.whisperx_output_dir)
        output_layout.addWidget(browse_output_btn)

        layout.addWidget(output_group)
        layout.addStretch()

        return tab

    def _deferred_whisperx_check(self):
        """
        Perform WhisperX dependency check after startup is complete.

        This is called via QTimer.singleShot to avoid subprocess calls during
        splash screen display. On Windows, subprocess calls that load PyTorch
        can briefly flash a console window even with CREATE_NO_WINDOW flag.
        """
        try:
            print("[DEBUG] >>> DEFERRED check_whisperx_deps_simple()")
            # Check dependencies and update UI (also returns True if installed)
            whisperx_installed = self.check_whisperx_deps_simple()
            print("[DEBUG] <<< DEFERRED check_whisperx_deps_simple() complete")

            # Auto-switch to appropriate tab based on installed dependencies
            if whisperx_installed:
                # WhisperX is installed - show WhisperX tab by default
                self.tab_widget.setCurrentIndex(self.whisperx_tab_index)
            elif self._check_faster_whisper_dependencies():
                # Faster-Whisper-XXL is ready - show Generate tab by default
                self.tab_widget.setCurrentIndex(self.generate_tab_index)
        except Exception as e:
            print(f"[DEBUG] Deferred WhisperX check error: {e}")

    def check_whisperx_deps_simple(self):
        """Check if WhisperX is installed in the self-contained virtual environment"""
        # Check if venv exists and has WhisperX installed
        try:
            app_dir = get_app_directory()
            # Debug output
            print(f"[WhisperX Check] App directory: {app_dir}")
            print(f"[WhisperX Check] sys.frozen: {getattr(sys, 'frozen', False)}")
            print(f"[WhisperX Check] sys.executable: {sys.executable}")
        except Exception as e:
            error_msg = f"Error getting app directory: {str(e)[:50]}"
            print(f"[WhisperX Check] {error_msg}")
            self.whisperx_deps_status.setText(f"❌ {error_msg}")
            self.whisperx_deps_status.setStyleSheet("color: red; font-size: 12px; font-weight: bold;")
            self.whisperx_install_btn.setVisible(True)
            self.whisperx_update_btn.setVisible(False)
            self.whisperx_reinstall_btn.setVisible(False)
            return False

        venv_dir = os.path.join(app_dir, "whisperx_env")
        print(f"[WhisperX Check] Looking for venv at: {venv_dir}")

        # Check for portable embedded Python first (python.exe directly in whisperx_env root)
        # This is the preferred portable structure that works regardless of where the folder is located
        portable_python = os.path.join(venv_dir, "python.exe") if sys.platform == "win32" else os.path.join(venv_dir, "bin", "python3")
        if os.path.exists(portable_python):
            venv_python = portable_python
            print(f"[WhisperX Check] Found portable Python at: {venv_python}")
        # Check for new venv structure (full Python installation with nested venv)
        elif sys.platform == "win32":
            nested_venv = os.path.join(venv_dir, "venv", "Scripts", "python.exe")
            if os.path.exists(nested_venv):
                venv_python = nested_venv
                print(f"[WhisperX Check] Found nested venv Python at: {venv_python}")
            else:
                # Check for legacy embedded Python (backward compatibility)
                embedded_python = os.path.join(venv_dir, "python_embedded", "python.exe")
                if os.path.exists(embedded_python):
                    venv_python = embedded_python
                    print(f"[WhisperX Check] Found legacy embedded Python at: {venv_python}")
                else:
                    # Fall back to traditional venv structure (used when running from source)
                    venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
                    print(f"[WhisperX Check] Checking for venv Python at: {venv_python}")
        else:
            # Linux/Mac: check nested venv first, then fallback
            nested_venv = os.path.join(venv_dir, "venv", "bin", "python")
            if os.path.exists(nested_venv):
                venv_python = nested_venv
                print(f"[WhisperX Check] Found nested venv Python at: {venv_python}")
            else:
                venv_python = os.path.join(venv_dir, "bin", "python")
                print(f"[WhisperX Check] Checking for venv Python at: {venv_python}")

        print(f"[WhisperX Check] Python exists: {os.path.exists(venv_python)}")

        if not os.path.exists(venv_python):
            self.whisperx_deps_status.setText("❌ WhisperX not installed")
            self.whisperx_deps_status.setStyleSheet("color: red; font-size: 12px; font-weight: bold;")
            self.whisperx_install_btn.setVisible(True)
            self.whisperx_update_btn.setVisible(False)
            self.whisperx_reinstall_btn.setVisible(False)
            if hasattr(self, 'whisperx_cuda_status_label'):
                self.whisperx_cuda_status_label.setVisible(False)
            if hasattr(self, 'whisperx_cuda_test_btn'):
                self.whisperx_cuda_test_btn.setVisible(False)
            if hasattr(self, 'whisperx_cuda_status_label'):
                self.whisperx_cuda_status_label.setVisible(False)
            if hasattr(self, 'whisperx_cuda_test_btn'):
                self.whisperx_cuda_test_btn.setVisible(False)
            if hasattr(self, 'whisperx_not_installed_label'):
                self.whisperx_not_installed_label.setVisible(True)
            return False

        # Try to import whisperx in the venv and get version
        try:
            import subprocess
            import json

            # Check if whisperx is installed and get version
            version_check = "\n".join([
                "import json",
                "import importlib.metadata",
                "status = {'installed': False, 'version': None}",
                "try:",
                "    import whisperx",
                "    status['installed'] = True",
                "    try:",
                "        status['version'] = importlib.metadata.version('whisperx')",
                "    except Exception:",
                "        status['version'] = 'unknown'",
                "except Exception as e:",
                "    status['error'] = str(e)",
                "print(json.dumps(status))",
            ])

            # Create environment with PYTHONPATH pointing to site-packages
            # This makes the portable env work regardless of drive letter
            env = get_clean_environment()
            site_packages = os.path.join(venv_dir, "Lib", "site-packages")
            if os.path.exists(site_packages):
                env["PYTHONPATH"] = site_packages
                print(f"[WhisperX Check] Setting PYTHONPATH to: {site_packages}")

            result = subprocess.run(
                [venv_python, "-c", version_check],
                capture_output=True,
                text=True,
                timeout=60,  # Increased from 5s - importing whisperx loads PyTorch which can take 30+ seconds
                env=env,
                startupinfo=get_subprocess_startup_info(),
                creationflags=get_subprocess_creation_flags()
            )

            if result.returncode == 0:
                try:
                    status_data = json.loads(result.stdout.strip())
                    if status_data.get('installed'):
                        version = status_data.get('version', 'unknown')
                        version_text = f" v{version}" if version and version != 'unknown' else ""
                        self.whisperx_deps_status.setText(f"✅ WhisperX{version_text}")
                        self.whisperx_deps_status.setStyleSheet("color: green; font-size: 12px; font-weight: bold;")
                        self.whisperx_install_btn.setVisible(False)
                        self.whisperx_update_btn.setVisible(True)  # Show update option when installed
                        self.whisperx_reinstall_btn.setVisible(True)  # Show reinstall option when installed
                        if hasattr(self, 'whisperx_not_installed_label'):
                            self.whisperx_not_installed_label.setVisible(False)
                    else:
                        # Installed flag is False
                        error_msg = status_data.get('error', 'unknown error')
                        print(f"[WhisperX Check] Import failed - error: {error_msg}")
                        print(f"[WhisperX Check] Full status_data: {status_data}")
                        self.whisperx_deps_status.setText("❌ WhisperX not installed in venv")
                        self.whisperx_deps_status.setStyleSheet("color: red; font-size: 12px; font-weight: bold;")
                        self.whisperx_install_btn.setVisible(True)
                        self.whisperx_update_btn.setVisible(False)
                        self.whisperx_reinstall_btn.setVisible(False)
                        if hasattr(self, 'whisperx_cuda_status_label'):
                            self.whisperx_cuda_status_label.setVisible(False)
                        if hasattr(self, 'whisperx_cuda_test_btn'):
                            self.whisperx_cuda_test_btn.setVisible(False)
                        if hasattr(self, 'whisperx_not_installed_label'):
                            self.whisperx_not_installed_label.setVisible(True)
                        return False
                except json.JSONDecodeError:
                    # Fall back to old behavior if JSON parsing fails
                    self.whisperx_deps_status.setText(f"✅ WhisperX is installed")
                    self.whisperx_deps_status.setStyleSheet("color: green; font-size: 12px; font-weight: bold;")
                    self.whisperx_install_btn.setVisible(False)
                    self.whisperx_update_btn.setVisible(True)  # Show update option when installed
                    self.whisperx_reinstall_btn.setVisible(True)  # Show reinstall option when installed
                # Show CUDA status and test button when installed
                if hasattr(self, 'whisperx_cuda_status_label'):
                    self.whisperx_cuda_status_label.setVisible(True)
                    self.whisperx_cuda_status_label.setText("CUDA status: not checked")
                    self.whisperx_cuda_status_label.setStyleSheet("color: #666; font-size: 12px;")
                if hasattr(self, 'whisperx_cuda_test_btn'):
                    self.whisperx_cuda_test_btn.setVisible(True)
                # Store venv python path for use during transcription
                if not hasattr(self, 'whisperx_venv_python'):
                    self.whisperx_venv_python = venv_python
                return True
            else:
                # returncode != 0, something went wrong
                print(f"[WhisperX Check] FAILED - Return code: {result.returncode}")
                print(f"[WhisperX Check] stdout: {result.stdout}")
                print(f"[WhisperX Check] stderr: {result.stderr}")
                self.whisperx_deps_status.setText("❌ WhisperX check failed")
                self.whisperx_deps_status.setStyleSheet("color: red; font-size: 12px; font-weight: bold;")
                self.whisperx_install_btn.setVisible(True)
                self.whisperx_update_btn.setVisible(False)
                self.whisperx_reinstall_btn.setVisible(False)
                if hasattr(self, 'whisperx_cuda_status_label'):
                    self.whisperx_cuda_status_label.setVisible(False)
                if hasattr(self, 'whisperx_cuda_test_btn'):
                    self.whisperx_cuda_test_btn.setVisible(False)
                return False
        except subprocess.TimeoutExpired:
            self.whisperx_deps_status.setText(f"❌ WhisperX check timed out")
            self.whisperx_deps_status.setStyleSheet("color: red; font-size: 12px; font-weight: bold;")
            self.whisperx_install_btn.setVisible(True)
            self.whisperx_update_btn.setVisible(False)
            self.whisperx_reinstall_btn.setVisible(False)
            if hasattr(self, 'whisperx_cuda_status_label'):
                self.whisperx_cuda_status_label.setVisible(False)
            if hasattr(self, 'whisperx_cuda_test_btn'):
                self.whisperx_cuda_test_btn.setVisible(False)
            return False
        except Exception as e:
            # Show more detailed error for debugging
            import traceback
            error_msg = str(e)
            full_trace = traceback.format_exc()

            # Truncate very long error messages for UI
            if len(error_msg) > 100:
                error_msg = error_msg[:100] + "..."
            self.whisperx_deps_status.setText(f"❌ Error checking WhisperX: {error_msg}")
            self.whisperx_deps_status.setStyleSheet("color: red; font-size: 12px; font-weight: bold;")

            # Log full traceback to console for debugging (safely)
            try:
                if hasattr(self, '_append_text_to_console') and callable(self._append_text_to_console):
                    self._append_text_to_console(f"\n=== WhisperX Check Error ===\n{full_trace}\n")
            except Exception:
                # If console logging fails, just print to stdout
                print(f"WhisperX check error:\n{full_trace}")

            self.whisperx_install_btn.setVisible(True)
            self.whisperx_update_btn.setVisible(False)
            self.whisperx_reinstall_btn.setVisible(False)
            if hasattr(self, 'whisperx_cuda_status_label'):
                self.whisperx_cuda_status_label.setVisible(False)
            if hasattr(self, 'whisperx_cuda_test_btn'):
                self.whisperx_cuda_test_btn.setVisible(False)
            return False

    def install_whisperx_simple(self, upgrade_version=None, use_cuda_override=None):
        """Install WhisperX in a self-contained virtual environment.

        Args:
            upgrade_version: If set (e.g. "3.8.2"), download that specific release instead
                of the default WHISPERX_VERSION constant. Used by the auto-upgrade path.
            use_cuda_override: If not None, skip the CUDA dialog and use this value directly.
                Used by the auto-upgrade path which already asked the user.
        """
        if use_cuda_override is not None:
            install_cuda = use_cuda_override
        else:
            # Ask user if they want CUDA support
            cuda_reply = QMessageBox.question(
                self,
                "CUDA Support",
                "Do you want to install WhisperX with NVIDIA CUDA GPU support?\n\n"
                "✅ WITH CUDA (Recommended if you have an NVIDIA GPU):\n"
                "   • Requires: NVIDIA GPU with CUDA support\n"
                "   • Download size: ~2-3GB\n"
                "   • Processing: 10-50x faster\n"
                "   • Installation time: 10-15 minutes\n\n"
                "❌ WITHOUT CUDA (CPU-only):\n"
                "   • Works on any computer\n"
                "   • Download size: ~500MB-1GB\n"
                "   • Processing: Slower (CPU only)\n"
                "   • Installation time: 5-10 minutes\n\n"
                "Do you have an NVIDIA GPU and want CUDA support?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            install_cuda = (cuda_reply == QMessageBox.StandardButton.Yes)

        self.whisperx_install_btn.setEnabled(False)
        self.whisperx_install_btn.setVisible(False)
        self.whisperx_cancel_btn.setVisible(True)
        self.whisperx_install_btn.setText("Installing...")
        self.whisperx_deps_status.setText("Installing WhisperX... This may take several minutes.")
        self.whisperx_deps_status.setStyleSheet("color: #ff9800; font-size: 12px;")

        # Create immediate log file BEFORE anything else
        immediate_log_path = None
        immediate_log = None

        # Simple write test function
        def try_write_log(path):
            try:
                f = open(path, 'w', encoding='utf-8', buffering=1)
                f.write("TEST LOG FILE CREATED\n")
                f.flush()
                return f
            except Exception as e:
                return None

        try:
            import datetime

            # Try multiple locations
            log_locations = []

            # Location 1: Next to executable/script
            try:
                if getattr(sys, 'frozen', False):
                    loc1 = os.path.dirname(sys.executable)
                else:
                    loc1 = get_app_directory()
                log_locations.append(loc1)
            except:
                pass

            # Location 2: Current working directory
            try:
                loc2 = os.getcwd()
                log_locations.append(loc2)
            except:
                pass

            # Location 3: User's home directory
            try:
                loc3 = os.path.expanduser("~")
                log_locations.append(loc3)
            except:
                pass

            # Location 4: Temp directory
            try:
                import tempfile
                loc4 = tempfile.gettempdir()
                log_locations.append(loc4)
            except:
                pass

            # Try each location
            for log_dir in log_locations:
                try:
                    immediate_log_path = os.path.join(log_dir, "whisperx_install_crash_debug.log")
                    immediate_log = try_write_log(immediate_log_path)
                    if immediate_log:
                        # Success! Now write actual content
                        immediate_log.write("=" * 70 + "\n")
                        immediate_log.write("WhisperX Installation Debug Log\n")
                        immediate_log.write("=" * 70 + "\n")
                        immediate_log.write(f"Timestamp: {datetime.datetime.now()}\n")
                        immediate_log.write(f"Running as .exe: {getattr(sys, 'frozen', False)}\n")
                        immediate_log.write(f"Python: {sys.version}\n")
                        immediate_log.write(f"Platform: {sys.platform}\n")
                        immediate_log.write(f"Executable: {sys.executable}\n")
                        immediate_log.write(f"Install CUDA: {install_cuda}\n")
                        immediate_log.write(f"Log location: {immediate_log_path}\n")
                        immediate_log.write("=" * 70 + "\n\n")
                        immediate_log.write("CHECKPOINT: Log file created successfully\n")
                        immediate_log.flush()

                        self._append_text_to_console(f"\n✓ CRASH DEBUG LOG CREATED: {immediate_log_path}\n")
                        self._append_text_to_console("If installation freezes/crashes, check this file for details.\n\n")
                        break  # Successfully created log
                except:
                    continue  # Try next location

            if not immediate_log:
                self._append_text_to_console(f"⚠ WARNING: Could not create crash debug log in any location!\n")
                self._append_text_to_console(f"Tried: {', '.join(log_locations)}\n\n")

        except Exception as e:
            self._append_text_to_console(f"⚠ Critical error setting up crash log: {str(e)}\n")
            self._append_text_to_console(f"Error type: {type(e).__name__}\n\n")

        self._append_text_to_console("=" * 60 + "\n")
        self._append_text_to_console("DEBUG: Pre-Installation Environment Check\n")
        self._append_text_to_console("=" * 60 + "\n")
        self._append_text_to_console(f"Running as .exe: {getattr(sys, 'frozen', False)}\n")
        self._append_text_to_console(f"Python: {sys.version.split()[0]}\n")
        self._append_text_to_console(f"Platform: {sys.platform}\n")
        self._append_text_to_console(f"Executable: {sys.executable}\n")

        # Check if we can even create a venv
        try:
            import venv
            self._append_text_to_console(f"venv module: Available\n")
        except:
            self._append_text_to_console(f"venv module: NOT AVAILABLE - THIS WILL FAIL!\n")

        self._append_text_to_console("=" * 60 + "\n\n")

        self._append_text_to_console("Installing WhisperX to self-contained environment...\n")
        self._append_text_to_console("Location: whisperx_env/ (within Scriptoria folder)\n")
        if install_cuda:
            self._append_text_to_console("Mode: WITH CUDA GPU support\n")
            self._append_text_to_console("This may take 10-15 minutes depending on your internet connection.\n")
        else:
            self._append_text_to_console("Mode: CPU-only (no CUDA)\n")
            self._append_text_to_console("This may take 5-10 minutes depending on your internet connection.\n")
        self._append_text_to_console("=" * 60 + "\n\n")

        # Use QThread to run installation in background
        from PyQt6.QtCore import QThread, pyqtSignal
        import time

        class InstallThread(QThread):
            finished = pyqtSignal(bool, str, str)  # success, message, venv_python_path
            progress = pyqtSignal(str)

            def __init__(self, parent_widget, install_cuda, crash_log, upgrade_version=None):
                super().__init__()
                self.parent_widget = parent_widget
                self.install_cuda = install_cuda
                self.crash_log = crash_log  # Pass the already-open log file
                self._cancelled = False  # Cancellation flag
                self.upgrade_version = upgrade_version  # None = use WHISPERX_VERSION constant

            def cancel(self):
                """Request cancellation of the installation"""
                self._cancelled = True

            def run(self):
                import subprocess
                import threading
                import queue
                import time

                # Only import venv if not running as frozen .exe
                # (PyInstaller doesn't bundle venv module, and we don't need it when frozen)
                if not getattr(sys, 'frozen', False):
                    import venv
                else:
                    venv = None  # Not needed when using --target method

                # Write to crash log immediately
                def crash_log(msg):
                    """Write to the pre-opened crash log"""
                    if self.crash_log:
                        try:
                            self.crash_log.write(msg)
                            self.crash_log.flush()
                        except:
                            pass

                crash_log("CHECKPOINT: Thread started\n")

                # Setup debug log file - MUST be first thing
                # Use a list to make it mutable in the closure
                log_file_container = [None]
                log_path = None

                # Define log_and_emit helper
                def log_and_emit(msg):
                    """Helper to write to both console and log file"""
                    self.progress.emit(msg)
                    crash_log(msg)  # Also write to crash log
                    if log_file_container[0]:
                        try:
                            log_file_container[0].write(msg)
                            log_file_container[0].flush()
                        except:
                            pass

                # Try to create log file
                crash_log("CHECKPOINT: About to create detailed log file\n")
                try:
                    # Try to get app directory
                    crash_log("CHECKPOINT: Getting app directory\n")
                    if getattr(sys, 'frozen', False):
                        app_dir = os.path.dirname(sys.executable)
                    else:
                        script_dir = os.path.dirname(os.path.abspath(__file__))
                        app_dir = os.path.dirname(script_dir)

                    log_path = os.path.join(app_dir, "whisperx_install_debug.log")
                    crash_log(f"CHECKPOINT: Will create detailed log at: {log_path}\n")
                    self.progress.emit(f"Attempting to create log file at: {log_path}\n")

                    # Create directory if needed
                    os.makedirs(app_dir, exist_ok=True)

                    # Open log file with line buffering
                    log_file_container[0] = open(log_path, 'w', encoding='utf-8', buffering=1)
                    log_and_emit(f"✓ Debug log file created successfully: {log_path}\n\n")

                except Exception as log_error:
                    # If we can't create log file, try alternate location
                    self.progress.emit(f"⚠ Could not create log file at {log_path}: {str(log_error)}\n")
                    self.progress.emit(f"   Error type: {type(log_error).__name__}\n")
                    try:
                        # Try temp directory
                        import tempfile
                        temp_dir = tempfile.gettempdir()
                        log_path = os.path.join(temp_dir, "whisperx_install_debug.log")
                        self.progress.emit(f"Trying alternate location: {log_path}\n")
                        log_file_container[0] = open(log_path, 'w', encoding='utf-8', buffering=1)
                        log_and_emit(f"✓ Debug log file created in temp directory: {log_path}\n\n")
                    except Exception as temp_error:
                        # Give up on log file
                        log_file_container[0] = None
                        self.progress.emit(f"⚠ Could not create log file anywhere: {str(temp_error)}\n")
                        self.progress.emit(f"   Error type: {type(temp_error).__name__}\n")
                        self.progress.emit("Continuing without log file...\n\n")

                try:
                    # ==================================================================
                    # DEBUG: Environment and System Information
                    # ==================================================================
                    log_and_emit("=" * 70 + "\n")
                    log_and_emit("DEBUG: Installation Environment Information\n")
                    log_and_emit("=" * 70 + "\n")

                    # Check if running as frozen executable
                    is_frozen = getattr(sys, 'frozen', False)
                    log_and_emit(f"Running as .exe: {is_frozen}\n")

                    if is_frozen:
                        log_and_emit(f"Executable path: {sys.executable}\n")
                        if hasattr(sys, '_MEIPASS'):
                            log_and_emit(f"PyInstaller temp dir (_MEIPASS): {sys._MEIPASS}\n")

                    # Python version and location
                    log_and_emit(f"Python version: {sys.version}\n")
                    log_and_emit(f"Python executable: {sys.executable}\n")
                    log_and_emit(f"Platform: {sys.platform}\n")

                    # Environment variables that might cause issues
                    env_vars_to_check = ['PYTHONHOME', 'PYTHONPATH', 'PYTHONEXECUTABLE', '_MEIPASS', 'PATH']
                    log_and_emit("\nEnvironment variables (before cleaning):\n")
                    for var in env_vars_to_check:
                        val = os.environ.get(var, '<not set>')
                        if var == 'PATH':
                            # Truncate PATH for readability
                            val = val[:200] + '...' if len(val) > 200 else val
                        log_and_emit(f"  {var}: {val}\n")

                    # Test clean environment
                    clean_env = get_clean_environment()
                    log_and_emit("\nEnvironment variables (after cleaning):\n")
                    for var in ['PYTHONHOME', 'PYTHONPATH', 'PYTHONEXECUTABLE']:
                        val = clean_env.get(var, '<removed>')
                        log_and_emit(f"  {var}: {val}\n")

                    log_and_emit("=" * 70 + "\n\n")

                    # Create whisperx_env directory in Scriptoria root
                    app_dir = get_app_directory()
                    whisperx_dir = os.path.join(app_dir, "whisperx_env")

                    self.progress.emit(f"App directory: {app_dir}\n")

                    # Check if running as frozen executable
                    is_frozen = getattr(sys, 'frozen', False)
                    crash_log(f"CHECKPOINT: is_frozen = {is_frozen}\n")

                    if is_frozen:
                        # Running as .exe - download pre-built WhisperX portable package
                        self.progress.emit(f"Running as .exe - downloading portable WhisperX package...\n")
                        crash_log("CHECKPOINT: Downloading portable WhisperX package\n")

                        # Check for cancellation
                        if self._cancelled:
                            crash_log("CHECKPOINT: Installation cancelled before download\n")
                            self.finished.emit(False, "Installation cancelled by user", "")
                            return

                        # Create the directory
                        os.makedirs(whisperx_dir, exist_ok=True)
                        crash_log(f"CHECKPOINT: Created directory: {whisperx_dir}\n")

                        # Download and extract portable package (NEW METHOD)
                        # Use log_and_emit so all download errors are written to the log file,
                        # not just shown in the UI dialog.
                        venv_python = download_and_extract_whisperx(
                            install_dir=whisperx_dir,
                            use_cuda=self.install_cuda,
                            progress_callback=log_and_emit,
                            cancel_check=lambda: self._cancelled,
                            target_version=self.upgrade_version
                        )

                        if not venv_python:
                            crash_log("CHECKPOINT: Portable package download/extract failed\n")
                            error_msg = (
                                "Failed to download or extract WhisperX portable package.\n\n"
                                "Please check:\n"
                                "1. Your internet connection\n"
                                "2. The portable packages have been built and uploaded to GitHub\n"
                                "3. The crash log for details\n\n"
                                "See whisperx_portable_builder/QUICKSTART.md for build instructions."
                            )
                            self.progress.emit(f"✗ {error_msg}\n")
                            self.finished.emit(False, error_msg, "")
                            return

                        crash_log(f"CHECKPOINT: Portable WhisperX runtime ready at: {venv_python}\n")
                        self.progress.emit("✓ Portable WhisperX environment is ready.\n\n")

                        self.progress.emit("Verifying portable installation...\n")
                        verify_code = "\n".join([
                            "import json",
                            "import importlib",
                            "import importlib.metadata",
                            "status = {'import_ok': False, 'version': None}",
                            "try:",
                            "    import whisperx",
                            "    status['import_ok'] = True",
                            "    status['version'] = getattr(whisperx, '__version__', None)",
                            "    if not status['version']:",
                            "        try:",
                            "            status['version'] = importlib.metadata.version('whisperx')",
                            "        except Exception:",
                            "            status['version'] = 'unknown'",
                            "except Exception as exc:",
                            "    status['error'] = repr(exc)",
                            "print(json.dumps(status))",
                        ])
                        verify_result = subprocess.run(
                            [venv_python, "-c", verify_code],
                            capture_output=True,
                            text=True,
                            timeout=30,
                            env=get_clean_environment(),
                            startupinfo=get_subprocess_startup_info(),
                            creationflags=get_subprocess_creation_flags()
                        )

                        if verify_result.returncode != 0:
                            error_msg = (
                                "Installation verification failed.\n\n"
                                f"Error output:\n{verify_result.stderr}\n\nStdout:\n{verify_result.stdout}"
                            )
                            crash_log(f"CHECKPOINT: Portable verification failed: {error_msg}\n")
                            self.progress.emit(f"\n{error_msg}\n")
                            self.finished.emit(False, error_msg, "")
                            return

                        try:
                            info = json.loads(verify_result.stdout.strip() or "{}")
                        except Exception:
                            info = {}

                        if not info.get("import_ok"):
                            error_msg = "Installation verification failed: whisperx import unsuccessful."
                            if info.get("error"):
                                error_msg += f"\n\nError: {info['error']}"
                            crash_log(f"CHECKPOINT: Portable verification import failed: {error_msg}\n")
                            self.progress.emit(f"\n{error_msg}\n")
                            self.finished.emit(False, error_msg, "")
                            return

                        reported_version = info.get("version") or "unknown"
                        crash_log(f"CHECKPOINT: Portable verification succeeded, version {reported_version}\n")
                        self.progress.emit(f"WhisperX import OK (version: {reported_version})\n")
                        self.finished.emit(True, "WhisperX installed successfully to self-contained environment!", venv_python)
                        return

                    else:
                        # Running from source - use traditional venv
                        self.progress.emit(f"Creating virtual environment at: {whisperx_dir}\n")
                        venv_dir = whisperx_dir  # Keep old variable name for compatibility

                    # Setup Python environment based on execution mode
                    if not is_frozen:
                        # Running from source - create traditional venv
                        crash_log(f"CHECKPOINT: Checking if venv exists at: {venv_dir}\n")
                        if not os.path.exists(venv_dir):
                            self.progress.emit("Creating new virtual environment...\n")
                            crash_log("CHECKPOINT: Creating venv with venv.create()\n")
                            try:
                                venv.create(venv_dir, with_pip=True)
                                crash_log("CHECKPOINT: venv.create() returned successfully\n")
                                self.progress.emit("✓ Virtual environment created successfully.\n\n")
                            except Exception as venv_error:
                                crash_log(f"CHECKPOINT: venv.create() failed: {type(venv_error).__name__}: {str(venv_error)}\n")
                                error_msg = f"Failed to create virtual environment: {str(venv_error)}"
                                self.finished.emit(False, error_msg, "")
                                return
                        else:
                            crash_log("CHECKPOINT: venv already exists\n")
                            self.progress.emit("✓ Using existing virtual environment.\n\n")

                        # Determine Python executable path in venv
                        if sys.platform == "win32":
                            venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
                        else:
                            venv_python = os.path.join(venv_dir, "bin", "python")

                        crash_log(f"CHECKPOINT: venv_python = {venv_python}\n")
                    else:
                        # Running as .exe - venv_python already set to sys.executable
                        self.progress.emit("✓ Using bundled Python (no venv needed when running as .exe)\n\n")
                        crash_log(f"CHECKPOINT: Using bundled Python: {venv_python}\n")

                    self.progress.emit(f"Looking for venv Python at: {venv_python}\n")

                    if not os.path.exists(venv_python):
                        # List what's actually in the directory
                        try:
                            if sys.platform == "win32":
                                scripts_dir = os.path.join(venv_dir, "Scripts")
                            else:
                                scripts_dir = os.path.join(venv_dir, "bin")

                            if os.path.exists(scripts_dir):
                                files = os.listdir(scripts_dir)
                                self.progress.emit(f"Files in {scripts_dir}:\n")
                                for f in files[:20]:  # Limit to first 20
                                    self.progress.emit(f"  - {f}\n")
                            else:
                                self.progress.emit(f"Directory does not exist: {scripts_dir}\n")
                        except Exception as list_error:
                            self.progress.emit(f"Could not list directory: {str(list_error)}\n")

                        error_msg = f"Failed to create virtual environment. Python not found at: {venv_python}"
                        self.progress.emit(f"✗ {error_msg}\n")
                        self.finished.emit(False, error_msg, "")
                        return

                    self.progress.emit(f"✓ Found venv Python at: {venv_python}\n\n")

                    # Test venv Python before proceeding
                    self.progress.emit("Testing venv Python executable...\n")
                    try:
                        test_result = subprocess.run(
                            [venv_python, "--version"],
                            capture_output=True,
                            text=True,
                            timeout=10,
                            env=get_clean_environment(),
                            startupinfo=get_subprocess_startup_info(),
                            creationflags=get_subprocess_creation_flags()
                        )
                        if test_result.returncode == 0:
                            self.progress.emit(f"✓ Venv Python test successful: {test_result.stdout.strip()}\n\n")
                        else:
                            self.progress.emit(f"✗ Venv Python test failed with code {test_result.returncode}\n")
                            self.progress.emit(f"Stdout: {test_result.stdout}\n")
                            self.progress.emit(f"Stderr: {test_result.stderr}\n\n")
                    except Exception as test_error:
                        self.progress.emit(f"✗ Could not test venv Python: {str(test_error)}\n")
                        import traceback
                        self.progress.emit(f"Traceback:\n{traceback.format_exc()}\n\n")

                    # Build pip install command base
                    # No need for --target with embedded Python or venv - both have their own site-packages
                    pip_target_args = []

                    # Check for cancellation
                    if self._cancelled:
                        crash_log("CHECKPOINT: Installation cancelled before pip upgrade\n")
                        self.finished.emit(False, "Installation cancelled by user", "")
                        return

                    # Upgrade pip first (stream output)
                    self.progress.emit("Upgrading pip...\n")
                    try:
                        pip_cmd = [venv_python, "-m", "pip", "install", "--upgrade", "pip"]
                        # Don't use --target for pip upgrade itself
                        pip_proc = subprocess.Popen(
                            pip_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            env=get_clean_environment(),
                            startupinfo=get_subprocess_startup_info(),
                            creationflags=get_subprocess_creation_flags()
                        )

                        for line in pip_proc.stdout:
                            if self._cancelled:
                                pip_proc.terminate()
                                pip_proc.wait(timeout=5)
                                self.finished.emit(False, "Installation cancelled by user", "")
                                return
                            self.progress.emit(line)

                        pip_proc.wait()
                        if self._cancelled:
                            self.finished.emit(False, "Installation cancelled by user", "")
                            return
                        if pip_proc.returncode != 0:
                            self.progress.emit(f"⚠ Warning: pip upgrade had issues (exit code {pip_proc.returncode}), continuing anyway...\n")
                        else:
                            self.progress.emit("✓ Pip upgraded successfully.\n")
                    except Exception as pip_error:
                        self.progress.emit(f"⚠ Warning: pip upgrade failed with error: {str(pip_error)}\n")
                        self.progress.emit("Continuing with existing pip version...\n")
                        import traceback
                        self.progress.emit(f"Traceback:\n{traceback.format_exc()}\n")

                    # Use configured WhisperX requirements
                    torch_version = WHISPERX_PYTORCH_VERSION
                    torchaudio_version = WHISPERX_TORCHAUDIO_VERSION

                    self.progress.emit(f"\nUsing WhisperX requirements:\n")
                    self.progress.emit(f"  PyTorch: {torch_version}\n")
                    self.progress.emit(f"  Torchaudio: {torchaudio_version}\n")
                    self.progress.emit(f"  CUDA: {WHISPERX_CUDA_VERSION} ({WHISPERX_CUDA_SHORT})\n")

                    # Optionally verify against PyPI (for logging/warning only)
                    try:
                        import urllib.request
                        import json as json_module

                        url = "https://pypi.org/pypi/whisperx/json"
                        with urllib.request.urlopen(url, timeout=10) as response:
                            data = json_module.loads(response.read().decode('utf-8'))

                        whisperx_version = data['info']['version']
                        self.progress.emit(f"\nLatest WhisperX on PyPI: {whisperx_version}\n")

                        # Check if our configured versions match PyPI
                        requires_dist = data['info'].get('requires_dist', [])
                        if requires_dist:
                            import re
                            for req in requires_dist:
                                if req and 'torch' in req.lower() and 'torchaudio' not in req.lower():
                                    match = re.match(r'torch\s*([~=><]+)\s*([\d.]+)', req, re.IGNORECASE)
                                    if match:
                                        operator, version = match.groups()
                                        if version != torch_version:
                                            self.progress.emit(f"  NOTE: PyPI requires torch{operator}{version}, but we're using {torch_version}\n")
                                elif req and 'torchaudio' in req.lower():
                                    match = re.match(r'torchaudio\s*([~=><]+)\s*([\d.]+)', req, re.IGNORECASE)
                                    if match:
                                        operator, version = match.groups()
                                        if version != torchaudio_version:
                                            self.progress.emit(f"  NOTE: PyPI requires torchaudio{operator}{version}, but we're using {torchaudio_version}\n")
                        self.progress.emit("\n")
                    except Exception as e:
                        self.progress.emit(f"(Could not verify against PyPI: {str(e)})\n\n")

                    # Check for cancellation
                    if self._cancelled:
                        crash_log("CHECKPOINT: Installation cancelled before PyTorch install\n")
                        self.finished.emit(False, "Installation cancelled by user", "")
                        return

                    # Install PyTorch with determined version
                    if self.install_cuda:
                        self.progress.emit("\nInstalling WhisperX with CUDA GPU support...\n")
                        self.progress.emit("This will download approximately 2-3GB of packages.\n")
                        self.progress.emit("Please wait, this may take 10-15 minutes...\n\n")

                        # Validate CUDA driver compatibility
                        cuda_version, cuda_short, warning = detect_cuda_version()
                        if warning:
                            self.progress.emit(f"\n{warning}\n\n")
                        else:
                            self.progress.emit(f"NVIDIA driver is compatible with CUDA {cuda_version}\n")

                        # First install PyTorch with CUDA support
                        self.progress.emit(f"Step 1/2: Installing PyTorch {torch_version} with CUDA {cuda_version} support...\n")
                        torch_cmd = [
                            venv_python,
                            "-m",
                            "pip",
                            "install",
                            "--no-cache-dir",
                            "--extra-index-url",
                            f"https://download.pytorch.org/whl/{cuda_short}",
                            f"torch=={torch_version}+{cuda_short}",
                            f"torchaudio=={torchaudio_version}+{cuda_short}",
                            "-v",
                        ] + pip_target_args  # Add --target if running as .exe

                        torch_proc = subprocess.Popen(
                            torch_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            env=get_clean_environment(),
                            startupinfo=get_subprocess_startup_info(),
                            creationflags=get_subprocess_creation_flags()
                        )
                    else:
                        self.progress.emit("\nInstalling WhisperX (CPU-only)...\n")
                        self.progress.emit("This will download approximately 500MB-1GB of packages.\n")
                        self.progress.emit("Please wait, this may take 5-10 minutes...\n\n")

                        # Install CPU-only PyTorch
                        self.progress.emit(f"Step 1/2: Installing PyTorch {torch_version} (CPU version)...\n")
                        torch_cmd = [venv_python, "-m", "pip", "install", f"torch=={torch_version}", f"torchaudio=={torchaudio_version}", "-v"] + pip_target_args
                        torch_proc = subprocess.Popen(
                            torch_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            env=get_clean_environment(),
                            startupinfo=get_subprocess_startup_info(),
                            creationflags=get_subprocess_creation_flags()
                        )

                    # Stream PyTorch installation
                    torch_queue = queue.Queue()
                    def read_torch_output():
                        for line in torch_proc.stdout:
                            torch_queue.put(('line', line))
                        torch_queue.put(('done', None))

                    torch_reader = threading.Thread(target=read_torch_output, daemon=True)
                    torch_reader.start()

                    while True:
                        if self._cancelled:
                            torch_proc.terminate()
                            torch_proc.wait(timeout=5)
                            self.finished.emit(False, "Installation cancelled by user", "")
                            return

                        try:
                            msg_type, content = torch_queue.get(timeout=15)
                            if msg_type == 'done':
                                break
                            elif msg_type == 'line':
                                self.progress.emit(content)
                        except queue.Empty:
                            if torch_proc.poll() is None:
                                self.progress.emit(f"[{time.strftime('%H:%M:%S')}] Still installing PyTorch... (this can take several minutes)\n")

                    torch_proc.wait()
                    if self._cancelled:
                        self.finished.emit(False, "Installation cancelled by user", "")
                        return
                    if torch_proc.returncode != 0:
                        self.finished.emit(False, "PyTorch installation failed. See console output for details.", "")
                        return

                    # Check for cancellation
                    if self._cancelled:
                        self.finished.emit(False, "Installation cancelled by user", "")
                        return

                    self.progress.emit("\nStep 2/2: Installing WhisperX...\n")

                    whisperx_cmd = [venv_python, "-m", "pip", "install", "whisperx", "--upgrade", "-v"] + pip_target_args
                    install_proc = subprocess.Popen(
                        whisperx_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        env=get_clean_environment(),
                        startupinfo=get_subprocess_startup_info(),
                        creationflags=get_subprocess_creation_flags()
                    )

                    output_queue = queue.Queue()
                    last_activity = [time.time()]  # Use list to allow modification in thread

                    def read_output():
                        for line in install_proc.stdout:
                            output_queue.put(('line', line))
                            last_activity[0] = time.time()
                        output_queue.put(('done', None))

                    reader_thread = threading.Thread(target=read_output, daemon=True)
                    reader_thread.start()

                    # Process output and show periodic status
                    silence_count = 0
                    while True:
                        if self._cancelled:
                            install_proc.terminate()
                            install_proc.wait(timeout=5)
                            self.finished.emit(False, "Installation cancelled by user", "")
                            return

                        try:
                            msg_type, content = output_queue.get(timeout=15)  # Check every 15 seconds
                            if msg_type == 'done':
                                break
                            elif msg_type == 'line':
                                self.progress.emit(content)
                                silence_count = 0
                        except queue.Empty:
                            # No output for 15 seconds, show activity indicator
                            silence_count += 1
                            if install_proc.poll() is None:  # Process still running
                                self.progress.emit(f"[{time.strftime('%H:%M:%S')}] Still installing packages... (this can take several minutes)\n")

                    install_proc.wait()

                    if self._cancelled:
                        self.finished.emit(False, "Installation cancelled by user", "")
                        return

                    if install_proc.returncode != 0:
                        self.finished.emit(False, "Installation failed. See console output for details.", "")
                        return

                    # Verify installation
                    self.progress.emit("\nVerifying installation...\n")
                    verify_code = "\n".join([
                        "import json",
                        "import importlib",
                        "import importlib.metadata",
                        "status = {'import_ok': False, 'version': None}",
                        "try:",
                        "    import whisperx",
                        "    status['import_ok'] = True",
                        "    status['version'] = getattr(whisperx, '__version__', None)",
                        "    if not status['version']:",
                        "        try:",
                        "            status['version'] = importlib.metadata.version('whisperx')",
                        "        except Exception:",
                        "            status['version'] = 'unknown'",
                        "except Exception as exc:",
                        "    status['error'] = repr(exc)",
                        "print(json.dumps(status))",
                    ])
                    verify_result = subprocess.run(
                        [venv_python, "-c", verify_code],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        env=get_clean_environment(),
                        startupinfo=get_subprocess_startup_info(),
                        creationflags=get_subprocess_creation_flags()
                    )

                    if verify_result.returncode != 0:
                        error_msg = f"Installation verification failed.\n\nError output:\n{verify_result.stderr}\n\nStdout:\n{verify_result.stdout}"
                        self.progress.emit(f"\n{error_msg}\n")
                        self.finished.emit(False, error_msg, "")
                        return

                    try:
                        info = json.loads(verify_result.stdout.strip() or "{}")
                    except Exception:
                        info = {}

                    if not info.get("import_ok"):
                        error_msg = "Installation verification failed: whisperx import unsuccessful."
                        if info.get("error"):
                            error_msg += f"\n\nError: {info['error']}"
                        self.progress.emit(f"\n{error_msg}\n")
                        self.finished.emit(False, error_msg, "")
                        return

                    reported_version = info.get("version") or "unknown"
                    self.progress.emit(f"WhisperX import OK (version: {reported_version})\n")
                    self.finished.emit(True, "WhisperX installed successfully to self-contained environment!", venv_python)

                except subprocess.TimeoutExpired:
                    error_msg = "Installation timed out"
                    self.progress.emit(f"\n{'=' * 70}\n")
                    self.progress.emit(f"ERROR: {error_msg}\n")
                    self.progress.emit(f"{'=' * 70}\n")
                    self.finished.emit(False, error_msg, "")
                except Exception as e:
                    import traceback
                    error_msg = f"Installation error: {str(e)}"
                    full_trace = traceback.format_exc()

                    self.progress.emit(f"\n{'=' * 70}\n")
                    self.progress.emit(f"CRITICAL ERROR - Installation Crashed\n")
                    self.progress.emit(f"{'=' * 70}\n")
                    self.progress.emit(f"Error type: {type(e).__name__}\n")
                    self.progress.emit(f"Error message: {str(e)}\n")
                    self.progress.emit(f"\nFull traceback:\n{full_trace}\n")
                    self.progress.emit(f"{'=' * 70}\n")

                    # Include environment info in crash report
                    self.progress.emit("\nEnvironment at time of crash:\n")
                    self.progress.emit(f"  Running as .exe: {getattr(sys, 'frozen', False)}\n")
                    self.progress.emit(f"  Python: {sys.version}\n")
                    self.progress.emit(f"  Platform: {sys.platform}\n")
                    self.progress.emit(f"  Executable: {sys.executable}\n")
                    self.progress.emit(f"{'=' * 70}\n")

                    self.finished.emit(False, error_msg, "")
                finally:
                    # Close log file
                    if log_file_container[0]:
                        try:
                            log_and_emit(f"\n{'=' * 70}\n")
                            log_and_emit(f"Installation process ended\n")
                            log_and_emit(f"Log file location: {log_path}\n")
                            log_and_emit(f"{'=' * 70}\n")
                            log_file_container[0].close()
                        except:
                            pass

        if immediate_log:
            immediate_log.write("\nCHECKPOINT: Creating installation thread\n")
            immediate_log.flush()

        self.install_thread = InstallThread(self, install_cuda, immediate_log, upgrade_version=upgrade_version)
        self.install_thread.progress.connect(self._append_text_to_console)
        self.install_thread.finished.connect(self.on_whisperx_install_finished)

        if immediate_log:
            immediate_log.write("CHECKPOINT: Starting installation thread\n")
            immediate_log.flush()

        self.install_thread.start()

        if immediate_log:
            immediate_log.write("CHECKPOINT: Thread start() called successfully\n")
            immediate_log.flush()

    def cancel_whisperx_installation(self):
        """Cancel the ongoing WhisperX installation"""
        if hasattr(self, 'install_thread') and self.install_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Cancel Installation",
                "Are you sure you want to cancel the WhisperX installation?\n\nAny partially downloaded files will be cleaned up.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                self._append_text_to_console("\n" + "=" * 60 + "\n")
                self._append_text_to_console("Cancelling installation...\n")
                self._append_text_to_console("=" * 60 + "\n\n")
                self.install_thread.cancel()

    def on_whisperx_install_finished(self, success: bool, message: str, venv_python: str = ""):
        """Handle WhisperX installation completion"""
        self.whisperx_install_btn.setEnabled(True)
        self.whisperx_install_btn.setVisible(True)
        self.whisperx_cancel_btn.setVisible(False)
        self.whisperx_install_btn.setText("📦 Install WhisperX")

        self._append_text_to_console("\n" + "=" * 60 + "\n")
        self._append_text_to_console(message + "\n")
        self._append_text_to_console("=" * 60 + "\n")

        if success:
            self.whisperx_deps_status.setText("✅ WhisperX installed successfully")
            self.whisperx_deps_status.setStyleSheet("color: green; font-size: 12px; font-weight: bold;")
            self.whisperx_install_btn.setVisible(False)
            self.whisperx_update_btn.setVisible(True)
            self.whisperx_reinstall_btn.setVisible(True)
            if hasattr(self, 'whisperx_cuda_status_label'):
                self.whisperx_cuda_status_label.setVisible(True)
                self.whisperx_cuda_status_label.setText("CUDA status: not checked")
                self.whisperx_cuda_status_label.setStyleSheet("color: #666; font-size: 12px;")
            if hasattr(self, 'whisperx_cuda_test_btn'):
                self.whisperx_cuda_test_btn.setVisible(True)
            # Store the venv python path
            if venv_python:
                self.whisperx_venv_python = venv_python
            QMessageBox.information(self, "Success", "WhisperX has been installed to a self-contained environment!\n\nLocation: whisperx_env/")
        else:
            self.whisperx_deps_status.setText("❌ Installation failed")
            self.whisperx_deps_status.setStyleSheet("color: red; font-size: 12px; font-weight: bold;")
            if hasattr(self, 'whisperx_cuda_status_label'):
                self.whisperx_cuda_status_label.setVisible(False)
            if hasattr(self, 'whisperx_cuda_test_btn'):
                self.whisperx_cuda_test_btn.setVisible(False)
            QMessageBox.critical(self, "Installation Failed", message)

    def reinstall_whisperx(self):
        """Reinstall WhisperX by deleting the venv and reinstalling"""
        # Show confirmation dialog
        reply = QMessageBox.question(
            self,
            "Reinstall WhisperX",
            "This will completely remove the current WhisperX installation and reinstall it with CUDA support.\n\n"
            "This will download ~2-3GB and may take 10-15 minutes.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        self._append_text_to_console("=" * 60 + "\n")
        self._append_text_to_console("Reinstalling WhisperX...\n")
        self._append_text_to_console("=" * 60 + "\n\n")

        # Disable buttons during reinstall
        self.whisperx_reinstall_btn.setEnabled(False)
        self.whisperx_reinstall_btn.setText("Reinstalling...")
        self.whisperx_deps_status.setText("Removing old installation...")
        self.whisperx_deps_status.setStyleSheet("color: #ff9800; font-size: 12px;")

        # Delete the venv folder
        import shutil
        app_dir = get_app_directory()
        venv_dir = os.path.join(app_dir, "whisperx_env")

        try:
            if os.path.exists(venv_dir):
                self._append_text_to_console(f"Deleting: {venv_dir}\n")
                shutil.rmtree(venv_dir)
                self._append_text_to_console("Old installation removed successfully.\n\n")
            else:
                self._append_text_to_console("No existing installation found.\n\n")

            # Reset status
            self.whisperx_deps_status.setText("❌ WhisperX not installed")
            self.whisperx_deps_status.setStyleSheet("color: red; font-size: 12px; font-weight: bold;")
            self.whisperx_update_btn.setVisible(False)
            self.whisperx_reinstall_btn.setVisible(False)
            self.whisperx_install_btn.setVisible(True)

            # Start fresh installation
            self._append_text_to_console("Starting fresh installation with CUDA support...\n\n")
            self.install_whisperx_simple()

        except Exception as e:
            self._append_text_to_console(f"Error during reinstall: {str(e)}\n")
            self.whisperx_deps_status.setText(f"❌ Reinstall failed: {str(e)}")
            self.whisperx_deps_status.setStyleSheet("color: red; font-size: 12px; font-weight: bold;")
            self.whisperx_reinstall_btn.setEnabled(True)
            self.whisperx_reinstall_btn.setText("🔃 Reinstall WhisperX")
            QMessageBox.critical(self, "Reinstall Failed", f"Failed to remove old installation:\n{str(e)}")

    def update_whisperx(self):
        """Check GitHub for a newer WhisperX portable release."""
        self._append_text_to_console("=" * 60 + "\n")
        self._append_text_to_console("Checking WhisperX GitHub releases...\n")
        self._append_text_to_console("=" * 60 + "\n\n")

        self.whisperx_update_btn.setEnabled(False)
        self.whisperx_update_btn.setText("Checking...")
        self.whisperx_deps_status.setText("Checking GitHub release for updates...")
        self.whisperx_deps_status.setStyleSheet("color: #ff9800; font-size: 12px;")

        def version_tuple(value: str) -> tuple[int, ...]:
            """Convert semantic version string into a comparable tuple."""
            if not value:
                return ()
            parts = []
            for part in re.split(r'[^0-9]+', value):
                if part:
                    try:
                        parts.append(int(part))
                    except ValueError:
                        break
            return tuple(parts)

        # Detect the actually-installed version from whisperx_env, not the code constant.
        # WHISPERX_VERSION is the version the *app* was built for, not what the user has installed.
        # Use the python path already found by check_whisperx_deps_simple() at startup.
        current_version = None
        _portable_python = getattr(self, 'whisperx_python', None) or getattr(self, 'whisperx_venv_python', None)
        if _portable_python and os.path.exists(_portable_python):
            try:
                _ver_result = subprocess.run(
                    [_portable_python, "-c",
                     "import importlib.metadata; print(importlib.metadata.version('whisperx'))"],
                    capture_output=True, text=True, timeout=10,
                    startupinfo=get_subprocess_startup_info(),
                    creationflags=get_subprocess_creation_flags()
                )
                if _ver_result.returncode == 0:
                    current_version = _ver_result.stdout.strip()
                    self._append_text_to_console(f"Installed version: {current_version}\n")
                else:
                    self._append_text_to_console(f"Version check failed: {_ver_result.stderr.strip()}\n")
            except Exception as _e:
                self._append_text_to_console(f"Version check error: {_e}\n")
        else:
            self._append_text_to_console("Could not locate whisperx_env python — version unknown.\n")
        if not current_version:
            current_version = WHISPERX_VERSION  # fallback to code constant

        release_url = f"https://github.com/{GITHUB_REPO}/releases"

        try:
            api_url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
            headers = {"Accept": "application/vnd.github+json"}
            response = requests.get(api_url, timeout=15, headers=headers)
            if response.status_code != 200:
                raise RuntimeError(f"GitHub API returned HTTP {response.status_code}")

            data = response.json()
            release_url = data.get("html_url") or release_url
            tag_name = data.get("tag_name") or ""
            name_field = data.get("name") or ""

            latest_version = ""
            tag_match = re.search(r'v(\d+(?:\.\d+){0,2})', tag_name)
            if tag_match:
                latest_version = tag_match.group(1)
            else:
                name_match = re.search(r'(\d+(?:\.\d+){1,})', name_field)
                if name_match:
                    latest_version = name_match.group(1)

            if not latest_version:
                latest_version = tag_name or name_field or ""

            display_name = tag_name or name_field or "Unknown release"
            self._append_text_to_console(f"Latest GitHub release: {display_name}\n")

            latest_tuple = version_tuple(latest_version)
            current_tuple = version_tuple(current_version)
            if not latest_tuple:
                raise RuntimeError("Could not determine version from latest release metadata.")

            if latest_tuple <= current_tuple:
                msg = f"WhisperX portable v{current_version} is the latest release.\n"
                self._append_text_to_console(msg)
                self.whisperx_deps_status.setText(f"✅ WhisperX is up to date (v{current_version})")
                self.whisperx_deps_status.setStyleSheet("color: green; font-size: 12px; font-weight: bold;")
                QMessageBox.information(
                    self,
                    "WhisperX Update",
                    f"No newer WhisperX portable release was found.\nCurrent version: v{current_version}"
                )
            else:
                self._append_text_to_console(
                    f"A newer WhisperX portable release is available!\n"
                    f"  Current: v{current_version}\n"
                    f"  Latest:  v{latest_version}\n\n"
                )
                self.whisperx_deps_status.setText(f"⚠ Update available: v{latest_version}")
                self.whisperx_deps_status.setStyleSheet("color: #ff9800; font-size: 12px; font-weight: bold;")

                # Ask whether they want CUDA or CPU for the upgrade
                cuda_reply = QMessageBox.question(
                    self,
                    "Upgrade WhisperX",
                    f"WhisperX v{latest_version} is available (current: v{current_version}).\n\n"
                    "Upgrade with NVIDIA CUDA GPU support?\n\n"
                    "Yes  → CUDA version (~3.5 GB download, 10-50x faster with NVIDIA GPU)\n"
                    "No   → CPU-only version (~800 MB download, works on any system)",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes
                )
                use_cuda = (cuda_reply == QMessageBox.StandardButton.Yes)

                confirm = QMessageBox.question(
                    self,
                    "Confirm Upgrade",
                    f"Upgrade WhisperX from v{current_version} to v{latest_version}?\n\n"
                    f"Mode: {'CUDA (NVIDIA GPU)' if use_cuda else 'CPU-only'}\n"
                    f"Download: ~{'3.5 GB' if use_cuda else '800 MB'}\n\n"
                    "The existing WhisperX installation will be removed first.\n"
                    "This may take 10-15 minutes.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )

                if confirm != QMessageBox.StandardButton.Yes:
                    return

                # Remove existing installation
                app_dir = get_app_directory()
                venv_dir = os.path.join(app_dir, "whisperx_env")
                try:
                    if os.path.exists(venv_dir):
                        import shutil as _shutil
                        self._append_text_to_console(f"Removing old installation: {venv_dir}\n")
                        _shutil.rmtree(venv_dir)
                        self._append_text_to_console("Old installation removed.\n\n")
                except Exception as rm_err:
                    QMessageBox.critical(self, "Upgrade Failed", f"Could not remove old installation:\n{rm_err}")
                    return

                # Reset UI so install flow works correctly
                self.whisperx_deps_status.setText(f"Upgrading to v{latest_version}...")
                self.whisperx_update_btn.setVisible(False)
                self.whisperx_reinstall_btn.setVisible(False)
                self.whisperx_install_btn.setVisible(True)

                # Kick off install with the specific new version
                self.install_whisperx_simple(upgrade_version=latest_version, use_cuda_override=use_cuda)

        except Exception as exc:
            error_msg = f"Failed to check GitHub releases: {exc}"
            self._append_text_to_console(error_msg + "\n")
            self.whisperx_deps_status.setText("❌ Update check failed")
            self.whisperx_deps_status.setStyleSheet("color: red; font-size: 12px; font-weight: bold;")
            QMessageBox.critical(self, "WhisperX Update", error_msg)
        finally:
            self.whisperx_update_btn.setEnabled(True)
            self.whisperx_update_btn.setText("🔄 Update WhisperX")
            if hasattr(self, "whisperx_reinstall_btn"):
                self.whisperx_reinstall_btn.setEnabled(True)

    def browse_whisperx_output_dir(self):
        """Browse for WhisperX output directory"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.whisperx_output_dir.text() or ""
        )
        if directory:
            self.whisperx_output_dir.setText(directory)

    def _ensure_one_whisperx_export_checked(self, changed: str):
        """Ensure at least one WhisperX export option is selected"""
        try:
            srt = self.whisperx_export_srt.isChecked()
            adobe = self.whisperx_export_adobe.isChecked()
            if not srt and not adobe:
                # Re-enable the opposite of what changed
                if changed == 'srt':
                    self.whisperx_export_adobe.blockSignals(True)
                    self.whisperx_export_adobe.setChecked(True)
                    self.whisperx_export_adobe.blockSignals(False)
                else:
                    self.whisperx_export_srt.blockSignals(True)
                    self.whisperx_export_srt.setChecked(True)
                    self.whisperx_export_srt.blockSignals(False)
                # Update formatting visibility after automatic changes
                self.update_whisperx_formatting_visibility()
        except Exception:
            pass

    def update_whisperx_formatting_visibility(self):
        """Show/hide WhisperX formatting options based on export selection"""
        try:
            # Show formatting options only when the Adobe Static Transcript export is selected
            if hasattr(self, 'whisperx_formatting_group'):
                self.whisperx_formatting_group.setVisible(self.whisperx_export_adobe.isChecked())
        except Exception:
            pass

    def update_whisperx_formatting_controls(self):
        """Enable/disable WhisperX gap threshold control based on format selection"""
        try:
            # Enable gap threshold only for Paragraph Form
            is_paragraph_form = self.whisperx_format_paragraph_form.isChecked()
            self.whisperx_paragraph_gap_spin.setEnabled(is_paragraph_form)
        except Exception:
            pass

    def is_adobe_export_enabled(self):
        """Check if Adobe export is enabled based on active tab"""
        try:
            current_tab_index = self.tab_widget.currentIndex()
            using_whisperx = (current_tab_index == self.whisperx_tab_index)

            if using_whisperx:
                return hasattr(self, 'whisperx_export_adobe') and self.whisperx_export_adobe.isChecked()
            else:
                return hasattr(self, 'export_adobe') and self.export_adobe.isChecked()
        except Exception:
            return False

    def _ensure_one_export_checked(self, changed: str):
        """Ensure at least one export option is selected. If a toggle would uncheck both, keep the other on."""
        try:
            srt = self.export_srt.isChecked()
            adobe = self.export_adobe.isChecked()
            if not srt and not adobe:
                # Re-enable the opposite of what changed
                if changed == 'srt':
                    self.export_adobe.blockSignals(True)
                    self.export_adobe.setChecked(True)
                    self.export_adobe.blockSignals(False)
                else:
                    self.export_srt.blockSignals(True)
                    self.export_srt.setChecked(True)
                    self.export_srt.blockSignals(False)
                # Update formatting visibility after automatic changes
                self.update_formatting_visibility()
        except Exception:
            pass

    def create_console_section(self):
        """Create modern progress panel with collapsible console output"""
        # Create the progress panel
        self.progress_panel = ProgressPanel(self)

        # Store reference to output_text for compatibility with existing code
        self.output_text = self.progress_panel.output_text

        # Apply custom black I-beam cursor for visibility (match Create Transcript tab)
        try:
            self._apply_custom_ibeam_cursor(self.output_text)
        except Exception:
            pass

        return self.progress_panel

    def create_button_section(self):
        """Create modern control buttons section"""
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 10, 0, 0)

        self.run_btn = QPushButton("▶ Generate Captions")
        self.run_btn.setMinimumHeight(44)
        self.run_btn.clicked.connect(self.start_transcription)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background: #28a745;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #218838;
            }
            QPushButton:disabled {
                background: #6c757d;
            }
        """)

        self.stop_btn = QPushButton("⏸ Stop")
        self.stop_btn.setMinimumHeight(44)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_transcription)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: #dc3545;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #c82333;
            }
            QPushButton:disabled {
                background: #6c757d;
            }
        """)

        layout.addWidget(self.run_btn, 3)
        layout.addWidget(self.stop_btn, 1)

        return layout

    def show_model_help_dialog(self):
        """Show a dialog explaining model trade-offs with a comparison table."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Model Trade-offs: Speed, Accuracy, Memory")
        dlg.setMinimumSize(780, 560)

        v = QVBoxLayout(dlg)
        title = QLabel("Whisper Models: Trade-offs")
        title.setStyleSheet("font-size:16px; font-weight:bold;")
        v.addWidget(title)

        browser = QTextBrowser(dlg)
        browser.setOpenExternalLinks(True)
        html = """
<div style='font-family:Segoe UI, sans-serif; color:#222; font-size:13px;'>
  <p>Here’s a breakdown / comparison of <b>distil-large-v3</b>, <b>large-v3-turbo</b>, <b>large-v3</b>, and <b>large-v2</b> in the context of Whisper models (or distilled variants thereof). What’s “better” depends on your trade-offs (speed, accuracy, resource constraints, multilingual vs English, etc.).</p>

  <h3 style='margin-top:14px;'>Key models & definitions</h3>
  <ul>
    <li><b>large-v2</b>: The “large” Whisper model as of version 2. Predecessor to v3. (<a href='https://huggingface.co/openai/whisper-large-v2'>Hugging Face</a>)</li>
    <li><b>large-v3</b>: Newer Whisper “large” with 128 Mel bins (vs 80) and more data, generally lower error vs v2. (<a href='https://huggingface.co/openai/whisper-large-v3'>Hugging Face</a>)</li>
    <li><b>large-v3-turbo</b>: Pruned/optimized large-v3 (decoder depth reduced) → faster, small accuracy drop. (<a href='https://huggingface.co/openai/whisper-large-v3-turbo'>Hugging Face</a>)</li>
    <li><b>distil-large-v3</b>: Distilled variant of large-v3 (often English-focused), retaining most accuracy with lower latency/memory. (<a href='https://huggingface.co/distil-whisper/distil-large-v3'>Hugging Face</a>)</li>
  </ul>

  <h3 style='margin-top:14px;'>Comparative trade-offs</h3>
  <table border='1' cellspacing='0' cellpadding='6' style='border-collapse:collapse; width:100%;'>
    <thead style='background:#f4f6f8;'>
      <tr>
        <th align='left'>Model</th>
        <th align='left'>Relative size / complexity</th>
        <th align='left'>Accuracy / error</th>
        <th align='left'>Speed / latency</th>
        <th align='left'>Multilingual / translation</th>
        <th align='left'>Best use case / caveats</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><b>large-v2</b></td>
        <td>Full large (baseline before v3)</td>
        <td>Strong; sometimes considered more stable in some language regimes</td>
        <td>Slower vs turbo/distilled</td>
        <td>Fully multilingual + translate</td>
        <td>Robust across many languages if you don’t mind heavier compute</td>
      </tr>
      <tr>
        <td><b>large-v3</b></td>
        <td>Similar scale to v2; architectural/data improvements</td>
        <td>Generally lower error than v2 (often 10–20% in some languages)</td>
        <td>Slower vs turbo/distilled</td>
        <td>Full multilingual + translate</td>
        <td>Choose when accuracy is top priority and compute is available</td>
      </tr>
      <tr>
        <td><b>large-v3-turbo</b></td>
        <td>Pruned/optimized decoder for speed</td>
        <td>Slightly degraded vs v3, often still close to v2</td>
        <td>Much faster, lower latency</td>
        <td>Multilingual; translation may degrade slightly</td>
        <td>Great middle ground: “almost as good”, significantly faster</td>
      </tr>
      <tr>
        <td><b>distil-large-v3</b></td>
        <td>Smaller via distillation</td>
        <td>Within ~1% WER of large-v3 for many long-form English tasks</td>
        <td>Substantially faster & lower memory</td>
        <td>Often English-focused (check multilingual needs)</td>
        <td>Constrained hardware or high throughput with strong accuracy</td>
      </tr>
    </tbody>
  </table>

  <h3 style='margin-top:14px;'>Recommendations</h3>
  <ul>
    <li><b>Max accuracy</b>: large-v3</li>
    <li><b>Speed / latency</b>: large-v3-turbo</li>
    <li><b>Constrained compute (English)</b>: distil-large-v3</li>
    <li><b>Multilingual/translate & stability</b>: test large-v3 vs large-v2 for your languages</li>
  </ul>
</div>
        """
        browser.setHtml(html)
        v.addWidget(browser)

        btns = QHBoxLayout()
        btns.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btns.addWidget(close_btn)
        v.addLayout(btns)

        dlg.exec()

    def show_vad_help_dialog(self):
        """Show a dialog explaining VAD method trade-offs with descriptions."""
        dlg = QDialog(self)
        dlg.setWindowTitle("VAD Methods: Speed, Accuracy, Compatibility")
        dlg.setMinimumSize(720, 480)

        v = QVBoxLayout(dlg)
        title = QLabel("Voice Activity Detection (VAD) Methods")
        title.setStyleSheet("font-size:16px; font-weight:bold;")
        v.addWidget(title)

        browser = QTextBrowser(dlg)
        browser.setOpenExternalLinks(True)
        html = """
<div style='font-family:Segoe UI, sans-serif; color:#222; font-size:13px;'>
  <p>Voice Activity Detection (VAD) filters out silent portions of audio before transcription, improving accuracy and reducing processing time. Choose the method that best fits your hardware and requirements.</p>

  <h3 style='margin-top:14px;'>Available VAD Methods</h3>
  <table border='1' cellspacing='0' cellpadding='8' style='border-collapse:collapse; width:100%;'>
    <thead style='background:#f4f6f8;'>
      <tr>
        <th align='left'>Method</th>
        <th align='left'>Speed</th>
        <th align='left'>Accuracy</th>
        <th align='left'>Hardware Support</th>
        <th align='left'>Best Use Case</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><b>silero_v4</b></td>
        <td>Fastest</td>
        <td>Good</td>
        <td>Widest compatibility (CPU/GPU)</td>
        <td>General use, older hardware, maximum compatibility</td>
      </tr>
      <tr>
        <td><b>pyannote_onnx_v3</b></td>
        <td>Fast</td>
        <td>Very Good</td>
        <td>CUDA GPUs, ONNX runtime</td>
        <td>Lite version of pyannote_v3 with CUDA acceleration</td>
      </tr>
      <tr>
        <td><b>pyannote_v3</b></td>
        <td>Moderate</td>
        <td>Best</td>
        <td>CUDA GPUs</td>
        <td>Maximum accuracy when CUDA is available</td>
      </tr>
    </tbody>
  </table>

  <h3 style='margin-top:14px;'>Recommendations</h3>
  <ul>
    <li><b>Maximum compatibility</b>: silero_v4 - works on all systems</li>
    <li><b>CUDA acceleration with good balance</b>: pyannote_onnx_v3</li>
    <li><b>Best accuracy with CUDA</b>: pyannote_v3</li>
  </ul>

  <h3 style='margin-top:14px;'>Hardware Notes</h3>
  <ul>
    <li><b>RTX 50 series (Blackwell)</b>: Fully supported in WhisperX. Faster-Whisper-XXL support pending PyTorch updates.</li>
    <li><b>Older CUDA GPUs</b>: pyannote_v3 or pyannote_onnx_v3 both work well</li>
    <li><b>CPU-only systems</b>: silero_v4 is your best option</li>
  </ul>
</div>
        """
        browser.setHtml(html)
        v.addWidget(browser)

        btns = QHBoxLayout()
        btns.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btns.addWidget(close_btn)
        v.addLayout(btns)

        dlg.exec()

    def show_compute_help_dialog(self):
        """Show a dialog explaining compute type trade-offs."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Compute Types: Precision, Speed, Memory")
        dlg.setMinimumSize(660, 420)

        v = QVBoxLayout(dlg)
        title = QLabel("Compute Type Selection")
        title.setStyleSheet("font-size:16px; font-weight:bold;")
        v.addWidget(title)

        browser = QTextBrowser(dlg)
        browser.setOpenExternalLinks(True)
        html = """
<div style='font-family:Segoe UI, sans-serif; color:#222; font-size:13px;'>
  <p>Compute types determine the numerical precision used during transcription. The choice affects speed, accuracy, and memory usage.</p>

  <h3 style='margin-top:14px;'>Available Compute Types</h3>
  <table border='1' cellspacing='0' cellpadding='8' style='border-collapse:collapse; width:100%;'>
    <thead style='background:#f4f6f8;'>
      <tr>
        <th align='left'>Type</th>
        <th align='left'>Precision</th>
        <th align='left'>Speed</th>
        <th align='left'>Memory Usage</th>
        <th align='left'>Device Support</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><b>float16</b></td>
        <td>Lower precision</td>
        <td>Fastest</td>
        <td>Lowest VRAM usage</td>
        <td>CUDA only (not CPU)</td>
      </tr>
      <tr>
        <td><b>float32</b></td>
        <td>High precision</td>
        <td>Slower</td>
        <td>Higher memory usage</td>
        <td>Both CPU and CUDA</td>
      </tr>
      <tr>
        <td><b>int8</b></td>
        <td>Quantized</td>
        <td>Very fast</td>
        <td>Very low memory</td>
        <td>Both CPU and CUDA</td>
      </tr>
    </tbody>
  </table>

  <h3 style='margin-top:14px;'>Recommendations</h3>
  <ul>
    <li><b>CUDA with sufficient VRAM</b>: float16 for best speed/memory balance</li>
    <li><b>CPU or low VRAM</b>: float32 for reliability</li>
    <li><b>Very limited memory</b>: int8 for maximum efficiency</li>
  </ul>

  <h3 style='margin-top:14px;'>Automatic Fallbacks</h3>
  <p>Scriptoria automatically adjusts compute types when needed:</p>
  <ul>
    <li>float16 → float32 when falling back to CPU</li>
    <li>Preserves your selection when using CUDA</li>
  </ul>
</div>
        """
        browser.setHtml(html)
        v.addWidget(browser)

        btns = QHBoxLayout()
        btns.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btns.addWidget(close_btn)
        v.addLayout(btns)

        dlg.exec()

    def show_device_help_dialog(self):
        """Show a dialog explaining device selection."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Device Selection: CPU vs GPU")
        dlg.setMinimumSize(640, 400)

        v = QVBoxLayout(dlg)
        title = QLabel("Device Selection Guide")
        title.setStyleSheet("font-size:16px; font-weight:bold;")
        v.addWidget(title)

        browser = QTextBrowser(dlg)
        browser.setOpenExternalLinks(True)
        html = """
<div style='font-family:Segoe UI, sans-serif; color:#222; font-size:13px;'>
  <p>Choose between CPU and GPU (CUDA) for transcription processing. Each has different trade-offs for speed, compatibility, and resource usage.</p>

  <h3 style='margin-top:14px;'>Device Comparison</h3>
  <table border='1' cellspacing='0' cellpadding='8' style='border-collapse:collapse; width:100%;'>
    <thead style='background:#f4f6f8;'>
      <tr>
        <th align='left'>Device</th>
        <th align='left'>Speed</th>
        <th align='left'>Compatibility</th>
        <th align='left'>Memory Requirements</th>
        <th align='left'>Best For</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><b>CPU</b></td>
        <td>Slower</td>
        <td>Universal (all systems)</td>
        <td>Uses system RAM</td>
        <td>Compatibility, no GPU, or fallback</td>
      </tr>
      <tr>
        <td><b>CUDA</b></td>
        <td>Much faster</td>
        <td>NVIDIA GPUs only</td>
        <td>Uses GPU VRAM</td>
        <td>Speed, batch processing, large models</td>
      </tr>
    </tbody>
  </table>

  <h3 style='margin-top:14px;'>Recommendations</h3>
  <ul>
    <li><b>NVIDIA GPU with 4GB+ VRAM</b>: Use CUDA for best performance</li>
    <li><b>No NVIDIA GPU or compatibility issues</b>: Use CPU</li>
    <li><b>Batch processing many files</b>: CUDA significantly faster</li>
    <li><b>Occasional single files</b>: CPU is fine</li>
  </ul>

  <h3 style='margin-top:14px;'>Automatic Fallbacks</h3>
  <p>Scriptoria includes smart fallback handling:</p>
  <ul>
    <li>Automatically retries with CPU if CUDA fails</li>
    <li>Remembers failures during your session</li>
    <li>Adjusts compute types automatically (float16 → float32 on CPU)</li>
  </ul>
</div>
        """
        browser.setHtml(html)
        v.addWidget(browser)

        btns = QHBoxLayout()
        btns.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btns.addWidget(close_btn)
        v.addLayout(btns)

        dlg.exec()

    def show_task_help_dialog(self):
        """Show a dialog explaining task selection."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Task Selection: Transcribe vs Translate")
        dlg.setMinimumSize(620, 380)

        v = QVBoxLayout(dlg)
        title = QLabel("Task Selection Guide")
        title.setStyleSheet("font-size:16px; font-weight:bold;")
        v.addWidget(title)

        browser = QTextBrowser(dlg)
        browser.setOpenExternalLinks(True)
        html = """
<div style='font-family:Segoe UI, sans-serif; color:#222; font-size:13px;'>
  <p>Choose the appropriate task based on whether you want to transcribe speech in the same language or translate it to English.</p>

  <h3 style='margin-top:14px;'>Task Options</h3>
  <table border='1' cellspacing='0' cellpadding='8' style='border-collapse:collapse; width:100%;'>
    <thead style='background:#f4f6f8;'>
      <tr>
        <th align='left'>Task</th>
        <th align='left'>Input</th>
        <th align='left'>Output</th>
        <th align='left'>Use Case</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><b>transcribe</b></td>
        <td>Audio in any language</td>
        <td>Text in the same language</td>
        <td>Converting speech to text in original language</td>
      </tr>
      <tr>
        <td><b>translate</b></td>
        <td>Audio in any language</td>
        <td>Text translated to English</td>
        <td>Converting foreign speech directly to English text</td>
      </tr>
    </tbody>
  </table>

  <h3 style='margin-top:14px;'>Examples</h3>
  <ul>
    <li><b>Spanish audio + transcribe</b> → Spanish subtitles</li>
    <li><b>Spanish audio + translate</b> → English subtitles</li>
    <li><b>English audio + transcribe</b> → English subtitles</li>
    <li><b>English audio + translate</b> → English subtitles (same as transcribe)</li>
  </ul>

  <h3 style='margin-top:14px;'>Recommendations</h3>
  <ul>
    <li><b>Same language output</b>: Use transcribe</li>
    <li><b>Need English subtitles from foreign audio</b>: Use translate</li>
    <li><b>English audio</b>: Use transcribe (translate produces same result)</li>
    <li><b>Multilingual content</b>: Consider transcribe + separate translation step</li>
  </ul>
</div>
        """
        browser.setHtml(html)
        v.addWidget(browser)

        btns = QHBoxLayout()
        btns.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btns.addWidget(close_btn)
        v.addLayout(btns)

        dlg.exec()

    def _apply_custom_ibeam_cursor(self, text_edit: QTextEdit):
        """Apply a custom black I-beam cursor to a QTextEdit viewport for consistency."""
        cursor_pixmap = QPixmap(12, 18)
        cursor_pixmap.fill(Qt.GlobalColor.transparent)

        painter = QPainter(cursor_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(Qt.GlobalColor.black)
        pen.setWidth(2)
        painter.setPen(pen)

        center_x = 6
        top_y = 2
        bottom_y = 16
        cap_width = 4

        painter.drawLine(center_x, top_y, center_x, bottom_y)
        painter.drawLine(center_x - cap_width//2, top_y, center_x + cap_width//2, top_y)
        painter.drawLine(center_x - cap_width//2, bottom_y, center_x + cap_width//2, bottom_y)
        painter.end()

        custom_cursor = QCursor(cursor_pixmap, center_x, center_x)
        text_edit.viewport().setCursor(custom_cursor)

    def on_file_dropped(self, file_path):
        """Handle single file drop/selection"""
        self.current_input_file = file_path
        file_name = truncate_filename(os.path.basename(file_path))

        # Update BOTH Generate tab and WhisperX tab labels
        self.current_file_label.setText(f"Selected: {file_name}")
        self.current_file_label.setStyleSheet("color: #28a745; font-size: 12px; font-weight: bold;")

        if hasattr(self, 'whisperx_current_file_label'):
            self.whisperx_current_file_label.setText(f"Selected: {file_name}")
            self.whisperx_current_file_label.setStyleSheet("color: #28a745; font-size: 12px; font-weight: bold;")

        # Auto-save this file selection
        self.save_settings_delayed()
        self._ensure_batch_mode_for_active_run()

    def on_files_dropped(self, file_paths):
        """Handle multiple files drop/selection for batch queue"""
        if len(file_paths) == 1:
            # Single file - check if we should add to batch or set as current
            single_file = file_paths[0]

            # If no current file and no batch files, set as current file
            if not self.current_input_file and not self.batch_files:
                self.on_file_dropped(single_file)
            else:
                # Add to batch queue (existing behavior for subsequent files)
                if single_file not in self.batch_files:
                    self.batch_files.append(single_file)
                    self.update_batch_display()
                    # Update current file display - only show "Batch Mode" if more than 1 file
                    if len(self.batch_files) > 1:
                        self.current_file_label.setText(f"Batch Mode: {len(self.batch_files)} files queued")
                        self.current_file_label.setStyleSheet("color: #007acc; font-size: 12px; font-weight: bold;")
                        if hasattr(self, 'whisperx_current_file_label'):
                            self.whisperx_current_file_label.setText(f"Batch Mode: {len(self.batch_files)} files queued")
                            self.whisperx_current_file_label.setStyleSheet("color: #007acc; font-size: 12px; font-weight: bold;")
                    else:
                        # Single file - show file name
                        file_name = truncate_filename(os.path.basename(single_file))
                        self.current_file_label.setText(f"Selected: {file_name}")
                        self.current_file_label.setStyleSheet("color: #28a745; font-size: 12px; font-weight: bold;")
                        if hasattr(self, 'whisperx_current_file_label'):
                            self.whisperx_current_file_label.setText(f"Selected: {file_name}")
                            self.whisperx_current_file_label.setStyleSheet("color: #28a745; font-size: 12px; font-weight: bold;")
                    self._ensure_batch_mode_for_active_run()
        else:
            # Multiple files - add all to batch queue
            added_count = 0
            for file_path in file_paths:
                if file_path not in self.batch_files:
                    self.batch_files.append(file_path)
                    added_count += 1

            if added_count > 0:
                self.update_batch_display()
                # Update current file display - only show "Batch Mode" if more than 1 file
                if len(self.batch_files) > 1:
                    self.current_file_label.setText(f"Batch Mode: {len(self.batch_files)} files queued")
                    self.current_file_label.setStyleSheet("color: #007acc; font-size: 12px; font-weight: bold;")
                    if hasattr(self, 'whisperx_current_file_label'):
                        self.whisperx_current_file_label.setText(f"Batch Mode: {len(self.batch_files)} files queued")
                        self.whisperx_current_file_label.setStyleSheet("color: #007acc; font-size: 12px; font-weight: bold;")
                else:
                    # Single file - show file name
                    file_name = truncate_filename(os.path.basename(self.batch_files[0]))
                    self.current_file_label.setText(f"Selected: {file_name}")
                    self.current_file_label.setStyleSheet("color: #28a745; font-size: 12px; font-weight: bold;")
                    if hasattr(self, 'whisperx_current_file_label'):
                        self.whisperx_current_file_label.setText(f"Selected: {file_name}")
                        self.whisperx_current_file_label.setStyleSheet("color: #28a745; font-size: 12px; font-weight: bold;")
                self._ensure_batch_mode_for_active_run()

    def add_batch_files(self):
        """Add multiple files to batch queue"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Audio/Video Files for Batch Processing",
            "",
            "Media Files (*.mp4 *.mp3 *.wav *.m4a *.flac *.ogg *.avi *.mov *.mkv *.webm *.aac);;All Files (*)"
        )

        if file_paths:
            # Add new files to batch (avoid duplicates)
            added = False
            for file_path in file_paths:
                if file_path not in self.batch_files:
                    self.batch_files.append(file_path)
                    added = True

            self.update_batch_display()

            # Update current file display - only show "Batch Mode" if more than 1 file
            if self.batch_files:
                if len(self.batch_files) > 1:
                    self.current_file_label.setText(f"Batch Mode: {len(self.batch_files)} files queued")
                    self.current_file_label.setStyleSheet("color: #007acc; font-size: 12px; font-weight: bold;")
                    if hasattr(self, 'whisperx_current_file_label'):
                        self.whisperx_current_file_label.setText(f"Batch Mode: {len(self.batch_files)} files queued")
                        self.whisperx_current_file_label.setStyleSheet("color: #007acc; font-size: 12px; font-weight: bold;")
                else:
                    # Single file - show file name
                    file_name = truncate_filename(os.path.basename(self.batch_files[0]))
                    self.current_file_label.setText(f"Selected: {file_name}")
                    self.current_file_label.setStyleSheet("color: #28a745; font-size: 12px; font-weight: bold;")
                    if hasattr(self, 'whisperx_current_file_label'):
                        self.whisperx_current_file_label.setText(f"Selected: {file_name}")
                        self.whisperx_current_file_label.setStyleSheet("color: #28a745; font-size: 12px; font-weight: bold;")

            if added:
                self._ensure_batch_mode_for_active_run()

    def clear_batch_files(self):
        """Clear all files from batch queue and stop batch processing if active"""
        # If batch processing is active, stop it
        if self.batch_mode_active:
            self.batch_mode_active = False
            self._batch_index = 0
            # Stop the current process if running
            if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
                self.stop_transcription()
            self.batch_progress_bar.setVisible(False)
            if hasattr(self, 'whisperx_batch_progress_bar'):
                self.whisperx_batch_progress_bar.setVisible(False)

        self.batch_files.clear()
        self.batch_results.clear()
        self.update_batch_display()

        # Reset current file display in BOTH tabs
        self.current_file_label.setText("No file selected")
        self.current_file_label.setStyleSheet("color: #666; font-size: 12px; font-style: italic;")

        if hasattr(self, 'whisperx_current_file_label'):
            self.whisperx_current_file_label.setText("No file selected")
            self.whisperx_current_file_label.setStyleSheet("color: #666; font-size: 12px; font-style: italic;")

        self.current_input_file = None

    def update_batch_display(self):
        """Update the batch display in BOTH tabs"""
        if not self.batch_files:
            self.clear_batch_btn.setEnabled(False)
            self.batch_progress_bar.setVisible(False)

            # Clear the "Batch Mode: X files queued" label
            self.current_file_label.setText("")
            self.current_file_label.setStyleSheet("")

            if hasattr(self, 'whisperx_clear_batch_btn'):
                self.whisperx_clear_batch_btn.setEnabled(False)
            if hasattr(self, 'whisperx_batch_progress_bar'):
                self.whisperx_batch_progress_bar.setVisible(False)
            if hasattr(self, 'whisperx_current_file_label'):
                self.whisperx_current_file_label.setText("")
                self.whisperx_current_file_label.setStyleSheet("")
        else:
            self.clear_batch_btn.setEnabled(True)

            if hasattr(self, 'whisperx_clear_batch_btn'):
                self.whisperx_clear_batch_btn.setEnabled(True)
            # Progress bar will be shown when processing starts

    def save_settings_delayed(self):
        """Save settings with a small delay to avoid excessive saves"""
        if hasattr(self, '_save_timer'):
            self._save_timer.stop()

        self._save_timer = QTimer()
        self._save_timer.timeout.connect(self.save_settings)
        self._save_timer.setSingleShot(True)
        self._save_timer.start(500)  # Save after 500ms of inactivity

    def on_diarization_toggled(self):
        """Diarization disabled: no-op"""
        return

    def on_engine_changed(self):
        """Handle transcription engine change"""
        if self.engine_faster_whisper.isChecked():
            # Faster-Whisper-XXL models (restricted set) using names recognized by the binary
            self.model_combo.clear()
            self.model_combo.addItems([
                'distil-large-v3.5',
                'large-v3-turbo',
                'large-v3',
                'large-v2'
            ])
            # Default to distil-large-v3.5 for performance
            self.model_combo.setCurrentText('distil-large-v3.5')
            # VAD: full list for Faster-Whisper
            try:
                self._refresh_vad_options()
            except Exception:
                pass
        else:
            # WhisperX models
            self.model_combo.clear()
            # Only allow Whisper V3 Turbo for WhisperX
            self.model_combo.addItems(['large-v3-turbo'])
            self.model_combo.setCurrentText('large-v3-turbo')
            # VAD: restrict to Silero V5 for WhisperX
            try:
                self._refresh_vad_options()
            except Exception:
                pass

        # Update dependency UI wiring only; no checks until user clicks Install
        try:
            if self.engine_whisperx.isChecked():
                self._update_whisperx_ui_state()
            else:
                # Lightweight filesystem check for Faster-Whisper (instant)
                self._check_faster_whisper_dependencies()
        except Exception:
            pass
        # Persist engine choice
        self.save_settings_delayed()
        # Keep diarization options as user-selected without engine-based disabling

    def _refresh_vad_options(self):
        """Refresh VAD combobox items based on selected engine."""
        if not hasattr(self, 'vad_method'):
            return
        # Preserve current if still valid
        current = self.vad_method.currentText() if self.vad_method.count() else None
        self.vad_method.blockSignals(True)
        self.vad_method.clear()
        if False:
            items = ['silero_v4']
        else:
            # Remove deprecated Silero V5 variants
            items = ['silero_v4', 'pyannote_v3', 'pyannote_onnx_v3']
        self.vad_method.addItems(items)
        if current in items:
            self.vad_method.setCurrentText(current)
        else:
            # Default to Silero V4
            try:
                self.vad_method.setCurrentText('silero_v4')
            except Exception:
                pass
        self.vad_method.blockSignals(False)

    def check_dependencies(self):
        """Check if dependencies are available (Faster-Whisper-XXL only)."""
        return self._check_faster_whisper_dependencies()

    # Dependency checks are intentionally not auto-triggered to keep startup fast.
    def _get_runtime_dir(self):
        return os.path.join(get_app_directory(), 'runtime', 'python312')

    def _get_runtime_python_exe(self):
        runtime_dir = self._get_runtime_dir()
        exe = os.path.join(runtime_dir, 'python.exe')
        if sys.platform == 'win32' and os.path.exists(exe):
            return exe
        alt = os.path.join(runtime_dir, 'pythonw.exe')
        return exe if os.path.exists(exe) else (alt if os.path.exists(alt) else None)

    def _whisperx_installed_lightcheck(self):
        """Lightweight check: consider WhisperX installed if our private runtime python exists."""
        exe = self._get_runtime_python_exe()
        return bool(exe and os.path.exists(exe))

    def _update_whisperx_ui_state(self):
        """Update Dependencies UI for WhisperX without heavy imports."""
        installed = self._whisperx_installed_lightcheck()
        try:
            self.download_deps_btn.clicked.disconnect()
        except Exception:
            pass
        if installed:
            # Cache interpreter path
            self.whisperx_python = self._get_runtime_python_exe()
            if isinstance(self.settings, dict):
                self.settings['whisperx_python'] = self.whisperx_python
                self.save_settings()
            # Show reinstall option
            self.deps_status_label.setText("✅ WhisperX runtime detected. Ready to use.")
            self.deps_status_label.setStyleSheet("color: green;")
            self.download_deps_btn.setText("🔁 Reinstall WhisperX")
            self.download_deps_btn.clicked.connect(self.reinstall_whisperx)
            self.download_deps_btn.setVisible(True)
            # Show CUDA controls; do not auto-check on UI load
            if hasattr(self, 'cuda_status_label') and hasattr(self, 'enable_cuda_btn'):
                self.cuda_status_label.setVisible(True)
                self.enable_cuda_btn.setVisible(True)
                if hasattr(self, 'refresh_cuda_btn'):
                    self.refresh_cuda_btn.setVisible(True)
                if hasattr(self, 'cuda_diag_btn'):
                    self.cuda_diag_btn.setVisible(True)
                # Show last-known status if available
                last_status = None
                last_name = None
                try:
                    if isinstance(self.settings, dict):
                        last_status = self.settings.get('cuda_status')
                        last_name = self.settings.get('cuda_gpu_name')
                except Exception:
                    pass
                if last_status is True:
                    self.cuda_status_label.setText(f"CUDA: enabled ({last_name or 'GPU'})")
                    self.cuda_status_label.setStyleSheet("color: green; font-size: 12px; padding: 4px 8px;")
                elif last_status is False:
                    self.cuda_status_label.setText("CUDA: not available in WhisperX runtime (using CPU)")
                    self.cuda_status_label.setStyleSheet("color: #d48b00; font-size: 12px; padding: 4px 8px;")
                else:
                    self.cuda_status_label.setText("CUDA: status unknown — click Refresh")
                    self.cuda_status_label.setStyleSheet("color: #666; font-size: 12px; padding: 4px 8px;")
        else:
            self.deps_status_label.setText("WhisperX selected. Click Install to set up.")
            self.deps_status_label.setStyleSheet("color: #666;")
            self.download_deps_btn.setText("📦 Install WhisperX (pip)")
            self.download_deps_btn.clicked.connect(self.install_whisperx_dependencies)
            self.download_deps_btn.setVisible(True)
            if hasattr(self, 'cuda_status_label') and hasattr(self, 'enable_cuda_btn'):
                self.cuda_status_label.setVisible(False)
                self.enable_cuda_btn.setVisible(False)
            if hasattr(self, 'refresh_cuda_btn'):
                self.refresh_cuda_btn.setVisible(False)
            if hasattr(self, 'cuda_diag_btn'):
                self.cuda_diag_btn.setVisible(False)

    def diagnose_cuda_runtime(self):
        """Run a detailed CUDA diagnostic inside the WhisperX runtime."""
        try:
            candidates = []
            if hasattr(self, "whisperx_venv_python") and self.whisperx_venv_python:
                candidates.append(self.whisperx_venv_python)
            if hasattr(self, "whisperx_python") and self.whisperx_python:
                candidates.append(self.whisperx_python)
            rt_private = self._get_runtime_python_exe()
            if rt_private:
                candidates.append(rt_private)

            rt = next((p for p in candidates if p and os.path.exists(p)), None)
            if not rt:
                QMessageBox.warning(self, "WhisperX Runtime Missing", "Please install WhisperX before running diagnostics.")
                return

            self._append_text_to_console(">>> Running WhisperX CUDA diagnostic...\n")
            self._append_text_to_console(f"Runtime: {rt}\n")

            diag_code = "\n".join([
                "import json",
                "import os",
                "import subprocess",
                "",
                "info = {}",
                "try:",
                "    import torch",
                "except Exception as exc:",
                "    info['torch_import_error'] = repr(exc)",
                "else:",
                "    info['torch_version'] = getattr(torch, '__version__', 'unknown')",
                "    info['cuda_build'] = getattr(getattr(torch, 'version', object()), 'cuda', None)",
                "    try:",
                "        avail = torch.cuda.is_available()",
                "    except Exception as exc:",
                "        info['cuda_available_error'] = repr(exc)",
                "        avail = False",
                "    info['cuda_available'] = bool(avail)",
                "    try:",
                "        count = torch.cuda.device_count()",
                "    except Exception as exc:",
                "        info['cuda_device_count_error'] = repr(exc)",
                "        count = 0",
                "    info['cuda_device_count'] = int(count)",
                "    if avail and count:",
                "        try:",
                "            info['cuda_device_name'] = torch.cuda.get_device_name(0)",
                "        except Exception as exc:",
                "            info['cuda_device_name_error'] = repr(exc)",
                "        try:",
                "            props = torch.cuda.get_device_properties(0)",
                "            info['cuda_total_memory_gb'] = round(props.total_memory / (1024**3), 2)",
                "        except Exception as exc:",
                "            info['cuda_properties_error'] = repr(exc)",
                "    else:",
                "        reasons = []",
                "        if getattr(getattr(torch, 'version', object()), 'cuda', None) is None:",
                "            reasons.append('PyTorch build is CPU-only (no CUDA runtime)')",
                "        env_val = os.environ.get('CUDA_VISIBLE_DEVICES')",
                "        if env_val in ('', '-1'):",
                "            reasons.append(f'CUDA_VISIBLE_DEVICES disables GPUs (value={env_val!r})')",
                "        try:",
                "            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)",
                "            if result.returncode != 0:",
                "                reasons.append(f'nvidia-smi failed (exit code {result.returncode})')",
                "        except FileNotFoundError:",
                "            reasons.append('nvidia-smi command not found (driver not installed?)')",
                "        except Exception as exc:",
                "            reasons.append(f'nvidia-smi check error: {exc}')",
                "        info['diagnostic_notes'] = reasons",
                "",
                "print(json.dumps(info))",
            ])

            try:
                result = subprocess.run(
                    [rt, "-c", diag_code],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    env=get_clean_environment(),
                    startupinfo=get_subprocess_startup_info(),
                    creationflags=get_subprocess_creation_flags()
                )
            except subprocess.TimeoutExpired:
                self._append_text_to_console("CUDA diagnostic timed out.\n")
                QMessageBox.warning(self, "CUDA Diagnostic", "CUDA diagnostic timed out after 30 seconds.")
                return

            stdout = (result.stdout or "").strip()
            stderr = (result.stderr or "").strip()

            if not stdout:
                msg = "CUDA diagnostic returned no output."
                if stderr:
                    msg += f"\n\nstderr:\n{stderr}"
                self._append_text_to_console(msg + "\n")
                QMessageBox.warning(self, "CUDA Diagnostic", msg)
                return

            try:
                info = json.loads(stdout)
            except json.JSONDecodeError:
                msg = "CUDA diagnostic produced unexpected output."
                if stdout:
                    msg += f"\n\nstdout:\n{stdout}"
                if stderr:
                    msg += f"\n\nstderr:\n{stderr}"
                self._append_text_to_console(msg + "\n")
                QMessageBox.warning(self, "CUDA Diagnostic", msg)
                return

            lines = []
            torch_error = info.get("torch_import_error")
            if torch_error:
                lines.append("Torch import failed inside WhisperX runtime:")
                lines.append(f"  {torch_error}")
                for lbl in [getattr(self, "whisperx_cuda_status_label", None), getattr(self, "cuda_status_label", None)]:
                    if lbl:
                        lbl.setText("CUDA: torch import failed in WhisperX runtime")
                        lbl.setStyleSheet("color: #d48b00; font-size: 12px; padding: 4px 8px;")
                try:
                    if hasattr(self, "enable_cuda_btn") and self.enable_cuda_btn:
                        self.enable_cuda_btn.setText("⚡ Enable GPU (CUDA) in WhisperX runtime")
                except Exception:
                    pass
                try:
                    if isinstance(self.settings, dict):
                        self.settings["cuda_status"] = False
                        self.settings["cuda_gpu_name"] = ""
                        self.save_settings()
                except Exception:
                    pass
            else:
                torch_version = info.get("torch_version", "unknown")
                cuda_build = info.get("cuda_build")
                lines.append(f"torch version: {torch_version}")
                lines.append(f"torch compiled with CUDA: {cuda_build if cuda_build else 'No'}")

                avail = info.get("cuda_available", False)
                if avail:
                    device_name = info.get("cuda_device_name") or "GPU"
                    mem_gb = info.get("cuda_total_memory_gb")
                    lines.append("CUDA available: Yes")
                    lines.append(f"device: {device_name}")
                    if mem_gb:
                        lines.append(f"memory: {mem_gb} GB")
                    for lbl in [getattr(self, "whisperx_cuda_status_label", None), getattr(self, "cuda_status_label", None)]:
                        if lbl:
                            lbl.setText(f"CUDA: enabled ({device_name})")
                            lbl.setStyleSheet("color: green; font-size: 12px; padding: 4px 8px;")
                    try:
                        if hasattr(self, "enable_cuda_btn") and self.enable_cuda_btn:
                            self.enable_cuda_btn.setText("Reinstall GPU (CUDA) wheels")
                    except Exception:
                        pass
                    try:
                        if isinstance(self.settings, dict):
                            self.settings["cuda_status"] = True
                            self.settings["cuda_gpu_name"] = device_name
                            self.save_settings()
                    except Exception:
                        pass
                else:
                    lines.append("CUDA available: No")
                    notes = info.get("diagnostic_notes") or []
                    if notes:
                        lines.append("Possible causes:")
                        for note in notes:
                            lines.append(f"  - {note}")
                    cuda_error = info.get("cuda_available_error")
                    if cuda_error:
                        lines.append(f"cuda.is_available raised: {cuda_error}")
                    device_err = info.get("cuda_device_count_error")
                    if device_err:
                        lines.append(f"device_count error: {device_err}")
                    for lbl in [getattr(self, "whisperx_cuda_status_label", None), getattr(self, "cuda_status_label", None)]:
                        if lbl:
                            lbl.setText("CUDA: not available in WhisperX runtime (using CPU)")
                            lbl.setStyleSheet("color: #d48b00; font-size: 12px; padding: 4px 8px;")
                    try:
                        if hasattr(self, "enable_cuda_btn") and self.enable_cuda_btn:
                            self.enable_cuda_btn.setText("⚡ Enable GPU (CUDA) in WhisperX runtime")
                    except Exception:
                        pass
                    try:
                        if isinstance(self.settings, dict):
                            self.settings["cuda_status"] = False
                            self.settings["cuda_gpu_name"] = ""
                            self.save_settings()
                    except Exception:
                        pass

            message = "\n".join(lines)
            self._append_text_to_console(message + "\n")
            if stderr:
                self._append_text_to_console(f"(stderr)\n{stderr}\n")

            QMessageBox.information(self, "CUDA Diagnostic", message)
        except Exception as exc:
            err_msg = f"CUDA diagnostic failed: {exc}"
            self._append_text_to_console(err_msg + "\n")
            QMessageBox.critical(self, "CUDA Diagnostic", err_msg)

    def start_cuda_status_check(self):
        """Start async CUDA availability check to avoid blocking UI."""
        try:
            rt = self._get_runtime_python_exe()
            if not (rt and os.path.exists(rt)):
                self.cuda_status_label.setText("")
                return
            # Show in-progress text
            self.cuda_status_label.setText("Checking CUDA status...")
            self.cuda_status_label.setStyleSheet("color: #666; font-size: 12px; padding: 4px 8px;")
            # Prepare QProcess
            self._cuda_check_proc = QProcess(self)
            code = (
                "import json,torch;\n"
                "avail=torch.cuda.is_available();\n"
                "name=torch.cuda.get_device_name(0) if avail else '';\n"
                "print(json.dumps({'avail':avail,'name':name}))\n"
            )
            self._cuda_check_proc.setProgram(rt)
            self._cuda_check_proc.setArguments(['-c', code])
            self._cuda_check_proc.finished.connect(self.on_cuda_check_finished)
            self._cuda_check_proc.start()
            # Timeout safeguard
            self._cuda_check_timeout = QTimer(self)
            self._cuda_check_timeout.setInterval(15000)
            self._cuda_check_timeout.setSingleShot(True)
            self._cuda_check_timeout.timeout.connect(self.on_cuda_check_timeout)
            self._cuda_check_timeout.start()
        except Exception:
            self.cuda_status_label.setText("CUDA: status check failed (setup)")
            self.cuda_status_label.setStyleSheet("color: #666; font-size: 12px; padding: 4px 8px;")

    def on_cuda_check_timeout(self):
        try:
            if hasattr(self, '_cuda_check_proc') and self._cuda_check_proc and self._cuda_check_proc.state() == QProcess.ProcessState.Running:
                self._cuda_check_proc.kill()
        except Exception:
            pass
        self.cuda_status_label.setText("CUDA: status check timed out")
        self.cuda_status_label.setStyleSheet("color: #666; font-size: 12px; padding: 4px 8px;")

    def on_cuda_check_finished(self, code, status):
        try:
            if hasattr(self, '_cuda_check_timeout') and self._cuda_check_timeout:
                self._cuda_check_timeout.stop()
        except Exception:
            pass
        try:
            out = bytes(self._cuda_check_proc.readAllStandardOutput()).decode('utf-8', errors='replace').strip()
            import json as _json
            info = _json.loads(out or '{}')
            if info.get('avail'):
                self.cuda_status_label.setText(f"CUDA: enabled ({info.get('name','GPU')})")
                self.cuda_status_label.setStyleSheet("color: green; font-size: 12px; padding: 4px 8px;")
                self.enable_cuda_btn.setText("Reinstall GPU (CUDA) wheels")
                # Cache
                try:
                    if isinstance(self.settings, dict):
                        self.settings['cuda_status'] = True
                        self.settings['cuda_gpu_name'] = info.get('name')
                        self.save_settings()
                except Exception:
                    pass
            else:
                self.cuda_status_label.setText("CUDA: not available in WhisperX runtime (using CPU)")
                self.cuda_status_label.setStyleSheet("color: #d48b00; font-size: 12px; padding: 4px 8px;")
                self.enable_cuda_btn.setText("⚡ Enable GPU (CUDA) in WhisperX runtime")
                try:
                    if isinstance(self.settings, dict):
                        self.settings['cuda_status'] = False
                        self.settings['cuda_gpu_name'] = ''
                        self.save_settings()
                except Exception:
                    pass
        except Exception:
            self.cuda_status_label.setText("CUDA: status check failed")
            self.cuda_status_label.setStyleSheet("color: #666; font-size: 12px; padding: 4px 8px;")

    def enable_cuda_for_whisperx(self):
        """Install CUDA-enabled PyTorch into the private runtime."""
        rt = self._get_runtime_python_exe()
        if not (rt and os.path.exists(rt)):
            QMessageBox.warning(self, "WhisperX Runtime Missing", "Please install WhisperX first, then enable CUDA.")
            return

        # Build CUDA install attempts (uninstall existing CPU wheels first)
        self._cuda_queue = []
        self._cuda_pins_added = False
        self._append_text_to_console("Enabling CUDA in WhisperX runtime...\n")
        self._cuda_queue.append((
            "Remove existing torch/torchaudio",
            [rt, '-m', 'pip', 'uninstall', '-y', 'torchaudio', 'torch']
        ))
        # Ensure pyannote stack is not present to avoid import-time crashes
        self._cuda_queue.append((
            "Remove pyannote packages",
            [rt, '-m', 'pip', 'uninstall', '-y', 'pyannote-audio', 'pyannote-core', 'pyannote-database', 'pyannote-metrics', 'pyannote-pipeline', 'speechbrain']
        ))
        # Install PyTorch with CUDA support (using configured WhisperX requirements)
        self._cuda_queue.append((
            f"Install PyTorch CUDA ({WHISPERX_CUDA_SHORT}, pinned)",
            [rt, '-m', 'pip', 'install', '--no-cache-dir', '--index-url',
             f'https://download.pytorch.org/whl/{WHISPERX_CUDA_SHORT}',
             f'torch=={WHISPERX_PYTORCH_VERSION}+{WHISPERX_CUDA_SHORT}',
             f'torchaudio=={WHISPERX_TORCHAUDIO_VERSION}+{WHISPERX_CUDA_SHORT}']
        ))
        self._cuda_queue.append((
            f"Install PyTorch CUDA ({WHISPERX_CUDA_SHORT}, unpinned)",
            [rt, '-m', 'pip', 'install', '--no-cache-dir', '--index-url',
             f'https://download.pytorch.org/whl/{WHISPERX_CUDA_SHORT}',
             'torch', 'torchaudio']
        ))
        self._cuda_verify_cmd = [rt, '-c', 'import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)']

        # Use QProcess chain
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self._on_cuda_step_finished)

        # Heartbeat
        try:
            if hasattr(self, '_install_heartbeat_timer') and self._install_heartbeat_timer is not None:
                self._install_heartbeat_timer.stop()
        except Exception:
            pass
        self._install_heartbeat_timer = QTimer(self)
        self._install_heartbeat_timer.setInterval(10000)
        self._install_heartbeat_timer.timeout.connect(lambda: self._append_text_to_console("(still enabling CUDA...)\n"))
        self._install_heartbeat_timer.start()

        self._start_next_cuda_step()

    def _start_next_cuda_step(self):
        try:
            if self._cuda_queue:
                desc, cmd = self._cuda_queue.pop(0)
                self._current_cuda_desc = desc
                self._append_text_to_console(f"\n>>> {desc}: {' '.join(cmd)}\n")
                self.process.start(cmd[0], cmd[1:])
            else:
                # Verify CUDA availability
                self._append_text_to_console("\n>>> Verifying CUDA availability...\n")
                self.process.start(self._cuda_verify_cmd[0], self._cuda_verify_cmd[1:])
        except Exception as e:
            self._append_text_to_console(f"CUDA enable error: {e}\n")

    def _on_cuda_step_finished(self, code, status):
        try:
            if hasattr(self, '_cuda_queue') and self._cuda_queue:
                if code == 0:
                    self._append_text_to_console("Step completed successfully.\n")
                    # If we just installed a CUDA build, pin WhisperX + pyannote to cuDNN9-compatible set
                    try:
                        if (isinstance(self._current_cuda_desc, str)
                                and self._current_cuda_desc.startswith("Install PyTorch CUDA")
                                and not getattr(self, '_cuda_pins_added', False)):
                            rt = self._get_runtime_python_exe()
                            # Clear remaining CUDA variant attempts
                            self._cuda_queue = []
                            # Add pinned installs (WhisperX only; no pyannote stack)
                            self._cuda_queue.append((
                                "Install WhisperX (pinned)",
                                [rt, '-m', 'pip', 'install', '--no-cache-dir', f'whisperx=={WHISPERX_VERSION}']
                            ))
                            self._cuda_pins_added = True
                    except Exception:
                        pass
                else:
                    self._append_text_to_console(f"Step failed with code {code}, trying next CUDA option...\n")
                self._start_next_cuda_step()
            else:
                # Verification completed
                try:
                    if hasattr(self, '_install_heartbeat_timer') and self._install_heartbeat_timer is not None:
                        self._install_heartbeat_timer.stop()
                except Exception:
                    pass
                if code == 0:
                    self._append_text_to_console("CUDA enabled successfully in WhisperX runtime.\n")
                    # If verification OK, enforce silero_v4 VAD by default
                    try:
                        if hasattr(self, 'vad_method'):
                            self.vad_method.setCurrentText('silero_v4')
                            self.save_settings()
                    except Exception:
                        pass
                else:
                    self._append_text_to_console("CUDA still not available. Check NVIDIA drivers and retry.\n")
                self.process = None
                # Refresh CUDA status asynchronously if available
                try:
                    if hasattr(self, 'start_cuda_status_check'):
                        self.start_cuda_status_check()
                except Exception:
                    pass
        except Exception as e:
            self._append_text_to_console(f"CUDA enable error: {e}\n")

    def _check_faster_whisper_dependencies(self):
        """Check Faster-Whisper-XXL dependencies"""
        # Determine executable name based on platform
        if sys.platform == "win32":
            self.executable_name = "faster-whisper-xxl.exe"
            self.files_to_check = [self.executable_name, "ffmpeg.exe"]
        else:
            self.executable_name = "faster-whisper-xxl"
            self.files_to_check = [self.executable_name, "ffmpeg"]

        # Check local bin directory
        local_executable_path = os.path.join(self.bin_dir, self.executable_name)
        all_files_in_bin = all(os.path.exists(os.path.join(self.bin_dir, f)) for f in self.files_to_check)

        if all_files_in_bin:
            self.executable_path = os.path.abspath(local_executable_path)
            self.deps_status_label.setText(f"✓ Dependencies ready: {self.executable_name}")
            self.deps_status_label.setStyleSheet("color: green;")
            try:
                self.download_deps_btn.setVisible(False)
            except Exception:
                pass
            return True

        # Check system PATH
        path_in_system = shutil.which(self.executable_name)
        if path_in_system:
            self.executable_path = path_in_system
            self.deps_status_label.setText(f"✓ Dependencies ready: {self.executable_name} (system)")
            self.deps_status_label.setStyleSheet("color: green;")
            try:
                self.download_deps_btn.setVisible(False)
            except Exception:
                pass
            return True

        # Dependencies not found
        self.deps_status_label.setText(f"⚠ Dependencies missing: {self.executable_name}")
        self.deps_status_label.setStyleSheet("color: orange;")
        # Rewire button for Faster-Whisper bundle download
        try:
            try:
                self.download_deps_btn.clicked.disconnect()
            except Exception:
                pass
            self.download_deps_btn.setText("📥 Download Dependencies (1.4GB)")
            self.download_deps_btn.clicked.connect(self.download_dependencies)
            self.download_deps_btn.setVisible(True)
        except Exception:
            pass
        return False

    def _check_whisperx_dependencies(self):
        """Check WhisperX (Python package) dependencies"""
        try:
            # Try to import whisperx
            import importlib

            # Prefer a Python 3.12 interpreter if available for WhisperX compatibility
            self._ensure_whisperx_python_interpreter()

            # Verify whisperx import on that interpreter
            if self.whisperx_python:
                import subprocess as _sub
                check = _sub.run([self.whisperx_python, '-c', 'import whisperx; print(1)'], capture_output=True, text=True, timeout=3)
                if check.returncode != 0:
                    raise ImportError(check.stderr.strip() or 'whisperx not installed in selected interpreter')
            else:
                importlib.import_module('whisperx')

            # Also ensure our CLI script exists
            cli_path = os.path.join(get_app_directory(), 'Data', 'whisperx_cli.py')
            if not os.path.exists(cli_path):
                self.deps_status_label.setText("⚠ WhisperX installed, but CLI not found")
                self.deps_status_label.setStyleSheet("color: orange;")
                self.download_deps_btn.setVisible(False)
                self.executable_path = None
                return False

            # Set executable_path to chosen Python interpreter for QProcess
            self.executable_path = self.whisperx_python or sys.executable
            self.deps_status_label.setText("✓ WhisperX ready (Python package detected)")
            self.deps_status_label.setStyleSheet("color: green;")

            # Hide or repurpose the button
            self.download_deps_btn.setVisible(False)
            return True
        except Exception:
            # Not installed — offer to install via pip
            self.executable_path = None
            self.deps_status_label.setText("❌ WhisperX not installed. Click to install via pip.")
            self.deps_status_label.setStyleSheet("color: red;")

            # Rewire the button for pip install
            try:
                # Disconnect previous connections, then connect to installer
                try:
                    self.download_deps_btn.clicked.disconnect()
                except Exception:
                    pass
                self.download_deps_btn.setText("📦 Install WhisperX (pip)")
                self.download_deps_btn.clicked.connect(self.install_whisperx_dependencies)
                self.download_deps_btn.setVisible(True)
            except Exception:
                pass
            return False

    def install_whisperx_dependencies(self):
        """Install whisperx via pip using the current Python interpreter"""
        try:
            # Ensure we have a compatible interpreter (prefer Python 3.12)
            self._ensure_whisperx_python_interpreter()
            py_exe = self.whisperx_python or sys.executable

            # If on Windows and selected interpreter is 3.13+, auto-download a private 3.12 runtime
            import subprocess as _sub
            py_vers = ''
            try:
                v = _sub.run([py_exe, '-c', 'import sys; print(sys.version_info[:2])'], capture_output=True, text=True, timeout=2)
                py_vers = v.stdout.strip() if v.returncode == 0 else ''
            except Exception:
                pass

            is_py313_or_newer = ('(3, 13)' in py_vers) or ('(3, 14)' in py_vers) or ('(3, 15)' in py_vers)

            # If no valid private interpreter or current is 3.13+, download private 3.12
            missing_private = (self.whisperx_python is None) or (not os.path.exists(self.whisperx_python))
            if sys.platform == 'win32' and (missing_private or is_py313_or_newer):
                # Download and prepare private Python 3.12 runtime, then continue
                runtime_dir = os.path.join(get_app_directory(), 'runtime', 'python312')
                self._append_text_to_console(f"Preparing private Python 3.12 runtime at: {runtime_dir}\n")

                self.progress_dialog = DownloadProgressDialog(self)
                self.runtime_downloader = PythonRuntimeDownloader(runtime_dir)
                self.runtime_downloader.progress.connect(self.progress_dialog.update_progress)

                def _runtime_done(success, msg, exe_path):
                    try:
                        self.progress_dialog.accept()
                    except Exception:
                        pass
                    if not success:
                        QMessageBox.critical(self, "Runtime Setup Failed", msg)
                        return
                    # Use the new interpreter
                    self.whisperx_python = exe_path
                    if isinstance(self.settings, dict):
                        self.settings['whisperx_python'] = exe_path
                        self.save_settings()
                    self._begin_whisperx_install(exe_path)

                self.runtime_downloader.finished.connect(_runtime_done)
                self.runtime_downloader.start()
                self.progress_dialog.exec()
            else:
                self._begin_whisperx_install(py_exe)
        except Exception as e:
            QMessageBox.critical(self, "Installation Error", f"Failed to start pip install: {str(e)}")

    def _begin_whisperx_install(self, py_exe: str):
        """Build and start the whisperx install sequence using the given interpreter."""
        self._append_text_to_console(f"Installing whisperx using: {py_exe}\n")

        # If this is our embedded runtime, ensure path isolation is relaxed for sdists
        try:
            runtime_dir = os.path.dirname(py_exe)
            if os.path.exists(runtime_dir):
                pth_files = [f for f in os.listdir(runtime_dir) if f.endswith('._pth')]
                for p in pth_files:
                    try:
                        os.remove(os.path.join(runtime_dir, p))
                    except Exception:
                        pass
                os.makedirs(os.path.join(runtime_dir, 'Lib', 'site-packages'), exist_ok=True)
                os.makedirs(os.path.join(runtime_dir, 'Scripts'), exist_ok=True)
        except Exception:
            pass

        # Detect target interpreter version
        import subprocess as _sub
        py_vers = ''
        try:
            v = _sub.run([py_exe, '-c', 'import sys; print(sys.version_info[:2])'], capture_output=True, text=True, timeout=2)
            py_vers = v.stdout.strip() if v.returncode == 0 else ''
        except Exception:
            pass

        is_py313_or_newer = ('(3, 13)' in py_vers) or ('(3, 14)' in py_vers) or ('(3, 15)' in py_vers)

        # Build install steps
        self._install_queue = []
        # Always ensure packaging toolchain first for embedded runtime
        self._install_queue.append((
            "Upgrade packaging tools (pip, setuptools, wheel)",
            [py_exe, '-m', 'pip', 'install', '--upgrade', '--no-cache-dir', 'pip', 'setuptools', 'wheel']
        ))
        # Preinstall problematic pure-Python sdists to avoid legacy setup.py metadata errors
        self._install_queue.append((
            "Preinstall helper deps (docopt, antlr runtime)",
            [py_exe, '-m', 'pip', 'install', 'docopt==0.6.2', 'antlr4-python3-runtime==4.9.3']
        ))
        if is_py313_or_newer:
            # Best-effort fallback for Python 3.13: install whisperx without deps then required pins
            self._append_text_to_console("Detected Python 3.13+. Using compatibility install path...\n")
            self._install_queue.append((
                "Install whisperx (no-deps)",
                [py_exe, '-m', 'pip', 'install', 'whisperx==3.2.0', '--no-deps']
            ))
            self._install_queue.append((
                "Install required deps (ctranslate2, faster-whisper)",
                [py_exe, '-m', 'pip', 'install', 'ctranslate2==4.6.0', 'faster-whisper==1.0.0']
            ))
        else:
            # Normal install path
            self._install_queue.append((
                "Install whisperx (standard)",
                [py_exe, '-m', 'pip', 'install', 'whisperx', '--upgrade', '--prefer-binary', '--no-cache-dir']
            ))

        # Final verification step
        self._verify_cmd = [py_exe, '-c', 'import whisperx; print("ok")']

        # Use QProcess to keep UI responsive
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self._on_install_step_finished)

        # Heartbeat so users know it's still working
        try:
            if hasattr(self, '_install_heartbeat_timer') and self._install_heartbeat_timer is not None:
                self._install_heartbeat_timer.stop()
        except Exception:
            pass
        from PyQt6.QtCore import QTimer
        self._install_heartbeat_timer = QTimer(self)
        self._install_heartbeat_timer.setInterval(10000)
        self._install_heartbeat_timer.timeout.connect(lambda: self._append_text_to_console("(still installing...)\n"))
        self._install_heartbeat_timer.start()

        # Start first step
        self._start_next_install_step()

    def _start_next_install_step(self):
        """Run the next queued install step or verify installation."""
        try:
            if self._install_queue:
                desc, cmd = self._install_queue.pop(0)
                self._append_text_to_console(f"\n>>> {desc}: {' '.join(cmd)}\n")
                self.process.start(cmd[0], cmd[1:])
            else:
                # Run verification
                self._append_text_to_console("\n>>> Verifying whisperx import...\n")
                self.process.start(self._verify_cmd[0], self._verify_cmd[1:])
        except Exception as e:
            self._append_text_to_console(f"Installer error: {e}\n")

    def _on_install_step_finished(self, code, status):
        """Handle completion of an install step or verification."""
        try:
            if self._install_queue:
                # We were in an install step
                if code == 0:
                    self._append_text_to_console("Step completed successfully.\n")
                    self._start_next_install_step()
                else:
                    self._append_text_to_console(f"Step failed with code {code}. Aborting.\n")
                    self.process = None
                    try:
                        if hasattr(self, '_install_heartbeat_timer') and self._install_heartbeat_timer is not None:
                            self._install_heartbeat_timer.stop()
                    except Exception:
                        pass
            else:
                # Verification step completed
                if code == 0:
                    self._append_text_to_console("Verification OK. whisperx is ready.\n")
                    # Re-check dependencies to update UI state and interpreter
                    self.check_dependencies()
                else:
                    self._append_text_to_console("Verification failed. whisperx import still failing.\n")
                self.process = None
                try:
                    if hasattr(self, '_install_heartbeat_timer') and self._install_heartbeat_timer is not None:
                        self._install_heartbeat_timer.stop()
                except Exception:
                    pass
        except Exception as e:
            self._append_text_to_console(f"Installer error: {e}\n")

    def _ensure_whisperx_python_interpreter(self):
        """Resolve and cache a Python 3.12 interpreter for WhisperX if available."""
        if self.whisperx_python:
            if os.path.exists(self.whisperx_python):
                return
            # Clear stale path if deleted
            self.whisperx_python = None

        # If a saved setting exists, trust it if still present
        try:
            if hasattr(self, 'settings') and isinstance(self.settings, dict):
                saved = self.settings.get('whisperx_python')
                if saved and os.path.exists(saved):
                    self.whisperx_python = saved
                    return
        except Exception:
            pass

        # Try to discover Python 3.12
        try:
            import shutil as _shutil
            import subprocess as _sub
            candidates = []
            if sys.platform == 'win32':
                py_launcher = _shutil.which('py')
                if py_launcher:
                    # Query the concrete path of Python 3.12
                    res = _sub.run([py_launcher, '-3.12', '-c', 'import sys;print(sys.executable)'], capture_output=True, text=True, timeout=3)
                    if res.returncode == 0:
                        p = res.stdout.strip()
                        if p and os.path.exists(p):
                            candidates.append(p)
                # Common Windows install paths
                user_py = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python312\python.exe")
                candidates.extend([
                    r"C:\\Python312\\python.exe",
                    user_py,
                    r"C:\\Program Files\\Python312\\python.exe",
                    r"C:\\Program Files (x86)\\Python312\\python.exe",
                ])
            else:
                # Unix-like
                for name in ['python3.12', '/usr/bin/python3.12', '/opt/homebrew/bin/python3.12', '/usr/local/bin/python3.12']:
                    if _shutil.which(name) or os.path.exists(name):
                        candidates.append(_shutil.which(name) or name)

            for cand in candidates:
                try:
                    res = _sub.run([cand, '-c', 'import sys; import platform; print(sys.version_info[:2])'], capture_output=True, text=True, timeout=2)
                    if res.returncode == 0 and '(3, 12)' in res.stdout:
                        self.whisperx_python = cand
                        # Cache to settings if available
                        if isinstance(self.settings, dict):
                            self.settings['whisperx_python'] = cand
                        break
                except Exception:
                    continue
        except Exception:
            pass

    def download_dependencies(self):
        """Download required dependencies"""
        if sys.platform == "win32":
            url = "https://github.com/Purfview/whisper-standalone-win/releases/download/Faster-Whisper-XXL/Faster-Whisper-XXL_r245.4_windows.7z"
        else:
            url = "https://github.com/Purfview/whisper-standalone-win/releases/download/Faster-Whisper-XXL/Faster-Whisper-XXL_r245.4_linux.7z"

        reply = QMessageBox.question(
            self, "Download Dependencies",
            f"Download faster-whisper-xxl executable and dependencies?\n\n"
            f"Size: ~1.4GB\nLocation: {self.bin_dir}\n\n"
            f"This is required for caption generation.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.progress_dialog = DownloadProgressDialog(self)
            self.downloader = DependencyDownloader(url, self.files_to_check, self.bin_dir)

            self.downloader.progress.connect(self.progress_dialog.update_progress)
            self.downloader.finished.connect(self.on_download_finished)

            self.downloader.start()

            if self.progress_dialog.exec() == QDialog.DialogCode.Rejected:
                self.downloader.cancel()

    def on_download_finished(self, success, message):
        """Handle download completion"""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.accept()

        if success:
            QMessageBox.information(self, "Success", "Dependencies downloaded successfully!")
            self.check_dependencies()
        else:
            QMessageBox.critical(self, "Download Failed", f"Failed to download dependencies:\n{message}")

    def detect_hardware(self):
        """Show hardware optimization dialog (matches reference implementation exactly)"""
        try:
            dialog = HardwareOptimizationDialog(self)
            result = dialog.exec()

            if result == QDialog.DialogCode.Accepted and dialog.user_accepted:
                # Apply the recommendations exactly like the reference
                self.apply_hardware_recommendations(dialog.recommendations, dialog.hardware_info)

                # Save hardware info
                self.hardware_info = dialog.hardware_info
                self.save_settings()

                # Show confirmation
                QMessageBox.information(
                    self, "Optimization Applied",
                    "Hardware-optimized settings have been applied!\n\n"
                    "Settings have been automatically saved."
                )
            else:
                # Update display even if user declined
                if hasattr(dialog, 'hardware_info'):
                    self.hardware_info = dialog.hardware_info
                    self.update_hardware_display()
                    self.save_settings()

        except Exception as e:
            QMessageBox.critical(self, "Hardware Detection Error", f"Error during hardware detection:\n{str(e)}")

    def apply_hardware_recommendations(self, recommendations, hardware_info):
        """Apply hardware recommendations to UI controls (matches reference)"""
        try:
            # Apply device setting
            if "device" in recommendations:
                self.device_combo.setCurrentText(recommendations["device"])

            # Apply compute type
            if "compute_type" in recommendations:
                self.compute_combo.setCurrentText(recommendations["compute_type"])

            # Apply model recommendation
            if "model" in recommendations:
                self.model_combo.setCurrentText(recommendations["model"])

            # If we're using Faster-Whisper (default), choose best model based on GPU
            # Default engine is Faster-Whisper; select best model without reading hidden radios
            try:
                if hardware_info.get('has_cuda'):
                    self.model_combo.setCurrentText('distil-large-v3.5')
                else:
                    self.model_combo.setCurrentText('large-v2')
            except Exception:
                pass

            # Apply VAD method
            if "vad_method" in recommendations:
                self.vad_method.setCurrentText(recommendations["vad_method"])

            # Diarization disabled

            # Update the hardware display
            self.update_hardware_display()

        except Exception as e:
            logging.error(f"Error applying hardware recommendations: {e}")

    def browse_file(self):
        """Browse for input file (legacy method, now handled by drag-drop)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Audio/Video File", "",
            "Media Files (*.mp3 *.wav *.mp4 *.avi *.mov *.mkv *.m4a *.aac *.flac *.ogg *.webm);;All Files (*)"
        )
        if file_path:
            self.on_file_dropped(file_path)

    def browse_output_dir(self):
        """Browse for output directory (legacy method)"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            if hasattr(self, 'output_dir'):
                self.output_dir.setText(dir_path)

    def _compute_output_dir_for_input(self, input_file: Optional[str]) -> str:
        """Compute output directory for provided input file respecting the output_dir override."""
        if hasattr(self, 'output_dir') and self.output_dir.text().strip():
            return self.output_dir.text().strip()
        if input_file:
            return os.path.dirname(input_file)
        return os.path.join(get_app_directory(), "output")

    def _resolve_job_context(self, input_file: Optional[str] = None, output_dir: Optional[str] = None) -> tuple[Optional[str], str]:
        """
        Resolve the input/output context for the active or most recent transcription job.

        Preference order for input:
          1. Explicit override argument.
          2. Currently running job (if QProcess still active).
          3. Pending post-processing job.
          4. Current UI selection (self.current_input_file).
          5. Last completed job.
        """
        resolved_input = input_file
        process_running = bool(self.process and self.process.state() != QProcess.ProcessState.NotRunning)

        if not resolved_input:
            if process_running and self._active_run_input_file:
                resolved_input = self._active_run_input_file
            elif getattr(self, 'postprocess_started', False) and not getattr(self, 'postprocess_done', False) and self._active_run_input_file:
                resolved_input = self._active_run_input_file
            elif self.current_input_file:
                resolved_input = self.current_input_file
            else:
                resolved_input = self._last_completed_input_file

        resolved_output = output_dir
        if not resolved_output:
            if resolved_input and resolved_input == self._active_run_input_file and self._active_run_output_dir:
                resolved_output = self._active_run_output_dir
            elif self._last_completed_output_dir:
                resolved_output = self._last_completed_output_dir
            else:
                resolved_output = self._compute_output_dir_for_input(resolved_input)

        return resolved_input, resolved_output

    def get_output_dir(self):
        """Get output directory - defaults to same directory as resolved job input file"""
        _, output_dir = self._resolve_job_context()
        return output_dir

    def _ensure_batch_mode_for_active_run(self):
        """Promote an active single transcription run to batch mode if new files are queued."""
        running = bool(self.process and self.process.state() == QProcess.ProcessState.Running)
        current_file = self._active_run_input_file or self.current_input_file

        if self.batch_mode_active:
            # Deduplicate while preserving existing order
            deduped_queue: List[str] = []
            for path in self.batch_files:
                if path and path not in deduped_queue:
                    deduped_queue.append(path)
            self.batch_files = deduped_queue

            total_files = len(self.batch_files)
            completed = min(self._batch_index, total_files)
            max_value = max(total_files, 1)

            self.batch_progress_bar.setMaximum(max_value)
            self.batch_progress_bar.setValue(completed)
            self.batch_progress_bar.setFormat(f"Processing {completed}/{total_files} files..." if total_files else "Processing 0/0 files...")
            self.batch_progress_bar.setVisible(total_files > 1)

            if hasattr(self, 'whisperx_batch_progress_bar'):
                self.whisperx_batch_progress_bar.setMaximum(max_value)
                self.whisperx_batch_progress_bar.setValue(completed)
                self.whisperx_batch_progress_bar.setFormat(f"Processing {completed}/{total_files} files..." if total_files else "Processing 0/0 files...")
                self.whisperx_batch_progress_bar.setVisible(total_files > 1)

            if total_files > 1:
                mode_text = f"Batch Mode: {total_files} files queued"
                self.current_file_label.setText(mode_text)
                self.current_file_label.setStyleSheet("color: #007acc; font-size: 12px; font-weight: bold;")
                if hasattr(self, 'whisperx_current_file_label'):
                    self.whisperx_current_file_label.setText(mode_text)
                    self.whisperx_current_file_label.setStyleSheet("color: #007acc; font-size: 12px; font-weight: bold;")
            return

        if not running:
            return

        normalized_queue: List[str] = []
        if current_file:
            normalized_queue.append(current_file)
        for path in self.batch_files:
            if path and path not in normalized_queue:
                normalized_queue.append(path)
        self.batch_files = normalized_queue

        total_files = len(self.batch_files)
        if total_files <= 1:
            return

        self.batch_mode_active = True
        self.batch_results = []
        self._batch_index = 0
        self._append_text_to_console("\nEntering batch mode: additional files queued during transcription.\n")

        self.batch_progress_bar.setMaximum(total_files)
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setFormat(f"Processing 0/{total_files} files...")
        self.batch_progress_bar.setVisible(True)

        if hasattr(self, 'whisperx_batch_progress_bar'):
            self.whisperx_batch_progress_bar.setMaximum(total_files)
            self.whisperx_batch_progress_bar.setValue(0)
            self.whisperx_batch_progress_bar.setFormat(f"Processing 0/{total_files} files...")
            self.whisperx_batch_progress_bar.setVisible(True)

        mode_text = f"Batch Mode: {total_files} files queued"
        self.current_file_label.setText(mode_text)
        self.current_file_label.setStyleSheet("color: #007acc; font-size: 12px; font-weight: bold;")
        if hasattr(self, 'whisperx_current_file_label'):
            self.whisperx_current_file_label.setText(mode_text)
            self.whisperx_current_file_label.setStyleSheet("color: #007acc; font-size: 12px; font-weight: bold;")

    def _activate_cleaner_tab(self) -> bool:
        """Switch UI to the Cleaner tab if available."""
        try:
            main_window = self.parent
            if not main_window or not hasattr(main_window, 'tabs'):
                return False

            tabs = main_window.tabs
            if not tabs:
                return False

            target_index = None
            for i in range(tabs.count()):
                if tabs.tabText(i) == 'Process Captions':
                    target_index = i
                    break
            if target_index is None:
                for i in range(tabs.count()):
                    label = tabs.tabText(i).lower()
                    if 'cleaner' in label or 'process' in label:
                        target_index = i
                        break

            if target_index is not None:
                tabs.setCurrentIndex(target_index)
                if hasattr(main_window, 'cleaner_input_text') and main_window.cleaner_input_text:
                    main_window.cleaner_input_text.setFocus()
                return True
        except Exception as e:
            self._append_text_to_console(f"Note: Could not switch to Cleaner tab automatically: {e}\n")
        return False

    def _apply_content_to_cleaner_input(self, new_content: str) -> Optional[str]:
        """Insert caption content into the cleaner input, prompting if existing text is present.

        Returns:
            Optional[str]: 'overwrite' or 'append' when content applied, None if cancelled or unavailable.
        """
        main_window = self.parent
        if not main_window or not hasattr(main_window, 'cleaner_input_text'):
            self._append_text_to_console("Warning: Could not access cleaner input for caption upload\n")
            return None

        self._activate_cleaner_tab()

        text_widget = main_window.cleaner_input_text
        existing_text = text_widget.toPlainText()
        has_existing = bool(existing_text.strip())
        action = "overwrite"

        if has_existing:
            if self._cleaner_insert_mode:
                if self._cleaner_insert_mode == "overwrite":
                    if self._cleaner_insert_mode_once:
                        action = "overwrite"
                    else:
                        action = "append"
                else:
                    action = self._cleaner_insert_mode
            else:
                # Use overlay dialog positioned over the Step 1 textbox
                dialog = CleanerInputOverlayDialog(text_widget, main_window)
                dialog.exec()

                if dialog.result_action is None:
                    self._append_text_to_console("Caption upload cancelled by user.\n")
                    return None

                action = dialog.result_action

                if action == "overwrite":
                    self._cleaner_insert_mode_once = True
                self._cleaner_insert_mode = action
        else:
            if self._cleaner_insert_mode is None:
                self._cleaner_insert_mode = "append"

        if action == "append" and has_existing:
            combined = existing_text.rstrip()
            if combined:
                combined += "\n\n"
            combined += new_content
            text_widget.setPlainText(combined)
        else:
            text_widget.setPlainText(new_content)
            if not has_existing and self._cleaner_insert_mode is None:
                self._cleaner_insert_mode = "append"

        main_window.cleaner_input_has_real_content = True
        if hasattr(main_window, 'placeholder_label'):
            main_window.placeholder_label.hide()

        if action == "overwrite":
            if self._cleaner_insert_mode_once:
                self._cleaner_insert_mode = "append"
                self._cleaner_insert_mode_once = False
        else:
            self._cleaner_insert_mode_once = False

        return action

    def _load_session_history_from_session(self):
        """Load stored caption history from the parent session data."""
        try:
            main_window = self.parent
            if not main_window:
                return
            if not hasattr(main_window, 'session_data') or main_window.session_data is None:
                main_window.session_data = {}
            raw_history = main_window.session_data.get('generate_captions_history', [])
            if isinstance(raw_history, list):
                sanitized = []
                for entry in raw_history:
                    if isinstance(entry, dict):
                        sanitized.append(entry)
                # Keep only the most recent 100 entries
                self.session_history = copy.deepcopy(sanitized[-100:])
            else:
                self.session_history = []
        except Exception as e:
            self.session_history = []
            self._append_text_to_console(f"Warning: Could not load caption history: {e}\n")

        # Refresh the static transcripts panel with loaded history
        self._refresh_transcripts_panel()

    def _sync_session_history_to_session(self):
        """Persist current history into the parent session data."""
        try:
            main_window = self.parent
            if not main_window:
                return
            if not hasattr(main_window, 'session_data') or main_window.session_data is None:
                main_window.session_data = {}
            main_window.session_data['generate_captions_history'] = copy.deepcopy(self.session_history)
            try:
                if hasattr(main_window, 'mark_changes_pending'):
                    main_window.mark_changes_pending()
                if hasattr(main_window, 'session_modified'):
                    main_window.session_modified = True
            except Exception:
                pass
        except Exception as e:
            self._append_text_to_console(f"Warning: Could not sync caption history to session: {e}\n")

    def _add_history_entry(self, input_file: Optional[str], output_dir: Optional[str], files: List[dict], success: bool = True, error: Optional[str] = None):
        """Record a caption generation event for the current session."""
        try:
            # Check for duplicates - if input_file and output_dir match an existing entry, skip
            for existing_entry in self.session_history:
                if (existing_entry.get("input_file") == input_file and
                    existing_entry.get("output_dir") == output_dir):
                    # Duplicate found - don't add
                    return

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            entry = {
                "timestamp": timestamp,
                "input_file": input_file,
                "output_dir": output_dir,
                "files": files or [],
                "success": success,
                "error": error
            }
            self.session_history.append(entry)
            if len(self.session_history) > 100:
                self.session_history = self.session_history[-100:]
            self._sync_session_history_to_session()
            # Refresh the static transcripts panel
            self._refresh_transcripts_panel()
        except Exception as e:
            self._append_text_to_console(f"Warning: Could not record session history entry: {e}\n")

    def show_session_history_dialog(self):
        """Display a dialog summarizing caption files generated during this session."""
        self._load_session_history_from_session()

        dialog = QDialog(self)
        dialog.setWindowTitle("Session Caption History")
        dialog.resize(640, 520)

        main_layout = QVBoxLayout(dialog)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        if not self.session_history:
            empty_label = QLabel("No captions have been generated in this session yet.")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            empty_label.setStyleSheet("color: #666; font-size: 13px;")
            main_layout.addWidget(empty_label, stretch=1)
        else:
            def handle_delete(index: int):
                if index < 0 or index >= len(self.session_history):
                    return
                try:
                    removed = self.session_history.pop(index)
                except Exception:
                    return
                self._sync_session_history_to_session()
                # Refresh the static transcripts panel
                self._refresh_transcripts_panel()
                self._append_text_to_console(f"History entry removed for: {os.path.basename((removed or {}).get('input_file') or '')}\n")
                dialog.accept()
                QTimer.singleShot(0, self.show_session_history_dialog)

            scroll = QScrollArea(dialog)
            scroll.setWidgetResizable(True)
            container = QWidget(dialog)  # Parent to prevent top-level window flash
            container_layout = QVBoxLayout(container)
            container_layout.setSpacing(12)

            for display_index, entry in enumerate(reversed(self.session_history)):
                actual_index = len(self.session_history) - 1 - display_index
                input_path = entry.get("input_file") or "Unknown file"
                base_name = os.path.basename(input_path) if input_path else "Unknown file"
                timestamp = entry.get("timestamp", "")
                success = entry.get("success", True)
                error = entry.get("error")

                header_icon = "✅" if success else "⚠"
                group_title = f"{header_icon} {base_name} • {timestamp}"
                group = QGroupBox(group_title)
                group.setStyleSheet("""
                    QGroupBox {
                        border: 1px solid #d0d5d9;
                        border-radius: 6px;
                        margin-top: 4px;
                        font-weight: bold;
                    }
                    QGroupBox::title {
                        subcontrol-origin: margin;
                        left: 10px;
                        padding: 2px 6px;
                    }
                """)
                vbox = QVBoxLayout(group)
                vbox.setSpacing(6)

                output_dir = entry.get("output_dir") or ""
                details = QLabel()
                details.setStyleSheet("color: #444; font-size: 12px;")
                if output_dir:
                    details.setTextFormat(Qt.TextFormat.RichText)
                    details.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse)
                    details.setOpenExternalLinks(False)
                    details.setText(f'Output folder: <a href="history-dir">{output_dir}</a>')
                    details.linkActivated.connect(lambda _, p=output_dir: self.open_file_location(p))
                else:
                    details.setText("Output folder unavailable")
                vbox.addWidget(details)

                if not success and error:
                    error_label = QLabel(f"Error: {error}")
                    error_label.setStyleSheet("color: #c92a2a; font-size: 12px;")
                    error_label.setWordWrap(True)
                    vbox.addWidget(error_label)

                files = entry.get("files") or []
                if files:
                    for file_info in files:
                        file_path = file_info.get("path")
                        file_label = file_info.get("label", "File")
                        name_label = QLabel()
                        name_label.setStyleSheet("color: #333; font-size: 12px;")
                        if file_path and os.path.exists(file_path):
                            file_full_path = file_path
                            name_label.setTextFormat(Qt.TextFormat.RichText)
                            name_label.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse)
                            name_label.setOpenExternalLinks(False)
                            display_name = os.path.basename(file_path) or file_path
                            name_label.setText(f"{file_label}: <a href=\"history-file\">{display_name}</a>")
                            name_label.linkActivated.connect(lambda _, p=file_full_path: self.open_file_location(p))
                        else:
                            display_name = os.path.basename(file_path) if file_path else 'Unavailable'
                            name_label.setText(f"{file_label}: {display_name}")
                        vbox.addWidget(name_label)
                else:
                    no_files = QLabel("No files were generated.")
                    no_files.setStyleSheet("color: #666; font-size: 12px;")
                    vbox.addWidget(no_files)

                delete_btn = QPushButton("Delete")
                delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                delete_btn.setStyleSheet("""
                    QPushButton {
                        background: #ffe8e8;
                        border: 1px solid #ffb3b3;
                        border-radius: 4px;
                        padding: 4px 12px;
                        font-size: 12px;
                        color: #c92a2a;
                        font-weight: 600;
                        margin-right: auto;
                    }
                    QPushButton:hover {
                        background: #ffd6d6;
                    }
                """)
                delete_btn.clicked.connect(lambda _, idx=actual_index: handle_delete(idx))
                vbox.addWidget(delete_btn, alignment=Qt.AlignmentFlag.AlignLeft)

                container_layout.addWidget(group)

            container_layout.addStretch(1)
            scroll.setWidget(container)
            main_layout.addWidget(scroll, stretch=1)

        close_btn = QPushButton("Close")
        close_btn.setMinimumHeight(36)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background: #007acc;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-size: 13px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #0062a3;
            }
        """)
        close_btn.clicked.connect(dialog.accept)

        main_layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignRight)
        dialog.exec()

    def open_file_location(self, path: str):
        """Open file location in system file manager.

        On Windows, selects the file when a file path is provided. On other
        platforms or when given a directory, opens the target folder normally.
        """
        try:
            import subprocess
            import platform

            if not path:
                self._append_text_to_console("Could not open file location: path unavailable\n")
                return

            system = platform.system()
            normalized_path = os.path.normpath(path)

            if system == "Windows":
                if os.path.isfile(normalized_path):
                    subprocess.run(["explorer", "/select,", normalized_path], check=True)
                else:
                    subprocess.run(["explorer", normalized_path], check=True)
            elif system == "Darwin":  # macOS
                target = normalized_path
                if os.path.isfile(normalized_path):
                    target = os.path.dirname(normalized_path)
                subprocess.run(["open", target], check=True)
            else:  # Linux and others
                target = normalized_path
                if os.path.isfile(normalized_path):
                    target = os.path.dirname(normalized_path)
                subprocess.run(["xdg-open", target], check=True)

            self._append_text_to_console(f"✓ Opened file location: {path}\n")

        except Exception as e:
            self._append_text_to_console(f"Could not open file location: {str(e)}\n")

    def show_completion_success_dialog(self, output_dir, srt_created, adobe_created):
        """Show success dialog after transcription completion with file type-specific messaging"""

        class CompletionDialog(QDialog):
            """Custom dialog with close confirmation"""
            def __init__(self, parent, srt_created, adobe_created):
                super().__init__(parent)
                self.srt_created = srt_created
                self.adobe_created = adobe_created
                self.proceeded = False

            def closeEvent(self, event):
                """Warn user if closing without proceeding"""
                if not self.proceeded:
                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Icon.Warning)
                    msg.setWindowTitle("Manual Upload Required")

                    warning_text = "If you don't press \"Yes, Proceed\", you will need to manually:\n\n"
                    warning_text += "• Upload .ref file(s) via File → Upload Reference Transcript\n"

                    if self.srt_created:
                        warning_text += "• Import SRT file(s) into Process Captions tab\n"

                    warning_text += "\nAre you sure you want to close?"

                    msg.setText(warning_text)
                    msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    msg.setDefaultButton(QMessageBox.StandardButton.No)

                    if msg.exec() == QMessageBox.StandardButton.Yes:
                        event.accept()
                    else:
                        event.ignore()
                else:
                    event.accept()

        dialog = CompletionDialog(self, srt_created, adobe_created)
        dialog.setWindowTitle("Transcription Completed!")
        dialog.setFixedSize(580, 480)
        dialog.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)

        # Main layout
        main_layout = QVBoxLayout(dialog)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        # Success icon and title section
        header_layout = QHBoxLayout()
        header_layout.setSpacing(15)

        # Success icon (checkmark)
        icon_label = QLabel("✓")
        icon_label.setStyleSheet("""
            QLabel {
                color: #4CAF50;
                font-size: 48px;
                font-weight: bold;
                min-width: 60px;
                max-width: 60px;
            }
        """)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(icon_label)

        # Title and subtitle
        text_layout = QVBoxLayout()
        text_layout.setSpacing(8)

        title_label = QLabel("Success!")
        title_label.setStyleSheet("""
            QLabel {
                color: #2E2E2E;
                font-size: 24px;
                font-weight: bold;
                margin: 0;
            }
        """)
        text_layout.addWidget(title_label)

        subtitle_label = QLabel("Caption generated successfully")
        subtitle_label.setStyleSheet("""
            QLabel {
                color: #666666;
                font-size: 14px;
                margin: 0;
            }
        """)
        text_layout.addWidget(subtitle_label)

        header_layout.addLayout(text_layout)
        header_layout.addStretch()
        main_layout.addLayout(header_layout)

        # Main message based on what was created
        message_text = ""

        if srt_created and adobe_created:
            message_text = ("✅ Successfully Generated:\n"
                           "• Caption File (.srt) - The generated captions used by Scriptoria as a foundation in Process Captions tab\n"
                           "• Reference files (.ref + .json) - Links Scriptoria and Premiere Pro for word-level matching\n\n"
                           "📋 Recommended Next Steps:\n\n"
                           "1. Click \"Yes, Proceed\" to auto-upload .ref and .srt to Scriptoria\n\n"
                           "2. In Premiere Pro, import the .json Static Transcript:\n"
                           "   • Text → Transcript → Import Static Transcript\n"
                           "   • Select the .json file from the output folder\n"
                           "   • This links your Premiere Pro sequence with Scriptoria\n\n"
                           "Ready to proceed?")
        elif srt_created:
            message_text = ("✅ Successfully Generated:\n"
                           "• Caption File (.srt) - The generated captions used by Scriptoria as a foundation in Process Captions tab\n"
                           "• Reference file (.ref) - For word-level accuracy in Scriptoria\n\n"
                           "📋 Recommended Next Steps:\n\n"
                           "Click \"Yes, Proceed\" to auto-upload .ref and .srt to Scriptoria\n\n"
                           "Ready to proceed?")
        elif adobe_created:
            message_text = ("✅ Successfully Generated:\n"
                           "• Reference files (.ref + .json) - Links Scriptoria and Premiere Pro for word-level matching\n\n"
                           "📋 Recommended Next Steps:\n\n"
                           "1. Click \"Yes, Proceed\" to auto-upload .ref to Scriptoria\n\n"
                           "2. In Premiere Pro, import the .json Static Transcript:\n"
                           "   • Text → Transcript → Import Static Transcript\n"
                           "   • Select the .json file from the output folder\n"
                           "   • This links your Premiere Pro sequence with Scriptoria\n\n"
                           "Ready to proceed?")

        message_label = QLabel(message_text)
        message_label.setStyleSheet("""
            QLabel {
                color: #444444;
                font-size: 13px;
                background-color: #F8F9FA;
                padding: 20px;
                border: 1px solid #E9ECEF;
                border-radius: 8px;
            }
        """)
        message_label.setWordWrap(True)
        message_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        main_layout.addWidget(message_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        # Open File Location button (left side)
        location_button = QPushButton("📁 Open File Location")
        location_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: 500;
                min-width: 140px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:pressed {
                background-color: #545b62;
            }
        """)
        location_button.setCursor(Qt.CursorShape.PointingHandCursor)
        location_button.clicked.connect(lambda: self.open_file_location(output_dir))  # Don't close dialog
        button_layout.addWidget(location_button)

        button_layout.addStretch()

        # Yes, Proceed button (primary action)
        proceed_button = QPushButton("Yes, Proceed")
        proceed_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: 600;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        proceed_button.setCursor(Qt.CursorShape.PointingHandCursor)
        proceed_button.clicked.connect(lambda: self.handle_proceed_button(dialog, output_dir, srt_created, adobe_created))
        proceed_button.setDefault(True)
        button_layout.addWidget(proceed_button)

        main_layout.addLayout(button_layout)

        # Show dialog
        dialog.exec()

    def handle_proceed_button(self, dialog, output_dir, srt_created, adobe_created):
        """Handle Yes, Proceed button - auto-upload files"""
        dialog.proceeded = True  # Mark as proceeded to skip warning
        dialog.accept()
        self._activate_cleaner_tab()

        # Auto-upload reference transcript REF (if it exists) for both cleaner input and reference transcript
        try:
            resolved_input, resolved_output = self._resolve_job_context(self._last_completed_input_file, output_dir)
            if not resolved_input:
                self._append_text_to_console("Warning: No completed file available for auto-upload\n")
                return

            base_name = os.path.splitext(os.path.basename(resolved_input))[0]
            if getattr(self, 'last_srt_text', '').strip():
                self.auto_upload_srt_text(base_name, self.last_srt_text)
            else:
                txt_path = self.get_expected_txt_path(resolved_input, resolved_output)
                if os.path.exists(txt_path):
                    self.auto_upload_txt_file(resolved_output, input_file=resolved_input)
                else:
                    self._append_text_to_console("Note: No SRT or REF text available for cleaner upload.\n")

            # Always keep reference transcript list in sync
            txt_path_ref = self.get_expected_txt_path(resolved_input, resolved_output)
            if os.path.exists(txt_path_ref):
                self.auto_upload_reference_transcript(resolved_output, input_file=resolved_input)
        except Exception as e:
            self._append_text_to_console(f"Warning: Could not auto-upload reference transcript REF: {e}\n")

        # Switch to the Cleaner (Process Captions) tab for next steps
        try:
            main_window = self.parent
            if main_window and hasattr(main_window, 'tabs'):
                tabs = main_window.tabs
                # Prefer an exact match first
                target_index = None
                for i in range(tabs.count()):
                    if tabs.tabText(i) == 'Process Captions':
                        target_index = i
                        break
                # Fallback: look for a likely cleaner tab by name
                if target_index is None:
                    for i in range(tabs.count()):
                        label = tabs.tabText(i).lower()
                        if 'cleaner' in label or 'process' in label:
                            target_index = i
                            break
                if target_index is not None:
                    tabs.setCurrentIndex(target_index)
                    # Give focus to the cleaner input if available
                    if hasattr(main_window, 'cleaner_input_text') and main_window.cleaner_input_text:
                        main_window.cleaner_input_text.setFocus()
        except Exception as e:
            # Do not block flow if tab switching fails
            self._append_text_to_console(f"Note: Could not switch to Cleaner tab automatically: {e}\n")

    def handle_batch_proceed_button(self, dialog):
        """Handle Yes, Proceed button for batch completion - auto-upload files and switch tabs"""
        dialog.proceeded = True  # Mark as proceeded to skip warning
        dialog.accept()
        self._activate_cleaner_tab()

        # Auto-upload batch results
        self.auto_upload_batch_results()

        # Switch to the Cleaner (Process Captions) tab for next steps
        try:
            main_window = self.parent
            if main_window and hasattr(main_window, 'tabs'):
                tabs = main_window.tabs
                # Prefer an exact match first
                target_index = None
                for i in range(tabs.count()):
                    if tabs.tabText(i) == 'Process Captions':
                        target_index = i
                        break
                # Fallback: look for a likely cleaner tab by name
                if target_index is None:
                    for i in range(tabs.count()):
                        label = tabs.tabText(i).lower()
                        if 'cleaner' in label or 'process' in label:
                            target_index = i
                            break
                if target_index is not None:
                    tabs.setCurrentIndex(target_index)
                    # Give focus to the cleaner input if available
                    if hasattr(main_window, 'cleaner_input_text') and main_window.cleaner_input_text:
                        main_window.cleaner_input_text.setFocus()
        except Exception as e:
            # Do not block flow if tab switching fails
            self._append_text_to_console(f"Note: Could not switch to Cleaner tab automatically: {e}\n")

    def auto_upload_reference_transcript(self, output_dir, input_file: Optional[str] = None):
        """Auto-upload reference transcript REF as reference transcript"""
        try:
            # Find the reference transcript REF file (better for reference transcripts than the Adobe Static Transcript)
            resolved_input, resolved_output = self._resolve_job_context(input_file, output_dir)
            if not resolved_input or not resolved_output:
                self._append_text_to_console("Warning: No input file available for reference transcript upload\n")
                return
            txt_file = self.get_expected_txt_path(resolved_input, resolved_output)

            if not os.path.exists(txt_file):
                self._append_text_to_console(f"Warning: Reference transcript REF not found: {txt_file}\n")
                return

            # Try to add reference transcript through proper channels
            main_window = self.parent
            if not main_window:
                self._append_text_to_console("Warning: No parent window found for reference transcript upload\n")
                return

            # Debug: Check what attributes the main window has
            self._append_text_to_console(f"Debug: Main window type: {type(main_window).__name__}\n")

            try:
                # Initialize session data structure if needed
                if not hasattr(main_window, 'session_data'):
                    main_window.session_data = {}
                if main_window.session_data is None:
                    main_window.session_data = {}
                if 'transcript_files' not in main_window.session_data:
                    main_window.session_data['transcript_files'] = []

                # Check if this file is already in the list
                existing_files = main_window.session_data['transcript_files']
                if any(f.get('path') == txt_file for f in existing_files):
                    self._append_text_to_console(f"Reference transcript already added: {os.path.basename(txt_file)}\n")
                    return

                # Add the new transcript file to the session data
                import time
                transcript_entry = {
                    'path': txt_file,
                    'last_modified': time.time()
                }
                main_window.session_data['transcript_files'].append(transcript_entry)

                # Mark session as modified and pending save
                try:
                    if hasattr(main_window, 'mark_changes_pending'):
                        main_window.mark_changes_pending()
                    if hasattr(main_window, 'session_modified'):
                        main_window.session_modified = True
                except Exception:
                    pass

                self._append_text_to_console(f"✓ Auto-uploaded reference transcript REF: {os.path.basename(txt_file)}\n")

            except Exception as e:
                self._append_text_to_console(f"Warning: Reference transcript auto-upload failed: {str(e)}\n")

        except Exception as e:
            self._append_text_to_console(f"Warning: Auto-upload of reference transcript failed: {str(e)}\n")

    def auto_upload_srt_file(self, output_dir, input_file: Optional[str] = None):
        """Auto-upload SRT file to cleaner input"""
        try:
            # Find the SRT file
            resolved_input, resolved_output = self._resolve_job_context(input_file, output_dir)
            if not resolved_input or not resolved_output:
                self._append_text_to_console("Warning: No input file available for SRT auto-upload\n")
                return
            base_name = os.path.splitext(os.path.basename(resolved_input))[0]

            # Look for SRT file in output directory
            srt_file = None
            possible_srt_files = [
                os.path.join(resolved_output, f"{base_name}.srt"),
                os.path.join(resolved_output, "srt", f"{base_name}.srt"),
            ]

            # Also check for any .srt files in the directory
            try:
                all_files = os.listdir(resolved_output)
                srt_files = [f for f in all_files if f.endswith('.srt')]
                if srt_files:
                    for srt_filename in srt_files:
                        possible_srt_files.append(os.path.join(resolved_output, srt_filename))
            except:
                pass

            for possible_file in possible_srt_files:
                if os.path.exists(possible_file):
                    srt_file = possible_file
                    break

            if not srt_file:
                self._append_text_to_console("Warning: SRT file not found for auto-upload\n")
                return

            # Access the main window to get the cleaner input text
            main_window = self.parent
            if not main_window or not hasattr(main_window, 'cleaner_input_text'):
                self._append_text_to_console("Warning: Could not access cleaner input for SRT upload\n")
                return

            # Read SRT file content
            with open(srt_file, 'r', encoding='utf-8') as f:
                srt_content = f.read().strip()

            if not srt_content:
                self._append_text_to_console("Warning: SRT file is empty\n")
                return

            # Create header for the file (like the existing drag-drop logic)
            filename = os.path.splitext(os.path.basename(srt_file))[0]  # Remove extension
            header = f"==={filename}==="
            content_with_header = header + "\n" + srt_content

            action = self._apply_content_to_cleaner_input(content_with_header)
            if not action:
                return

            verb = "Appended" if action == "append" else "Loaded"
            self._append_text_to_console(f"✓ {verb} SRT in cleaner input: {os.path.basename(srt_file)}\n")

        except Exception as e:
            self._append_text_to_console(f"Warning: Auto-upload of SRT file failed: {str(e)}\n")

    def auto_upload_txt_file(self, output_dir, input_file: Optional[str] = None):
        """Auto-upload reference transcript REF file to cleaner input"""
        try:
            # Find the reference transcript REF file
            resolved_input, resolved_output = self._resolve_job_context(input_file, output_dir)
            if not resolved_input or not resolved_output:
                self._append_text_to_console("Warning: No input file available for TXT auto-upload\n")
                return
            txt_file = self.get_expected_txt_path(resolved_input, resolved_output)

            if not os.path.exists(txt_file):
                self._append_text_to_console("Warning: Reference transcript REF not found for auto-upload\n")
                return

            # Access the main window to get the cleaner input text
            main_window = self.parent
            if not main_window or not hasattr(main_window, 'cleaner_input_text'):
                self._append_text_to_console("Warning: Could not access cleaner input for TXT upload\n")
                return

            # Read TXT file content
            with open(txt_file, 'r', encoding='utf-8') as f:
                txt_content = f.read().strip()

            if not txt_content:
                self._append_text_to_console("Warning: TXT file is empty\n")
                return

            # Create header for the file (like the existing drag-drop logic)
            filename = os.path.splitext(os.path.basename(txt_file))[0]  # Remove extension
            header = f"==={filename}==="
            content_with_header = header + "\n" + txt_content

            action = self._apply_content_to_cleaner_input(content_with_header)
            if not action:
                return

            verb = "Appended" if action == "append" else "Loaded"
            self._append_text_to_console(f"✓ {verb} reference transcript in cleaner input: {os.path.basename(txt_file)}\n")

        except Exception as e:
            self._append_text_to_console(f"Warning: Auto-upload of TXT file failed: {str(e)}\n")

    def auto_upload_srt_text(self, base_name: str, srt_text: str):
        """Auto-upload SRT transcription text to the cleaner tab."""
        try:
            srt_text = (srt_text or '').strip()
            if not srt_text:
                self._append_text_to_console("Warning: No SRT text available for cleaner upload\n")
                return

            header = f"==={base_name}==="
            content_with_header = header + "\n" + srt_text

            action = self._apply_content_to_cleaner_input(content_with_header)
            if not action:
                return

            verb = "Appended" if action == "append" else "Loaded"
            self._append_text_to_console(f"✓ {verb} SRT to cleaner input\n")

        except Exception as e:
            self._append_text_to_console(f"Warning: Auto-upload of SRT text failed: {str(e)}\n")


    def _get_compute_type_for_device(self, device, selected_compute_type):
        """
        Determine appropriate compute type based on device.
        CPU doesn't support float16, so fall back to float32.

        Args:
            device: The device being used ('cuda' or 'cpu')
            selected_compute_type: The user's selected compute type

        Returns:
            The compute type to use
        """
        if device.lower() == 'cpu' and selected_compute_type == 'float16':
            return 'float32'
        return selected_compute_type

    def start_transcription(self):
        """Start the transcription process"""
        # Determine files to process
        files_to_process = self.batch_files if self.batch_files else ([self.current_input_file] if self.current_input_file else [])

        if not files_to_process:
            QMessageBox.warning(self, "Input Required", "Please select an input file using the drag-drop area or add files to the batch queue.")
            return

        # Validate all files exist
        missing_files = [f for f in files_to_process if not os.path.exists(f)]
        if missing_files:
            QMessageBox.warning(self, "Files Not Found", f"The following files do not exist:\n" + "\n".join(missing_files))
            return

        # Check which tab is active to determine engine
        current_tab_index = self.tab_widget.currentIndex()
        using_whisperx = (current_tab_index == self.whisperx_tab_index)

        # Check dependencies based on active tab
        if using_whisperx:
            if not self.check_whisperx_deps_simple():
                QMessageBox.critical(self, "Dependencies Missing", "Please install WhisperX from the WhisperX tab first.")
                return
        else:
            # Using Faster-Whisper-XXL or on Dependencies/Generate tab
            if self.engine_faster_whisper.isChecked():
                if not self._check_faster_whisper_dependencies():
                    QMessageBox.critical(self, "Dependencies Missing", "Please download Faster-Whisper dependencies first from the Dependencies tab.")
                    return
            else:
                if not self._check_whisperx_dependencies():
                    QMessageBox.critical(self, "Dependencies Missing", "Please install WhisperX dependencies first from the Dependencies tab.")
                    return

        # Reset cleaner insertion mode for this transcription run
        self._cleaner_insert_mode = None
        self._cleaner_insert_mode_once = False
        self._load_session_history_from_session()

        # Initialize batch processing
        if len(files_to_process) > 1:
            self.batch_mode_active = True
            self.batch_files = files_to_process
            self.batch_results = []
            self._batch_index = 0

            # Setup progress bars for BOTH tabs
            self.batch_progress_bar.setMaximum(len(files_to_process))
            self.batch_progress_bar.setValue(0)
            self.batch_progress_bar.setFormat(f"Processing 0/{len(files_to_process)} files...")
            self.batch_progress_bar.setVisible(True)

            if hasattr(self, 'whisperx_batch_progress_bar'):
                self.whisperx_batch_progress_bar.setMaximum(len(files_to_process))
                self.whisperx_batch_progress_bar.setValue(0)
                self.whisperx_batch_progress_bar.setFormat(f"Processing 0/{len(files_to_process)} files...")
                self.whisperx_batch_progress_bar.setVisible(True)

            self._append_text_to_console(f"Starting batch processing of {len(files_to_process)} files...\n" + "="*50 + "\n")

            # For WhisperX, use single-command batch mode (model loaded once)
            if using_whisperx:
                self._run_whisperx_batch(files_to_process)
                return
        else:
            self.batch_mode_active = False

        # Start processing the first (or only) file
        self._process_next_in_batch()

    def _run_whisperx_batch(self, files_to_process):
        """Run WhisperX batch mode - all files in one command (model loaded once)"""
        # Build command with all files
        cli_path = os.path.join(get_app_directory(), 'Data', 'whisperx_cli.py')
        if not os.path.exists(cli_path):
            QMessageBox.critical(self, "Error", f"WhisperX CLI not found: {cli_path}")
            return

        py_exe, extra_env = self._get_whisperx_execution_context(purpose="cli")
        self._whisperx_execution_env = extra_env or None
        self._whisperx_execution_python = py_exe

        if not (py_exe and os.path.exists(py_exe)):
            QMessageBox.critical(self, "WhisperX Not Available", "Please install WhisperX from the WhisperX tab first.")
            return

        cmd = [py_exe]
        if getattr(sys, 'frozen', False) and py_exe == sys.executable:
            cmd.extend(["--scriptoria-run-python", cli_path])
        else:
            cmd.append(cli_path)

        # Add all input files
        cmd.extend(files_to_process)

        # Use WhisperX tab settings
        selected_device = self.whisperx_device_combo.currentText()
        selected_compute = self.whisperx_compute_combo.currentText()

        options = {
            "-m": self.whisperx_model_combo.currentText(),
            "--task": self.whisperx_task_combo.currentText(),
            "-l": self.whisperx_language_combo.currentText() if self.whisperx_language_combo.currentText() != 'auto' else None,
            "--compute_type": selected_compute,
            "--device": selected_device,
        }

        # Output formats
        formats = []
        if self.whisperx_export_srt.isChecked():
            formats.append("srt")
        if self.whisperx_export_adobe.isChecked():
            formats.append("json")
            options["--word_timestamps"] = "True"

        # VAD settings
        if self.whisperx_vad_filter.isChecked():
            options["--vad_filter"] = "True"
            options["--vad_method"] = self.whisperx_vad_method.currentText()

        # Diarization
        if self.whisperx_enable_diarization.isChecked():
            options["--diarize"] = "True"
            if self.whisperx_hf_token.text().strip():
                options["--hf_token"] = self.whisperx_hf_token.text().strip()
            if hasattr(self, 'whisperx_num_speakers'):
                choice = self.whisperx_num_speakers.currentText()
                if choice and choice.lower() != "auto":
                    options["--num_speakers"] = choice

        for key, value in options.items():
            if value is not None:
                cmd.extend([key, str(value)])

        if formats:
            cmd.extend(["--output_format"] + formats)
        else:
            cmd.extend(["--output_format", "srt"])

        # Clear console and start
        self.output_text.clear()

        # Start the progress panel
        show_diarization = self.whisperx_enable_diarization.isChecked() if hasattr(self, 'whisperx_enable_diarization') else False
        if hasattr(self, 'progress_panel') and self.progress_panel:
            self.progress_panel.start_processing(f"Batch: {len(files_to_process)} files", show_diarization=show_diarization)

        self._append_text_to_console(f"Running WhisperX batch mode ({len(files_to_process)} files)...\n")
        self._append_text_to_console(f"Model will be loaded ONCE for all files.\n")
        self._append_text_to_console("="*50 + "\n")

        command_str = " ".join([f'"{arg}"' if " " in arg else arg for arg in cmd[:5]]) + f" ... ({len(files_to_process)} files)"
        self._append_text_to_console(f"Command: {command_str}\n" + "="*50 + "\n")

        # Track files for post-processing
        self._whisperx_batch_files = files_to_process
        self._whisperx_batch_completed_count = 0  # Track completed files for progress bar
        self.batch_results = []  # Reset batch results for completion dialog

        # Initialize batch progress bar
        num_files = len(files_to_process)
        self.batch_progress_bar.setMaximum(num_files)
        self.batch_progress_bar.setValue(0)
        self.batch_progress_bar.setFormat(f"Processing 0/{num_files} files...")
        self.batch_progress_bar.setVisible(num_files > 1)
        if hasattr(self, 'whisperx_batch_progress_bar'):
            self.whisperx_batch_progress_bar.setMaximum(num_files)
            self.whisperx_batch_progress_bar.setValue(0)
            self.whisperx_batch_progress_bar.setFormat(f"Processing 0/{num_files} files...")
            self.whisperx_batch_progress_bar.setVisible(num_files > 1)

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.stop_requested = False
        self.transcription_completed_successfully = False
        self.output_buffer = ""

        # Reset post-processing flags
        self.postprocess_started = False
        self.postprocess_done = False
        self._adobe_converted = False

        # Start process
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self._on_whisperx_batch_finished)
        self.process.errorOccurred.connect(self.on_process_error)

        try:
            hf_cache_dir = get_hf_cache_directory()
            env = QProcessEnvironment.systemEnvironment()
            # Force unbuffered Python output so we get real-time progress updates
            env.insert("PYTHONUNBUFFERED", "1")
            if hf_cache_dir:
                env.insert("HF_HOME", hf_cache_dir)
                env.insert("HF_HUB_CACHE", hf_cache_dir)
                env.insert("HUGGINGFACE_HUB_CACHE", hf_cache_dir)
                env.insert("HF_HUB_DISABLE_SYMLINKS", "1")
                env.insert("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            if hasattr(self, "whisperx_hf_token"):
                token_value = self.whisperx_hf_token.text().strip()
                if token_value:
                    env.insert("HF_TOKEN", token_value)
            if extra_env:
                for key, value in extra_env.items():
                    if value is None:
                        continue
                    value_str = str(value)
                    if key.upper() == "PYTHONPATH":
                        existing = env.value("PYTHONPATH")
                        if existing:
                            value_str = f"{value_str}{os.pathsep}{existing}"
                    env.insert(key, value_str)
            self.process.setProcessEnvironment(env)
        except Exception:
            pass

        self.process.start(cmd[0], cmd[1:])

    def _on_whisperx_batch_finished(self, exit_code, exit_status):
        """Handle completion of WhisperX batch processing"""
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.process = None

        self._append_text_to_console("="*50 + "\n")

        success = exit_code == 0 and not self.stop_requested
        if hasattr(self, 'progress_panel') and self.progress_panel:
            self.progress_panel.stop_processing(success=success)

        if self.stop_requested:
            self._append_text_to_console("Batch processing stopped by user.\n")
        elif exit_code == 0:
            self._append_text_to_console("WhisperX batch processing completed successfully!\n")

            # Fallback: ensure Adobe conversion and history for any missed files
            # (Primary processing happens per-file in _process_whisperx_batch_file_completion)
            if hasattr(self, '_whisperx_batch_files'):
                for input_file in self._whisperx_batch_files:
                    output_dir = os.path.dirname(input_file)
                    base_name = os.path.splitext(os.path.basename(input_file))[0]
                    adobe_path = os.path.join(output_dir, f"{base_name}_adobe.json")

                    # Only convert if Adobe file doesn't exist yet (wasn't created per-file)
                    if self.whisperx_export_adobe.isChecked() and not os.path.exists(adobe_path):
                        try:
                            self.convert_to_adobe_format(input_file=input_file, output_dir=output_dir)
                        except Exception as e:
                            self._append_text_to_console(f"Warning: Adobe conversion failed for {os.path.basename(input_file)}: {e}\n")

                    # Add to history (duplicate detection in _add_history_entry prevents duplicates)
                    files_created = []
                    srt_path = os.path.join(output_dir, f"{base_name}.srt")
                    ref_path = os.path.join(output_dir, f"{base_name}_reference_transcript.ref")
                    if os.path.exists(srt_path):
                        files_created.append({"path": srt_path, "label": "SRT Captions"})
                    if os.path.exists(adobe_path):
                        files_created.append({"path": adobe_path, "label": "Adobe Static Transcript"})
                    if os.path.exists(ref_path):
                        files_created.append({"path": ref_path, "label": "Reference Transcript"})
                    if files_created:
                        self._add_history_entry(input_file, output_dir, files_created, success=True)
        else:
            self._append_text_to_console(f"Batch processing failed with exit code {exit_code}\n")

        # Update batch progress bar
        if hasattr(self, '_whisperx_batch_files'):
            num_files = len(self._whisperx_batch_files)
            self.batch_progress_bar.setValue(num_files)
            self.batch_progress_bar.setFormat(f"Completed {num_files}/{num_files} files!")
            if hasattr(self, 'whisperx_batch_progress_bar'):
                self.whisperx_batch_progress_bar.setValue(num_files)
                self.whisperx_batch_progress_bar.setFormat(f"Completed {num_files}/{num_files} files!")
            QTimer.singleShot(2000, lambda: self.batch_progress_bar.setVisible(False))
            if hasattr(self, 'whisperx_batch_progress_bar'):
                QTimer.singleShot(2000, lambda: self.whisperx_batch_progress_bar.setVisible(False))

        self.batch_mode_active = False

        # Clear the batch queue after completion
        self.batch_files.clear()
        self.update_batch_display()

        # Show batch completion dialog for WhisperX batch mode
        if exit_code == 0 and not self.stop_requested:
            self.show_batch_completion_dialog()

    def _process_whisperx_batch_file_completion(self):
        """Process history and Adobe transcript for each file as it completes in WhisperX batch mode"""
        try:
            # Get the file index (completed count - 1 since we just incremented it)
            completed_idx = getattr(self, '_whisperx_batch_completed_count', 1) - 1
            if completed_idx < 0 or completed_idx >= len(self._whisperx_batch_files):
                return

            input_file = self._whisperx_batch_files[completed_idx]
            output_dir = os.path.dirname(input_file)
            base_name = os.path.splitext(os.path.basename(input_file))[0]

            # Construct expected output paths
            json_path = os.path.join(output_dir, f"{base_name}.json")
            srt_path = os.path.join(output_dir, f"{base_name}.srt")
            adobe_path = os.path.join(output_dir, f"{base_name}_adobe.json")
            ref_path = os.path.join(output_dir, f"{base_name}_reference_transcript.ref")

            # Track what was created
            srt_created = os.path.exists(srt_path)
            adobe_created = False

            # Create Adobe static transcript if enabled (this also creates the .ref file)
            if self.is_adobe_export_enabled() and os.path.exists(json_path):
                try:
                    self.convert_to_adobe_format(input_file=input_file, output_dir=output_dir)
                    if os.path.exists(adobe_path):
                        adobe_created = True
                        self._append_text_to_console(f"Created Adobe Static Transcript: {os.path.basename(adobe_path)}\n")
                except Exception as e:
                    self._append_text_to_console(f"Warning: Could not create Adobe transcript: {e}\n")

            # Build files list for history (after conversion, so we have correct files)
            files_created = []
            if srt_created:
                files_created.append({"path": srt_path, "label": "SRT Captions"})
            if os.path.exists(adobe_path):
                files_created.append({"path": adobe_path, "label": "Adobe Static Transcript"})
            if os.path.exists(ref_path):
                files_created.append({"path": ref_path, "label": "Reference Transcript"})

            # Add to batch_results for the completion dialog
            self.batch_results.append({
                'file_path': input_file,
                'output_dir': output_dir,
                'srt_created': srt_created,
                'adobe_created': adobe_created
            })

            # Add history entry for this file
            if files_created:
                self._add_history_entry(input_file, output_dir, files_created, success=True)

        except Exception as e:
            self._append_text_to_console(f"Warning: Error processing batch file completion: {e}\n")

    def _process_next_in_batch(self):
        """Process the next file in batch queue"""
        if self.batch_mode_active:
            if self._batch_index >= len(self.batch_files):
                # All files processed - show batch completion
                self._handle_batch_completion()
                return

            current_file = self.batch_files[self._batch_index]
            self._append_text_to_console(f"Processing file {self._batch_index + 1}/{len(self.batch_files)}: {os.path.basename(current_file)}\n")
            self.run_transcription(current_file)
        else:
            # Single file processing
            current_file = self.batch_files[0] if self.batch_files else self.current_input_file
            self.run_transcription(current_file)

    def _handle_batch_completion(self):
        """Handle completion of batch processing"""
        # Hide progress bars and show completion status in BOTH tabs
        self.batch_progress_bar.setValue(len(self.batch_files))
        self.batch_progress_bar.setFormat(f"Completed {len(self.batch_files)}/{len(self.batch_files)} files!")
        QTimer.singleShot(2000, lambda: self.batch_progress_bar.setVisible(False))  # Hide after 2 seconds

        if hasattr(self, 'whisperx_batch_progress_bar'):
            self.whisperx_batch_progress_bar.setValue(len(self.batch_files))
            self.whisperx_batch_progress_bar.setFormat(f"Completed {len(self.batch_files)}/{len(self.batch_files)} files!")
            QTimer.singleShot(2000, lambda: self.whisperx_batch_progress_bar.setVisible(False))

        self._append_text_to_console("\n" + "="*50 + "\n")
        self._append_text_to_console(f"Transcription completed! Processed {len(self.batch_results)} files.\n")

        # Count successful and failed results
        srt_files = [r for r in self.batch_results if r['srt_created']]
        adobe_files = [r for r in self.batch_results if r['adobe_created']]
        failed_files = [r for r in self.batch_results if 'error' in r]

        self._append_text_to_console(f"✅ SRT files created: {len(srt_files)}\n")
        self._append_text_to_console(f"✅ Adobe Static Transcript files created: {len(adobe_files)}\n")
        if failed_files:
            self._append_text_to_console(f"❌ Failed files: {len(failed_files)}\n")
            for failed in failed_files:
                self._append_text_to_console(f"   - {os.path.basename(failed['file_path'])}: {failed.get('error', 'Unknown error')}\n")

        # Show batch success dialog
        self.show_batch_completion_dialog()

    def show_batch_completion_dialog(self):
        """Show success dialog after batch completion with auto-upload option"""

        class BatchCompletionDialog(QDialog):
            """Custom dialog with close confirmation"""
            def __init__(self, parent):
                super().__init__(parent)
                self.proceeded = False

            def closeEvent(self, event):
                """Warn user if closing without proceeding"""
                if not self.proceeded:
                    msg = QMessageBox(self)
                    msg.setIcon(QMessageBox.Icon.Warning)
                    msg.setWindowTitle("Manual Upload Required")

                    warning_text = "If you don't press \"Yes, Proceed\", you will need to manually:\n\n"
                    warning_text += "• Upload .ref file(s) via File → Upload Reference Transcript\n"
                    warning_text += "• Import SRT file(s) into Process Captions tab\n"
                    warning_text += "\nAre you sure you want to close?"

                    msg.setText(warning_text)
                    msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                    msg.setDefaultButton(QMessageBox.StandardButton.No)

                    if msg.exec() == QMessageBox.StandardButton.Yes:
                        event.accept()
                    else:
                        event.ignore()
                else:
                    event.accept()

        dialog = BatchCompletionDialog(self)
        dialog.setWindowTitle("Batch Transcription Completed!")
        dialog.setFixedSize(580, 480)
        dialog.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)

        # Main layout
        main_layout = QVBoxLayout(dialog)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        # Success icon and title section
        header_layout = QHBoxLayout()
        header_layout.setSpacing(15)

        # Success icon (checkmark)
        icon_label = QLabel("🎉")
        icon_label.setStyleSheet("""
            QLabel {
                color: #4CAF50;
                font-size: 48px;
                font-weight: bold;
                min-width: 60px;
                max-width: 60px;
            }
        """)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header_layout.addWidget(icon_label)

        # Title and subtitle
        title_layout = QVBoxLayout()
        title_layout.setSpacing(5)

        title = QLabel("Transcription Completed!")
        title.setStyleSheet("""
            QLabel {
                color: #333333;
                font-size: 18px;
                font-weight: bold;
            }
        """)

        subtitle = QLabel(f"Successfully processed {len(self.batch_results)} files")
        subtitle.setStyleSheet("""
            QLabel {
                color: #666666;
                font-size: 14px;
            }
        """)

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        header_layout.addLayout(title_layout)

        main_layout.addLayout(header_layout)

        # Results summary
        srt_files = [r for r in self.batch_results if r['srt_created']]
        adobe_files = [r for r in self.batch_results if r['adobe_created']]

        summary_text = f"✅ Successfully Generated:\n"
        if srt_files:
            summary_text += f"• {len(srt_files)} Caption File(s) (.srt) - The generated captions used by Scriptoria as a foundation in Process Captions tab\n"
        if adobe_files:
            summary_text += f"• {len(adobe_files)} Reference file pair(s) (.ref + .json) - Links Scriptoria and Premiere Pro\n"
        else:
            summary_text += f"• {len(self.batch_results)} Reference file(s) (.ref) - For word-level accuracy in Scriptoria\n"

        summary_text += f"\n📋 Recommended Next Steps:\n\n"
        if srt_files:
            summary_text += "1. Click \"Yes, Proceed\" to auto-upload .ref and .srt to Scriptoria\n\n"
        else:
            summary_text += "1. Click \"Yes, Proceed\" to auto-upload .ref to Scriptoria\n\n"
        if adobe_files:
            summary_text += f"2. In Premiere Pro, import the .json Static Transcript(s):\n"
            summary_text += "   • Text → Transcript → Import Static Transcript\n"
            summary_text += "   • Select each .json file from the output folder(s)\n"
            summary_text += "   • This links your Premiere Pro sequences with Scriptoria\n\n"
        summary_text += "Ready to proceed?"

        message_label = QLabel(summary_text)
        message_label.setStyleSheet("""
            QLabel {
                color: #444444;
                font-size: 13px;
                background-color: #F8F9FA;
                padding: 20px;
                border: 1px solid #E9ECEF;
                border-radius: 8px;
            }
        """)
        message_label.setWordWrap(True)
        message_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        main_layout.addWidget(message_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(12)

        # Open File Locations button (left side)
        open_location_btn = QPushButton("📁 Open File Locations")
        open_location_btn.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: 500;
                min-width: 160px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:pressed {
                background-color: #545b62;
            }
        """)
        open_location_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_location_btn.clicked.connect(lambda: self.open_batch_file_locations())
        button_layout.addWidget(open_location_btn)

        button_layout.addStretch()

        # Yes, Proceed button (primary action)
        yes_btn = QPushButton("Yes, Proceed")
        yes_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                font-size: 13px;
                font-weight: 600;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        yes_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        yes_btn.setDefault(True)
        yes_btn.clicked.connect(lambda: self.handle_batch_proceed_button(dialog))
        button_layout.addWidget(yes_btn)

        main_layout.addLayout(button_layout)

        # Show dialog
        dialog.exec()

    def open_batch_file_locations(self):
        """Open file locations for all batch results"""
        opened_dirs = set()
        for result in self.batch_results:
            output_dir = result['output_dir']
            if output_dir not in opened_dirs:
                try:
                    if sys.platform == "win32":
                        os.startfile(os.path.normpath(output_dir))
                    elif sys.platform == "darwin":
                        subprocess.run(["open", output_dir])
                    else:
                        subprocess.run(["xdg-open", output_dir])
                    opened_dirs.add(output_dir)
                except Exception as e:
                    self._append_text_to_console(f"Warning: Could not open directory {output_dir}: {e}\n")

    def auto_upload_batch_results(self):
        """Auto-upload all batch results to Scriptoria"""
        try:
            self._activate_cleaner_tab()
            # Upload SRT files to cleaner tab (SRT is always created for input, regardless of export checkbox)
            cleaner_files_uploaded = 0
            for result in self.batch_results:
                # Always try SRT first (it's always created for the cleaner input tab)
                srt_path = self.get_expected_srt_path(result['file_path'], result['output_dir'])
                if os.path.exists(srt_path):
                    self.auto_upload_srt_file(result['output_dir'], input_file=result['file_path'])
                    cleaner_files_uploaded += 1
                else:
                    # Fallback to REF if SRT doesn't exist for some reason
                    txt_path = self.get_expected_txt_path(result['file_path'], result['output_dir'])
                    if os.path.exists(txt_path):
                        self.auto_upload_txt_file(result['output_dir'], input_file=result['file_path'])
                        cleaner_files_uploaded += 1

            # Upload all REF files as reference transcripts (separate from cleaner input)
            transcript_files_uploaded = 0
            for result in self.batch_results:
                txt_path = self.get_expected_txt_path(result['file_path'], result['output_dir'])
                if os.path.exists(txt_path):
                    self.auto_upload_reference_transcript(result['output_dir'], input_file=result['file_path'])
                    transcript_files_uploaded += 1

            if cleaner_files_uploaded > 0:
                self._append_text_to_console(f"✅ Auto-uploaded {cleaner_files_uploaded} files to Process Captions tab\n")
            self._append_text_to_console(f"✅ Auto-uploaded {transcript_files_uploaded} reference transcript REF files as reference transcripts\n")

            # Clear batch after successful upload
            self.batch_files.clear()
            self.batch_results.clear()
            self.batch_mode_active = False
            self._batch_index = 0
            self.update_batch_display()

        except Exception as e:
            self._append_text_to_console(f"Warning: Batch auto-upload encountered errors: {str(e)}\n")

    def get_expected_srt_path(self, input_file, output_dir):
        """Get expected SRT file path for a given input file"""
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        return os.path.join(output_dir, f"{base_name}.srt")

    def get_expected_adobe_path(self, input_file, output_dir):
        """Get expected Adobe Static Transcript file path for a given input file"""
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        return os.path.join(output_dir, f"{base_name}_adobe.json")

    def get_expected_txt_path(self, input_file, output_dir):
        """Get expected reference transcript REF file path for a given input file"""
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        return os.path.join(output_dir, f"{base_name}_reference_transcript.ref")

    def create_paragraph_segments(self, whisper_data, pause_threshold=3.0, min_words_per_paragraph=10):
        """
        Create paragraph-formatted segments from Whisper JSON data.

        Args:
            whisper_data: Original Whisper JSON data
            pause_threshold: Minimum pause duration (in seconds) to create new paragraph
            min_words_per_paragraph: Minimum words required per paragraph

        Returns:
            List of paragraph segments with combined text and timing
        """
        if not whisper_data.get('segments'):
            return []

        paragraph_segments = []
        current_paragraph = {
            'text': '',
            'words': [],
            'start': None,
            'end': None,
            'speaker': None,
            'word_count': 0
        }

        for i, segment in enumerate(whisper_data['segments']):
            # Get segment text and timing
            segment_text = segment.get('text', '').strip()
            segment_start = segment.get('start', 0.0)
            segment_end = segment.get('end', segment_start)
            segment_speaker = segment.get('speaker', 'SPEAKER_00')
            segment_words = segment.get('words', [])

            if not segment_text:
                continue

            # Check if we should start a new paragraph
            should_start_new_paragraph = False

            if current_paragraph['text']:  # Not the first segment
                # Check for speaker change
                if segment_speaker != current_paragraph['speaker']:
                    should_start_new_paragraph = True
                # Check for long pause
                elif current_paragraph['end'] and (segment_start - current_paragraph['end']) > pause_threshold:
                    should_start_new_paragraph = True
                # Check if current paragraph is getting too long (>50 words) and this segment ends with punctuation
                elif (current_paragraph['word_count'] > 50 and
                      segment_text.rstrip().endswith(('.', '!', '?'))):
                    should_start_new_paragraph = True

            # If we should start a new paragraph and current paragraph meets minimum requirements
            if should_start_new_paragraph and current_paragraph['word_count'] >= min_words_per_paragraph:
                paragraph_segments.append(current_paragraph.copy())
                current_paragraph = {
                    'text': '',
                    'words': [],
                    'start': None,
                    'end': None,
                    'speaker': None,
                    'word_count': 0
                }

            # Add this segment to current paragraph
            if not current_paragraph['text']:  # First segment in paragraph
                current_paragraph['start'] = segment_start
                current_paragraph['speaker'] = segment_speaker
                current_paragraph['text'] = segment_text
            else:
                # Add space if previous text doesn't end with punctuation or current doesn't start with punctuation
                if (not current_paragraph['text'].rstrip().endswith(('.', '!', '?', ':', ';')) and
                    not segment_text.lstrip().startswith(('.', ',', '!', '?', ':', ';'))):
                    current_paragraph['text'] += ' '
                current_paragraph['text'] += segment_text

            current_paragraph['end'] = segment_end
            current_paragraph['words'].extend(segment_words)
            current_paragraph['word_count'] += len(segment_words)

            # IMPORTANT: Keep track of segments that made up this paragraph
            # This helps preserve the relationship between text and words
            if 'source_segments' not in current_paragraph:
                current_paragraph['source_segments'] = []
            current_paragraph['source_segments'].append(segment)

        # Add the final paragraph if it has content
        if current_paragraph['text'] and current_paragraph['word_count'] >= min_words_per_paragraph:
            paragraph_segments.append(current_paragraph)
        elif current_paragraph['text'] and paragraph_segments:
            # If final paragraph is too short, merge it with the last paragraph
            paragraph_segments[-1]['text'] += ' ' + current_paragraph['text']
            paragraph_segments[-1]['end'] = current_paragraph['end']
            paragraph_segments[-1]['words'].extend(current_paragraph['words'])
        elif current_paragraph['text']:
            # If it's the only paragraph, keep it regardless of length
            paragraph_segments.append(current_paragraph)

        return paragraph_segments

    def _words_to_plain_text(self, words: list[dict]) -> str:
        """Reconstruct plain text from Adobe word tokens, preserving speaker labels."""
        pieces: list[str] = []
        pending_prefix = ""

        def is_opening_punct(tok: str) -> bool:
            return tok in {"(", "[", "{", '"', "'", "\u201c", "\u2018", "«"}

        for w in words:
            text = (w.get('original_text') or w.get('text') or '').strip()
            if not text:
                continue

            # Speaker labels are intentionally omitted for reference transcripts
            is_punct = (w.get('type') == 'punctuation')
            if is_punct:
                if is_opening_punct(text):
                    pending_prefix += text
                else:
                    if not pieces:
                        pending_prefix += text
                    else:
                        pieces[-1] += text
                continue

            if pending_prefix:
                text = pending_prefix + text
                pending_prefix = ""
            pieces.append(text)

        if pending_prefix:
            if pieces:
                pieces[-1] += pending_prefix
            else:
                pieces.append(pending_prefix)

        return ' '.join(pieces)

    def _normalize_whisperx_speakers(self, data: dict) -> dict:
        """Normalize WhisperX speaker IDs into SPEAKER_01, SPEAKER_02, ... and ensure word end times."""
        try:
            if not data or 'segments' not in data:
                return data

            normalized = copy.deepcopy(data)
            alias_map: dict[str, str] = {}

            def alias(raw: Optional[str]) -> Optional[str]:
                if not raw:
                    return None
                if raw not in alias_map:
                    alias_map[raw] = f"SPEAKER_{len(alias_map) + 1:02d}"
                return alias_map[raw]

            for segment in normalized.get('segments', []):
                seg_raw = segment.get('speaker') or segment.get('speaker_id')
                seg_alias = alias(seg_raw)
                if seg_alias:
                    segment['speaker'] = seg_alias

                words = segment.get('words', []) or []
                seg_end = float(segment.get('end', segment.get('start', 0.0)))
                for idx, word in enumerate(words):
                    word_alias = alias(word.get('speaker') or seg_raw)
                    if word_alias:
                        word['speaker'] = word_alias
                    try:
                        start_val = float(word.get('start', 0.0))
                    except Exception:
                        start_val = 0.0
                    end_val = word.get('end')
                    if end_val is not None:
                        try:
                            end_val = float(end_val)
                        except Exception:
                            end_val = None
                    if end_val is None or end_val <= start_val:
                        next_start = None
                        if idx + 1 < len(words):
                            try:
                                next_start = float(words[idx + 1].get('start', start_val))
                            except Exception:
                                next_start = None
                        if next_start is not None and next_start > start_val:
                            end_val = next_start
                        elif seg_end > start_val:
                            end_val = seg_end
                        else:
                            end_val = start_val
                    word['end'] = end_val
                    duration_val = max(end_val - start_val, 0.0)
                    word['duration'] = duration_val

            return normalized
        except Exception:
            return data

    def _build_srt_text(self, segments: list[dict]) -> str:
        """Generate SRT formatted text from normalized segments."""
        lines: list[str] = []
        for idx, segment in enumerate(segments or [], 1):
            start_time = self.format_timestamp(segment.get('start', 0.0))
            end_time = self.format_timestamp(segment.get('end', segment.get('start', 0.0)))
            text = (segment.get('text') or '').strip()
            speaker = segment.get('speaker')
            if speaker:
                text = f"[{speaker}] {text}" if text else f"[{speaker}]"
            lines.append(f"{idx}\n{start_time} --> {end_time}\n{text}\n")
        return "\n".join(lines).strip()

    def generate_paragraph_txt(self, input_file, output_dir, adobe_data, paragraph_form: bool = False, word_gap_seconds: float = 3.0):
        """
        Generate a TXT file from Adobe Static Transcript data.
        - If paragraph_form is False: output a single continuous block (no timestamps).
        - If paragraph_form is True: group into paragraphs and prefix each with a timestamp, new paragraph when gap between consecutive words exceeds word_gap_seconds.

        Args:
            input_file: Path to the original input file
            output_dir: Output directory for the TXT file
            adobe_data: Adobe Static Transcript data produced by converter

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Generate TXT file path
            txt_file = self.get_expected_txt_path(input_file, output_dir)

            # Collect words across all segments
            all_words: list[dict] = []
            for seg in adobe_data.get('segments', []):
                seg_speaker = seg.get('speaker')
                ws = seg.get('words', [])
                if ws:
                    for w in ws:
                        if seg_speaker and not w.get('speaker'):
                            w['speaker'] = seg_speaker
                        try:
                            start_val = float(w.get('start', 0.0))
                            duration_val = float(w.get('duration', 0.0))
                            w['end'] = start_val + duration_val
                        except Exception:
                            w['end'] = float(w.get('start', 0.0))
                    all_words.extend(ws)

            if not all_words:
                self._append_text_to_console("Warning: No words found for TXT generation\n")
                return False

            # Build TXT content
            if not paragraph_form:
                # Single continuous block
                text = self._words_to_plain_text(all_words)
                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                self._append_text_to_console(f"✓ Continuous TXT saved: {txt_file}\n")
                return True
            else:
                # Paragraph Form with timestamps, break on word_gap_seconds
                paragraphs: list[list[dict]] = []
                current: list[dict] = []
                last_end: Optional[float] = None
                current_speaker: Optional[str] = None
                for w in all_words:
                    start = float(w.get('start', 0.0))
                    end = float(w.get('end', start + float(w.get('duration', 0.0))))
                    speaker = w.get('speaker')
                    gap_trigger = last_end is not None and (start - last_end) > float(word_gap_seconds)
                    speaker_trigger = current and current_speaker and speaker and speaker != current_speaker
                    if (gap_trigger or speaker_trigger) and current:
                        paragraphs.append(current)
                        current = []
                    current.append(w)
                    last_end = end
                    if speaker:
                        current_speaker = speaker
                if current:
                    paragraphs.append(current)

                # Build lines with timestamp ranges
                lines: list[str] = []
                for para_words in paragraphs:
                    if not para_words:
                        continue
                    para_text = self._words_to_plain_text(para_words)
                    start_time = para_words[0].get('start')
                    end_time = para_words[-1].get('end')
                    timestamp_range = self.format_srt_timestamp_range(start_time, end_time)
                    lines.append(f"{timestamp_range}\n{para_text}")

                with open(txt_file, 'w', encoding='utf-8') as f:
                    f.write("\n\n".join(lines))

                self._append_text_to_console(f"✓ Paragraph Form TXT saved: {txt_file}\n")
                return True

        except Exception as e:
            self._append_text_to_console(f"Error generating continuous TXT: {str(e)}\n")
            return False

    def format_timestamp(self, seconds):
        """Format seconds into standard SRT timestamp format HH:MM:SS,mmm"""
        if seconds is None:
            return "00:00:00,000"

        total_seconds = int(seconds)
        milliseconds = int((seconds - total_seconds) * 1000)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60

        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

    def format_srt_timestamp_range(self, start_seconds, end_seconds):
        """Format start and end seconds into SRT timestamp range format"""
        start_ts = self.format_timestamp(start_seconds)
        end_ts = self.format_timestamp(end_seconds)
        return f"{start_ts} --> {end_ts}"

    def _get_audio_duration(self, file_path):
        """Get audio/video duration in seconds using ffprobe"""
        import subprocess
        import shutil

        try:
            # Try to find ffprobe
            ffprobe_path = shutil.which('ffprobe')
            if not ffprobe_path:
                # Check common locations
                common_paths = [
                    r'C:\ffmpeg\bin\ffprobe.exe',
                    r'C:\Program Files\ffmpeg\bin\ffprobe.exe',
                    '/usr/bin/ffprobe',
                    '/usr/local/bin/ffprobe'
                ]
                for path in common_paths:
                    if os.path.exists(path):
                        ffprobe_path = path
                        break

            if not ffprobe_path:
                return None

            # Run ffprobe to get duration
            # Hide console window on Windows to prevent flash
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            result = subprocess.run(
                [ffprobe_path, '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', file_path],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=creationflags
            )

            if result.returncode == 0 and result.stdout.strip():
                duration = float(result.stdout.strip())
                return duration
        except Exception:
            pass

        return None

    def run_transcription(self, input_file):
        """Run the transcription process"""
        self.current_input_file = input_file
        self._active_run_input_file = input_file
        self._active_run_output_dir = self._compute_output_dir_for_input(input_file)

        command = self.build_command(input_file)
        if not command:
            self._active_run_input_file = None
            self._active_run_output_dir = None
            return

        # Don't interfere with process cleanup in batch mode
        # The process should be None by the time we get here in batch processing

        # Clear console and show command (but preserve batch progress in batch mode)
        if not self.batch_mode_active:
            self.output_text.clear()

        # Start the progress panel with file info
        # Determine if diarization is enabled based on active tab and settings
        show_diarization = False
        current_tab_index = self.tab_widget.currentIndex()
        if current_tab_index == self.whisperx_tab_index:
            # WhisperX tab - check whisperx diarization checkbox
            if hasattr(self, 'whisperx_enable_diarization'):
                show_diarization = self.whisperx_enable_diarization.isChecked()
        else:
            # Faster-Whisper-XXL tab - check if using WhisperX engine with diarization
            if hasattr(self, 'engine_whisperx') and self.engine_whisperx.isChecked():
                if hasattr(self, 'enable_diarization'):
                    show_diarization = self.enable_diarization.isChecked()
            # Faster-Whisper-XXL doesn't support diarization, so keep False

        if hasattr(self, 'progress_panel') and self.progress_panel:
            # Try to get audio duration for accurate progress tracking
            audio_duration = self._get_audio_duration(input_file)
            if audio_duration:
                self.progress_panel.set_audio_duration(audio_duration)
            self.progress_panel.start_processing(input_file, show_diarization=show_diarization)

        self._append_text_to_console(f"Running transcription...\nFile: {input_file}\n" + "="*50 + "\n")

        # Show the actual command for debugging
        command_str = " ".join([f'"{arg}"' if " " in arg else arg for arg in command])
        self._append_text_to_console(f"Command: {command_str}\n" + "="*50 + "\n")

        # Reset post-processing flags for this run
        self.postprocess_started = False
        self.postprocess_done = False
        self._adobe_converted = False

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.stop_requested = False
        self.transcription_completed_successfully = False

        # Start process
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.on_finished)
        self.process.errorOccurred.connect(self.on_process_error)

        try:
            hf_cache_dir = get_hf_cache_directory()
            env = QProcessEnvironment.systemEnvironment()
            if hf_cache_dir:
                env.insert("HF_HOME", hf_cache_dir)
                env.insert("HF_HUB_CACHE", hf_cache_dir)
                env.insert("HUGGINGFACE_HUB_CACHE", hf_cache_dir)
                env.insert("HF_HUB_DISABLE_SYMLINKS", "1")
                env.insert("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            if hasattr(self, "whisperx_hf_token"):
                token_value = self.whisperx_hf_token.text().strip()
                if token_value:
                    env.insert("HF_TOKEN", token_value)
            extra_env = getattr(self, "_whisperx_execution_env", None)
            if extra_env:
                for key, value in extra_env.items():
                    if value is None:
                        continue
                    value_str = str(value)
                    if key.upper() == "PYTHONPATH":
                        existing = env.value("PYTHONPATH")
                        if existing:
                            value_str = f"{value_str}{os.pathsep}{existing}"
                    env.insert(key, value_str)
            self.process.setProcessEnvironment(env)
        except Exception:
            pass

        self.process.start(command[0], command[1:])

    def _get_whisperx_execution_context(self, purpose: str = "cli"):
        """Return interpreter path and env overrides for WhisperX execution."""
        app_dir = get_app_directory()
        venv_dir = os.path.join(app_dir, "whisperx_env")

        extra_env = {}

        # Check for portable Python first (python.exe directly in whisperx_env root)
        # This is the preferred portable structure that works regardless of drive letter
        portable_python = os.path.join(venv_dir, "python.exe") if sys.platform == "win32" else os.path.join(venv_dir, "bin", "python3")
        portable_site_packages = os.path.join(venv_dir, "Lib", "site-packages")

        if os.path.exists(portable_python):
            # Set PYTHONPATH for portable env to work regardless of drive letter
            if os.path.exists(portable_site_packages):
                extra_env["PYTHONPATH"] = portable_site_packages
            self.whisperx_venv_python = portable_python
            return portable_python, extra_env

        # Check for new venv structure (nested venv)
        venv_python = os.path.join(venv_dir, "venv", "Scripts", "python.exe") if sys.platform == "win32" else os.path.join(venv_dir, "venv", "bin", "python")
        if os.path.exists(venv_python):
            self.whisperx_venv_python = venv_python
            return venv_python, extra_env

        # Check for legacy embedded Python (backward compatibility)
        embedded_root = os.path.join(venv_dir, "python_embedded")
        embedded_site_packages = os.path.join(embedded_root, "Lib", "site-packages")
        embedded_python = os.path.join(embedded_root, "python.exe")

        is_frozen = getattr(sys, 'frozen', False)

        # Legacy embedded Python handling (for backward compatibility)
        if is_frozen and os.path.exists(embedded_site_packages):
            pythonpath_entries = [embedded_root, embedded_site_packages]
            pythonpath = os.pathsep.join([p for p in pythonpath_entries if p and os.path.exists(p)])
            if pythonpath:
                extra_env["PYTHONPATH"] = pythonpath

            if purpose == "cli":
                py_exe = sys.executable
                return py_exe, extra_env

            if os.path.exists(embedded_python):
                return embedded_python, extra_env

        if not hasattr(self, 'whisperx_venv_python') or not self.whisperx_venv_python:
            # Fall back to legacy embedded Python
            if os.path.exists(embedded_python):
                self.whisperx_venv_python = embedded_python
            # Fall back to traditional venv structure
            elif sys.platform == "win32":
                self.whisperx_venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
            else:
                self.whisperx_venv_python = os.path.join(venv_dir, "bin", "python")

        py_exe = getattr(self, 'whisperx_venv_python', None)
        return py_exe, extra_env

    def build_command(self, input_file):
        """Build the command for transcription (engine-aware)"""
        self._whisperx_execution_env = None
        self._whisperx_execution_python = None

        # Check if WhisperX tab is active
        current_tab_index = self.tab_widget.currentIndex()
        using_whisperx = (current_tab_index == self.whisperx_tab_index)

        # Build for WhisperX engine (Python CLI) or Faster-Whisper-XXL (binary)
        if using_whisperx:
            # CLI script path
            cli_path = os.path.join(get_app_directory(), 'Data', 'whisperx_cli.py')
            if not os.path.exists(cli_path):
                QMessageBox.critical(self, "Error", f"WhisperX CLI not found: {cli_path}")
                return None

            py_exe, extra_env = self._get_whisperx_execution_context(purpose="cli")
            self._whisperx_execution_env = extra_env or None
            self._whisperx_execution_python = py_exe

            if not (py_exe and os.path.exists(py_exe)):
                QMessageBox.critical(self, "WhisperX Not Available", "Please install WhisperX from the WhisperX tab first.")
                return None

            cmd = [py_exe]
            if getattr(sys, 'frozen', False) and py_exe == sys.executable:
                cmd.extend(["--scriptoria-run-python", cli_path])
            else:
                cmd.append(cli_path)
            cmd.append(input_file)

            # Use WhisperX tab settings
            selected_device = self.whisperx_device_combo.currentText()
            selected_compute = self.whisperx_compute_combo.currentText()

            options = {
                "-m": self.whisperx_model_combo.currentText(),
                "--task": self.whisperx_task_combo.currentText(),
                "-l": self.whisperx_language_combo.currentText() if self.whisperx_language_combo.currentText() != 'auto' else None,
                "--compute_type": selected_compute,
                "--device": selected_device,
                "--output_dir": self.whisperx_output_dir.text() if self.whisperx_output_dir.text() else os.path.dirname(input_file)
            }

            # Output formats from WhisperX tab
            formats = []

            # Always create SRT files (checkbox is always checked but hidden)
            if self.whisperx_export_srt.isChecked():
                formats.append("srt")

            if self.whisperx_export_adobe.isChecked():
                formats.append("json")
                options["--word_timestamps"] = "True"

            # VAD settings
            if self.whisperx_vad_filter.isChecked():
                options["--vad_filter"] = "True"
                options["--vad_method"] = self.whisperx_vad_method.currentText()

            # Diarization (WhisperX-specific)
            if self.whisperx_enable_diarization.isChecked():
                options["--diarize"] = "True"
                if self.whisperx_hf_token.text().strip():
                    options["--hf_token"] = self.whisperx_hf_token.text().strip()
                if hasattr(self, 'whisperx_num_speakers'):
                    choice = self.whisperx_num_speakers.currentText()
                    if choice and choice.lower() != "auto":
                        options["--num_speakers"] = choice

            # Ensure Hugging Face cache/token env vars apply even for manual reiterations
            for key, value in options.items():
                if value is not None:
                    cmd.extend([key, str(value)])

            if formats:
                cmd.extend(["--output_format"] + formats)
            else:
                cmd.extend(["--output_format", "srt"])

            return cmd
        else:
            # Faster-Whisper-XXL path (existing)
            if not self.executable_path or not os.path.exists(self.executable_path):
                QMessageBox.critical(self, "Error", f"Executable not found: {self.executable_path}")
                return None

            if not os.access(self.executable_path, os.X_OK):
                QMessageBox.critical(self, "Error", f"Executable does not have execute permissions: {self.executable_path}")
                return None

            cmd = [self.executable_path, input_file]

            # Map any legacy/alternate model labels to names recognized by the binary
            model_name = self.model_combo.currentText()
            legacy_map = {
                'faster-distil-whisper-large-v3.5': 'distil-large-v3.5',
                'faster-whisper-large-v3-turbo': 'large-v3-turbo'
            }
            model_name = legacy_map.get(model_name, model_name)

            # Use selected device and compute type
            selected_device = self.device_combo.currentText()
            effective_compute_type = self._get_compute_type_for_device(selected_device, self.compute_combo.currentText())

            output_dir = self._compute_output_dir_for_input(input_file)
            options = {
                "-m": model_name,
                "--task": self.task_combo.currentText(),
                "-l": self.language_combo.currentText() if self.language_combo.currentText() != 'auto' else None,
                "--compute_type": effective_compute_type,
                "--device": selected_device,
                "--output_dir": output_dir
            }

            formats = []

            # Always create SRT files (checkbox is always checked but hidden)
            if self.export_srt.isChecked():
                formats.append("srt")

            if self.export_adobe.isChecked():
                formats.append("json")
                options["--word_timestamps"] = "True"

            if self.vad_filter.isChecked():
                options["--vad_filter"] = "True"
                options["--vad_method"] = self.vad_method.currentText()

            # Diarization disabled

            for key, value in options.items():
                if value is not None:
                    cmd.extend([key, str(value)])

            if formats:
                cmd.extend(["--output_format"] + formats)
            else:
                cmd.extend(["--output_format", "srt"])

            return cmd

    def validate_hf_token(self):
        """Validate the Hugging Face token and ensure diarization model access."""
        try:
            token = self.whisperx_hf_token.text().strip() if hasattr(self, 'whisperx_hf_token') else ''
            if not token:
                QMessageBox.warning(self, "Missing Token", "Enter a Hugging Face token before validating.")
                return

            py_exe, extra_env = self._get_whisperx_execution_context(purpose="utility")
            if not (py_exe and os.path.exists(py_exe)):
                QMessageBox.warning(self, "Runtime Missing", "WhisperX runtime not found. Install WhisperX first.")
                return

            self.whisperx_validate_token_btn.setEnabled(False)
            if hasattr(self, 'whisperx_hf_status_label'):
                self.whisperx_hf_status_label.setText("Validating Hugging Face token...")
                self.whisperx_hf_status_label.setStyleSheet("color: #666; font-size: 11px;")

            self._hf_validate_proc = QProcess(self)
            tools_script = os.path.join(get_app_directory(), 'Data', 'whisperx_hf_tools.py')
            if not os.path.exists(tools_script):
                raise FileNotFoundError(f"Hugging Face helper script missing: {tools_script}")

            env = QProcessEnvironment.systemEnvironment()
            hf_cache_dir = get_hf_cache_directory()
            if hf_cache_dir:
                env.insert("HF_HOME", hf_cache_dir)
                env.insert("HF_HUB_CACHE", hf_cache_dir)
                env.insert("HUGGINGFACE_HUB_CACHE", hf_cache_dir)
                env.insert("HF_HUB_DISABLE_SYMLINKS", "1")
                env.insert("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            env.insert("HF_TOKEN", token)
            if extra_env:
                for key, value in extra_env.items():
                    if value is None:
                        continue
                    value_str = str(value)
                    if key.upper() == "PYTHONPATH":
                        existing = env.value("PYTHONPATH")
                        if existing:
                            value_str = f"{value_str}{os.pathsep}{existing}"
                    env.insert(key, value_str)
            self._hf_validate_proc.setProcessEnvironment(env)
            self._hf_validate_proc.finished.connect(self.on_hf_validate_finished)
            command = [py_exe]
            if getattr(sys, 'frozen', False) and py_exe == sys.executable:
                command.extend(["--scriptoria-run-python", tools_script])
            else:
                command.append(tools_script)
            command.extend(["validate", "--token", token])
            self._hf_validate_proc.start(command[0], command[1:])
        except Exception as exc:
            if hasattr(self, 'whisperx_validate_token_btn'):
                self.whisperx_validate_token_btn.setEnabled(True)
            if hasattr(self, 'whisperx_hf_status_label'):
                self.whisperx_hf_status_label.setText(f"Validation failed: {exc}")
                self.whisperx_hf_status_label.setStyleSheet("color: #d48b00; font-size: 11px;")
            QMessageBox.critical(self, "Token Validation Failed", str(exc))

    def on_hf_validate_finished(self, exit_code, status):
        try:
            stdout = bytes(self._hf_validate_proc.readAllStandardOutput()).decode('utf-8', errors='replace')
            stderr = bytes(self._hf_validate_proc.readAllStandardError()).decode('utf-8', errors='replace')
        except Exception:
            stdout = ""
            stderr = ""

        if hasattr(self, 'whisperx_validate_token_btn'):
            self.whisperx_validate_token_btn.setEnabled(True)

        result = {}
        try:
            result = json.loads(stdout.strip() or "{}")
        except Exception:
            result = {}

        if result.get("ok"):
            user = result.get("user") or "your account"
            if result.get("model_access"):
                message = f"Token validated for {user}. Diarization model access confirmed."
                color = "green"
                QMessageBox.information(self, "Token Validated", message)
                self.prefetch_diarization_model(token=self.whisperx_hf_token.text().strip())
            else:
                detail = result.get("model_message") or "Access to pyannote/speaker-diarization-3.1 is restricted."
                message = f"Token valid for {user}, but diarization model access is blocked."
                color = "#d48b00"
                QMessageBox.warning(
                    self,
                    "Model Access Required",
                    f"{message}\n\nDetails:\n{detail}\n\nAccept the license at:\nhttps://huggingface.co/pyannote/speaker-diarization-3.1"
                )
        else:
            error_text = result.get("error") or stderr or "Unknown error."
            message = "Token validation failed. See details."
            color = "#d48b00"
            QMessageBox.critical(self, "Token Validation Failed", f"Could not validate the token.\n\nDetails:\n{error_text}")

        if hasattr(self, 'whisperx_hf_status_label'):
            self.whisperx_hf_status_label.setText(message)
            self.whisperx_hf_status_label.setStyleSheet(f"color: {color}; font-size: 11px;")

        self._hf_validate_proc = None

    def prefetch_diarization_model(self, token: str):
        """Download the diarization model into the local HF cache."""
        if not token:
            return
        try:
            if not hasattr(self, '_hf_prefetch_proc'):
                self._hf_prefetch_proc = None
            if self._hf_prefetch_proc is not None:
                return  # already running

            py_exe, extra_env = self._get_whisperx_execution_context(purpose="utility")
            if not (py_exe and os.path.exists(py_exe)):
                return

            tools_script = os.path.join(get_app_directory(), 'Data', 'whisperx_hf_tools.py')
            if not os.path.exists(tools_script):
                return

            self._hf_prefetch_proc = QProcess(self)
            env = QProcessEnvironment.systemEnvironment()
            hf_cache_dir = get_hf_cache_directory()
            if hf_cache_dir:
                env.insert("HF_HOME", hf_cache_dir)
                env.insert("HF_HUB_CACHE", hf_cache_dir)
                env.insert("HUGGINGFACE_HUB_CACHE", hf_cache_dir)
                env.insert("HF_HUB_DISABLE_SYMLINKS", "1")
                env.insert("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            env.insert("HF_TOKEN", token)
            if extra_env:
                for key, value in extra_env.items():
                    if value is None:
                        continue
                    value_str = str(value)
                    if key.upper() == "PYTHONPATH":
                        existing = env.value("PYTHONPATH")
                        if existing:
                            value_str = f"{value_str}{os.pathsep}{existing}"
                    env.insert(key, value_str)
            self._hf_prefetch_proc.setProcessEnvironment(env)
            self._hf_prefetch_proc.finished.connect(self.on_hf_prefetch_finished)
            command = [py_exe]
            if getattr(sys, 'frozen', False) and py_exe == sys.executable:
                command.extend(["--scriptoria-run-python", tools_script])
            else:
                command.append(tools_script)
            command.extend(["prefetch", "--token", token])
            self._hf_prefetch_proc.start(command[0], command[1:])
            if hasattr(self, 'whisperx_hf_status_label'):
                self.whisperx_hf_status_label.setText("Downloading diarization model (background)...")
                self.whisperx_hf_status_label.setStyleSheet("color: #666; font-size: 11px;")
        except Exception as exc:
            self._hf_prefetch_proc = None
            if hasattr(self, 'whisperx_hf_status_label'):
                self.whisperx_hf_status_label.setText(f"Diarization download failed: {exc}")
                self.whisperx_hf_status_label.setStyleSheet("color: #d48b00; font-size: 11px;")

    def on_hf_prefetch_finished(self, exit_code, status):
        try:
            stdout = bytes(self._hf_prefetch_proc.readAllStandardOutput()).decode('utf-8', errors='replace')
        except Exception:
            stdout = ""
        try:
            stderr = bytes(self._hf_prefetch_proc.readAllStandardError()).decode('utf-8', errors='replace')
        except Exception:
            stderr = ""

        result = {}
        try:
            result = json.loads(stdout.strip() or "{}")
        except Exception:
            result = {}

        if result.get("ok"):
            path = result.get("path") or get_hf_cache_directory()
            message = f"Diarization model downloaded to cache ({path})."
            color = "green"
        else:
            message = f"Diarization download failed. {result.get('error') or stderr or ''}"
            color = "#d48b00"

        if hasattr(self, 'whisperx_hf_status_label'):
            self.whisperx_hf_status_label.setText(message)
            self.whisperx_hf_status_label.setStyleSheet(f"color: {color}; font-size: 11px;")

        self._hf_prefetch_proc = None

    def stop_transcription(self):
        """Stop the transcription process"""
        self.stop_requested = True
        if self.process and self.process.state() == QProcess.ProcessState.Running:
            self._append_text_to_console("\nTerminating process...\n")
            self.process.terminate()
            if not self.process.waitForFinished(2000):
                self.process.kill()

    def handle_stdout(self):
        """Handle stdout from process"""
        if not self.process:
            return

        data = self.process.readAllStandardOutput()
        text = bytes(data).decode('utf-8', errors='replace')
        self._append_text_to_console(text)

        # Debug: Log any output containing "Removed" to see what's happening
        if "Removed" in text or "removed" in text:
            import time
            current = time.time()
            ms = int((current - int(current)) * 1000)
            timestamp = f"{time.strftime('%H:%M:%S')}.{ms:03d}"
            self._append_text_to_console(f"[DEBUG] 'Removed' detected in STDOUT at {timestamp}: {repr(text[:100])}\n")

        # Debug: Log when we see the final "removed JSON" message
        if "Removed original JSON file:" in text:
            import time
            current = time.time()
            ms = int((current - int(current)) * 1000)
            timestamp = f"{time.strftime('%H:%M:%S')}.{ms:03d}"
            self._append_text_to_console(f"[DEBUG] Final output detected at {timestamp}, process state: {self.process.state()}\n")

            # Start monitoring process state
            def check_process_state():
                if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
                    current = time.time()
                    ms = int((current - int(current)) * 1000)
                    timestamp = f"{time.strftime('%H:%M:%S')}.{ms:03d}"
                    self._append_text_to_console(f"[DEBUG] Process still running at {timestamp}, state: {self.process.state()}\n")
                    QTimer.singleShot(5000, check_process_state)  # Check again in 5 seconds
                else:
                    current = time.time()
                    ms = int((current - int(current)) * 1000)
                    timestamp = f"{time.strftime('%H:%M:%S')}.{ms:03d}"
                    self._append_text_to_console(f"[DEBUG] Process finished at {timestamp}\n")

            QTimer.singleShot(5000, check_process_state)  # Start checking in 5 seconds

        # Check for success indicators
        success_markers = [
            "Transcription complete",
            "saved to",
            "Subtitles are written",
            "Operation finished"
        ]
        lowered = text.lower()
        if any(marker.lower() in lowered for marker in success_markers):
            self.transcription_completed_successfully = True
            # Start post-processing early but only announce once the final marker appears
            if self.is_adobe_export_enabled():
                op_finished = ("operation finished" in lowered)
                if not self.postprocess_started:
                    # Begin early (silently) to reduce delay
                    self._begin_postprocessing(self._active_run_input_file, self._active_run_output_dir)
                    if op_finished:
                        self._append_text_to_console("Post-processing: preparing Adobe Static Transcript and reference transcript...\n")
                else:
                    if op_finished and not getattr(self, 'postprocess_done', False):
                        self._append_text_to_console("Post-processing: preparing Adobe Static Transcript and reference transcript...\n")

        # Update batch progress bar when a file completes (WhisperX batch mode)
        if "transcription complete" in lowered and hasattr(self, '_whisperx_batch_files'):
            if hasattr(self, '_whisperx_batch_completed_count'):
                self._whisperx_batch_completed_count += 1
                num_files = len(self._whisperx_batch_files)
                completed = self._whisperx_batch_completed_count
                self.batch_progress_bar.setValue(completed)
                self.batch_progress_bar.setFormat(f"Completed {completed}/{num_files} files...")
                if hasattr(self, 'whisperx_batch_progress_bar'):
                    self.whisperx_batch_progress_bar.setValue(completed)
                    self.whisperx_batch_progress_bar.setFormat(f"Completed {completed}/{num_files} files...")

                # Update history and create Adobe transcript for this file immediately
                self._process_whisperx_batch_file_completion()

    def handle_stderr(self):
        """Handle stderr from process"""
        if not self.process:
            return

        data = self.process.readAllStandardError()
        text = bytes(data).decode('utf-8', errors='replace')
        self._append_text_to_console(text)

        # Debug: Log any output containing "Removed" to see what's happening
        if "Removed" in text or "removed" in text:
            import time
            current = time.time()
            ms = int((current - int(current)) * 1000)
            timestamp = f"{time.strftime('%H:%M:%S')}.{ms:03d}"
            self._append_text_to_console(f"[DEBUG] 'Removed' detected in STDERR at {timestamp}: {repr(text[:100])}\n")

        # Debug: Log when we see the final "removed JSON" message in stderr
        if "Removed original JSON file:" in text:
            import time
            current = time.time()
            ms = int((current - int(current)) * 1000)
            timestamp = f"{time.strftime('%H:%M:%S')}.{ms:03d}"
            self._append_text_to_console(f"[DEBUG] Final output detected in STDERR at {timestamp}, process state: {self.process.state()}\n")

            # Start monitoring process state
            def check_process_state():
                if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
                    current = time.time()
                    ms = int((current - int(current)) * 1000)
                    timestamp = f"{time.strftime('%H:%M:%S')}.{ms:03d}"
                    self._append_text_to_console(f"[DEBUG] Process still running at {timestamp}, state: {self.process.state()}\n")
                    QTimer.singleShot(5000, check_process_state)  # Check again in 5 seconds
                else:
                    current = time.time()
                    ms = int((current - int(current)) * 1000)
                    timestamp = f"{time.strftime('%H:%M:%S')}.{ms:03d}"
                    self._append_text_to_console(f"[DEBUG] Process finished at {timestamp}\n")

            QTimer.singleShot(1000, check_process_state)  # Start checking in 1 second

    def on_finished(self, exit_code, exit_status):
        """Handle process completion"""

        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        # Clear process reference immediately to avoid conflicts with batch processing
        self.process = None

        self._append_text_to_console("="*50 + "\n")

        # Stop the progress panel with appropriate status
        success = (exit_code == 0 or self.transcription_completed_successfully) and not self.stop_requested
        if hasattr(self, 'progress_panel') and self.progress_panel:
            self.progress_panel.stop_processing(success=success)

        # Check for CUDA compatibility issues and provide user guidance
        if exit_code != 0 and not self.stop_requested:
            # Check if this was a CUDA error
            output_text = self.output_buffer.lower()
            cuda_error_patterns = [
                "cuda error",
                "runtimeerror: cuda",
                "no kernel image is available for execution",
                "cuda out of memory",
                "cublas runtime error"
            ]
            has_cuda_error = any(pattern in output_text for pattern in cuda_error_patterns)

            if has_cuda_error:
                self._append_text_to_console("\n" + "="*50 + "\n")
                self._append_text_to_console("🚨 CUDA COMPATIBILITY ISSUE DETECTED\n")
                self._append_text_to_console("\nThis error is often caused by VAD method compatibility issues.\n")
                self._append_text_to_console("\n💡 SOLUTIONS TO TRY:\n")
                self._append_text_to_console("   1. Switch VAD method: Try silero_v4 or pyannote_onnx_v3\n")
                self._append_text_to_console("   2. Use CPU: Change Device to 'cpu' for guaranteed compatibility\n")
                self._append_text_to_console("   3. Different compute type: Try float32 or int8 instead of float16\n")
                self._append_text_to_console("\n🔧 Quick fix: Use Device=CPU + VAD=silero_v4 for reliable operation\n")
                self._append_text_to_console("="*50 + "\n")

        if self.stop_requested:
            self._append_text_to_console("Process stopped by user.\n")
        elif exit_code == 0 or self.transcription_completed_successfully:
            self._append_text_to_console("Transcription completed successfully!\n")

            # Handle completion - determine what files were actually created
            input_file, output_dir = self._resolve_job_context(self._active_run_input_file, self._active_run_output_dir)
            if not input_file:
                input_file = self.current_input_file
            base_name = os.path.splitext(os.path.basename(input_file or ""))[0]

            # Record context for downstream UI actions
            self._last_completed_input_file = input_file
            self._last_completed_output_dir = output_dir

            # In batch mode, SRT files are always created (forced in build_command)
            # Check filesystem for verification
            expected_srt = None
            try:
                expected_srt = self.get_expected_srt_path(input_file, output_dir)
                srt_created = os.path.exists(expected_srt)
            except Exception:
                expected_srt = None
                srt_created = False
            adobe_created = False
            expected_adobe = None

            # Handle Adobe conversion if requested (skip if file already exists)
            if self.is_adobe_export_enabled() and input_file:
                try:
                    expected_adobe = self.get_expected_adobe_path(input_file, output_dir)
                    adobe_created = os.path.exists(expected_adobe)

                    # Only do conversion if file doesn't exist
                    if not adobe_created:
                        self._append_text_to_console("Post-processing: converting to Adobe Static Transcript and building reference transcript...\n")
                        adobe_created = self.convert_to_adobe_format(input_file=input_file, output_dir=output_dir)
                except Exception as e:
                    expected_adobe = None
                    adobe_created = False

            # Fallback: if neither exists, use checkbox state to decide messaging
            if not srt_created and not adobe_created:
                # Check which tab is active to determine which checkboxes to use
                current_tab_index = self.tab_widget.currentIndex()
                using_whisperx = (current_tab_index == self.whisperx_tab_index)

                if using_whisperx:
                    if hasattr(self, 'whisperx_export_srt') and self.whisperx_export_srt.isChecked():
                        srt_created = True
                    elif hasattr(self, 'whisperx_export_adobe') and self.whisperx_export_adobe.isChecked():
                        adobe_created = True
                else:
                    if hasattr(self, 'export_srt') and self.export_srt.isChecked():
                        srt_created = True
                    elif hasattr(self, 'export_adobe') and self.export_adobe.isChecked():
                        adobe_created = True

            file_records: List[dict] = []
            if expected_srt and os.path.exists(expected_srt):
                file_records.append({"label": "SRT", "path": expected_srt})
            if expected_adobe and os.path.exists(expected_adobe):
                file_records.append({"label": "Adobe Static Transcript", "path": expected_adobe})
            try:
                txt_path = self.get_expected_txt_path(input_file, output_dir)
                if os.path.exists(txt_path):
                    file_records.append({"label": "Reference TXT", "path": txt_path})
            except Exception:
                pass

            self._add_history_entry(input_file, output_dir, file_records, success=True)

            # Store results for batch processing
            if self.batch_mode_active:
                self.batch_results.append({
                    'file_path': input_file,
                    'output_dir': output_dir,
                    'srt_created': srt_created,
                    'adobe_created': adobe_created
                })

                # Update progress bars in BOTH tabs
                self._batch_index += 1
                self.batch_progress_bar.setValue(self._batch_index)
                self.batch_progress_bar.setFormat(f"Processing {self._batch_index}/{len(self.batch_files)} files...")
    
                if hasattr(self, 'whisperx_batch_progress_bar'):
                    self.whisperx_batch_progress_bar.setValue(self._batch_index)
                    self.whisperx_batch_progress_bar.setFormat(f"Processing {self._batch_index}/{len(self.batch_files)} files...")
    
                # Move to next file in batch
                self._process_next_in_batch()
            else:
                # Single file - show success dialog
                self.show_completion_success_dialog(output_dir, srt_created, adobe_created)
        else:
            status_str = "Crashed" if exit_status == QProcess.ExitStatus.CrashExit else "Failed"
            self._append_text_to_console(f"Process {status_str} with exit code {exit_code}.\n")

            # Handle batch processing even on failure
            if self.batch_mode_active:
                # Record the failure
                self.batch_results.append({
                    'file_path': self._active_run_input_file or self.current_input_file,
                    'output_dir': self._active_run_output_dir or self._compute_output_dir_for_input(self._active_run_input_file or self.current_input_file),
                    'srt_created': False,
                    'adobe_created': False,
                    'error': f"Process {status_str} with exit code {exit_code}"
                })

                # Update progress bars in BOTH tabs
                self._batch_index += 1
                self.batch_progress_bar.setValue(self._batch_index)
                self.batch_progress_bar.setFormat(f"Processing {self._batch_index}/{len(self.batch_files)} files...")
    
                if hasattr(self, 'whisperx_batch_progress_bar'):
                    self.whisperx_batch_progress_bar.setValue(self._batch_index)
                    self.whisperx_batch_progress_bar.setFormat(f"Processing {self._batch_index}/{len(self.batch_files)} files...")
    
                # Move to next file in batch
                self._process_next_in_batch()

            history_input = self._active_run_input_file or self.current_input_file
            history_output = self._active_run_output_dir or self._compute_output_dir_for_input(history_input)
            self._add_history_entry(history_input, history_output, [], success=False, error=f"Process {status_str} with exit code {exit_code}")

        # Final cleanup (self.process already set to None at the beginning)
        self.stop_requested = False

    def _begin_postprocessing(self, input_file: Optional[str] = None, output_dir: Optional[str] = None):
        """Start polling for the JSON output and run Adobe conversion early to reduce perceived delay."""
        self.postprocess_started = True

        try:
            # Determine expected Whisper JSON path using stable job context
            input_file, output_dir = self._resolve_job_context(input_file, output_dir)
            if not input_file or not output_dir:
                self._append_text_to_console("[DEBUG] Post-processing skipped: unable to resolve input/output context\n")
                return

            base_name = os.path.splitext(os.path.basename(input_file))[0]
            json_file = os.path.join(output_dir, f"{base_name}.json")

            state = {"attempts": 0, "last_size": -1, "stable": 0}

            def poll_for_json():
                # Already done elsewhere
                if getattr(self, '_adobe_converted', False):
                    self._append_text_to_console(f"[DEBUG] Polling: Adobe already converted, stopping\n")
                    self.postprocess_done = True
                    return

                try:
                    self._append_text_to_console(f"[DEBUG] Polling attempt {state['attempts']}, checking: {json_file}\n")
                    if os.path.exists(json_file):
                        size = os.path.getsize(json_file)
                        self._append_text_to_console(f"[DEBUG] JSON exists, size: {size}, last_size: {state['last_size']}, stable: {state['stable']}\n")
                        if size == state["last_size"]:
                            state["stable"] += 1
                        else:
                            state["stable"] = 0
                            state["last_size"] = size

                        # Consider the file stable after two consecutive identical sizes (~400ms)
                        if state["stable"] >= 2:
                            self._append_text_to_console(f"[DEBUG] JSON stable, starting conversion\n")
                            ok = self.convert_to_adobe_format(input_file=input_file, output_dir=output_dir)
                            self.postprocess_done = True
                            # If conversion failed, keep trying until on_finished fallback
                            if not ok:
                                self._append_text_to_console("Post-processing: early conversion failed, will retry after process exits.\n")
                            return
                    else:
                        self._append_text_to_console(f"[DEBUG] JSON file not found yet\n")

                    state["attempts"] += 1
                    # Try up to ~10 seconds (50 * 200ms)
                    if state["attempts"] < 50:
                        QTimer.singleShot(200, poll_for_json)
                    else:
                        self._append_text_to_console("Post-processing: waiting for output files timed out; will finalize after process exit.\n")
                except Exception as e:
                    self._append_text_to_console(f"Post-processing error while polling: {e}\n")

            QTimer.singleShot(200, poll_for_json)
        except Exception as e:
            self._append_text_to_console(f"Post-processing could not start: {e}\n")

    def on_process_error(self, error):
        """Handle process errors"""
        if error == QProcess.ProcessError.Crashed and self.transcription_completed_successfully:
            return

        error_messages = {
            QProcess.ProcessError.FailedToStart: "Failed to start process",
            QProcess.ProcessError.Crashed: "Process crashed",
            QProcess.ProcessError.Timedout: "Process timed out",
            QProcess.ProcessError.ReadError: "Read error",
            QProcess.ProcessError.WriteError: "Write error",
            QProcess.ProcessError.UnknownError: "Unknown error"
        }

        error_message = error_messages.get(error, "Unspecified error")
        self._append_text_to_console(f"\nPROCESS ERROR: {error_message}\n")

    def convert_to_adobe_format(self, input_file: Optional[str] = None, output_dir: Optional[str] = None):
        """Convert JSON output to Adobe format"""
        try:
            self.last_srt_text = ""

            # Find the JSON file that was just created
            input_file, output_dir = self._resolve_job_context(input_file, output_dir)
            if not input_file or not output_dir:
                self._append_text_to_console("Warning: Cannot convert to Adobe format without an input file context\n")
                return False

            base_name = os.path.splitext(os.path.basename(input_file))[0]

            # Use the predictable JSON filename that faster-whisper creates
            json_file = os.path.join(output_dir, f"{base_name}.json")

            if not os.path.exists(json_file):
                # List all files in output directory for debugging
                try:
                    files_in_dir = os.listdir(output_dir)
                    self._append_text_to_console(f"Warning: Expected JSON file not found: {os.path.basename(json_file)}\n")
                    self._append_text_to_console(f"Files in output directory: {files_in_dir}\n")
                except:
                    self._append_text_to_console(f"Warning: Expected JSON file not found: {os.path.basename(json_file)}\n")
                return False

            # Import conversion logic from whisper_to_adobe_converter
            from Data import whisper_to_adobe_converter_logic

            adobe_file = os.path.join(output_dir, f"{base_name}_adobe.json")

            with open(json_file, 'r', encoding='utf-8') as f:
                whisper_data = json.load(f)

            whisper_data = self._normalize_whisperx_speakers(whisper_data)

            self.last_srt_text = self._build_srt_text(whisper_data.get('segments', []))

            # Persist normalized JSON for downstream tools prior to conversion
            try:
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(whisper_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                self._append_text_to_console(f"Warning: Could not rewrite normalized JSON: {e}\n")

            # Determine formatting mode based on active tab
            current_tab_index = self.tab_widget.currentIndex()
            using_whisperx = (current_tab_index == self.whisperx_tab_index)

            if using_whisperx:
                use_paragraph_form = hasattr(self, 'whisperx_format_paragraph_form') and self.whisperx_format_paragraph_form.isChecked()
            else:
                use_paragraph_form = hasattr(self, 'format_paragraph_form') and self.format_paragraph_form.isChecked()

            if use_paragraph_form:
                # Build paragraph segments based on word-level gaps (>3s)
                # Flatten words and construct paragraph boundaries
                # We'll reuse whisper segments but group by gaps when building Adobe
                # Create simple paragraph segments structure: start, end, words
                try:
                    if using_whisperx:
                        gap = float(self.whisperx_paragraph_gap_spin.value()) if hasattr(self, 'whisperx_paragraph_gap_spin') else 2.0
                    else:
                        gap = float(self.paragraph_gap_spin.value()) if hasattr(self, 'paragraph_gap_spin') else 3.0
                except Exception:
                    gap = 3.0
                flat_segments = whisper_data.get('segments', []) or []
                # Flatten words list preserving order
                all_words = []
                for seg in flat_segments:
                    for w in seg.get('words', []) or []:
                        all_words.append(w)

                paragraph_segments = []
                current = {"start": None, "end": None, "words": []}
                last_end = None
                for w in all_words:
                    w_start = float(w.get('start', 0.0))
                    w_end = float(w.get('end', w_start + float(w.get('duration', 0.0))))
                    if current["start"] is None:
                        current["start"] = w_start
                    # Decide paragraph break on gap
                    if last_end is not None and (w_start - last_end) > gap and current["words"]:
                        current["end"] = last_end
                        paragraph_segments.append(current)
                        current = {"start": w_start, "end": None, "words": []}
                    current["words"].append(w)
                    last_end = w_end
                if current["words"]:
                    current["end"] = last_end
                    paragraph_segments.append(current)

                adobe_data = whisper_to_adobe_converter_logic.convert_whisper_to_adobe(
                    whisper_data,
                    paragraph_segments=paragraph_segments,
                    single_segment=False,
                )

                # Generate Paragraph Form TXT from Adobe Static Transcript data
                self.generate_paragraph_txt(input_file, output_dir, adobe_data, paragraph_form=True, word_gap_seconds=gap)
            else:
                # Convert to Adobe format as single continuous segment (no paragraph breaks)
                # This ensures perfect text consistency with the continuous TXT file
                adobe_data = whisper_to_adobe_converter_logic.convert_whisper_to_adobe(
                    whisper_data,
                    paragraph_segments=None,
                    single_segment=True,
                )

                # Generate continuous TXT from Adobe Static Transcript data
                self.generate_paragraph_txt(input_file, output_dir, adobe_data, paragraph_form=False)

            with open(adobe_file, 'w', encoding='utf-8') as f:
                json.dump(adobe_data, f, indent=2, ensure_ascii=False)

            self._append_text_to_console(f"✓ Adobe format saved: {adobe_file}\n")

            # Delete the original JSON file, keep only the Adobe version
            try:
                os.remove(json_file)
                self._append_text_to_console(f"✓ Removed original JSON file: {os.path.basename(json_file)}\n")

                # Debug: Check process state when post-processing completes
                import time
                current = time.time()
                ms = int((current - int(current)) * 1000)
                timestamp = f"{time.strftime('%H:%M:%S')}.{ms:03d}"
                if self.process:
                    self._append_text_to_console(f"[DEBUG] Post-processing complete at {timestamp}, process state: {self.process.state()}\n")

                    # Only terminate process if NOT in batch mode
                    # In batch mode, the next file will start immediately and needs the process slot
                    if not self.batch_mode_active:
                        # Start monitoring and terminate process after completion
                        def monitor_and_terminate():
                            if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
                                current = time.time()
                                ms = int((current - int(current)) * 1000)
                                timestamp = f"{time.strftime('%H:%M:%S')}.{ms:03d}"
                                self._append_text_to_console(f"[DEBUG] Process still running after completion, terminating at {timestamp}\n")

                                # Terminate the process since it has completed its work but isn't exiting
                                self.process.terminate()

                                # Give it 3 seconds to terminate gracefully, then kill if needed
                                def force_kill_if_needed():
                                    if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
                                        self._append_text_to_console(f"[DEBUG] Process didn't terminate gracefully, force killing\n")
                                        self.process.kill()

                                QTimer.singleShot(3000, force_kill_if_needed)
                            else:
                                current = time.time()
                                ms = int((current - int(current)) * 1000)
                                timestamp = f"{time.strftime('%H:%M:%S')}.{ms:03d}"
                                self._append_text_to_console(f"[DEBUG] Process exited naturally at {timestamp}\n")

                        QTimer.singleShot(2000, monitor_and_terminate)  # Give it 2 seconds, then terminate

            except Exception as e:
                self._append_text_to_console(f"Warning: Could not delete original JSON file: {str(e)}\n")

            # Mark as converted to avoid duplicate work on process finish
            self._adobe_converted = True

            return True

        except Exception as e:
            self._append_text_to_console(f"Warning: Adobe conversion failed: {str(e)}\n")
            return False

    def _append_text_to_console(self, text):
        """Append text to console output and update progress panel"""
        # Use progress panel's append method which handles both console and progress parsing
        if hasattr(self, 'progress_panel') and self.progress_panel:
            self.progress_panel.append_output(text)
        else:
            # Fallback for direct output_text access
            cursor = self.output_text.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText(text)
            self.output_text.setTextCursor(cursor)
            self.output_text.ensureCursorVisible()

        # Also store in output buffer for error detection
        self.output_buffer += text

    def save_settings(self):
        """Save current settings with persistence"""
        try:
            settings = {
                "model": self.model_combo.currentText(),
                "task": self.task_combo.currentText(),
                "language": self.language_combo.currentText(),
                "compute_type": self.compute_combo.currentText(),
                "device": self.device_combo.currentText(),
                "engine_faster_whisper": self.engine_faster_whisper.isChecked(),
                "vad_filter": self.vad_filter.isChecked(),
                "vad_method": self.vad_method.currentText(),
                "enable_diarization": self.enable_diarization.isChecked() if hasattr(self, 'enable_diarization') else False,
                "diarization_method": self.diarization_method.currentText() if hasattr(self, 'diarization_method') else "",
                "num_speakers": self.num_speakers.currentText() if hasattr(self, 'num_speakers') else "auto",
                "export_srt": self.export_srt.isChecked(),
                "export_adobe": self.export_adobe.isChecked(),
                "format_paragraph_form": self.format_paragraph_form.isChecked() if hasattr(self, 'format_paragraph_form') else False,
                "format_paragraph_gap_seconds": float(self.paragraph_gap_spin.value()) if hasattr(self, 'paragraph_gap_spin') else 3.0,
                "hardware_info": self.hardware_info,
                "dependencies_ready": bool(self.executable_path and os.path.exists(self.executable_path)),
                "last_hardware_check": getattr(self, '_last_hardware_check', None),
                "engine": "faster_whisper",
                "whisperx_python": self.whisperx_python,
                "whisperx_model": self.whisperx_model_combo.currentText() if hasattr(self, 'whisperx_model_combo') else None,
                "whisperx_task": self.whisperx_task_combo.currentText() if hasattr(self, 'whisperx_task_combo') else None,
                "whisperx_language": self.whisperx_language_combo.currentText() if hasattr(self, 'whisperx_language_combo') else None,
                "whisperx_device": self.whisperx_device_combo.currentText() if hasattr(self, 'whisperx_device_combo') else None,
                "whisperx_compute_type": self.whisperx_compute_combo.currentText() if hasattr(self, 'whisperx_compute_combo') else None,
                "whisperx_vad_filter": self.whisperx_vad_filter.isChecked() if hasattr(self, 'whisperx_vad_filter') else None,
                "whisperx_vad_method": self.whisperx_vad_method.currentText() if hasattr(self, 'whisperx_vad_method') else None,
                "whisperx_enable_diarization": self.whisperx_enable_diarization.isChecked() if hasattr(self, 'whisperx_enable_diarization') else None,
                "whisperx_export_adobe": self.whisperx_export_adobe.isChecked() if hasattr(self, 'whisperx_export_adobe') else None,
                "whisperx_export_srt": self.whisperx_export_srt.isChecked() if hasattr(self, 'whisperx_export_srt') else None,
                "whisperx_paragraph_form": self.whisperx_format_paragraph_form.isChecked() if hasattr(self, 'whisperx_format_paragraph_form') else None,
                "whisperx_paragraph_gap_seconds": float(self.whisperx_paragraph_gap_spin.value()) if hasattr(self, 'whisperx_paragraph_gap_spin') else None,
                "whisperx_output_dir": self.whisperx_output_dir.text() if hasattr(self, 'whisperx_output_dir') else "",
                "whisperx_hf_token": self.whisperx_hf_token.text() if hasattr(self, 'whisperx_hf_token') else "",
                "whisperx_num_speakers": self.whisperx_num_speakers.currentText() if hasattr(self, 'whisperx_num_speakers') else "Auto"
            }

            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2, default=str)
        except Exception as e:
            logging.warning(f"Could not save settings: {e}")

    def load_settings(self):
        """Load saved settings with persistence"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)

                # Load engine preference
                engine_faster_whisper = settings.get("engine_faster_whisper", True)
                self.engine_faster_whisper.setChecked(engine_faster_whisper)
                self.engine_whisperx.setChecked(not engine_faster_whisper)

                # Populate model list based on engine, then apply saved
                self.on_engine_changed()
                self.model_combo.setCurrentText(settings.get("model", 'small' if engine_faster_whisper else 'large-v3-turbo'))
                self.task_combo.setCurrentText(settings.get("task", "transcribe"))
                self.language_combo.setCurrentText(settings.get("language", "auto"))
                self.compute_combo.setCurrentText(settings.get("compute_type", "int8"))
                self.device_combo.setCurrentText(settings.get("device", "cpu"))
                self.vad_filter.setChecked(settings.get("vad_filter", True))
                self.vad_method.setCurrentText(settings.get("vad_method", "silero_v4"))

                # Load diarization settings
                if hasattr(self, 'enable_diarization'):
                    self.enable_diarization.setChecked(settings.get("enable_diarization", False))
                if hasattr(self, 'diarization_method'):
                    self.diarization_method.setCurrentText(settings.get("diarization_method", "pyannote_v3.1"))
                if hasattr(self, 'num_speakers'):
                    self.num_speakers.setCurrentText(settings.get("num_speakers", "2"))

                # Defaults: Adobe on, SRT always on (now hidden in UI)
                self.export_srt.setChecked(True)  # Always True, ignore saved settings
                self.export_adobe.setChecked(settings.get("export_adobe", True))
                # Enforce at least one option
                self._ensure_one_export_checked('adobe')

                # Update diarization options visibility based on loaded setting
                self.on_diarization_toggled()

                # Reset output directory to default (don't restore between sessions)
                if hasattr(self, 'output_dir'):
                    self.output_dir.setText("")

                # Restore hardware info if available
                if settings.get("hardware_info"):
                    self.hardware_info = settings["hardware_info"]
                    self.update_hardware_display()

                # Don't restore input file - user should select fresh each time

                # Restore WhisperX interpreter preference
                self.whisperx_python = settings.get("whisperx_python")

                # Restore WhisperX-specific settings
                try:
                    if hasattr(self, 'whisperx_model_combo'):
                        self.whisperx_model_combo.setCurrentText(settings.get("whisperx_model", 'large-v3-turbo'))
                    if hasattr(self, 'whisperx_task_combo'):
                        self.whisperx_task_combo.setCurrentText(settings.get("whisperx_task", 'transcribe'))
                    if hasattr(self, 'whisperx_language_combo'):
                        self.whisperx_language_combo.setCurrentText(settings.get("whisperx_language", 'auto'))
                    if hasattr(self, 'whisperx_device_combo'):
                        self.whisperx_device_combo.setCurrentText(settings.get("whisperx_device", 'cuda'))
                    if hasattr(self, 'whisperx_compute_combo'):
                        self.whisperx_compute_combo.setCurrentText(settings.get("whisperx_compute_type", 'float16'))
                    if hasattr(self, 'whisperx_vad_filter'):
                        self.whisperx_vad_filter.setChecked(settings.get("whisperx_vad_filter", True))
                    if hasattr(self, 'whisperx_vad_method'):
                        self.whisperx_vad_method.setCurrentText(settings.get("whisperx_vad_method", 'pyannote_v3'))
                    if hasattr(self, 'whisperx_enable_diarization'):
                        self.whisperx_enable_diarization.setChecked(settings.get("whisperx_enable_diarization", False))
                    if hasattr(self, 'whisperx_export_adobe'):
                        self.whisperx_export_adobe.setChecked(settings.get("whisperx_export_adobe", True))
                    if hasattr(self, 'whisperx_export_srt'):
                        self.whisperx_export_srt.setChecked(True)  # Always True, ignore saved settings
                    if hasattr(self, 'whisperx_format_paragraph_form') and settings.get("whisperx_paragraph_form") is not None:
                        if settings.get("whisperx_paragraph_form"):
                            self.whisperx_format_paragraph_form.setChecked(True)
                        else:
                            self.whisperx_format_single_segment.setChecked(True)
                    if hasattr(self, 'whisperx_paragraph_gap_spin') and settings.get("whisperx_paragraph_gap_seconds") is not None:
                        try:
                            self.whisperx_paragraph_gap_spin.setValue(float(settings.get("whisperx_paragraph_gap_seconds", 2.0)))
                        except Exception:
                            pass
                    if hasattr(self, 'whisperx_output_dir'):
                        self.whisperx_output_dir.setText(settings.get("whisperx_output_dir", ""))
                    if hasattr(self, 'whisperx_hf_token'):
                        self.whisperx_hf_token.setText(settings.get("whisperx_hf_token", ""))
                    if hasattr(self, 'whisperx_num_speakers') and settings.get("whisperx_num_speakers") is not None:
                        try:
                            choice = settings.get("whisperx_num_speakers", "Auto")
                            if choice not in [self.whisperx_num_speakers.itemText(i) for i in range(self.whisperx_num_speakers.count())]:
                                self.whisperx_num_speakers.addItem(str(choice))
                            self.whisperx_num_speakers.setCurrentText(str(choice))
                        except Exception:
                            self.whisperx_num_speakers.setCurrentIndex(0)
                except Exception:
                    pass

                # Restore formatting preference
                if settings.get("format_paragraph_form", False):
                    self.format_paragraph_form.setChecked(True)
                else:
                    self.format_single_segment.setChecked(True)

                # Restore gap threshold and update enable state
                try:
                    gap_val = float(settings.get("format_paragraph_gap_seconds", 3.0))
                    self.paragraph_gap_spin.setValue(gap_val)
                except Exception:
                    pass
                self.update_formatting_controls()

        except Exception as e:
            logging.warning(f"Could not load settings: {e}")

        # Final reset: Ensure output directory is always cleared on startup
        if hasattr(self, 'output_dir'):
            self.output_dir.setText("")
        if hasattr(self, 'whisperx_output_dir'):
            # Preserve user-specific output directory for WhisperX only if provided
            if not getattr(self, 'whisperx_output_dir', None).text():
                self.whisperx_output_dir.setText("")

    def update_formatting_controls(self):
        """Enable/disable formatting controls based on selection."""
        enable_gap = False
        try:
            enable_gap = self.format_paragraph_form.isChecked()
        except Exception:
            pass
        try:
            self.paragraph_gap_spin.setEnabled(enable_gap)
        except Exception:
            pass

    def update_formatting_visibility(self):
        """Show/hide formatting group based on Adobe export checkbox"""
        try:
            if hasattr(self, 'formatting_group'):
                is_adobe_checked = self.export_adobe.isChecked()
                self.formatting_group.setVisible(is_adobe_checked)
        except Exception:
            pass

    def update_hardware_display(self):
        """Update hardware status display from saved info"""
        if self.hardware_info:
            if self.hardware_info.get("has_cuda"):
                status = f"✓ CUDA GPU: {self.hardware_info.get('gpu_name', 'Unknown')} ({self.hardware_info.get('gpu_memory_gb', 0):.1f}GB)"
                self.hardware_status_label.setStyleSheet("color: green; font-size: 13px; padding: 8px;")

                # Update WhisperX CUDA status label as well
                if hasattr(self, 'whisperx_cuda_status_label'):
                    cuda_status = f"✓ CUDA available: {self.hardware_info.get('gpu_name', 'Unknown')}"
                    self.whisperx_cuda_status_label.setText(cuda_status)
                    self.whisperx_cuda_status_label.setStyleSheet("color: green; font-size: 12px; font-weight: bold;")
                    self.whisperx_cuda_status_label.setVisible(True)
            else:
                status = "CPU only (no CUDA GPU detected)"
                self.hardware_status_label.setStyleSheet("color: orange; font-size: 13px; padding: 8px;")

                # Update WhisperX CUDA status label as well
                if hasattr(self, 'whisperx_cuda_status_label'):
                    self.whisperx_cuda_status_label.setText("CPU only (no CUDA detected)")
                    self.whisperx_cuda_status_label.setStyleSheet("color: orange; font-size: 12px;")
                    self.whisperx_cuda_status_label.setVisible(True)

            self.hardware_status_label.setText(status)

    def closeEvent(self, event):
        """Handle widget close"""
        self.save_settings()
        super().closeEvent(event)
