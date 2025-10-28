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


# ============================================================================
# Configuration
# ============================================================================

WHISPERX_VERSION = "3.7.4"
PACKAGE_VERSION = f"v{WHISPERX_VERSION}"

# GitHub release URLs
GITHUB_REPO = "YOUR_USERNAME/scriptoria-whisperx-standalone"  # UPDATE THIS!
GITHUB_RELEASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/whisperx-{PACKAGE_VERSION}"

# Package files
CUDA_PACKAGE = f"whisperx_portable_win64_cuda128_{PACKAGE_VERSION}.zip"
CPU_PACKAGE = f"whisperx_portable_win64_cpu_{PACKAGE_VERSION}.zip"

CUDA_URL = f"{GITHUB_RELEASE_URL}/{CUDA_PACKAGE}"
CPU_URL = f"{GITHUB_RELEASE_URL}/{CPU_PACKAGE}"

# Expected package sizes (for progress display)
CUDA_SIZE_MB = 2500  # ~2.5 GB
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


# ============================================================================
# Main Installation Function
# ============================================================================

def download_and_extract_whisperx(
    install_dir,
    use_cuda=None,
    progress_callback=None,
    cancel_check=None
):
    """
    Download and extract pre-built WhisperX portable package.

    This replaces the complex Python installation process with a simple
    download and extract operation.

    Args:
        install_dir: Directory to extract WhisperX environment to
        use_cuda: True for CUDA version, False for CPU, None for auto-detect
        progress_callback: Optional function(str) to call with progress messages
        cancel_check: Optional function() that returns True if cancelled

    Returns:
        str: Path to python.exe in the extracted environment, or None if failed
    """

    def emit(msg):
        """Send progress message to callback."""
        if progress_callback:
            progress_callback(msg)

    def is_cancelled():
        """Check if operation was cancelled."""
        return cancel_check() if cancel_check else False

    # Auto-detect CUDA if not specified
    if use_cuda is None:
        use_cuda = has_nvidia_gpu()
        if use_cuda:
            emit("✓ NVIDIA GPU detected - downloading CUDA version\n")
        else:
            emit("ℹ No NVIDIA GPU detected - downloading CPU version\n")

    # Select appropriate package
    package_url = CUDA_URL if use_cuda else CPU_URL
    package_name = CUDA_PACKAGE if use_cuda else CPU_PACKAGE
    expected_size_mb = CUDA_SIZE_MB if use_cuda else CPU_SIZE_MB

    emit(f"\n{'='*60}\n")
    emit(f"WhisperX Portable Package Installer\n")
    emit(f"{'='*60}\n\n")
    emit(f"Package: {package_name}\n")
    emit(f"Expected size: ~{expected_size_mb} MB\n")
    emit(f"Download URL: {package_url}\n")
    emit(f"Install directory: {install_dir}\n\n")

    # Create install directory
    os.makedirs(install_dir, exist_ok=True)

    # Download path
    download_path = os.path.join(install_dir, package_name)
    python_exe = os.path.join(install_dir, "Scripts", "python.exe")

    # Check if already installed
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

    # Download package
    emit(f"\n{'='*60}\n")
    emit(f"Downloading WhisperX portable package\n")
    emit(f"{'='*60}\n\n")
    emit(f"This may take 5-15 minutes depending on your internet speed...\n\n")

    try:
        def download_progress(block_num, block_size, total_size):
            """Report download progress."""
            if is_cancelled():
                raise Exception("Download cancelled by user")

            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, (downloaded * 100) // total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)

                # Report every 2%
                if block_num % 100 == 0 or percent >= 100:
                    emit(f"  Downloaded: {mb_downloaded:.1f} MB / {mb_total:.1f} MB ({percent}%)\n")

        # Download the file
        urllib.request.urlretrieve(package_url, download_path, reporthook=download_progress)

        if is_cancelled():
            if os.path.exists(download_path):
                os.remove(download_path)
            emit("\n✗ Download cancelled by user\n")
            return None

        emit(f"\n✓ Download complete: {download_path}\n\n")

    except urllib.error.HTTPError as e:
        emit(f"\n✗ Download failed: HTTP {e.code}\n")
        emit(f"  URL: {package_url}\n")
        emit(f"  Error: {e.reason}\n\n")
        emit("  Please check:\n")
        emit("  1. The package has been built and uploaded to GitHub Releases\n")
        emit("  2. The URL is correct in the code\n")
        emit("  3. Your internet connection is working\n")
        return None

    except urllib.error.URLError as e:
        emit(f"\n✗ Download failed: {e.reason}\n")
        emit("  Please check your internet connection and try again.\n")
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

        # Test WhisperX
        result = subprocess.run(
            [python_exe, "-c", "import whisperx; print(f'WhisperX {whisperx.__version__}')"],
            capture_output=True,
            text=True,
            timeout=30,
            startupinfo=get_subprocess_startup_info(),
            creationflags=get_subprocess_creation_flags()
        )

        if result.returncode == 0:
            emit(f"✓ {result.stdout.strip()}\n")
        else:
            emit(f"⚠ WhisperX import test failed\n")

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
