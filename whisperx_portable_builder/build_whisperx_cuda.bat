@echo off
REM ============================================================================
REM WhisperX Portable Environment Builder - CUDA Version
REM ============================================================================
REM This script creates a self-contained WhisperX environment with CUDA support
REM for users with NVIDIA GPUs.
REM
REM Requirements:
REM   - Windows 10/11
REM   - Python 3.12 or 3.13 installed on the system
REM   - ~10 GB free disk space
REM   - Internet connection
REM
REM Output:
REM   - whisperx_portable_win64_cuda128_v3.4.2.zip (~2.5 GB)
REM ============================================================================

setlocal enabledelayedexpansion

echo ============================================================================
echo WhisperX Portable Environment Builder - CUDA Version
echo ============================================================================
echo.

REM Configuration - Update these when WhisperX dependencies change
set PYTHON_VERSION=3.13.3
set WHISPERX_VERSION=3.7.4
set PYTORCH_VERSION=2.8.0
set TORCHAUDIO_VERSION=2.8.0
set CUDA_VERSION=cu128
set PACKAGE_VERSION=v3.7.4

REM Build directories
set BUILD_DIR=%~dp0build_cuda
set OUTPUT_DIR=%~dp0output
set ENV_DIR=%BUILD_DIR%\whisperx_portable
set PYTHON_DIR=%ENV_DIR%\python313

echo Configuration:
echo   Python: %PYTHON_VERSION%
echo   WhisperX: %WHISPERX_VERSION%
echo   PyTorch: %PYTORCH_VERSION% (CUDA %CUDA_VERSION%)
echo   Build directory: %BUILD_DIR%
echo   Output directory: %OUTPUT_DIR%
echo.

REM Clean up previous builds
if exist "%BUILD_DIR%" (
    echo Cleaning up previous build...
    rmdir /s /q "%BUILD_DIR%"
)

if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
)

mkdir "%BUILD_DIR%"
mkdir "%ENV_DIR%"

echo ============================================================================
echo Step 1: Creating Python virtual environment
echo ============================================================================
echo.

REM Create venv using system Python
python -m venv "%ENV_DIR%"

if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    echo Make sure Python 3.12+ is installed and available in PATH
    exit /b 1
)

echo Virtual environment created successfully
echo.

echo ============================================================================
echo Step 2: Upgrading pip, setuptools, and wheel
echo ============================================================================
echo.

"%ENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel

if errorlevel 1 (
    echo ERROR: Failed to upgrade pip
    exit /b 1
)

echo.

echo ============================================================================
echo Step 3: Installing PyTorch with CUDA %CUDA_VERSION% support
echo ============================================================================
echo This may take 5-10 minutes depending on your internet connection...
echo.

"%ENV_DIR%\Scripts\python.exe" -m pip install torch==%PYTORCH_VERSION% torchaudio==%TORCHAUDIO_VERSION% --index-url https://download.pytorch.org/whl/%CUDA_VERSION%

if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    exit /b 1
)

echo PyTorch installed successfully
echo.

echo ============================================================================
echo Step 4: Installing WhisperX %WHISPERX_VERSION%
echo ============================================================================
echo.

"%ENV_DIR%\Scripts\python.exe" -m pip install whisperx==%WHISPERX_VERSION%

if errorlevel 1 (
    echo ERROR: Failed to install WhisperX
    exit /b 1
)

echo WhisperX installed successfully
echo.

echo ============================================================================
echo Step 4a: Patching SpeechBrain lazy importer for Windows paths
echo ============================================================================
echo.
echo Fixing infinite recursion bug in speechbrain/utils/importutils.py...

"%ENV_DIR%\Scripts\python.exe" -c "from pathlib import Path; p = Path(r'%ENV_DIR%\Lib\site-packages\speechbrain\utils\importutils.py'); c = p.read_text(); old = 'importer_frame.filename.endswith(\"/inspect.py\")'; new = 'importer_frame.filename.replace(\"\\\\\", \"/\").endswith(\"/inspect.py\")'; c_new = c.replace(old, new); assert c != c_new, 'Pattern not found!'; p.write_text(c_new); print('Patch applied successfully')"

if errorlevel 1 (
    echo WARNING: Failed to patch SpeechBrain - diarization may not work on Windows
    echo Continuing anyway...
)

echo ============================================================================
echo Step 5: Verifying installation
echo ============================================================================
echo.

echo Testing Python version...
"%ENV_DIR%\Scripts\python.exe" --version

echo.
echo Testing WhisperX import...
"%ENV_DIR%\Scripts\python.exe" -c "import whisperx; print(f'WhisperX {whisperx.__version__} imported successfully')"

if errorlevel 1 (
    echo ERROR: WhisperX verification failed
    exit /b 1
)

echo.
echo Testing PyTorch CUDA...
"%ENV_DIR%\Scripts\python.exe" -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo.

echo ============================================================================
echo Step 6: Creating portable package
echo ============================================================================
echo.

set OUTPUT_FILE=%OUTPUT_DIR%\whisperx_portable_win64_cuda128_%PACKAGE_VERSION%.zip

echo Compressing environment to: %OUTPUT_FILE%
echo This may take 5-10 minutes...
echo.

REM Use PowerShell to create zip (built-in on Windows 10+)
powershell -Command "Compress-Archive -Path '%ENV_DIR%\*' -DestinationPath '%OUTPUT_FILE%' -CompressionLevel Optimal -Force"

if errorlevel 1 (
    echo ERROR: Failed to create zip file
    echo.
    echo Alternative: You can manually zip the contents of:
    echo   %ENV_DIR%
    echo And save it as:
    echo   %OUTPUT_FILE%
    exit /b 1
)

echo.
echo ============================================================================
echo Build Complete!
echo ============================================================================
echo.

REM Get file size
for %%A in ("%OUTPUT_FILE%") do set FILE_SIZE=%%~zA
set /a FILE_SIZE_MB=FILE_SIZE/1048576

echo Package created: %OUTPUT_FILE%
echo Package size: %FILE_SIZE_MB% MB
echo.
echo Next steps:
echo   1. Test the package by extracting and running:
echo      Scripts\python.exe -c "import whisperx"
echo.
echo   2. Upload to GitHub Releases:
echo      - Create a new release with tag: whisperx-%PACKAGE_VERSION%
echo      - Upload: %OUTPUT_FILE%
echo.
echo   3. Update download URL in generate_captions.py
echo.

REM Optional: Clean up build directory
echo.
set /p CLEANUP="Delete build directory to save space? (Y/N): "
if /i "%CLEANUP%"=="Y" (
    echo Cleaning up build directory...
    rmdir /s /q "%BUILD_DIR%"
    echo Build directory deleted
)

echo.
echo Done!
pause
