# WhisperX Portable Environment Builder

This folder contains scripts to build self-contained, portable WhisperX environments for Windows.

## Overview

The portable packages are complete Python environments with WhisperX and all dependencies pre-installed. Users just download and extract - no Python installation required.

## Package Variants

### 1. CUDA Version (~2.5 GB)
- **File:** `whisperx_portable_win64_cuda128_v3.7.4.zip`
- **For:** Users with NVIDIA GPUs (RTX 20/30/40 series, GTX 16 series)
- **Includes:** PyTorch with CUDA 12.8 support
- **Performance:** 10-20x faster than CPU

### 2. CPU Version (~800 MB)
- **File:** `whisperx_portable_win64_cpu_v3.7.4.zip`
- **For:** Users without NVIDIA GPU, or with AMD/Intel GPUs
- **Includes:** PyTorch CPU-only version
- **Performance:** Slower but works on all systems

## Build Process

### Prerequisites
- Windows 10/11 with Python 3.12+ installed
- ~10 GB free disk space (temporary build files)
- Internet connection for downloading packages

### Manual Build

1. **Build CUDA version:**
   ```bash
   .\build_whisperx_cuda.bat
   ```

2. **Build CPU version:**
   ```bash
   .\build_whisperx_cpu.bat
   ```

### Automated Build (GitHub Actions)

The `.github/workflows/build_whisperx_portable.yml` workflow automatically builds both versions when:
- You push a tag matching `whisperx-v*` (e.g., `whisperx-v3.4.2`)
- Or manually trigger the workflow

Artifacts are uploaded to GitHub Releases.

## Package Contents

Each portable package contains:
```
whisperx_portable/
├── python313/              # Full Python 3.13 installation
│   ├── python.exe
│   ├── pythonw.exe
│   ├── python313.dll
│   └── ...
├── Scripts/                # Python scripts and executables
│   ├── pip.exe
│   ├── whisperx.exe       # WhisperX CLI (if available)
│   └── ...
└── Lib/
    └── site-packages/
        ├── whisperx/      # WhisperX package
        ├── torch/         # PyTorch
        ├── torchaudio/    # Audio processing
        └── ...            # All dependencies
```

## Usage in Scriptoria

When a user clicks "Install WhisperX" in Scriptoria:

1. App detects if user has NVIDIA GPU
2. Downloads appropriate package (CUDA or CPU)
3. Extracts to `whisperx_env/`
4. Ready to use - no installation!

## Updating

To update to a new WhisperX version:

1. Update version numbers in build scripts
2. Run build scripts or trigger GitHub Actions
3. Upload new packages to GitHub Releases
4. Update download URLs in `generate_captions.py`

## Testing

After building, test the package:

```bash
# Extract the zip
unzip whisperx_portable_win64_cuda128_v3.7.4.zip -d test_env

# Test Python
test_env\Scripts\python.exe --version

# Test WhisperX import
test_env\Scripts\python.exe -c "import whisperx; print(whisperx.__version__)"

# Test basic transcription (requires audio file)
test_env\Scripts\python.exe -c "import whisperx; print('WhisperX ready!')"
```

## File Hosting

Options for hosting the built packages:

1. **GitHub Releases** (recommended)
   - Free
   - 2GB file limit per file
   - Automatic CDN
   - URL: `https://github.com/YOUR_USERNAME/scriptoria/releases/download/whisperx-v3.7.4/whisperx_portable_win64_cuda128_v3.7.4.zip`

2. **Your own server/CDN**
   - Full control
   - No size limits
   - Bandwidth costs

3. **Cloud storage** (OneDrive, Google Drive, Dropbox)
   - Requires direct download links
   - May have bandwidth limits

## Troubleshooting

### Build fails with "out of disk space"
- Need ~10GB free for temporary files during build
- Clean up old build directories

### Package won't extract
- Ensure 7-Zip or WinRAR is installed
- File may be corrupted - re-download

### Python won't run after extraction
- Check antivirus didn't quarantine files
- Ensure extracted to a path without spaces or special characters
