# Integration Guide: Portable WhisperX Installer

This guide shows how to integrate the portable WhisperX installer into your `generate_captions.py`.

## Overview

The new approach **replaces** the complex Python installation process with a simple download-and-extract operation.

### Before (Complex)
1. Download Python installer
2. Run Python installer silently (can fail with error 1638)
3. Create virtual environment
4. Upgrade pip
5. Install PyTorch
6. Install WhisperX
7. **Total time: 15-20 minutes**

### After (Simple)
1. Download pre-built package
2. Extract zip file
3. Done!
4. **Total time: 5-10 minutes**

---

## Step 1: Update Configuration in generate_captions.py

Replace the existing Python installer configuration with portable package URLs:

```python
# ============================================================================
# WhisperX Portable Package Configuration
# ============================================================================

WHISPERX_VERSION = "3.4.2"
PACKAGE_VERSION = f"v{WHISPERX_VERSION}"

# GitHub repository for pre-built packages
# IMPORTANT: Update this to your actual GitHub username/repo!
GITHUB_REPO = "YOUR_USERNAME/scriptoria-whisperx-standalone"
GITHUB_RELEASE_URL = f"https://github.com/{GITHUB_REPO}/releases/download/whisperx-{PACKAGE_VERSION}"

# Package files
CUDA_PACKAGE = f"whisperx_portable_win64_cuda128_{PACKAGE_VERSION}.zip"
CPU_PACKAGE = f"whisperx_portable_win64_cpu_{PACKAGE_VERSION}.zip"

CUDA_URL = f"{GITHUB_RELEASE_URL}/{CUDA_PACKAGE}"
CPU_URL = f"{GITHUB_RELEASE_URL}/{CPU_PACKAGE}"
```

---

## Step 2: Add the Portable Installer Function

Copy the `download_and_extract_whisperx()` function from `portable_installer.py` into your `generate_captions.py`.

You can place it **before** your existing `setup_embedded_python()` function.

---

## Step 3: Update the Installation Thread

In your `InstallWhisperXThread` class, replace the call to `setup_embedded_python()` with `download_and_extract_whisperx()`:

### Find this code (around line 3020):

```python
if is_frozen:
    # Running as .exe - set up embedded Python environment
    self.progress.emit(f"Running as .exe - setting up embedded Python environment...\n")
    crash_log("CHECKPOINT: Setting up embedded Python\n")

    # Check for cancellation
    if self._cancelled:
        crash_log("CHECKPOINT: Installation cancelled before setup\n")
        self.finished.emit(False, "Installation cancelled by user", "")
        return

    # Create the directory
    os.makedirs(whisperx_dir, exist_ok=True)
    crash_log(f"CHECKPOINT: Created directory: {whisperx_dir}\n")

    # Set up embedded Python (download, extract, configure)
    venv_python = setup_embedded_python(
        whisperx_dir,
        self.progress.emit,
        lambda: self._cancelled  # Pass cancellation check
    )
```

### Replace with:

```python
if is_frozen:
    # Running as .exe - download pre-built WhisperX environment
    self.progress.emit(f"Running as .exe - downloading portable WhisperX package...\n")
    crash_log("CHECKPOINT: Downloading portable package\n")

    # Check for cancellation
    if self._cancelled:
        crash_log("CHECKPOINT: Installation cancelled before download\n")
        self.finished.emit(False, "Installation cancelled by user", "")
        return

    # Create the directory
    os.makedirs(whisperx_dir, exist_ok=True)
    crash_log(f"CHECKPOINT: Created directory: {whisperx_dir}\n")

    # Download and extract portable package
    venv_python = download_and_extract_whisperx(
        install_dir=whisperx_dir,
        use_cuda=self.install_cuda,  # Use the user's choice
        progress_callback=self.progress.emit,
        cancel_check=lambda: self._cancelled
    )
```

---

## Step 4: Optional Cleanup

You can now **delete or comment out** these old functions (they're no longer needed):

- `download_python_installer()`
- `setup_embedded_python()`
- `PYTHON_VERSION` constant
- `PYTHON_INSTALLER_URL` constant

---

## Step 5: Build and Upload Packages

### Option A: Manual Build (Windows Machine)

1. Open PowerShell in the `whisperx_portable_builder` folder
2. Run the build scripts:
   ```powershell
   .\build_whisperx_cuda.bat
   .\build_whisperx_cpu.bat
   ```
3. Wait for builds to complete (~15-30 minutes total)
4. Packages will be in `whisperx_portable_builder/output/`

### Option B: Automated Build (GitHub Actions)

1. Push the workflow file to GitHub:
   ```bash
   git add .github/workflows/build-scriptoria-whisperx-standalone.yml
   git commit -m "Add WhisperX portable build workflow"
   git push
   ```

2. Create and push a tag:
   ```bash
   git tag whisperx-v3.4.2
   git push origin whisperx-v3.4.2
   ```

3. GitHub Actions will automatically build both packages and create a release

---

## Step 6: Update GitHub Repository URL

After building and uploading:

1. Get your GitHub release URL (e.g., `https://github.com/YOUR_USERNAME/scriptoria-whisperx-standalone/releases/download/whisperx-v3.4.2/`)
2. Update `GITHUB_REPO` in your code:
   ```python
   GITHUB_REPO = "YOUR_USERNAME/scriptoria-whisperx-standalone"
   ```

---

## Testing

Test the new installer:

1. Delete your existing `whisperx_env` folder (if any)
2. Run Scriptoria
3. Click "Install WhisperX"
4. Monitor the progress - you should see:
   - GPU detection
   - Download progress (0-100%)
   - Extraction progress (0-100%)
   - Verification tests
   - "Installation Complete!"

---

## Troubleshooting

### "Download failed: HTTP 404"
- The package hasn't been uploaded to GitHub Releases yet
- Or the URL is incorrect
- Check the releases page: `https://github.com/YOUR_USERNAME/scriptoria-whisperx-standalone/releases`

### "Downloaded file is not a valid zip archive"
- Download was corrupted - try again
- Antivirus may have interfered

### "python.exe not found at expected location"
- Extraction didn't complete
- Zip structure is wrong (should extract directly, not in a subfolder)

### Download is very slow
- GitHub's CDN is usually fast, but can vary by location
- Consider hosting on your own CDN if needed

---

## Updating WhisperX

When a new WhisperX version is released:

1. Update version numbers in build scripts
2. Rebuild packages (manually or via GitHub Actions tag)
3. Upload to new GitHub release
4. Update `WHISPERX_VERSION` in `generate_captions.py`
5. Done!

Users will automatically get the new version on next install.

---

## Benefits of This Approach

✅ **No Windows Installer conflicts** - no error 1638
✅ **Faster installation** - pre-built vs building locally
✅ **More reliable** - same tested environment for everyone
✅ **Better UX** - just download and extract
✅ **Easier maintenance** - rebuild once, distribute to all
✅ **Smaller initial app size** - WhisperX downloaded only when needed
✅ **Works offline** - after first download, can work without internet
