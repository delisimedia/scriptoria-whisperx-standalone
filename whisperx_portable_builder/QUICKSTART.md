# Quick Start: Building WhisperX Portable Packages

## Prerequisites

- Windows 10/11 machine
- Python 3.12+ installed and in PATH
- ~10 GB free disk space
- Internet connection

## Option 1: Manual Build (Recommended for First Time)

### Step 1: Build the Packages

Open PowerShell or Command Prompt in the `whisperx_portable_builder` folder:

```bash
# Build CUDA version (~2.5 GB, takes 15-20 minutes)
.\build_whisperx_cuda.bat

# Build CPU version (~800 MB, takes 10-15 minutes)
.\build_whisperx_cpu.bat
```

**What happens:**
1. Creates a Python virtual environment
2. Installs PyTorch (CUDA or CPU version)
3. Installs WhisperX and all dependencies
4. Compresses everything into a zip file
5. Saves to `output/` folder

### Step 2: Upload to GitHub

1. Go to your GitHub repository
2. Click "Releases" → "Create a new release"
3. Set tag to `whisperx-v3.7.4`
4. Upload both zip files:
   - `whisperx_portable_win64_cuda128_v3.7.4.zip`
   - `whisperx_portable_win64_cpu_v3.7.4.zip`
5. Publish release

### Step 3: Update Your Code

In `generate_captions.py`, update the GitHub repository URL:

```python
GITHUB_REPO = "YOUR_USERNAME/scriptoria"  # or separate repo name
```

### Step 4: Test!

1. Delete any existing `whisperx_env` folder
2. Run Scriptoria
3. Click "Install WhisperX"
4. Watch it download and extract automatically!

---

## Option 2: Automated Build (GitHub Actions)

### Step 1: Push the Workflow

```bash
git add .github/workflows/build-scriptoria-whisperx-standalone.yml
git commit -m "Add WhisperX portable build automation"
git push
```

### Step 2: Trigger the Build

Create and push a tag:

```bash
git tag whisperx-v3.7.4
git push origin whisperx-v3.7.4
```

**OR** manually trigger in GitHub:
1. Go to Actions tab
2. Select "Build Scriptoria WhisperX Standalone"
3. Click "Run workflow"
4. Enter version: `3.7.4`
5. Click "Run workflow"

### Step 3: Wait for Build

GitHub Actions will:
1. Build both CUDA and CPU versions (runs in parallel)
2. Create a GitHub Release automatically
3. Upload both packages to the release

**Build time:** ~20-30 minutes

### Step 4: Verify

1. Check the Actions tab for build status
2. When complete, check Releases - you should see the new release with both zip files
3. Update `GITHUB_REPO` in your code if needed
4. Test!

---

## Testing the Packages

### Test Before Integrating

1. Download one of the built packages
2. Extract to a test folder
3. Run tests:

```bash
# Test Python
test_folder\Scripts\python.exe --version

# Test WhisperX
test_folder\Scripts\python.exe -c "import whisperx; print(whisperx.__version__)"

# Test PyTorch (CUDA version)
test_folder\Scripts\python.exe -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

If all tests pass, you're good to go!

---

## File Sizes

Expected sizes after building:

| Package | Compressed | Extracted |
|---------|-----------|-----------|
| CUDA    | ~2.5 GB   | ~6 GB     |
| CPU     | ~800 MB   | ~2.5 GB   |

---

## Troubleshooting

### "python: command not found"
- Python is not in PATH
- Install Python 3.12+ and add to PATH

### "Virtual environment creation failed"
- Python version too old (need 3.12+)
- Run: `python --version` to check

### Build script fails during PyTorch install
- Internet connection issue
- Try again or use a VPN if PyTorch CDN is blocked in your region

### Zip file too large for GitHub
- GitHub has 2GB limit per file
- CUDA version might exceed this (2.5GB)
- Options:
  1. Split the file (not recommended)
  2. Host on your own server/CDN
  3. Use Git LFS (Large File Storage)

### GitHub Actions quota exceeded
- Free accounts: 2000 minutes/month
- Each build uses ~60 minutes (both CUDA + CPU)
- Limit builds or use manual builds

---

## Next Steps

After successfully building and testing:

1. ✅ Integrate into `generate_captions.py` (see INTEGRATION_GUIDE.md)
2. ✅ Test the full installation flow in Scriptoria
3. ✅ Update documentation for users
4. ✅ Set up automatic rebuilds when WhisperX updates

---

## Updating for New WhisperX Versions

When WhisperX releases a new version:

1. Update version in build scripts:
   ```batch
   set WHISPERX_VERSION=3.5.0
   set PYTORCH_VERSION=2.8.0  # Check WhisperX requirements
   ```

2. Rebuild packages (manual or via GitHub Actions)

3. Upload to new release: `whisperx-v3.5.0`

4. Update `generate_captions.py`:
   ```python
   WHISPERX_VERSION = "3.5.0"
   ```

5. Users get new version automatically on next install!

---

## Cost Analysis

### Manual Builds
- **Cost:** $0 (uses your own machine)
- **Time:** ~30 minutes per build
- **Internet:** ~3 GB download for packages

### GitHub Actions
- **Cost:** Free (within 2000 min/month limit)
- **Time:** ~30 minutes automated
- **Storage:** Free (releases are free storage)

### Hosting
- **GitHub Releases:** Free, 2GB file limit, CDN included
- **Your server:** Variable cost, no limits
- **Cloud storage:** Usually free tier available

---

## Support

If you encounter issues:

1. Check the logs in the build output
2. Verify Python version: `python --version`
3. Check disk space: Should have ~10 GB free
4. Test internet connection to PyTorch CDN
5. Check GitHub Actions logs if using automation

For WhisperX-specific issues, see: https://github.com/m-bain/whisperX
