#!/usr/bin/env python3
"""
WhisperX CLI wrapper for Scriptoria

Purpose:
- Provide a simple command-line interface that mirrors the existing
  faster-whisper-xxl executable flags closely enough for Scriptoria's
  Generate Captions UI to invoke.
- Produce identical outputs: `<basename>.srt` and `<basename>.json`
  in the specified `--output_dir`.
"""

import argparse
import json
import os
import sys
import time
import traceback
import copy
import typing
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
    try:
        from huggingface_hub.utils import LocalEntryNotFoundError
    except Exception:
        class LocalEntryNotFoundError(Exception):
            """Fallback placeholder if huggingface_hub does not expose LocalEntryNotFoundError."""
            pass
except Exception:
    snapshot_download = None
    class LocalEntryNotFoundError(Exception):
        """Fallback when huggingface_hub is unavailable."""
        pass

# Pre-setup DLL search paths for embedded Python on Windows before importing torch
if os.name == 'nt':
    try:
        rt_dir = Path(sys.executable).parent
        torch_lib_guess = rt_dir / 'Lib' / 'site-packages' / 'torch' / 'lib'
        if torch_lib_guess.is_dir():
            try:
                os.add_dll_directory(str(torch_lib_guess))
            except Exception:
                pass
            os.environ['PATH'] = str(torch_lib_guess) + os.pathsep + os.environ.get('PATH', '')
    except Exception:
        pass

try:
    from torch.serialization import add_safe_globals
    from omegaconf import DictConfig, ListConfig
    try:
        from omegaconf.base import ContainerMetadata
    except Exception:
        ContainerMetadata = None

    globals_to_allow = [DictConfig, ListConfig]
    if ContainerMetadata is not None:
        globals_to_allow.append(ContainerMetadata)

    globals_to_allow.append(typing.Any)
    add_safe_globals(globals_to_allow)
except Exception:
    pass


try:
    import torch
    _original_torch_load = torch.load

    def _torch_load_allowlist(*args, **kwargs):
        kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _torch_load_allowlist
except Exception:
    pass

def seconds_to_srt_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(segments, output_path: Path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, seg in enumerate(segments, 1):
            start = seconds_to_srt_time(float(seg.get('start', 0.0)))
            end = seconds_to_srt_time(float(seg.get('end', 0.0)))
            text = (seg.get('text') or '').strip()
            # Add diarization label if present
            spk = seg.get('speaker')
            if spk:
                text = f"[{spk}] {text}"
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="WhisperX CLI for Scriptoria")
    p.add_argument('input_files', nargs='+', help='Audio/Video file(s) to transcribe (supports batch processing)')
    p.add_argument('-m', '--model', default='large-v3-turbo', help='Model name (e.g., large-v3-turbo)')
    p.add_argument('--task', default='transcribe', help='Task (transcribe/translate) - translate not implemented')
    p.add_argument('-l', '--language', default=None, help='Language code (None for auto)')
    p.add_argument('--compute_type', default=None, help='Compute type (e.g., float16, int8, int8_float16)')
    p.add_argument('--device', default='cpu', help='Device (cuda/cpu)')
    p.add_argument('--output_dir', default=None, help='Output directory (per-file if not specified)')
    p.add_argument('--output_format', nargs='+', default=['srt'], help='One or more of: srt json')
    p.add_argument('--word_timestamps', default='False', help='Enable word timestamps when exporting json')
    # Accepted for compatibility (ignored or best-effort):
    p.add_argument('--vad_filter', default=None)
    p.add_argument('--vad_method', default=None)
    p.add_argument('--diarize', default=None, help='Enable diarization (value ignored; presence enables)')
    p.add_argument('--num_speakers', default=None)
    p.add_argument('--hf_token', default=os.environ.get('HF_TOKEN'))
    return p.parse_args(argv)


def ts():
    return time.strftime('%H:%M:%S')


def log(msg: str):
    print(f"[{ts()}] {msg}")


def normalize_speakers(result: dict) -> dict:
    """Map diarization speaker IDs to SPEAKER_01, SPEAKER_02, ..."""
    try:
        if not result or 'segments' not in result:
            return result
        normalized = copy.deepcopy(result)
        alias_map: dict[str, str] = {}

        def alias(raw: str | None) -> str | None:
            if not raw:
                return None
            if raw not in alias_map:
                alias_map[raw] = f"SPEAKER_{len(alias_map) + 1:02d}"
            return alias_map[raw]

        for segment in normalized.get('segments', []):
            seg_alias = alias(segment.get('speaker') or segment.get('speaker_id'))
            if seg_alias:
                segment['speaker'] = seg_alias
            words = segment.get('words') or []
            seg_end = float(segment.get('end', segment.get('start', 0.0)))
            last_word_end = seg_end
            for idx, word in enumerate(words):
                word_alias = alias(word.get('speaker') or segment.get('speaker'))
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

                next_start = None
                if idx + 1 < len(words):
                    try:
                        ns = words[idx + 1].get('start')
                        if ns is not None:
                            next_start = float(ns)
                    except Exception:
                        next_start = None
                if next_start is None or next_start <= start_val:
                    next_start = seg_end if seg_end > start_val else None

                if end_val is None or end_val <= start_val:
                    if next_start is not None and next_start > start_val:
                        end_val = next_start
                    elif seg_end > start_val:
                        end_val = seg_end
                    else:
                        end_val = start_val

                word['end'] = end_val
                duration_val = max(end_val - start_val, 0.0)
                word['duration'] = duration_val
                if end_val > last_word_end:
                    last_word_end = end_val

            if words:
                segment['end'] = last_word_end
        return normalized
    except Exception:
        return result


def process_single_file(input_path: Path, args, model, whisperx, device, align_device, diarize_model, num_speakers):
    """Process a single audio/video file with pre-loaded models."""
    log(f"Processing: {input_path.name}")

    # Determine output directory
    out_dir = Path(args.output_dir) if args.output_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    base = input_path.stem

    # Load audio
    log("Loading audio...")
    audio = whisperx.load_audio(str(input_path))

    # Transcribe
    log("Transcribing...")
    t1 = time.perf_counter()
    sys.stdout.flush()

    result = model.transcribe(
        audio,
        batch_size=16,
        language=(None if (args.language in (None, '', 'auto')) else args.language),
        print_progress=True
    )

    sys.stdout.flush()
    log(f"Transcription done in {time.perf_counter()-t1:.1f}s; segments={len(result.get('segments', []))}")

    # Alignment for word timestamps
    if args.word_timestamps.lower() == 'true' or 'json' in [f.lower() for f in args.output_format]:
        log(f"Aligning for word-level timestamps... (device={align_device})")
        t2 = time.perf_counter()
        model_a, metadata = whisperx.load_align_model(
            language_code=result.get('language', args.language or 'en'),
            device=align_device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            align_device,
            return_char_alignments=True,
        )
        log(f"Alignment done in {time.perf_counter()-t2:.1f}s")

    # Optional diarization
    if diarize_model is not None:
        try:
            log("Running diarization...")
            t3 = time.perf_counter()
            diarize_kwargs = {}
            if num_speakers and num_speakers > 0:
                diarize_kwargs['min_speakers'] = num_speakers
                diarize_kwargs['max_speakers'] = num_speakers
            try:
                diarize_segments = diarize_model(audio, **diarize_kwargs)
            except Exception:
                diarize_segments = diarize_model(str(input_path), **diarize_kwargs)

            result = whisperx.assign_word_speakers(diarize_segments, result)
            log(f"Diarization done in {time.perf_counter()-t3:.1f}s")
        except Exception as e:
            log(f"Diarization skipped (error: {e})")

    result = normalize_speakers(result)

    # Prepare output
    segments = result.get('segments', [])
    language = result.get('language') or (args.language or 'en')

    # Write JSON if requested
    if any(fmt.lower() == 'json' for fmt in args.output_format):
        json_path = out_dir / f"{base}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'language': language,
                'segments': segments
            }, f, indent=2, ensure_ascii=False)
        log(f"Saved JSON: {json_path}")

    # Write SRT if requested
    if any(fmt.lower() == 'srt' for fmt in args.output_format):
        srt_path = out_dir / f"{base}.srt"
        write_srt(segments, srt_path)
        log(f"Saved SRT: {srt_path}")

    log(f"Transcription complete. Files saved to {out_dir}")
    return True


def main(argv=None) -> int:
    args = parse_args(argv)
    log("WhisperX CLI starting")
    log(f"Python: {sys.executable}")

    # Validate all input files exist
    input_paths = []
    for f in args.input_files:
        p = Path(f)
        if not p.exists():
            print(f"ERROR: input file does not exist: {p}")
            return 2
        input_paths.append(p)

    is_batch = len(input_paths) > 1
    if is_batch:
        log(f"Batch mode: {len(input_paths)} files to process")

    num_speakers = None
    if args.num_speakers not in (None, "", "0", 0, "auto"):
        try:
            num_speakers = int(args.num_speakers)
        except Exception:
            log(f"Warning: could not interpret --num_speakers='{args.num_speakers}'")
            num_speakers = None

    # Import whisperx lazily and provide actionable error
    try:
        t_import = time.perf_counter()
        import torch  # noqa: F401
        try:
            if hasattr(torch.backends, 'cuda'):
                torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends.cuda, 'allow_tf32'):  # PyTorch 2.x
                    torch.backends.cuda.allow_tf32 = True
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        # On Windows + embedded Python, ensure torch DLLs (e.g., cuDNN) are discoverable
        try:
            if os.name == 'nt':
                torch_dir = os.path.dirname(torch.__file__)
                torch_lib = os.path.join(torch_dir, 'lib')
                if os.path.isdir(torch_lib):
                    try:
                        os.add_dll_directory(torch_lib)  # Python 3.8+
                    except Exception:
                        pass
                    os.environ['PATH'] = torch_lib + os.pathsep + os.environ.get('PATH', '')
        except Exception:
            pass
        log(f"Imported torch in {time.perf_counter()-t_import:.2f}s")
        t_import = time.perf_counter()
        import whisperx
        log(f"Imported whisperx in {time.perf_counter()-t_import:.2f}s")
        t_import = time.perf_counter()
        import ctranslate2 as ctrans
        log(f"Imported ctranslate2 in {time.perf_counter()-t_import:.2f}s")
        try:
            from huggingface_hub import constants as hf_constants
            hf_cache = os.environ.get('HF_HOME') or os.environ.get('HF_HUB_CACHE') or getattr(hf_constants, 'HF_HUB_CACHE', None)
        except Exception:
            hf_cache = None
    except Exception as e:
        log("WhisperX not installed or failed to import. Please install with:")
        log(f"  {sys.executable} -m pip install whisperx")
        log(f"Import error: {e}")
        return 3

    # Device + compute type
    try:
        import torch
        device = 'cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu'
        log(f"torch.cuda.is_available = {torch.cuda.is_available()}")
        if device == 'cuda':
            try:
                gpu_name = torch.cuda.get_device_name(0)
                total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                log(f"CUDA available: {gpu_name} ({total_mem_gb:.1f} GB)")
            except Exception:
                pass
    except Exception:
        device = 'cpu'

    # Pick a supported compute type with graceful fallback
    requested_ct = args.compute_type
    if requested_ct is None:
        requested_ct = 'float16' if device == 'cuda' else 'int8'

    def pick_compute_type(dev: str, requested: str) -> tuple[str, str]:
        try:
            supported = ctrans.get_supported_compute_types(dev)
        except Exception:
            supported = []
        # Preference order by device
        prefs = ['float16', 'int8_float16', 'int8', 'float32'] if dev == 'cuda' else ['int8', 'float32', 'int16']
        # If requested is supported, use it
        if requested in supported:
            return dev, requested
        # Otherwise pick the first preferred that is supported
        for ct in prefs:
            if ct in supported:
                print(f"Requested compute_type '{requested}' not supported on {dev}. Falling back to '{ct}'.")
                return dev, ct
        # If nothing supported for this device, try CPU
        if dev == 'cuda':
            try:
                cpu_supported = ctrans.get_supported_compute_types('cpu')
            except Exception:
                cpu_supported = []
            for ct in ['int8', 'float32', 'int16']:
                if ct in cpu_supported:
                    print("CUDA compute types unavailable. Falling back to CPU with compute_type '", ct, "'.")
                    return 'cpu', ct
        # Last resort
        return dev, requested

    device, compute_type = pick_compute_type(device, requested_ct)
    log(f"Using device={device}, compute_type={compute_type}")
    if hf_cache:
        log(f"HF cache: {hf_cache}")
    if args.diarize is not None:
        log("Speaker diarization: enabled")
    else:
        log("Speaker diarization: disabled")

    # Load main model (once for all files)
    log("Loading WhisperX model... (first run may download ~2–3GB)")
    t0 = time.perf_counter()
    model = whisperx.load_model(
        args.model,
        device,
        compute_type=compute_type,
        language=(None if (args.language in (None, '', 'auto')) else args.language)
    )
    log(f"Model ready in {time.perf_counter()-t0:.1f}s")

    # Determine alignment device
    align_device = device
    try:
        import torch
        cudnn_ok = getattr(torch.backends, 'cudnn', None) and torch.backends.cudnn.is_available()
        if device == 'cuda' and not cudnn_ok:
            align_device = 'cpu'
            log("cuDNN not available for PyTorch ops; alignment will use CPU.")
    except Exception:
        if device == 'cuda':
            align_device = 'cpu'
            log("PyTorch cuDNN check failed; alignment will use CPU.")

    # Load diarization model once if requested
    diarize_model = None
    if args.diarize is not None:
        token_in_use = args.hf_token or os.environ.get("HF_TOKEN")
        if not token_in_use:
            log("Diarization requested but no Hugging Face token provided; skipping diarization.")
        else:
            safe_hint = f"{token_in_use[:6]}…" if isinstance(token_in_use, str) and len(token_in_use) > 6 else "provided"
            log(f"Using Hugging Face token ({safe_hint}) for diarization downloads.")
            try:
                log("Loading diarization model (downloads on first use)...")
                t3 = time.perf_counter()

                # Ensure diarization model cached locally
                cache_dir = (
                    os.environ.get("HF_HOME")
                    or os.environ.get("HF_HUB_CACHE")
                    or os.environ.get("HUGGINGFACE_HUB_CACHE")
                )
                if snapshot_download is not None:
                    try:
                        log("Checking local cache for diarization model...")
                        snapshot_download(
                            "pyannote/speaker-diarization-3.1",
                            token=token_in_use,
                            cache_dir=cache_dir,
                            local_files_only=True,
                        )
                        log("Diarization model found in local cache.")
                    except LocalEntryNotFoundError:
                        log("Local cache missing diarization model; downloading from Hugging Face...")
                        snapshot_download(
                            "pyannote/speaker-diarization-3.1",
                            token=token_in_use,
                            cache_dir=cache_dir,
                            local_files_only=False,
                        )
                        log("Diarization model downloaded successfully.")

                # WhisperX API compatibility: try attribute, then module import
                DiarizationPipeline = None
                try:
                    DiarizationPipeline = getattr(whisperx, 'DiarizationPipeline')
                except Exception:
                    pass
                if DiarizationPipeline is None:
                    try:
                        from whisperx.diarize import DiarizationPipeline as _DP
                        DiarizationPipeline = _DP
                        log("Using whisperx.diarize.DiarizationPipeline (compat mode)")
                    except Exception:
                        pass
                if DiarizationPipeline is None:
                    raise RuntimeError("DiarizationPipeline not available in this WhisperX version")

                # Choose diarization device
                diar_device = device
                try:
                    import torch
                    if device == 'cuda' and (not getattr(torch.backends, 'cudnn', None) or not torch.backends.cudnn.is_available()):
                        diar_device = 'cpu'
                        log("cuDNN not available; diarization will use CPU.")
                except Exception:
                    if device == 'cuda':
                        diar_device = 'cpu'

                # pyannote>=3.x renamed use_auth_token → token; try both
                try:
                    diarize_model = DiarizationPipeline(use_auth_token=token_in_use, device=diar_device)
                except TypeError:
                    diarize_model = DiarizationPipeline(token=token_in_use, device=diar_device)
                log(f"Diarization model ready in {time.perf_counter()-t3:.1f}s")
            except Exception as e:
                log(f"Diarization model loading failed (error: {e}); diarization will be skipped.")
                log(traceback.format_exc())
                diarize_model = None

    # Process each file
    success_count = 0
    fail_count = 0
    for idx, input_path in enumerate(input_paths):
        if is_batch:
            log(f"\n{'='*50}")
            log(f"File {idx + 1}/{len(input_paths)}: {input_path.name}")
            log(f"{'='*50}")

        try:
            if process_single_file(input_path, args, model, whisperx, device, align_device, diarize_model, num_speakers):
                success_count += 1
        except Exception as e:
            log(f"ERROR processing {input_path.name}: {e}")
            log(traceback.format_exc())
            fail_count += 1

    # Summary for batch mode
    if is_batch:
        log(f"\n{'='*50}")
        log(f"Batch complete: {success_count} succeeded, {fail_count} failed")
        log(f"{'='*50}")

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
