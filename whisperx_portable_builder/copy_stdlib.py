"""
Copy Python standard library and extension DLLs into the portable venv.

Without this, the venv cannot run on a machine other than the one it was built
on, because pyvenv.cfg points to the build-host's Python installation.

After copying:
  - whisperx_portable/Lib/ contains the full stdlib (+ existing site-packages)
  - whisperx_portable/DLLs/ contains extension module DLLs (_ssl.pyd, etc.)

Usage:
    python copy_stdlib.py <base_python_dir> <portable_dir>
    e.g.
    python copy_stdlib.py "C:/hostedtoolcache/windows/Python/3.13.3/x64" whisperx_portable
"""

import sys
import os
import shutil


def main():
    if len(sys.argv) < 3:
        print("Usage: copy_stdlib.py <base_python_dir> <portable_dir>")
        sys.exit(1)

    base_python = sys.argv[1]
    portable_dir = sys.argv[2]

    # ── 1. Copy extension-module DLLs ─────────────────────────────────────────
    src_dlls = os.path.join(base_python, "DLLs")
    dst_dlls = os.path.join(portable_dir, "DLLs")

    if os.path.isdir(src_dlls):
        os.makedirs(dst_dlls, exist_ok=True)
        dll_count = 0
        for name in os.listdir(src_dlls):
            s = os.path.join(src_dlls, name)
            d = os.path.join(dst_dlls, name)
            try:
                shutil.copy2(s, d)
                dll_count += 1
            except Exception as exc:
                print(f"  Warning: could not copy DLLs/{name}: {exc}")
        print(f"Copied {dll_count} files into DLLs/")
    else:
        print(f"Warning: DLLs directory not found at {src_dlls}")

    # ── 2. Copy stdlib into Lib/ (skip site-packages – venv already has those) ─
    src_lib = os.path.join(base_python, "Lib")
    dst_lib = os.path.join(portable_dir, "Lib")

    if not os.path.isdir(src_lib):
        print(f"ERROR: stdlib Lib/ not found at {src_lib}")
        sys.exit(1)

    os.makedirs(dst_lib, exist_ok=True)
    lib_count = 0
    for name in os.listdir(src_lib):
        if name.lower() == "site-packages":
            continue  # keep the venv's installed packages intact

        s = os.path.join(src_lib, name)
        d = os.path.join(dst_lib, name)
        try:
            if os.path.isdir(s):
                if not os.path.exists(d):
                    shutil.copytree(s, d)
                # If it already exists (e.g. test/ was installed by a package),
                # merge rather than overwrite.
                else:
                    for root, dirs, files in os.walk(s):
                        rel = os.path.relpath(root, s)
                        dest_root = os.path.join(d, rel)
                        os.makedirs(dest_root, exist_ok=True)
                        for f in files:
                            sf = os.path.join(root, f)
                            df = os.path.join(dest_root, f)
                            if not os.path.exists(df):
                                shutil.copy2(sf, df)
            else:
                shutil.copy2(s, d)
            lib_count += 1
        except Exception as exc:
            print(f"  Warning: could not copy Lib/{name}: {exc}")

    print(f"Copied {lib_count} stdlib items into Lib/")
    print("Python stdlib and DLLs successfully bundled into portable package.")


if __name__ == "__main__":
    main()
