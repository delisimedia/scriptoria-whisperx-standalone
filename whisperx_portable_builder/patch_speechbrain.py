"""
Patch SpeechBrain's lazy importer for Windows path separators.

SpeechBrain's importutils.py checks whether a frame's filename ends with
"/inspect.py" to detect the stdlib inspect module. On Windows the path uses
backslashes, so the check always fails and causes infinite AttributeError
recursion. This patch normalises the filename to forward slashes first.

Handles two forms that appear across SpeechBrain versions:

  Old (0.5.x, single-line):
    if importer_frame is not None and importer_frame.filename.endswith("/inspect.py"):
        raise AttributeError()

  New (1.0.x, multi-line):
    if importer_frame is not None and importer_frame.filename.endswith(
        "/inspect.py"
    ):
        raise AttributeError()

Usage:
    python patch_speechbrain.py <path_to_whisperx_portable>
"""

import re
import sys
from pathlib import Path

portable_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("whisperx_portable")

target = portable_dir / "Lib" / "site-packages" / "speechbrain" / "utils" / "importutils.py"

if not target.exists():
    print(f"SpeechBrain importutils.py not found at {target} — skipping patch.")
    sys.exit(0)

src = target.read_text(encoding="utf-8")

# Already patched?
if 'replace("\\\\", "/")' in src:
    print("SpeechBrain patch already applied — nothing to do.")
    sys.exit(0)

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
# if importer_frame is not None and importer_frame.filename.endswith(
#     "/inspect.py"
# ):
#     raise AttributeError()
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
    print("WARNING: SpeechBrain patch pattern not found — the source may have")
    print("already been fixed upstream or uses an unrecognised layout.")
    print("Continuing without patching (will fail at runtime if the bug is present).")
    sys.exit(0)  # non-fatal: don't break the build

target.write_text(new_src, encoding="utf-8")
print(f"SpeechBrain patch applied successfully ({count} substitution(s)) at {target}")
