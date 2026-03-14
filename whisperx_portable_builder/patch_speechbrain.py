"""
Patch SpeechBrain's lazy importer for Windows path separators.

SpeechBrain's importutils.py checks whether a frame's filename ends with
"/inspect.py" to detect the stdlib inspect module. On Windows the path uses
backslashes, so the check always fails and causes infinite AttributeError
recursion. This patch normalises the filename to forward slashes first.

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

# Pattern matches the original one-liner guard that uses a raw "/" path check
pattern = (
    r'(?m)^( +)if importer_frame is not None'
    r' and importer_frame\.filename\.endswith\("/inspect\.py"\):\r?\n'
    r'\1    raise AttributeError\(\)'
)

replacement = (
    r'\1if importer_frame is not None:\n'
    r'\1    _fname = importer_frame.filename.replace("\\\\", "/")\n'
    r'\1    if _fname.endswith("/inspect.py"):\n'
    r'\1        raise AttributeError()'
)

new_src, count = re.subn(pattern, replacement, src)

if count == 0:
    # Check if the patch was already applied
    if 'replace("\\\\", "/")' in src or "replace(\"\\\\\\\\\", \"/\")" in src:
        print("SpeechBrain patch already applied — nothing to do.")
        sys.exit(0)
    print("ERROR: SpeechBrain patch pattern not found and patch not already applied.")
    print("The SpeechBrain source may have changed. Manual inspection required.")
    sys.exit(1)

target.write_text(new_src, encoding="utf-8")
print(f"SpeechBrain patch applied successfully ({count} substitution(s)) at {target}")
