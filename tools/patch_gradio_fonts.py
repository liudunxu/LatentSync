"""Patch installed Gradio frontend to avoid loading Google Fonts from CDN.

Run once after `pip install -r requirements.txt` / `pip install gradio`:

    python tools/patch_gradio_fonts.py

This replaces Google Fonts URLs in Gradio's minified frontend JS with empty
data URIs and removes <link rel="preconnect"> tags from HTML templates.  The
page will then fall back to system fonts (see gradio_finetune.py CSS).
"""
import re
import shutil
import sys
from pathlib import Path

import gradio

GRADIO_ROOT = Path(gradio.__file__).resolve().parent
EMPTY_CSS_DATA_URI = 'data:text/css;base64,Lg=='  # harmless "."


def patch_js_files() -> int:
    """Replace Google Fonts CSS URLs with local empty data URIs."""
    patched = 0
    js_dir = GRADIO_ROOT / "templates" / "frontend" / "assets"
    if not js_dir.exists():
        print(f"[warn] assets dir not found: {js_dir}")
        return 0

    targets = [
        "https://fonts.googleapis.com/css2?family=Source+Sans+Pro",
        "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono",
    ]

    for path in js_dir.rglob("*.js"):
        text = path.read_text(encoding="utf-8")
        if not any(t in text for t in targets):
            continue
        bak = path.with_suffix(path.suffix + ".googlefonts.bak")
        if not bak.exists():
            shutil.copy2(path, bak)
        new_text = text
        for target in targets:
            # Keep the opening quote and replace the rest of the URL up to the closing quote.
            new_text = re.sub(
                re.escape(target) + r'[^"\']*',
                EMPTY_CSS_DATA_URI,
                new_text,
            )
        path.write_text(new_text, encoding="utf-8")
        print(f"[patched] {path}")
        patched += 1
    return patched


def patch_html_files() -> int:
    """Remove Google Fonts preconnect <link> tags from HTML templates."""
    patched = 0
    html_dirs = [
        GRADIO_ROOT / "templates" / "frontend",
        GRADIO_ROOT / "_frontend_code" / "lite",
    ]
    for html_dir in html_dirs:
        if not html_dir.exists():
            continue
        for path in html_dir.rglob("*.html"):
            text = path.read_text(encoding="utf-8")
            if "fonts.googleapis.com" not in text and "fonts.gstatic.com" not in text:
                continue
            bak = path.with_suffix(path.suffix + ".googlefonts.bak")
            if not bak.exists():
                shutil.copy2(path, bak)
            # Remove preconnect/link tags pointing to Google Fonts/gstatic.
            new_text = re.sub(
                r'<link[^>]*(?:fonts\.googleapis\.com|fonts\.gstatic\.com)[^>]*>',
                '',
                text,
                flags=re.IGNORECASE,
            )
            path.write_text(new_text, encoding="utf-8")
            print(f"[patched] {path}")
            patched += 1
    return patched


def main() -> int:
    print(f"Gradio root: {GRADIO_ROOT}")
    js_patched = patch_js_files()
    html_patched = patch_html_files()
    total = js_patched + html_patched
    if total == 0:
        print("No Google Fonts references found; nothing to patch.")
    else:
        print(f"Patched {total} file(s). Restart Gradio to take effect.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
