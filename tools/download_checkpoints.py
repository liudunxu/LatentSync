#!/usr/bin/env python3
"""Robustly download LatentSync checkpoints with resume and verification.

huggingface-cli occasionally reports success before the file is fully
written (especially for multi-GB checkpoints over unstable links). This
script uses huggingface_hub's resume-aware downloader and optionally
verifies the file size against the remote metadata.

Usage:
    python tools/download_checkpoints.py
    python tools/download_checkpoints.py --verify-size
    LATENTSYNC_CKPT_REPO=ByteDance/LatentSync-1.5 python tools/download_checkpoints.py
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from huggingface_hub import hf_hub_download, HfApi
    from huggingface_hub.utils import RepositoryNotFoundError, RevisionNotFoundError
except ImportError as e:
    print(f"ERROR: huggingface_hub is not installed: {e}")
    print("Run: pip install huggingface-hub")
    sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = os.environ.get("LATENTSYNC_CKPT_REPO", "ByteDance/LatentSync-1.6")

# (repo_filename, local_dir_under_repo_root)
DEFAULT_FILES: List[Tuple[str, Path]] = [
    ("latentsync_unet.pt", REPO_ROOT / "checkpoints"),
    ("whisper/tiny.pt", REPO_ROOT / "checkpoints"),
]


def _remote_size(repo_id: str, filename: str, repo_type: str = "model") -> Optional[int]:
    """Get expected file size in bytes from HuggingFace Hub."""
    try:
        info = HfApi().repo_info(repo_id, repo_type=repo_type)
        # repo_info doesn't list files; use hf_hub_url + HEAD or file metadata
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        import requests
        r = requests.head(url, allow_redirects=True, timeout=30)
        if r.status_code == 200:
            length = r.headers.get("Content-Length")
            if length:
                return int(length)
    except Exception as e:
        print(f"  ⚠️ could not fetch remote size for {filename}: {e}")
    return None


def _format_size(n: Optional[int]) -> str:
    if n is None:
        return "?"
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def download_one(
    repo_id: str,
    filename: str,
    local_dir: Path,
    verify_size: bool = False,
    max_retries: int = 3,
) -> Optional[Path]:
    """Download a single file, resume if partial, and optionally verify size."""
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📥 {repo_id}/{filename} -> {local_dir}")

    expected_size: Optional[int] = None
    if verify_size:
        expected_size = _remote_size(repo_id, filename)
        print(f"   expected size: {_format_size(expected_size)}")

    last_error: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            # hf_hub_download resumes automatically from the cache when possible.
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="model",
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            break
        except Exception as e:
            last_error = e
            print(f"   attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                print("   retrying ...")
    else:
        print(f"   ❌ failed after {max_retries} attempts: {last_error}")
        return None

    final_path = Path(downloaded_path)
    actual_size = final_path.stat().st_size
    print(f"   local size: {_format_size(actual_size)}")

    if expected_size is not None and actual_size != expected_size:
        print(
            f"   ❌ size mismatch: expected {_format_size(expected_size)}, "
            f"got {_format_size(actual_size)}"
        )
        print("   deleting partial file; run again to resume")
        final_path.unlink(missing_ok=True)
        return None

    print(f"   ✅ saved to {final_path}")
    return final_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download LatentSync checkpoints with resume and size verification."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO,
        help="HuggingFace repo to download from (default: %(default)s)",
    )
    parser.add_argument(
        "--verify-size",
        action="store_true",
        help="Compare downloaded file size with remote Content-Length",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Override files to download, e.g. 'latentsync_unet.pt whisper/tiny.pt'",
    )
    args = parser.parse_args()

    files: List[Tuple[str, Path]] = DEFAULT_FILES
    if args.files:
        files = [(f, REPO_ROOT / "checkpoints") for f in args.files]

    failed = False
    for filename, local_dir in files:
        path = download_one(
            repo_id=args.repo_id,
            filename=filename,
            local_dir=local_dir,
            verify_size=args.verify_size,
        )
        if path is None:
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
