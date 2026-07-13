"""Track which videos have been processed by which preprocess step.

Persists a JSON manifest in each output directory. The pipeline
calls mark_done() after a video passes a step and has_done() before
re-processing, enabling:

  - Resume from a crash: skip videos that already passed
  - Incremental: when new videos are added, only process the new ones
  - Observability: per-step counts, fail lists, runtime stats

Usage:
    from preprocess.manifest import Manifest

    m = Manifest("/data/voxceleb2/resampled")
    m.mark_done("video1.mp4", meta={"fps": 25})
    if m.has_done("video1.mp4"):
        skip(video1)
    stats = m.stats()  # {step_name: {done, failed, total}}

Manifest file format (.preprocess_manifest.json):
{
  "schema_version": 1,
  "steps": {
    "remove_broken": {
      "video1.mp4": {"ts": 1234567890, "meta": {}},
      "video2.mp4": {"ts": 1234567891, "meta": {"error": "..."}}
    },
    "resample_fps": { ... },
    ...
  }
}
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


MANIFEST_FILENAME = ".preprocess_manifest.json"
SCHEMA_VERSION = 1


class Manifest:
    """Per-output-dir tracker for preprocess steps.

    Thread-safe-enough for multiprocessing.Pool children: each child
    process should call .load() fresh and writes are atomic via
    tmp-file rename.
    """

    def __init__(self, output_dir: str | Path, autosave: bool = True) -> None:
        self.output_dir = Path(output_dir)
        self.path = self.output_dir / MANIFEST_FILENAME
        self.autosave = autosave
        self.data: Dict[str, Any] = self._load()

    # ---- (de)serialization ----

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": SCHEMA_VERSION, "steps": {}}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("schema_version") != SCHEMA_VERSION:
                print(
                    f"[Manifest] schema mismatch in {self.path} (have "
                    f"{data.get('schema_version')}, want {SCHEMA_VERSION}). "
                    "Starting fresh."
                )
                return {"schema_version": SCHEMA_VERSION, "steps": {}}
            return data
        except (json.JSONDecodeError, OSError) as e:
            print(f"[Manifest] {self.path} corrupted ({e}); starting fresh.")
            return {"schema_version": SCHEMA_VERSION, "steps": {}}

    def save(self) -> None:
        if not self.autosave:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, sort_keys=True)
        os.replace(tmp, self.path)

    # ---- core API ----

    def _step_bucket(self, step: str) -> Dict[str, Any]:
        return self.data["steps"].setdefault(step, {})

    def mark_done(
        self,
        step: str,
        video_id: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record that `video_id` successfully passed `step`."""
        bucket = self._step_bucket(step)
        bucket[video_id] = {
            "ts": int(time.time()),
            "status": "done",
            "meta": meta or {},
        }
        self.save()

    def mark_failed(
        self,
        step: str,
        video_id: str,
        error: str,
    ) -> None:
        """Record that `video_id` failed at `step` (so we don't retry forever)."""
        bucket = self._step_bucket(step)
        bucket[video_id] = {
            "ts": int(time.time()),
            "status": "failed",
            "meta": {"error": error[:500]},
        }
        self.save()

    def has_done(self, step: str, video_id: str) -> bool:
        return (
            self.data.get("steps", {}).get(step, {}).get(video_id, {}).get("status")
            == "done"
        )

    def has_failed(self, step: str, video_id: str) -> bool:
        return (
            self.data.get("steps", {}).get(step, {}).get(video_id, {}).get("status")
            == "failed"
        )

    def done_ids(self, step: str) -> Set[str]:
        bucket = self.data.get("steps", {}).get(step, {})
        return {k for k, v in bucket.items() if v.get("status") == "done"}

    def failed_ids(self, step: str) -> Set[str]:
        bucket = self.data.get("steps", {}).get(step, {})
        return {k for k, v in bucket.items() if v.get("status") == "failed"}

    def reset_step(self, step: str) -> None:
        """Wipe all entries for a step (e.g. when re-running with new params)."""
        self.data["steps"].pop(step, None)
        self.save()

    def reset(self) -> None:
        """Wipe the entire manifest."""
        self.data = {"schema_version": SCHEMA_VERSION, "steps": {}}
        self.save()

    # ---- observability ----

    def stats(self, step: Optional[str] = None) -> Dict[str, Dict[str, int]]:
        """Return {step: {done, failed, pending=0}} counts.

        For overall summary, pass no argument. `pending` is left to the
        caller since it requires knowing the candidate video list.
        """
        out: Dict[str, Dict[str, int]] = {}
        steps = [step] if step else list(self.data.get("steps", {}).keys())
        for s in steps:
            bucket = self.data.get("steps", {}).get(s, {})
            done = sum(1 for v in bucket.values() if v.get("status") == "done")
            failed = sum(1 for v in bucket.values() if v.get("status") == "failed")
            out[s] = {"done": done, "failed": failed, "total": len(bucket)}
        return out

    def filter_pending(self, step: str, candidates: Iterable[str]) -> List[str]:
        """Return the subset of `candidates` that has NOT been done yet.

        A video is "pending" if it has no entry, or its entry is "failed".
        """
        done = self.done_ids(step)
        return [v for v in candidates if v not in done]

    def __repr__(self) -> str:
        s = self.stats()
        lines = [f"Manifest({self.output_dir})"]
        for step, c in s.items():
            lines.append(f"  {step}: done={c['done']} failed={c['failed']}")
        return "\n".join(lines)
