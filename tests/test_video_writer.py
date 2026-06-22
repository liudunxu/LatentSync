"""Unit tests for the ffmpeg pipe video writer."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np


class TestVideoWriter(unittest.TestCase):
    def _import_writer(self):
        try:
            from latentsync.utils.util import write_video_via_ffmpeg
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional dependency missing: {exc.name}")
        return write_video_via_ffmpeg

    def test_write_video_via_ffmpeg_roundtrip(self):
        write_video_via_ffmpeg = self._import_writer()

        frames = []
        for i in range(8):
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            frame[:, :, 0] = i * 30  # varying red channel
            frames.append(frame)
        video_frames = np.stack(frames, axis=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.mp4")
            write_video_via_ffmpeg(out_path, video_frames, fps=8)
            self.assertTrue(os.path.isfile(out_path))
            self.assertGreater(os.path.getsize(out_path), 0)

    def test_empty_frames_raises(self):
        write_video_via_ffmpeg = self._import_writer()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.mp4")
            with self.assertRaises(ValueError):
                write_video_via_ffmpeg(out_path, np.array([]), fps=8)


if __name__ == "__main__":
    unittest.main()
