"""Unit tests for Audio2Feature embed cache key generation."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch


class TestAudioEmbedCache(unittest.TestCase):
    def _import_class(self):
        try:
            from latentsync.whisper.audio2feature import Audio2Feature
        except ModuleNotFoundError as exc:
            self.skipTest(f"Audio2Feature import failed: {exc.name}")
        return Audio2Feature

    def _make_temp_audio(self, content: bytes, suffix: str = ".wav") -> str:
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(content)
        except Exception:
            os.remove(path)
            raise
        return path

    def test_cache_path_uses_real_extension(self):
        Audio2Feature = self._import_class()
        cache_dir = "/tmp/cache"
        for suffix in (".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"):
            audio_path = f"/tmp/test_audio{suffix}"
            path = Audio2Feature._compute_cache_path(cache_dir, audio_path)
            self.assertTrue(
                path.endswith("_embeds.pt"),
                f"Expected _embeds.pt suffix for {suffix}, got {path}",
            )
            self.assertIn("test_audio", path)

    def test_same_content_same_cache_name(self):
        Audio2Feature = self._import_class()
        content = b"fake audio content"
        path1 = self._make_temp_audio(content, ".wav")
        path2 = self._make_temp_audio(content, ".wav")
        try:
            cache1 = Audio2Feature._compute_cache_path("/tmp", path1)
            cache2 = Audio2Feature._compute_cache_path("/tmp", path2)
            self.assertEqual(cache1, cache2)
        finally:
            os.remove(path1)
            os.remove(path2)

    def test_different_content_different_cache_name(self):
        Audio2Feature = self._import_class()
        path1 = self._make_temp_audio(b"content A", ".wav")
        path2 = self._make_temp_audio(b"content B", ".wav")
        try:
            cache1 = Audio2Feature._compute_cache_path("/tmp", path1)
            cache2 = Audio2Feature._compute_cache_path("/tmp", path2)
            self.assertNotEqual(cache1, cache2)
        finally:
            os.remove(path1)
            os.remove(path2)

    def test_long_stem_truncated(self):
        Audio2Feature = self._import_class()
        long_stem = "a" * 200
        audio_path = f"/tmp/{long_stem}.wav"
        cache = Audio2Feature._compute_cache_path("/tmp", audio_path)
        basename = os.path.basename(cache)
        # stem portion should be truncated to 64 chars + hash + suffix.
        self.assertLessEqual(len(basename), 64 + 1 + 8 + len("_embeds.pt"))

    def test_audio2feat_uses_cache_on_second_call(self):
        Audio2Feature = self._import_class()
        content = b"fake audio for cache hit"
        audio_path = self._make_temp_audio(content, ".wav")
        cache_dir = tempfile.mkdtemp()
        try:
            dummy_feat = torch.randn(100, 384)
            with mock.patch("latentsync.whisper.audio2feature.load_model"):
                encoder = Audio2Feature(
                    model_path="dummy.pt",
                    device="cpu",
                    audio_embeds_cache_dir=cache_dir,
                )
                encoder._audio2feat = mock.Mock(return_value=dummy_feat)

                # First call should compute and save cache.
                out1 = encoder.audio2feat(audio_path)
                self.assertTrue(torch.equal(out1, dummy_feat))
                self.assertEqual(encoder._audio2feat.call_count, 1)

                # Second call should load from cache without calling _audio2feat.
                out2 = encoder.audio2feat(audio_path)
                self.assertTrue(torch.equal(out2, dummy_feat))
                self.assertEqual(encoder._audio2feat.call_count, 1)
        finally:
            os.remove(audio_path)
            for f in os.listdir(cache_dir):
                os.remove(os.path.join(cache_dir, f))
            os.rmdir(cache_dir)


if __name__ == "__main__":
    unittest.main()
