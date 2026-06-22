"""Unit tests for quality_preset_override mapping in api.py."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestQualityPreset(unittest.TestCase):
    def _import(self):
        try:
            from api import QUALITY_PRESETS, _apply_quality_preset, LipSyncRequest
        except ModuleNotFoundError as exc:
            self.skipTest(f"api import failed: {exc.name}")
        return QUALITY_PRESETS, _apply_quality_preset, LipSyncRequest

    def test_known_presets_have_required_fields(self):
        QUALITY_PRESETS, _, _ = self._import()
        for name, preset in QUALITY_PRESETS.items():
            for field in (
                "color_match_strength",
                "mouth_detail_strength",
                "mouth_sharpen_strength",
                "mouth_temporal_stabilization_strength",
            ):
                self.assertIn(field, preset, f"{name} preset missing {field}")
                self.assertGreaterEqual(preset[field], 0.0)
                self.assertLessEqual(preset[field], 1.0)

    def test_preset_maps_correctly(self):
        _, _apply_quality_preset, LipSyncRequest = self._import()
        payload = LipSyncRequest(
            video_url="http://example.com/v.mp4",
            audio_url="http://example.com/a.wav",
            quality_preset_override="sharp",
        )
        overrides = _apply_quality_preset(payload)
        self.assertEqual(overrides["mouth_sharpen_strength"], 0.45)
        self.assertEqual(overrides["mouth_temporal_stabilization_strength"], 0.10)

    def test_no_preset_returns_empty(self):
        _, _apply_quality_preset, LipSyncRequest = self._import()
        payload = LipSyncRequest(
            video_url="http://example.com/v.mp4",
            audio_url="http://example.com/a.wav",
        )
        overrides = _apply_quality_preset(payload)
        self.assertEqual(overrides, {})

    def test_unknown_preset_warns_and_returns_empty(self):
        _, _apply_quality_preset, LipSyncRequest = self._import()
        payload = LipSyncRequest(
            video_url="http://example.com/v.mp4",
            audio_url="http://example.com/a.wav",
            quality_preset_override="ultra_hd",
        )
        overrides = _apply_quality_preset(payload)
        self.assertEqual(overrides, {})

    def test_explicit_field_overrides_preset(self):
        _, _apply_quality_preset, LipSyncRequest = self._import()
        payload = LipSyncRequest(
            video_url="http://example.com/v.mp4",
            audio_url="http://example.com/a.wav",
            quality_preset_override="sharp",
            mouth_sharpen_strength=0.05,
        )
        overrides = _apply_quality_preset(payload)
        self.assertEqual(overrides["mouth_sharpen_strength"], 0.05)
        self.assertEqual(overrides["mouth_detail_strength"], 0.75)


if __name__ == "__main__":
    unittest.main()
