"""Unit tests for adaptive composite quality fallback helpers."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch


class TestAdaptiveQualityFallback(unittest.TestCase):
    def _import_pipeline(self):
        try:
            from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        except ModuleNotFoundError as exc:
            self.skipTest(f"pipeline optional dependency missing: {exc.name}")
        return LipsyncPipeline

    def _make_face(self, value: float, dtype=torch.float32):
        """Create a 512x512 RGB face in [-1, 1] filled with ``value``."""
        return torch.full((3, 512, 512), value, dtype=dtype)

    def test_compute_frame_quality_score_identical_faces(self):
        LipsyncPipeline = self._import_pipeline()
        face = self._make_face(0.0)
        score = LipsyncPipeline._compute_frame_quality_score(
            face, face, yaw=0.0, identity_sim=0.8, audio_scale=1.0, mouth_temporal_delta=0.0
        )
        # Identical faces with good side signals should be near 1.0.
        self.assertGreater(score, 0.95)

    def test_compute_frame_quality_score_worse_with_yaw(self):
        LipsyncPipeline = self._import_pipeline()
        face = self._make_face(0.0)
        frontal = LipsyncPipeline._compute_frame_quality_score(
            face, face, yaw=0.0, identity_sim=0.8
        )
        side = LipsyncPipeline._compute_frame_quality_score(
            face, face, yaw=45.0, identity_sim=0.8
        )
        self.assertGreater(frontal, side)

    def test_compute_frame_quality_score_worse_with_identity(self):
        LipsyncPipeline = self._import_pipeline()
        face = self._make_face(0.0)
        good_id = LipsyncPipeline._compute_frame_quality_score(
            face, face, identity_sim=0.9
        )
        bad_id = LipsyncPipeline._compute_frame_quality_score(
            face, face, identity_sim=0.3
        )
        self.assertGreater(good_id, bad_id)

    def test_adaptive_threshold_respects_max_ratio(self):
        LipsyncPipeline = self._import_pipeline()
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        already_skipped = [False] * len(scores)
        # Base threshold would skip 4/10 (scores < 0.5). Cap at 20% -> 2 skipped.
        fallback = LipsyncPipeline._adaptive_quality_threshold(
            scores, base_threshold=0.5, max_fallback_ratio=0.2, already_skipped=already_skipped
        )
        self.assertEqual(sum(fallback), 2)
        self.assertTrue(fallback[0])
        self.assertTrue(fallback[1])
        self.assertFalse(fallback[-1])

    def test_adaptive_threshold_zero_max_ratio_disables(self):
        LipsyncPipeline = self._import_pipeline()
        scores = [0.0, 0.0, 1.0, 1.0]
        fallback = LipsyncPipeline._adaptive_quality_threshold(
            scores, base_threshold=0.5, max_fallback_ratio=0.0, already_skipped=[False] * 4
        )
        self.assertEqual(sum(fallback), 0)

    def test_hysteresis_suppresses_isolated_fallback(self):
        LipsyncPipeline = self._import_pipeline()
        fallback = [False, False, True, False, False]
        out = LipsyncPipeline._apply_quality_hysteresis(fallback, hysteresis_frames=2)
        self.assertEqual(out, [False, False, False, False, False])

    def test_hysteresis_keeps_boundary_run(self):
        LipsyncPipeline = self._import_pipeline()
        fallback = [True, False, False, False, True]
        out = LipsyncPipeline._apply_quality_hysteresis(fallback, hysteresis_frames=2)
        # Boundary single-frame runs are kept.
        self.assertTrue(out[0])
        self.assertTrue(out[-1])

    def test_hysteresis_keeps_long_run(self):
        LipsyncPipeline = self._import_pipeline()
        fallback = [False, True, True, True, False]
        out = LipsyncPipeline._apply_quality_hysteresis(fallback, hysteresis_frames=2)
        self.assertEqual(out, [False, True, True, True, False])

    def test_hysteresis_default_four_frames(self):
        LipsyncPipeline = self._import_pipeline()
        # A 4-frame internal run should be suppressed with the new default.
        fallback = [False, True, True, True, True, False]
        out = LipsyncPipeline._apply_quality_hysteresis(fallback, hysteresis_frames=4)
        self.assertEqual(out, [False] * len(fallback))

    def test_hysteresis_keeps_five_frame_run(self):
        LipsyncPipeline = self._import_pipeline()
        # A 5-frame internal run should be kept with hysteresis=4.
        fallback = [False, True, True, True, True, True, False]
        out = LipsyncPipeline._apply_quality_hysteresis(fallback, hysteresis_frames=4)
        self.assertEqual(out, [False, True, True, True, True, True, False])

    def test_mouth_region_diff_normalized_zero_for_identical(self):
        LipsyncPipeline = self._import_pipeline()
        face = self._make_face(0.5)
        self.assertEqual(LipsyncPipeline._mouth_region_diff_normalized(face, face), 0.0)

    def test_mouth_region_diff_normalized_extreme(self):
        LipsyncPipeline = self._import_pipeline()
        black = self._make_face(-1.0)
        white = self._make_face(1.0)
        diff = LipsyncPipeline._mouth_region_diff_normalized(black, white)
        # Full [-1,1] flip -> diff = 1.0
        self.assertAlmostEqual(diff, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
