"""Regression tests for temporal lip-sync boundary handling."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestTemporalContinuity(unittest.TestCase):
    def test_continuity_break_blocks_cross_batch_previous_face(self):
        import torch

        try:
            from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        except ModuleNotFoundError as exc:
            self.skipTest(f"pipeline optional dependency missing: {exc.name}")

        current = torch.zeros(1, 3, 8, 8)
        previous = torch.ones(3, 8, 8)

        blended, _last_face, _last_valid = LipsyncPipeline._smooth_face_sequence(
            current,
            prev_face=previous,
            prev_valid=True,
            inference_skip_mask=[False],
            continuity_break_mask=[False],
        )
        self.assertGreater(blended.mean().item(), 0.0)

        isolated, _last_face, _last_valid = LipsyncPipeline._smooth_face_sequence(
            current,
            prev_face=previous,
            prev_valid=True,
            inference_skip_mask=[False],
            continuity_break_mask=[True],
        )
        self.assertTrue(torch.equal(isolated, current))

    def test_mouth_region_diff_distinguishes_faces(self):
        """Identical crops -> 0, opposite extremes -> close to 1."""
        import torch

        try:
            from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        except ModuleNotFoundError as exc:
            self.skipTest(f"pipeline optional dependency missing: {exc.name}")

        # Aligned faces are uint8 512x512 in production, so test at that size.
        black = torch.zeros(3, 512, 512, dtype=torch.uint8)
        white = torch.full((3, 512, 512), 255, dtype=torch.uint8)
        mid = torch.full((3, 512, 512), 100, dtype=torch.uint8)

        # Identical faces -> diff == 0
        self.assertEqual(LipsyncPipeline._mouth_region_diff(black, black), 0.0)
        self.assertEqual(LipsyncPipeline._mouth_region_diff(mid, mid), 0.0)

        # Black vs white in [0, 255] = full-scale flip -> diff ~= 1.0
        diff_extreme = LipsyncPipeline._mouth_region_diff(black, white)
        self.assertGreater(diff_extreme, 0.95)

        # Black vs mid -> diff ~= 100/255 ~= 0.39
        diff_mid = LipsyncPipeline._mouth_region_diff(black, mid)
        self.assertGreater(diff_mid, 0.35)
        self.assertLess(diff_mid, 0.45)

    def test_mouth_region_diff_returns_zero_on_shape_mismatch(self):
        """Defensive: None / different shapes / wrong rank must not raise."""
        import torch

        try:
            from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        except ModuleNotFoundError as exc:
            self.skipTest(f"pipeline optional dependency missing: {exc.name}")

        a = torch.zeros(3, 512, 512, dtype=torch.uint8)
        b = torch.zeros(3, 256, 256, dtype=torch.uint8)
        c = torch.zeros(3, 512, 512, 512, dtype=torch.uint8)  # wrong rank

        # None inputs -> 0.0
        self.assertEqual(LipsyncPipeline._mouth_region_diff(None, a), 0.0)
        self.assertEqual(LipsyncPipeline._mouth_region_diff(a, None), 0.0)
        self.assertEqual(LipsyncPipeline._mouth_region_diff(None, None), 0.0)

        # Shape mismatch -> 0.0
        self.assertEqual(LipsyncPipeline._mouth_region_diff(a, b), 0.0)

        # Wrong rank -> 0.0
        self.assertEqual(LipsyncPipeline._mouth_region_diff(a, c), 0.0)

    def test_source_frame_scene_cut_score(self):
        import numpy as np

        try:
            from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        except ModuleNotFoundError as exc:
            self.skipTest(f"pipeline optional dependency missing: {exc.name}")

        black = np.zeros((96, 128, 3), dtype=np.uint8)
        white = np.full((96, 128, 3), 255, dtype=np.uint8)
        dim = np.full((96, 128, 3), 8, dtype=np.uint8)

        self.assertEqual(LipsyncPipeline._source_frame_scene_cut_score(black, black), 0.0)

        hard_cut = LipsyncPipeline._source_frame_scene_cut_score(black, white)
        self.assertGreater(hard_cut, 0.95)

        small_lighting_shift = LipsyncPipeline._source_frame_scene_cut_score(black, dim)
        self.assertLess(small_lighting_shift, 0.45)


if __name__ == "__main__":
    unittest.main()
