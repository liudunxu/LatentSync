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


if __name__ == "__main__":
    unittest.main()
