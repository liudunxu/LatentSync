"""Tests for source-frame index mapping in loop_video / restore_video."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch


class TestLoopSourceIndices(unittest.TestCase):
    """Verify that loop_video builds the expected source-index map and that
    restore_video uses it to look up the correct source frame.
    """

    def _build_expected_source_indices(self, n_source: int, n_output: int):
        """Reproduce the loop_video source-index construction without models."""
        if n_output <= n_source:
            return list(range(n_output))
        indices = []
        i = 0
        while len(indices) < n_output:
            if i % 2 == 0:
                indices += list(range(n_source))
            else:
                indices += list(range(n_source - 1, -1, -1))
            i += 1
        return indices[:n_output]

    def test_non_loop_case_identity_indices(self):
        self.assertEqual(self._build_expected_source_indices(10, 8), list(range(8)))

    def test_loop_case_forward_reverse(self):
        indices = self._build_expected_source_indices(3, 10)
        # 3 source frames -> pattern 0,1,2,2,1,0,0,1,2,2
        self.assertEqual(indices, [0, 1, 2, 2, 1, 0, 0, 1, 2, 2])

    def test_restore_video_uses_source_indices(self):
        try:
            from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        except ModuleNotFoundError as exc:
            self.skipTest(f"pipeline optional dependency missing: {exc.name}")

        # 2 source frames, 4 output positions mapped [0, 1, 1, 0].
        source_frames = np.array([
            np.full((4, 4, 3), 0, dtype=np.uint8),
            np.full((4, 4, 3), 255, dtype=np.uint8),
        ])
        faces = torch.zeros(4, 3, 4, 4)
        boxes = [[0, 0, 4, 4]] * 4
        affine_matrices = [np.eye(3).astype(np.float32)] * 4
        skip_mask = [True, True, True, True]
        source_indices = [0, 1, 1, 0]

        restored = LipsyncPipeline.restore_video(
            faces,
            source_frames,
            boxes,
            affine_matrices,
            skip_mask=skip_mask,
            source_indices=source_indices,
        )
        # All frames skipped -> output should be source frames in index order.
        self.assertTrue(np.array_equal(restored[0], source_frames[0]))
        self.assertTrue(np.array_equal(restored[1], source_frames[1]))
        self.assertTrue(np.array_equal(restored[2], source_frames[1]))
        self.assertTrue(np.array_equal(restored[3], source_frames[0]))

    def test_restore_video_legacy_path_without_indices(self):
        try:
            from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        except ModuleNotFoundError as exc:
            self.skipTest(f"pipeline optional dependency missing: {exc.name}")

        source_frames = np.array([
            np.full((4, 4, 3), 0, dtype=np.uint8),
            np.full((4, 4, 3), 255, dtype=np.uint8),
        ])
        faces = torch.zeros(2, 3, 4, 4)
        boxes = [[0, 0, 4, 4]] * 2
        affine_matrices = [np.eye(3).astype(np.float32)] * 2
        skip_mask = [True, True]

        restored = LipsyncPipeline.restore_video(
            faces,
            source_frames,
            boxes,
            affine_matrices,
            skip_mask=skip_mask,
        )
        # Without source_indices it should use video_frames[:len(faces)].
        self.assertTrue(np.array_equal(restored[0], source_frames[0]))
        self.assertTrue(np.array_equal(restored[1], source_frames[1]))


if __name__ == "__main__":
    unittest.main()
