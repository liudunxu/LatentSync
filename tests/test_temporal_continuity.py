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

    def test_upper_face_region_diff_distinguishes_faces(self):
        """Identical crops -> 0, opposite extremes -> close to 1.

        The upper-face diff samples the forehead/upper-cheek band, so it is
        stable under mouth motion and used for the continuity-break check
        instead of the mouth band (which trips on laughs / teeth flashes).
        """
        import torch

        try:
            from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        except ModuleNotFoundError as exc:
            self.skipTest(f"pipeline optional dependency missing: {exc.name}")

        black = torch.zeros(3, 512, 512, dtype=torch.uint8)
        white = torch.full((3, 512, 512), 255, dtype=torch.uint8)
        mid = torch.full((3, 512, 512), 100, dtype=torch.uint8)

        # Identical -> 0
        self.assertEqual(LipsyncPipeline._upper_face_region_diff(black, black), 0.0)
        self.assertEqual(LipsyncPipeline._upper_face_region_diff(mid, mid), 0.0)

        # Full flip -> ~= 1.0
        diff_extreme = LipsyncPipeline._upper_face_region_diff(black, white)
        self.assertGreater(diff_extreme, 0.95)

        # Black vs mid -> ~= 100/255 ~= 0.39
        diff_mid = LipsyncPipeline._upper_face_region_diff(black, mid)
        self.assertGreater(diff_mid, 0.35)
        self.assertLess(diff_mid, 0.45)

    def test_upper_face_region_diff_ignores_mouth_band(self):
        """A change confined to the mouth band must NOT register as upper-face
        content change -- this is the whole point of switching the continuity
        check off the mouth ROI."""
        import torch

        try:
            from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        except ModuleNotFoundError as exc:
            self.skipTest(f"pipeline optional dependency missing: {exc.name}")

        base = torch.zeros(3, 512, 512, dtype=torch.uint8)
        # Flip ONLY the mouth band (y 0.55..0.74, x 0.30..0.70) to white.
        mouth_only = base.clone()
        H, W = 512, 512
        y0, y1 = int(H * 0.55), int(H * 0.74)
        x0, x1 = int(W * 0.30), int(W * 0.70)
        mouth_only[:, y0:y1, x0:x1] = 255

        # Upper-face diff sees no change (forehead/cheeks identical).
        self.assertEqual(LipsyncPipeline._upper_face_region_diff(base, mouth_only), 0.0)
        # The mouth-band diff DOES see it (sanity check the contrast).
        self.assertGreater(LipsyncPipeline._mouth_region_diff(base, mouth_only), 0.5)

    def test_upper_face_region_diff_returns_zero_on_shape_mismatch(self):
        """Defensive: None / different shapes / wrong rank must not raise."""
        import torch

        try:
            from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        except ModuleNotFoundError as exc:
            self.skipTest(f"pipeline optional dependency missing: {exc.name}")

        a = torch.zeros(3, 512, 512, dtype=torch.uint8)
        b = torch.zeros(3, 256, 256, dtype=torch.uint8)
        c = torch.zeros(3, 512, 512, 512, dtype=torch.uint8)

        self.assertEqual(LipsyncPipeline._upper_face_region_diff(None, a), 0.0)
        self.assertEqual(LipsyncPipeline._upper_face_region_diff(a, None), 0.0)
        self.assertEqual(LipsyncPipeline._upper_face_region_diff(None, None), 0.0)
        self.assertEqual(LipsyncPipeline._upper_face_region_diff(a, b), 0.0)
        self.assertEqual(LipsyncPipeline._upper_face_region_diff(a, c), 0.0)

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

    def test_smooth_face_sequence_vectorized_matches_loop(self):
        import torch

        try:
            from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        except ModuleNotFoundError as exc:
            self.skipTest(f"pipeline optional dependency missing: {exc.name}")

        torch.manual_seed(0)
        face_crops = torch.randn(4, 3, 64, 64)
        prev_face = torch.randn(3, 64, 64)

        # Loop path forces vectorized=False by supplying a region_mask.
        loop_out, _, _ = LipsyncPipeline._smooth_face_sequence(
            face_crops,
            prev_face=prev_face,
            prev_valid=True,
            inference_skip_mask=[False] * 4,
            continuity_break_mask=[False] * 4,
            region_mask=torch.ones(4, 1, 64, 64),
        )

        # Fast path triggers with no masks/breaks/skips.
        fast_out, _, _ = LipsyncPipeline._smooth_face_sequence(
            face_crops,
            prev_face=prev_face,
            prev_valid=True,
            inference_skip_mask=[False] * 4,
            continuity_break_mask=[False] * 4,
            region_mask=None,
        )

        self.assertTrue(torch.allclose(loop_out, fast_out, atol=1e-5))

    def test_source_scene_cut_after_marks_hard_boundaries(self):
        import numpy as np

        try:
            from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        except ModuleNotFoundError as exc:
            self.skipTest(f"pipeline optional dependency missing: {exc.name}")

        black = np.zeros((32, 32, 3), dtype=np.uint8)
        white = np.full((32, 32, 3), 255, dtype=np.uint8)
        frames = np.stack([black, black, white, white], axis=0)

        cuts = LipsyncPipeline._compute_source_scene_cut_after(frames, threshold=0.45)

        self.assertEqual(cuts, [False, True, False])

    def test_shot_routing_manifest_splits_on_scene_cut(self):
        try:
            from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        except ModuleNotFoundError as exc:
            self.skipTest(f"pipeline optional dependency missing: {exc.name}")

        effective_skip = [False, False, True, True, True, False]
        source_indices = list(range(6))
        source_scene_cut_after = [False, True, False, False, True]

        manifest = LipsyncPipeline._build_shot_routing_manifest(
            effective_skip,
            source_indices,
            source_scene_cut_after,
            fps=25.0,
            pre_skip_mask=effective_skip,
        )

        self.assertEqual(manifest["shots_total"], 3)
        self.assertEqual(manifest["latentsync_shots"], 1)
        self.assertEqual(manifest["passthrough_shots"], 1)
        self.assertEqual(manifest["mixed_shots"], 1)
        self.assertEqual([shot["route"] for shot in manifest["shots"]], [
            "latentsync",
            "passthrough",
            "mixed",
        ])


if __name__ == "__main__":
    unittest.main()
