"""Unit tests for AlignRestore paste-back behavior."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

try:
    import kornia
except ModuleNotFoundError:
    kornia = None


class TestRestoreImgFeather(unittest.TestCase):
    def _make_restorer(self, resolution=512, device="cpu"):
        if kornia is None:
            self.skipTest("kornia not installed")
        try:
            from latentsync.utils.affine_transform import AlignRestore
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional dependency missing: {exc.name}")
        return AlignRestore(resolution=resolution, device=device, dtype=torch.float32)

    def _make_affine(self):
        # Identity-like affine that maps the aligned face back to the frame.
        return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    def test_feather_sigma_scales_with_resolution(self):
        """At 512 sigma should be 4; at 1080 sigma should be > 4."""
        restorer = self._make_restorer()
        for (h, w), expected_min_sigma in ((512, 512), 4.0), ((1080, 1920), 8.0):
            input_img = np.zeros((h, w, 3), dtype=np.uint8)
            face = torch.full((3, 512, 512), 0.5, dtype=torch.float32)
            paste_mask = torch.ones((1, 512, 512), dtype=torch.float32)
            affine = self._make_affine()

            # Patch the kornia blur call to capture its arguments.
            captured = {}

            def fake_blur(inp, kernel_size, sigma):
                captured["kernel_size"] = kernel_size
                captured["sigma"] = sigma
                return inp

            import latentsync.utils.affine_transform as at_mod
            original_blur = at_mod.kornia.filters.gaussian_blur2d
            at_mod.kornia.filters.gaussian_blur2d = fake_blur
            try:
                _ = restorer.restore_img(input_img, face, affine, paste_mask_512=paste_mask)
            finally:
                at_mod.kornia.filters.gaussian_blur2d = original_blur

            self.assertAlmostEqual(
                captured["sigma"][0], expected_min_sigma, places=1,
                msg=f"resolution ({h},{w}) should use sigma >= {expected_min_sigma}, got {captured['sigma']}"
            )
            self.assertGreaterEqual(captured["kernel_size"][0], 21)

    def test_restore_img_output_shape(self):
        restorer = self._make_restorer()
        h, w = 720, 1280
        input_img = np.zeros((h, w, 3), dtype=np.uint8)
        face = torch.full((3, 512, 512), 0.5, dtype=torch.float32)
        paste_mask = torch.ones((1, 512, 512), dtype=torch.float32)
        affine = self._make_affine()
        out = restorer.restore_img(input_img, face, affine, paste_mask_512=paste_mask)
        self.assertEqual(out.shape, (h, w, 3))


if __name__ == "__main__":
    unittest.main()
