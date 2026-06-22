"""Unit tests for the post-processing chain in :class:`LipsyncPipeline`.

Covers the four static helpers that drive paste-back, color match, and
mouth-region blending. All of these are pure torch tensor ops, so the
tests run on CPU and need no model / no GPU.

Why these exist: the post-processing chain is the main quality-tuning
lever in the project (see ``api.py`` defaults), and until now only
``_smooth_face_sequence`` and ``_mouth_region_diff`` were covered. These
tests give us a CPU safety net so default values can be tuned without
breaking the visible behaviour.

Run with::

    python -m tests.test_post_processing

or via ``pytest tests/``.
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestColorMatch(unittest.TestCase):
    """``_match_color_to_reference`` is the mean+std color transfer from the
    generated face to the reference face, applied inside the generated
    region mask.
    """

    def _import(self):
        try:
            import torch  # noqa: F401

            from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        except ModuleNotFoundError as exc:
            self.skipTest(f"pipeline optional dependency missing: {exc.name}")
        return __import__("torch"), LipsyncPipeline

    def test_strength_zero_returns_input_unchanged(self):
        torch, LipsyncPipeline = self._import()
        x = torch.randn(3, 64, 64)
        r = torch.randn(3, 64, 64)
        m = torch.ones(1, 64, 64)
        out = LipsyncPipeline._match_color_to_reference(x, r, m, strength=0.0)
        self.assertTrue(torch.equal(out, x))

    def test_shape_mismatch_returns_face_unchanged(self):
        torch, LipsyncPipeline = self._import()
        x = torch.randn(3, 64, 64)
        r = torch.randn(3, 32, 32)  # different shape
        m = torch.ones(1, 64, 64)
        out = LipsyncPipeline._match_color_to_reference(x, r, m, strength=0.6)
        self.assertTrue(torch.equal(out, x))

    def test_mask_all_zero_leaves_face_unchanged(self):
        # If no generated region, nothing should change.
        torch, LipsyncPipeline = self._import()
        x = torch.randn(3, 32, 32)
        r = torch.zeros(3, 32, 32)
        m = torch.zeros(1, 32, 32)
        out = LipsyncPipeline._match_color_to_reference(x, r, m, strength=0.6)
        # With no masked pixels the per-channel mean/var are 0/0 (clamped
        # to 1e-6), so the affine becomes scale=1, shift=0 and the
        # strength=0.6 blend reduces to the input.
        diff = (out - x).abs().max().item()
        self.assertLess(diff, 1e-5)

    def test_mean_std_aligns_inside_mask(self):
        # Inside the mask the per-channel mean/std of the output should
        # move toward the reference's mean/std, with the rate controlled
        # by `strength`. Outside the mask it should be untouched.
        torch, LipsyncPipeline = self._import()
        x = torch.zeros(3, 32, 32)  # generated is mid-gray
        x += torch.tensor([0.10, -0.05, 0.02]).view(3, 1, 1)  # slight tint
        r = torch.zeros(3, 32, 32)  # reference is mid-gray
        r += torch.tensor([-0.20, 0.20, 0.00]).view(3, 1, 1)  # opposite tint
        m = torch.ones(1, 32, 32)  # whole face is generated
        out = LipsyncPipeline._match_color_to_reference(x, r, m, strength=1.0)
        # With strength=1, output stats should match the reference's stats.
        out_mean = out.mean(dim=(1, 2))
        ref_mean = r.mean(dim=(1, 2))
        for c in range(3):
            self.assertAlmostEqual(out_mean[c].item(), ref_mean[c].item(), places=4)

    def test_partial_strength_interpolates(self):
        # strength=0.5 should move halfway between input and full transfer.
        # Use a large tone delta (>= 0.4) so the adaptive scaling saturates
        # to 1.0 and strength acts as the exact interpolation coefficient.
        torch, LipsyncPipeline = self._import()
        x = torch.full((3, 32, 32), 0.1)
        r = torch.full((3, 32, 32), 0.5)
        m = torch.ones(1, 32, 32)
        out = LipsyncPipeline._match_color_to_reference(x, r, m, strength=0.5)
        # Input mean 0.1, ref mean 0.5, so halfway ~= 0.3.
        out_mean = out.mean(dim=(1, 2)).mean().item()
        self.assertAlmostEqual(out_mean, 0.3, places=4)

    def test_adaptive_strength_attenuates_on_small_tone_delta(self):
        # When the generated face is already close to the reference tone,
        # the adaptive scaling should attenuate the transfer below the
        # requested strength so a near-match is not over-corrected.
        torch, LipsyncPipeline = self._import()
        x = torch.full((3, 32, 32), 0.10)
        r = torch.full((3, 32, 32), 0.12)  # tiny delta (0.02 << 0.4)
        m = torch.ones(1, 32, 32)
        out = LipsyncPipeline._match_color_to_reference(x, r, m, strength=0.6)
        out_mean = out.mean(dim=(1, 2)).mean().item()
        # With delta 0.02, diff_norm ~= 0.05, strength_scale ~= 0.43,
        # eff_strength ~= 0.6 * 0.43 ~= 0.26 -> out ~= 0.10 + 0.26*0.02 ~= 0.105.
        # The output should move only a tiny amount toward the reference,
        # much less than the 0.6 strength would normally produce (0.112).
        self.assertLess(abs(out_mean - 0.10), 0.02)
        self.assertLess(out_mean, 0.112)

    def test_batched_input(self):
        # (B, 3, H, W) with (B, 1, H, W) mask must not raise.
        torch, LipsyncPipeline = self._import()
        x = torch.randn(2, 3, 32, 32)
        r = torch.randn(2, 3, 32, 32)
        m = torch.ones(2, 1, 32, 32)
        out = LipsyncPipeline._match_color_to_reference(x, r, m, strength=0.5)
        self.assertEqual(out.shape, x.shape)


class TestMouthDetailRestore(unittest.TestCase):
    """``_restore_reference_detail`` adds back the reference's high-frequency
    detail outside the central mouth core, so cheeks/chin keep their
    original skin texture and only the lip aperture / lip contour remain
    driven by the inpainter's output.
    """

    def _import(self):
        try:
            import torch  # noqa: F401

            from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        except ModuleNotFoundError as exc:
            self.skipTest(f"pipeline optional dependency missing: {exc.name}")
        return __import__("torch"), LipsyncPipeline

    def test_strength_zero_returns_input_unchanged(self):
        torch, LipsyncPipeline = self._import()
        x = torch.randn(3, 64, 64)
        r = torch.randn(3, 64, 64)
        m = torch.ones(1, 64, 64)
        out = LipsyncPipeline._restore_reference_detail(x, r, m, strength=0.0)
        self.assertTrue(torch.equal(out, x))

    def test_shape_mismatch_returns_face_unchanged(self):
        torch, LipsyncPipeline = self._import()
        x = torch.randn(3, 64, 64)
        r = torch.randn(3, 32, 32)
        m = torch.ones(1, 64, 64)
        out = LipsyncPipeline._restore_reference_detail(x, r, m, strength=0.5)
        self.assertTrue(torch.equal(out, x))

    def test_core_protected_cheeks_changed(self):
        # Build a synthetic face where the inpainter's output is a flat
        # mid-tone and the reference has rich high-frequency content.
        # Outside the central core the function should add back the
        # reference's detail; inside the core it should leave the input
        # alone.
        torch, LipsyncPipeline = self._import()
        H = W = 128
        x = torch.zeros(1, 3, H, W)
        r = torch.zeros(1, 3, H, W)
        # Reference has a checkerboard at the cheek (top-half) region.
        for yy in range(0, 64, 4):
            for xx in range(0, 64, 4):
                r[0, :, yy:yy + 2, xx:xx + 2] = 0.5
        m = torch.ones(1, 1, H, W)  # entire face is "generated"
        out = LipsyncPipeline._restore_reference_detail(x, r, m, strength=1.0)
        # Cheek region (y < 50%) should now have non-trivial content from r.
        cheek_region = out[0, :, :64, :64]
        cheek_change = cheek_region.abs().max().item()
        self.assertGreater(cheek_change, 0.1)
        # Mouth center (~y=66%) is the protected core: input was 0, so
        # the output should still be ~0 in the central mouth aperture.
        H_center = H // 2
        W_center = W // 2
        half_box = 8
        core_box = out[0, :, H_center - half_box:H_center + half_box, W_center - half_box:W_center + half_box]
        # Core protection is "soft" (the mask falls off), so the core
        # itself should have small absolute change relative to cheeks.
        core_change = core_box.abs().max().item()
        self.assertLess(core_change, cheek_change)


class TestMouthCoreMask(unittest.TestCase):
    """``_mouth_core_mask`` is the central-mouth mask used to protect the
    lip aperture / contour from detail restoration. The returned mask is
    the inpaint mask multiplied by a soft elliptical falloff centered at
    the mouth.
    """

    def _import(self):
        try:
            import torch  # noqa: F401

            from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        except ModuleNotFoundError as exc:
            self.skipTest(f"pipeline optional dependency missing: {exc.name}")
        return __import__("torch"), LipsyncPipeline

    def test_default_center_is_in_mask_center(self):
        # At the default (0.50, 0.66) the elliptical falloff should be
        # fully active (== 1) at the mouth center and fall off to 0 far away.
        torch, LipsyncPipeline = self._import()
        m = torch.ones(1, 1, 128, 128)
        core = LipsyncPipeline._mouth_core_mask(m)
        self.assertEqual(core.shape, m.shape)
        # Center pixel should be 1 (or very close, given the soft falloff).
        center_y = int(0.66 * 128)
        center_x = int(0.50 * 128)
        center_val = core[0, 0, center_y, center_x].item()
        self.assertGreater(center_val, 0.95)
        # Far from the center (e.g. forehead) the value should be 0.
        forehead_val = core[0, 0, 4, 64].item()
        self.assertLess(forehead_val, 0.05)

    def test_mask_zero_yields_zero_core(self):
        # When the inpaint mask is empty everywhere, the core mask should
        # also be empty everywhere (multiplicative gating).
        torch, LipsyncPipeline = self._import()
        m = torch.zeros(1, 1, 128, 128)
        core = LipsyncPipeline._mouth_core_mask(m)
        self.assertEqual(core.shape, m.shape)
        self.assertEqual(core.max().item(), 0.0)

    def test_custom_center_moves_falloff(self):
        # A non-default center should move the bright region of the falloff.
        torch, LipsyncPipeline = self._import()
        m = torch.ones(1, 1, 128, 128)
        # Place the mouth center at (0.30, 0.40) -- well above-and-left of
        # the default.
        core = LipsyncPipeline._mouth_core_mask(
            m, mouth_center_norm=(0.30, 0.40)
        )
        # Custom center should be ~1.
        custom_y = int(0.40 * 128)
        custom_x = int(0.30 * 128)
        custom_val = core[0, 0, custom_y, custom_x].item()
        self.assertGreater(custom_val, 0.95)
        # Default center should now be near 0 (outside the new ellipse).
        default_y = int(0.66 * 128)
        default_x = int(0.50 * 128)
        default_val = core[0, 0, default_y, default_x].item()
        self.assertLess(default_val, 0.05)

    def test_output_range_in_unit_interval(self):
        # The soft falloff is a clamped linear function, so the output is
        # always in [0, 1] on the inpaint mask.
        torch, LipsyncPipeline = self._import()
        m = torch.ones(1, 1, 128, 128)
        core = LipsyncPipeline._mouth_core_mask(m)
        self.assertGreaterEqual(core.min().item(), 0.0)
        self.assertLessEqual(core.max().item(), 1.0)


class TestDynamicMouthMask(unittest.TestCase):
    """``generate_dynamic_mouth_mask`` is the per-frame paste-back mask
    built from aligned mouth landmarks. Returns ``(1, H, W)`` with
    ``1 = keep (preserve)`` and ``0 = inpaint (regenerate)``.
    """

    def _import(self):
        try:
            import torch  # noqa: F401

            from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        except ModuleNotFoundError as exc:
            self.skipTest(f"pipeline optional dependency missing: {exc.name}")
        return __import__("torch"), LipsyncPipeline

    def _mouth_info(self, cx_px, cy_px, hw_px, hh_px):
        # Real production values: center/half_widths are pixel coords on
        # the aligned 512x512 face. The helper normalizes to resolution.
        return {
            "center_x": float(cx_px),
            "center_y": float(cy_px),
            "half_width": float(hw_px),
            "half_height": float(hh_px),
        }

    def test_output_shape_and_convention(self):
        torch, LipsyncPipeline = self._import()
        info = self._mouth_info(64, 80, 16, 8)
        mask = LipsyncPipeline.generate_dynamic_mouth_mask(info, resolution=128)
        # (1, H, W) per docstring
        self.assertEqual(mask.shape, (1, 128, 128))
        # Convention: 1 = keep (preserve), 0 = inpaint
        # So the value at the center of the mouth should be <= 0 (inpaint)
        # and the value at the forehead should be ~1 (keep).
        self.assertLess(mask[0, 80, 64].item(), 0.5)
        self.assertGreater(mask[0, 4, 64].item(), 0.5)

    def test_none_mouth_info_uses_fallback(self):
        torch, LipsyncPipeline = self._import()
        mask = LipsyncPipeline.generate_dynamic_mouth_mask(None, resolution=128)
        # Same shape; should not raise.
        self.assertEqual(mask.shape, (1, 128, 128))
        # Forehead should still be kept.
        self.assertGreater(mask[0, 4, 64].item(), 0.5)

    def test_feather_off_yields_harder_edges(self):
        # When the Gaussian feather is disabled the inpaint region
        # boundary becomes a hard step. A pixel just outside the ellipse
        # should be exactly 1 (keep) and a pixel just inside should be
        # exactly 0 (inpaint).
        torch, LipsyncPipeline = self._import()
        info = self._mouth_info(64, 80, 16, 8)
        mask_smooth = LipsyncPipeline.generate_dynamic_mouth_mask(
            info, resolution=128, feather_sigma_px=7.0
        )
        mask_hard = LipsyncPipeline.generate_dynamic_mouth_mask(
            info, resolution=128, feather_sigma_px=0.0
        )
        # Probe a strip at y=80 (mouth centerline) moving outward in x.
        # With the hard mask, all values must be exactly 0 or 1.
        strip = mask_hard[0, 80, :]
        self.assertTrue(((strip == 0.0) | (strip == 1.0)).all().item())
        # Smoothed mask has a soft transition; the same strip should
        # contain at least one value strictly in (0, 1).
        strip_s = mask_smooth[0, 80, :]
        any_intermediate = ((strip_s > 0.0) & (strip_s < 1.0)).any().item()
        self.assertTrue(any_intermediate)

    def test_output_in_unit_interval(self):
        torch, LipsyncPipeline = self._import()
        info = self._mouth_info(64, 80, 16, 8)
        mask = LipsyncPipeline.generate_dynamic_mouth_mask(info, resolution=128)
        # After the post-feather `torch.where` clamps and the final
        # `1 - inpaint` flip, the values must be in [0, 1].
        self.assertGreaterEqual(mask.min().item(), 0.0)
        self.assertLessEqual(mask.max().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
