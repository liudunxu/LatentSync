"""Unit tests for the yaw side-face filter helpers in :class:`LipsyncPipeline`.

Covers the three static helpers that drive the yaw episode / warn-run /
rate-stabilization logic. All are pure list/number ops on inputs supplied
by the caller, so the tests run on CPU and need no model / no GPU / no
InsightFace / no video.

Why these exist: the yaw filter is the most-tweaked lever in the pipeline
(see ``api.py`` defaults history -- 45° → 22° → 30° → 45° → 30°). The
helpers were extracted from inline code in ``affine_transform_video`` to
make the per-frame logic unit-testable, so default values can be tuned
without breaking the visible behaviour.

Run with::

    python -m tests.test_yaw_filter

or via ``pytest tests/``.
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class _FilterHelperBase(unittest.TestCase):
    """Common import logic for the static helpers under test."""

    def _import(self):
        try:
            from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        except ModuleNotFoundError as exc:
            self.skipTest(f"pipeline optional dependency missing: {exc.name}")
        return LipsyncPipeline


class TestApplyEpisodePad(_FilterHelperBase):
    """``_apply_episode_pad`` extends ``skip_mask`` into the warn-band
    transition zone around each yaw-skip episode.
    """

    def _make_masks(self, n: int):
        return ([False] * n, [False] * n)

    def test_pre_post_pad_zero_is_noop(self):
        LipsyncPipeline = self._import()
        n = 10
        skip_mask, continuity_break_mask = self._make_masks(n)
        # Frames 4,5,6 are yaw-skipped; rest are warn-band.
        yaw_skip_reasons = [False] * n
        for i in (4, 5, 6):
            yaw_skip_reasons[i] = True
        yaws: List[Optional[float]] = [25.0 if 2 <= i <= 8 else 5.0 for i in range(n)]
        # Frames 4,5,6 already skipped.
        for i in (4, 5, 6):
            skip_mask[i] = True
        extra = LipsyncPipeline._apply_episode_pad(
            skip_mask, continuity_break_mask, yaws, yaw_skip_reasons,
            pre_pad=0, post_pad=0, warn_threshold=22.5,
        )
        self.assertEqual(extra, 0)
        # Nothing else should be marked skipped.
        self.assertEqual(sum(skip_mask), 3)

    def test_pre_pad_extends_into_warn_band(self):
        LipsyncPipeline = self._import()
        # Layout: idx 0..4 are warn-band (25°), 5..7 are skipped (35°), 8..12 frontal.
        n = 13
        skip_mask, continuity_break_mask = self._make_masks(n)
        yaw_skip_reasons = [False] * n
        for i in (5, 6, 7):
            yaw_skip_reasons[i] = True
        yaws = [25.0] * 5 + [35.0] * 3 + [5.0] * 5
        for i in (5, 6, 7):
            skip_mask[i] = True
        extra = LipsyncPipeline._apply_episode_pad(
            skip_mask, continuity_break_mask, yaws, yaw_skip_reasons,
            pre_pad=3, post_pad=0, warn_threshold=22.5,
        )
        # Pre pad should pick up idx 2,3,4 (3 frames just before run at idx 5).
        # Idx 0,1 stay (they're outside the pre_pad window of 3).
        self.assertEqual(extra, 3)
        self.assertTrue(skip_mask[2])
        self.assertTrue(skip_mask[3])
        self.assertTrue(skip_mask[4])
        self.assertTrue(skip_mask[5])
        self.assertTrue(skip_mask[6])
        self.assertTrue(skip_mask[7])
        self.assertFalse(skip_mask[0])
        self.assertFalse(skip_mask[1])
        # Continuity break mask should be set on newly-skipped frames.
        self.assertTrue(continuity_break_mask[2])
        self.assertTrue(continuity_break_mask[3])
        self.assertTrue(continuity_break_mask[4])

    def test_post_pad_extends_into_warn_band(self):
        LipsyncPipeline = self._import()
        # Layout: idx 0..4 frontal, 5..7 skipped, 8..12 warn-band.
        n = 13
        skip_mask, continuity_break_mask = self._make_masks(n)
        yaw_skip_reasons = [False] * n
        for i in (5, 6, 7):
            yaw_skip_reasons[i] = True
        yaws = [5.0] * 5 + [35.0] * 3 + [25.0] * 5
        for i in (5, 6, 7):
            skip_mask[i] = True
        extra = LipsyncPipeline._apply_episode_pad(
            skip_mask, continuity_break_mask, yaws, yaw_skip_reasons,
            pre_pad=0, post_pad=3, warn_threshold=22.5,
        )
        # Post pad should pick up idx 8,9,10.
        self.assertEqual(extra, 3)
        self.assertTrue(skip_mask[8])
        self.assertTrue(skip_mask[9])
        self.assertTrue(skip_mask[10])
        self.assertFalse(skip_mask[11])
        self.assertFalse(skip_mask[12])

    def test_pre_pad_clipped_at_sequence_start(self):
        LipsyncPipeline = self._import()
        # Run at the very start; pre_pad window would extend to idx -3, but
        # we should clip at 0 and only mark the run itself (idx 0,1,2 are
        # already skipped; nothing extra to mark).
        n = 10
        skip_mask, continuity_break_mask = self._make_masks(n)
        yaw_skip_reasons = [True] * 3 + [False] * 7
        yaws = [35.0] * 3 + [5.0] * 7
        for i in (0, 1, 2):
            skip_mask[i] = True
        extra = LipsyncPipeline._apply_episode_pad(
            skip_mask, continuity_break_mask, yaws, yaw_skip_reasons,
            pre_pad=3, post_pad=0, warn_threshold=22.5,
        )
        self.assertEqual(extra, 0)

    def test_post_pad_clipped_at_sequence_end(self):
        LipsyncPipeline = self._import()
        # Run at the very end; post_pad window clips at n-1.
        n = 10
        skip_mask, continuity_break_mask = self._make_masks(n)
        yaw_skip_reasons = [False] * 7 + [True] * 3
        yaws = [5.0] * 7 + [35.0] * 3
        for i in (7, 8, 9):
            skip_mask[i] = True
        extra = LipsyncPipeline._apply_episode_pad(
            skip_mask, continuity_break_mask, yaws, yaw_skip_reasons,
            pre_pad=0, post_pad=3, warn_threshold=22.5,
        )
        self.assertEqual(extra, 0)

    def test_does_not_extend_into_below_warn_band(self):
        LipsyncPipeline = self._import()
        # Pre-window has frames in the warn band (>= warn_threshold) and
        # some below. Only warn-band frames should be added.
        n = 10
        skip_mask, continuity_break_mask = self._make_masks(n)
        yaw_skip_reasons = [False] * n
        for i in (5, 6):
            yaw_skip_reasons[i] = True
        yaws = [10.0, 25.0, 25.0, 15.0, 5.0, 35.0, 35.0, 5.0, 5.0, 5.0]
        for i in (5, 6):
            skip_mask[i] = True
        extra = LipsyncPipeline._apply_episode_pad(
            skip_mask, continuity_break_mask, yaws, yaw_skip_reasons,
            pre_pad=3, post_pad=0, warn_threshold=22.5,
        )
        # Only idx 1, 2 are in warn band; idx 0 (10°), 3 (15°), 4 (5°) below.
        self.assertEqual(extra, 2)
        self.assertTrue(skip_mask[1])
        self.assertTrue(skip_mask[2])
        self.assertFalse(skip_mask[0])
        self.assertFalse(skip_mask[3])

    def test_respects_already_skipped_frames(self):
        LipsyncPipeline = self._import()
        # Frame 4 already skipped (e.g. detect_fail); pad should not double-mark.
        n = 10
        skip_mask, continuity_break_mask = self._make_masks(n)
        yaw_skip_reasons = [False] * n
        for i in (5, 6):
            yaw_skip_reasons[i] = True
        yaws = [25.0] * 5 + [35.0] * 2 + [5.0] * 3
        skip_mask[4] = True  # already skipped for other reason
        for i in (5, 6):
            skip_mask[i] = True
        extra = LipsyncPipeline._apply_episode_pad(
            skip_mask, continuity_break_mask, yaws, yaw_skip_reasons,
            pre_pad=3, post_pad=0, warn_threshold=22.5,
        )
        # Idx 4 already skipped (not counted in extra). Only idx 1,2,3 newly added.
        self.assertEqual(extra, 3)
        self.assertEqual(sum(skip_mask), 6)  # 4 (pre-existing) + 5,6 (run) + 1,2,3 (new)

    def test_warn_threshold_zero_disables(self):
        LipsyncPipeline = self._import()
        n = 10
        skip_mask, continuity_break_mask = self._make_masks(n)
        yaw_skip_reasons = [False] * 5 + [True] * 2 + [False] * 3
        yaws = [25.0] * 5 + [35.0] * 2 + [5.0] * 3
        for i in (5, 6):
            skip_mask[i] = True
        extra = LipsyncPipeline._apply_episode_pad(
            skip_mask, continuity_break_mask, yaws, yaw_skip_reasons,
            pre_pad=3, post_pad=3, warn_threshold=0.0,
        )
        self.assertEqual(extra, 0)


class TestApplyWarnRunSkip(_FilterHelperBase):
    """``_apply_warn_run_skip`` skips a contiguous run of warn-band frames
    when it lasts long enough.
    """

    def test_run_below_min_run_is_kept(self):
        LipsyncPipeline = self._import()
        # 3 warn-band frames, min_run=5: nothing should be skipped.
        n = 10
        skip_mask = [False] * n
        continuity_break_mask = [False] * n
        yaws = [5.0] * 3 + [25.0] * 3 + [5.0] * 4
        extra = LipsyncPipeline._apply_warn_run_skip(
            skip_mask, continuity_break_mask, yaws,
            warn_threshold=22.5, min_run_frames=5,
        )
        self.assertEqual(extra, 0)
        self.assertEqual(sum(skip_mask), 0)

    def test_run_at_min_run_is_skipped(self):
        LipsyncPipeline = self._import()
        # 5 warn-band frames, min_run=5: all 5 should be skipped.
        n = 10
        skip_mask = [False] * n
        continuity_break_mask = [False] * n
        yaws = [5.0] * 2 + [25.0] * 5 + [5.0] * 3
        extra = LipsyncPipeline._apply_warn_run_skip(
            skip_mask, continuity_break_mask, yaws,
            warn_threshold=22.5, min_run_frames=5,
        )
        self.assertEqual(extra, 5)
        for i in (2, 3, 4, 5, 6):
            self.assertTrue(skip_mask[i])

    def test_run_above_min_run_is_skipped(self):
        LipsyncPipeline = self._import()
        # 8 warn-band frames, min_run=5: all 8 should be skipped.
        n = 10
        skip_mask = [False] * n
        continuity_break_mask = [False] * n
        yaws = [25.0] * 8 + [5.0] * 2
        extra = LipsyncPipeline._apply_warn_run_skip(
            skip_mask, continuity_break_mask, yaws,
            warn_threshold=22.5, min_run_frames=5,
        )
        self.assertEqual(extra, 8)

    def test_already_skipped_frame_breaks_run(self):
        LipsyncPipeline = self._import()
        # 6 warn-band frames, but frame in the middle is already skipped --
        # that splits the run into two pieces of length 3 and 2.
        n = 10
        skip_mask = [False] * n
        continuity_break_mask = [False] * n
        skip_mask[4] = True  # already skipped for other reason
        yaws = [25.0] * 10
        extra = LipsyncPipeline._apply_warn_run_skip(
            skip_mask, continuity_break_mask, yaws,
            warn_threshold=22.5, min_run_frames=5,
        )
        # Neither sub-run reaches 5; nothing new is skipped.
        self.assertEqual(extra, 0)

    def test_detect_fail_breaks_run(self):
        LipsyncPipeline = self._import()
        # Frame 5 has yaw=None (detect fail); splits the run.
        n = 10
        skip_mask = [False] * n
        continuity_break_mask = [False] * n
        yaws: List[Optional[float]] = [25.0] * 5 + [None] + [25.0] * 4
        extra = LipsyncPipeline._apply_warn_run_skip(
            skip_mask, continuity_break_mask, yaws,
            warn_threshold=22.5, min_run_frames=5,
        )
        self.assertEqual(extra, 0)

    def test_min_run_zero_disables(self):
        LipsyncPipeline = self._import()
        n = 10
        skip_mask = [False] * n
        continuity_break_mask = [False] * n
        yaws = [25.0] * 10
        extra = LipsyncPipeline._apply_warn_run_skip(
            skip_mask, continuity_break_mask, yaws,
            warn_threshold=22.5, min_run_frames=0,
        )
        self.assertEqual(extra, 0)

    def test_warn_threshold_zero_disables(self):
        LipsyncPipeline = self._import()
        n = 10
        skip_mask = [False] * n
        continuity_break_mask = [False] * n
        yaws = [25.0] * 10
        extra = LipsyncPipeline._apply_warn_run_skip(
            skip_mask, continuity_break_mask, yaws,
            warn_threshold=0.0, min_run_frames=5,
        )
        self.assertEqual(extra, 0)


class TestStabilizeYawForRate(_FilterHelperBase):
    """``_stabilize_yaw_for_rate`` collapses landmark sign-flip jitter on
    near-frontal faces to 0 for the rate computation.
    """

    def test_prev_yaw_none_passthrough(self):
        LipsyncPipeline = self._import()
        out = LipsyncPipeline._stabilize_yaw_for_rate(2.0, prev_yaw=None)
        self.assertEqual(out, 2.0)

    def test_both_below_floor_collapses_to_zero(self):
        LipsyncPipeline = self._import()
        # |yaw| < 3 and |prev| < 3 -- both in noise band.
        out = LipsyncPipeline._stabilize_yaw_for_rate(2.0, prev_yaw=-1.0)
        self.assertEqual(out, 0.0)
        out = LipsyncPipeline._stabilize_yaw_for_rate(-2.5, prev_yaw=2.5)
        self.assertEqual(out, 0.0)

    def test_current_below_prev_above_keeps_current(self):
        LipsyncPipeline = self._import()
        # prev=5° is above the floor; current=1° should pass through.
        out = LipsyncPipeline._stabilize_yaw_for_rate(1.0, prev_yaw=5.0)
        self.assertEqual(out, 1.0)

    def test_current_above_prev_below_keeps_current(self):
        LipsyncPipeline = self._import()
        # current=5° is above the floor; prev=1° should pass through.
        out = LipsyncPipeline._stabilize_yaw_for_rate(5.0, prev_yaw=1.0)
        self.assertEqual(out, 5.0)

    def test_both_above_floor_keeps_current(self):
        LipsyncPipeline = self._import()
        # Real turn -- both above floor, pass through.
        out = LipsyncPipeline._stabilize_yaw_for_rate(15.0, prev_yaw=10.0)
        self.assertEqual(out, 15.0)

    def test_at_floor_boundary_is_above(self):
        LipsyncPipeline = self._import()
        # sign_floor=3.0, |x| < 3.0 is the only condition; |x|=3.0 is NOT below.
        out = LipsyncPipeline._stabilize_yaw_for_rate(3.0, prev_yaw=-3.0)
        self.assertEqual(out, 3.0)

    def test_custom_floor(self):
        LipsyncPipeline = self._import()
        # With floor=5.0, |x|=4.0 is below for both.
        out = LipsyncPipeline._stabilize_yaw_for_rate(4.0, prev_yaw=-4.0, sign_floor=5.0)
        self.assertEqual(out, 0.0)
        # But |x|=6.0 is above, pass through.
        out = LipsyncPipeline._stabilize_yaw_for_rate(6.0, prev_yaw=-4.0, sign_floor=5.0)
        self.assertEqual(out, 6.0)


class TestRateSkipIntegration(_FilterHelperBase):
    """End-to-end (still CPU-only) check that the rate calculation using
    ``_stabilize_yaw_for_rate`` correctly avoids false rate-skip on a
    frontal jitter sequence and correctly fires on a real turn.
    """

    def _simulate_rate_skip(self, yaws, rate_threshold=28.0, sign_floor=3.0):
        LipsyncPipeline = self._import()
        rate_skips = 0
        prev_yaw: Optional[float] = None
        for yaw in yaws:
            if prev_yaw is not None:
                stabilized = LipsyncPipeline._stabilize_yaw_for_rate(yaw, prev_yaw, sign_floor)
                rate = abs(stabilized - prev_yaw)
                if rate > rate_threshold:
                    rate_skips += 1
            prev_yaw = yaw
        return rate_skips

    def test_sign_jitter_frontal_no_false_rate_skip(self):
        # 5 frames of jitter, all within the sign_floor band.
        rate_skips = self._simulate_rate_skip([1.0, -2.0, 1.5, -1.0, 2.0])
        self.assertEqual(rate_skips, 0)

    def test_real_turn_still_triggers_rate_skip(self):
        # Frame 3 (12° → 20°): rate 8°, below 28° threshold.
        # Frame 4 (20° → 35°): rate 15°, below 28° threshold.
        # With threshold 10°: frame 4 should fire (rate 15 > 10).
        rate_skips = self._simulate_rate_skip(
            [5.0, 8.0, 12.0, 20.0, 35.0], rate_threshold=10.0,
        )
        self.assertEqual(rate_skips, 1)  # only the 20→35 frame

    def test_just_below_skip_threshold_no_rate_skip(self):
        # Steady 28-32°, real motion, all rate < 3°/frame.
        rate_skips = self._simulate_rate_skip(
            [28.0, 29.0, 30.0, 31.0, 32.0], rate_threshold=28.0,
        )
        self.assertEqual(rate_skips, 0)

    def test_sign_floor_only_when_both_below(self):
        # Frame 1: +5° (above floor), prev=+1° (below).
        # Stabilization should NOT collapse (current above floor).
        # rate = |5 - 1| = 4°.
        LipsyncPipeline = self._import()
        stabilized = LipsyncPipeline._stabilize_yaw_for_rate(5.0, prev_yaw=1.0)
        self.assertEqual(stabilized, 5.0)
        # Then frame 2: +1° (below), prev=+5° (above).
        # Stabilization should NOT collapse (prev above floor).
        stabilized = LipsyncPipeline._stabilize_yaw_for_rate(1.0, prev_yaw=5.0)
        self.assertEqual(stabilized, 1.0)


class TestComputeBlendZone(_FilterHelperBase):
    """``_compute_blend_zone`` returns a per-frame blend coefficient for
    cross-fading the inpaint output with the source frame at side-face
    boundaries.
    """

    def test_empty_input(self):
        LipsyncPipeline = self._import()
        out = LipsyncPipeline._compute_blend_zone([], fade_frames=3)
        self.assertEqual(out, [])

    def test_fade_frames_zero_disables(self):
        LipsyncPipeline = self._import()
        skip_mask = [False, True, True, False, False]
        out = LipsyncPipeline._compute_blend_zone(skip_mask, fade_frames=0)
        self.assertEqual(out, [0.0] * 5)

    def test_blend_at_boundary_zero_disables(self):
        LipsyncPipeline = self._import()
        skip_mask = [False, True, True, False, False]
        out = LipsyncPipeline._compute_blend_zone(
            skip_mask, fade_frames=3, blend_at_boundary=0.0,
        )
        self.assertEqual(out, [0.0] * 5)

    def test_no_skips_anywhere(self):
        LipsyncPipeline = self._import()
        skip_mask = [False] * 10
        out = LipsyncPipeline._compute_blend_zone(skip_mask, fade_frames=3)
        self.assertEqual(out, [0.0] * 10)

    def test_all_skips(self):
        LipsyncPipeline = self._import()
        skip_mask = [True] * 10
        out = LipsyncPipeline._compute_blend_zone(skip_mask, fade_frames=3)
        # All frames are skip frames themselves -> blend is 0 (the
        # skip branch in restore_video handles them with pure source).
        self.assertEqual(out, [0.0] * 10)

    def test_single_skip_frame_ramps_symmetrically(self):
        LipsyncPipeline = self._import()
        # Skip at index 5; ramp on each side.
        skip_mask = [False] * 11
        skip_mask[5] = True
        out = LipsyncPipeline._compute_blend_zone(skip_mask, fade_frames=3)
        # Skip frame itself: 0.0
        self.assertEqual(out[5], 0.0)
        # Frames at distance 1, 2, 3 from the skip.
        # 0.5 * (1 - d/3) with d=1: 0.333, d=2: 0.167, d=3: 0.0
        # Left side
        self.assertAlmostEqual(out[4], 0.5 * (1 - 1/3), places=6)
        self.assertAlmostEqual(out[3], 0.5 * (1 - 2/3), places=6)
        self.assertAlmostEqual(out[2], 0.5 * (1 - 3/3), places=6)  # = 0
        # Right side (symmetric)
        self.assertAlmostEqual(out[6], 0.5 * (1 - 1/3), places=6)
        self.assertAlmostEqual(out[7], 0.5 * (1 - 2/3), places=6)
        self.assertAlmostEqual(out[8], 0.5 * (1 - 3/3), places=6)  # = 0
        # Far from any skip: still 0
        self.assertEqual(out[0], 0.0)
        self.assertEqual(out[1], 0.0)
        self.assertEqual(out[9], 0.0)
        self.assertEqual(out[10], 0.0)

    def test_skip_block_at_start(self):
        LipsyncPipeline = self._import()
        # Skip frames at idx 0,1,2 -- ramp only to the right.
        skip_mask = [True, True, True, False, False, False, False, False]
        out = LipsyncPipeline._compute_blend_zone(skip_mask, fade_frames=3)
        # Skip frames: 0
        self.assertEqual(out[0], 0.0)
        self.assertEqual(out[1], 0.0)
        self.assertEqual(out[2], 0.0)
        # Right ramp
        self.assertAlmostEqual(out[3], 0.5 * (1 - 1/3), places=6)
        self.assertAlmostEqual(out[4], 0.5 * (1 - 2/3), places=6)
        self.assertEqual(out[5], 0.0)
        self.assertEqual(out[6], 0.0)
        self.assertEqual(out[7], 0.0)

    def test_skip_block_at_end(self):
        LipsyncPipeline = self._import()
        # Skip frames at idx 5,6,7 -- ramp only to the left.
        skip_mask = [False, False, False, False, False, True, True, True]
        out = LipsyncPipeline._compute_blend_zone(skip_mask, fade_frames=3)
        # Skip frames: 0
        self.assertEqual(out[5], 0.0)
        self.assertEqual(out[6], 0.0)
        self.assertEqual(out[7], 0.0)
        # Left ramp
        self.assertAlmostEqual(out[4], 0.5 * (1 - 1/3), places=6)
        self.assertAlmostEqual(out[3], 0.5 * (1 - 2/3), places=6)
        self.assertEqual(out[2], 0.0)
        self.assertEqual(out[1], 0.0)
        self.assertEqual(out[0], 0.0)

    def test_two_separate_skip_blocks(self):
        LipsyncPipeline = self._import()
        # Two skip blocks; each gets its own ramp. The blend for an
        # inpaint frame is determined by its min distance to ANY skip,
        # not to a specific block -- a frame between two blocks picks
        # up the closer block's ramp.
        skip_mask = [False, False, True, True, False, False, False, True, True, False, False]
        out = LipsyncPipeline._compute_blend_zone(skip_mask, fade_frames=3)
        # Block 1: skip at 2,3. Block 2: skip at 7,8.
        self.assertEqual(out[2], 0.0)
        self.assertEqual(out[3], 0.0)
        self.assertEqual(out[7], 0.0)
        self.assertEqual(out[8], 0.0)
        # Idx 1: dist=1 to block 1 -> peak
        self.assertAlmostEqual(out[1], 0.5 * (1 - 1/3), places=6)
        # Idx 4: dist=1 to block 1 -> peak (right side)
        self.assertAlmostEqual(out[4], 0.5 * (1 - 1/3), places=6)
        # Idx 5: dist=2 to block 1 AND dist=2 to block 2 -> 0.167
        self.assertAlmostEqual(out[5], 0.5 * (1 - 2/3), places=6)
        # Idx 6: dist=3 to block 1, dist=1 to block 2 -> min=1 -> 0.333.
        # This is the subtle case the test catches: a frame BETWEEN two
        # skip blocks can still be in the fade zone of the closer one.
        self.assertAlmostEqual(out[6], 0.5 * (1 - 1/3), places=6)
        # Idx 9: dist=1 to block 2 -> peak
        self.assertAlmostEqual(out[9], 0.5 * (1 - 1/3), places=6)
        # Idx 0: dist=2 to block 1 (idx 0 -> 2). Inside fade window -> small blend.
        self.assertAlmostEqual(out[0], 0.5 * (1 - 2/3), places=6)
        # Idx 10: dist=2 to block 2 (idx 10 -> 8). Inside fade window -> small blend.
        self.assertAlmostEqual(out[10], 0.5 * (1 - 2/3), places=6)

    def test_far_inpaint_block_in_middle_stays_zero(self):
        LipsyncPipeline = self._import()
        # Long inpaint block in the middle far from any skip. With
        # fade_frames=3, only the 3 frames closest to each skip block
        # have non-zero blend; frames 5, 6, 7 are far enough from BOTH
        # blocks that the ramp has decayed to 0.
        skip_mask = [True, True, True, False, False, False, False, False, False, False, True, True, True]
        out = LipsyncPipeline._compute_blend_zone(skip_mask, fade_frames=3)
        # Left skip block at 0-2.
        self.assertAlmostEqual(out[3], 0.5 * (1 - 1/3), places=6)  # d=1
        self.assertAlmostEqual(out[4], 0.5 * (1 - 2/3), places=6)  # d=2
        self.assertEqual(out[5], 0.0)  # d=3 from left -> formula gives 0
        # Middle: far from both blocks.
        self.assertEqual(out[6], 0.0)  # d=4 from left, d=4 from right
        self.assertEqual(out[7], 0.0)  # d=5 from left, d=3 from right -> formula gives 0
        # Right skip block at 10-12. Idx 8 is d=2 from right, inside the
        # fade window -- small but non-zero blend. The test catches
        # "stay zero" only on the dead-zone in the middle.
        self.assertAlmostEqual(out[8], 0.5 * (1 - 2/3), places=6)  # d=2 from right
        self.assertAlmostEqual(out[9], 0.5 * (1 - 1/3), places=6)  # d=1 from right
        self.assertEqual(out[10], 0.0)
        self.assertEqual(out[11], 0.0)
        self.assertEqual(out[12], 0.0)

    def test_custom_blend_at_boundary(self):
        LipsyncPipeline = self._import()
        # Symmetric [F, T, F]: both inpaint frames are dist=1 from the
        # skip, so both get the same peak blend coefficient.
        # With blend_at_boundary=0.3, fade_frames=3: peak = 0.3 * (1-1/3) = 0.2
        skip_mask = [False, True, False]
        out = LipsyncPipeline._compute_blend_zone(
            skip_mask, fade_frames=3, blend_at_boundary=0.3,
        )
        self.assertEqual(out[1], 0.0)  # skip frame itself
        self.assertAlmostEqual(out[0], 0.3 * (1 - 1/3), places=6)
        self.assertAlmostEqual(out[2], 0.3 * (1 - 1/3), places=6)

    def test_peak_is_capped_below_blend_at_boundary(self):
        LipsyncPipeline = self._import()
        # For fade_frames=N, the peak (at d=1) is blend_at_boundary
        # * (N-1)/N, never reaching blend_at_boundary itself.
        skip_mask = [False, True]
        out = LipsyncPipeline._compute_blend_zone(
            skip_mask, fade_frames=3, blend_at_boundary=0.5,
        )
        # Closest inpaint frame at d=1 -> 0.5 * 2/3 = 0.333
        self.assertAlmostEqual(out[0], 0.5 * (1 - 1/3), places=6)
        self.assertLess(out[0], 0.5)  # peak is strictly less than the cap
        # For larger fade_frames the peak approaches the cap but never
        # reaches it.
        out2 = LipsyncPipeline._compute_blend_zone(
            skip_mask, fade_frames=10, blend_at_boundary=0.5,
        )
        self.assertAlmostEqual(out2[0], 0.5 * 9/10, places=6)
        self.assertLess(out2[0], 0.5)


class TestShotPassthroughGuard(_FilterHelperBase):
    """Shot-level guard upgrades high-risk shots to source passthrough."""

    def _two_shot_frames(self):
        first = np.zeros((6, 16, 16, 3), dtype=np.uint8)
        second = np.full((6, 16, 16, 3), 255, dtype=np.uint8)
        return np.concatenate([first, second], axis=0)

    def test_high_bad_ratio_forces_only_that_shot(self):
        LipsyncPipeline = self._import()
        frames = self._two_shot_frames()
        skip_mask = [
            True, True, True, False, False, False,
            False, False, True, False, False, False,
        ]
        continuity = [False] * len(skip_mask)

        stats = LipsyncPipeline._apply_shot_passthrough_guard(
            skip_mask,
            continuity,
            frames,
            list(range(len(skip_mask))),
            scene_cut_threshold=0.45,
            skip_ratio_threshold=0.45,
            min_shot_frames=4,
            min_bad_frames=2,
        )

        self.assertEqual(stats, {"shots": 1, "frames": 3})
        self.assertEqual(skip_mask[:6], [True] * 6)
        self.assertEqual(skip_mask[6:], [False, False, True, False, False, False])
        self.assertEqual(continuity[:6], [True] * 6)
        self.assertEqual(continuity[6:], [False] * 6)

    def test_low_bad_ratio_is_kept(self):
        LipsyncPipeline = self._import()
        frames = self._two_shot_frames()
        skip_mask = [
            True, False, False, False, False, False,
            False, True, False, False, False, False,
        ]
        continuity = [False] * len(skip_mask)
        original = skip_mask[:]

        stats = LipsyncPipeline._apply_shot_passthrough_guard(
            skip_mask,
            continuity,
            frames,
            list(range(len(skip_mask))),
            scene_cut_threshold=0.45,
            skip_ratio_threshold=0.45,
            min_shot_frames=4,
            min_bad_frames=2,
        )

        self.assertEqual(stats, {"shots": 0, "frames": 0})
        self.assertEqual(skip_mask, original)
        self.assertEqual(continuity, [False] * len(skip_mask))

    def test_short_shot_is_kept(self):
        LipsyncPipeline = self._import()
        frames = np.zeros((3, 16, 16, 3), dtype=np.uint8)
        skip_mask = [True, True, False]
        continuity = [False] * len(skip_mask)

        stats = LipsyncPipeline._apply_shot_passthrough_guard(
            skip_mask,
            continuity,
            frames,
            list(range(len(skip_mask))),
            scene_cut_threshold=0.45,
            skip_ratio_threshold=0.45,
            min_shot_frames=4,
            min_bad_frames=2,
        )

        self.assertEqual(stats, {"shots": 0, "frames": 0})
        self.assertEqual(skip_mask, [True, True, False])



if __name__ == "__main__":
    unittest.main()
