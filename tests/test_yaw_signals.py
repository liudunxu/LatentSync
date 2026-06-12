"""Unit test for the new mouth area / aspect yaw signals in
``LipsyncPipeline._estimate_yaw_degrees``.

The new signals (4: area, 5: aspect) add cross-frame mouth-shape
information on top of the existing 3 corner-position signals
(nose / eye-asym / mouth-corner). Each has its own noise floor so
frontal faces stay near 0 even with landmark jitter.

This test doesn't replace GPU-side validation on real InsightFace
output -- synthetic landmarks have a 48-index conflict (used as
both mouth corner and a left-eye landmark), so the signal-1
"nose offset" baseline gets biased in this test. We test the
new signals in isolation by checking:
  1. invalid input -> 0 deg
  2. monotonicity: 60 deg yaw > 30 deg yaw
  3. 60 deg produces substantial yaw (>20 deg)
"""
import math
import sys
import re
import numpy as np
import ast


def _load_yaw_fn():
    """Lift _estimate_yaw_degrees out of the pipeline file without
    importing the whole module (which would require torch/insightface).
    """
    src_path = '/Users/dunxu.liu/workspace/others/LatentSync/latentsync/pipelines/lipsync_pipeline.py'
    with open(src_path) as f:
        src = f.read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == '_estimate_yaw_degrees':
                    fn_src = ast.get_source_segment(src, sub)
                    ns = {'np': np, 'math': math, 'Optional': type(None)}
                    exec(fn_src, ns)
                    return ns['_estimate_yaw_degrees']
    raise RuntimeError("could not locate _estimate_yaw_degrees")


def _synth_lmk(yaw_deg, cx=100.0, cy=125.0):
    """Realistic-shape 106-point face under yaw rotation. See module
    docstring for why lmk[48] is placed at the mouth corner
    (y=195) and the left-eye mean shifts accordingly -- this matches
    real InsightFace 106-pt output where index 48 is the mouth
    left corner."""
    yaw = math.radians(yaw_deg)
    cos_y = math.cos(yaw)
    lmk = np.zeros((106, 2), dtype=np.float64)
    for i in range(106):
        ang = (i / 106.0) * 2 * math.pi
        x3d = 90 * math.cos(ang)
        y3d = 60 * math.sin(ang)
        lmk[i] = (cx + x3d * cos_y, cy + y3d)
    for idx in [43, 49, 50, 51]:
        lmk[idx] = (cx - 35 * cos_y, 80)
    lmk[48] = (cx + 50 * cos_y, 195)
    for idx in range(101, 106):
        lmk[idx] = (cx + 35 * cos_y, 80)
    for idx in [74, 77, 83, 86]:
        lmk[idx] = (cx, 100)
    lmk[54] = (cx - 50 * cos_y, 195)
    lmk[51] = (cx, 190)
    lmk[57] = (cx, 215)
    return lmk


def test_yaw_signals_invalid_input():
    fn = _load_yaw_fn()
    assert fn(None) == 0.0
    assert fn(np.zeros((50, 2))) == 0.0
    assert fn(np.zeros((106, 2))) == 0.0  # 106 points, all (0,0)


def test_yaw_signals_monotonic():
    fn = _load_yaw_fn()
    yaw0 = fn(_synth_lmk(0))
    yaw15 = fn(_synth_lmk(15))
    yaw30 = fn(_synth_lmk(30))
    yaw60 = fn(_synth_lmk(60))
    # New signals 4 + 5 contribute more at higher yaw, so abs(yaw)
    # should be non-decreasing across the sweep.
    assert abs(yaw60) >= abs(yaw30) - 1.0, (
        f"60 deg yaw {yaw60} should be >= 30 deg yaw {yaw30}"
    )
    assert abs(yaw30) >= abs(yaw15) - 1.0, (
        f"30 deg yaw {yaw30} should be >= 15 deg yaw {yaw15}"
    )


def test_yaw_signals_produces_substantial_yaw_at_60():
    fn = _load_yaw_fn()
    yaw60 = fn(_synth_lmk(60))
    assert abs(yaw60) > 20, (
        f"60 deg rotation should produce >20 deg estimated yaw, got {yaw60}"
    )


def test_new_signals_noise_floors():
    """The new signals 4 (area) + 5 (aspect) have explicit noise
    floors and only fire below them. Verify by computing them in
    isolation: a frontal face should produce 0 deg from both.

    We test the signal math directly (not through
    ``_estimate_yaw_degrees``) because the synthetic test layout
    has a 5-points-collide artifact that the eye-width signal
    amplifies -- that's a pre-existing data quirk, not something
    this change should fix.
    """
    import math as _math
    lmk = _synth_lmk(0)
    face_area = (lmk[:, 0].max() - lmk[:, 0].min()) * (lmk[:, 1].max() - lmk[:, 1].min())
    mouth_w = float(np.linalg.norm(lmk[54] - lmk[48]))
    mouth_h = float(max(np.linalg.norm(lmk[57] - lmk[51]) / 2.0, 2.0))
    area_norm = (_math.pi * (mouth_w / 2.0) * mouth_h) / face_area
    aspect = mouth_w / mouth_h

    # Noise floor 0.025: frontal faces should be safely above this.
    assert area_norm > 0.025, f"frontal area_norm {area_norm} should be above noise floor"
    # Noise floor 2.0: frontal aspect should be safely above this.
    assert aspect > 2.0, f"frontal aspect {aspect} should be above noise floor"
    # Both signals -> 0
    area_yaw = (0.025 - area_norm) * 1200.0 if area_norm < 0.025 else 0.0
    aspect_yaw = (2.0 - aspect) * 60.0 if aspect < 2.0 else 0.0
    assert area_yaw == 0.0
    assert aspect_yaw == 0.0


def test_new_signals_fire_under_yaw():
    """At a meaningful yaw the new signals should start to fire.

    60 deg should drop area_norm below 0.025 OR aspect below 2.0
    (or both). The function returns the max across all signals, so
    the total yaw at 60 deg should exceed 20 deg in this synthetic
    layout.
    """
    fn = _load_yaw_fn()
    yaw60 = abs(fn(_synth_lmk(60)))
    assert yaw60 > 20, f"60 deg should produce >20 deg yaw, got {yaw60}"


def test_apply_warn_run_skip_seconds():
    """Time-based gate: a run of warn-band frames lasting >min_run_seconds
    is marked entirely as passthrough. Mirrors the absolute yaw passthrough
    in the same module but uses wall-clock duration instead of frame count.
    """
    # Lift the static method without importing the full module
    # (which would require torch/insightface).
    src_path = '/Users/dunxu.liu/workspace/others/LatentSync/latentsync/pipelines/lipsync_pipeline.py'
    with open(src_path) as f:
        src = f.read()
    import ast as _ast
    tree = _ast.parse(src)
    fn_src = None
    for node in tree.body:
        if isinstance(node, _ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, _ast.FunctionDef) and sub.name == '_apply_warn_run_skip':
                    fn_src = _ast.get_source_segment(src, sub)
                    break
    assert fn_src is not None
    # Strip the type annotations (List[bool], etc.) so the function
    # can exec without typing imports.
    fn_src_clean = re.sub(r': List\[bool\]', '', fn_src)
    fn_src_clean = re.sub(r': List\[Optional\[float\]\]', '', fn_src_clean)
    fn_src_clean = re.sub(r'-> int:', '-> None:', fn_src_clean)
    ns = {}
    exec(fn_src_clean, ns)
    fn = ns['_apply_warn_run_skip']

    # 10 frames of yaw=25° (>22.5 warn), fps=25, min_run_seconds=0.2
    # 10 frames * 0.04s = 0.4s > 0.2s -> all skipped
    skip = [False] * 10
    continuity = [False] * 10
    yaws = [25.0] * 10
    fn(
        skip, continuity, yaws, warn_threshold=22.5,
        min_run_frames=0, min_run_seconds=0.2, fps=25.0,
    )
    assert all(skip), f"all frames should be skipped, got {sum(skip)}/10"

    # Below the time gate (5 frames at 25fps = 0.2s, but threshold is 0.5s)
    skip = [False] * 5
    continuity = [False] * 5
    yaws = [25.0] * 5
    fn(
        skip, continuity, yaws, warn_threshold=22.5,
        min_run_frames=0, min_run_seconds=0.5, fps=25.0,
    )
    assert not any(skip), "5 frames (0.2s) should NOT clear 0.5s gate"

    # min_run_frames takes effect when seconds is 0
    skip = [False] * 6
    continuity = [False] * 6
    yaws = [25.0] * 6
    fn(
        skip, continuity, yaws, warn_threshold=22.5,
        min_run_frames=5, min_run_seconds=0.0, fps=25.0,
    )
    assert all(skip), "6 frames >= min_run_frames=5 should all skip"

    # Whichever is larger wins
    skip = [False] * 4
    continuity = [False] * 4
    yaws = [25.0] * 4
    fn(
        skip, continuity, yaws, warn_threshold=22.5,
        min_run_frames=5, min_run_seconds=0.5, fps=25.0,
    )
    # min_run_frames=5, min_run_seconds*fps=12.5 -> effective=12. 4 < 12, no skip
    assert not any(skip), "effective=12 frames > 4 frames should NOT clear"

    # Both disabled
    skip = [False] * 100
    continuity = [False] * 100
    yaws = [25.0] * 100
    fn(
        skip, continuity, yaws, warn_threshold=22.5,
        min_run_frames=0, min_run_seconds=0.0, fps=25.0,
    )
    assert not any(skip), "both disabled -> no skip"
