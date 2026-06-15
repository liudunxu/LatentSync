# AGENTS.md

> **Note:** This file is mirrored at `CLAUDE.md` (for Claude Code). The
> content is identical; whichever file your tool reads, you follow the
> same guidelines.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with
project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

## Project-Specific Notes (LatentSync)

### Testing
- **Don't run inference locally.** End-to-end and inference tests run on
  the user's remote GPU box. The user pulls the branch, runs the job,
  and reports logs back.
- Local checks are limited to:
  - Syntax: `python3 -m py_compile <file>`
  - Import sanity: `python3 -c "import latentsync.pipelines.lipsync_pipeline"`
  - Unit tests: `pytest tests/`
- Do not run `inference.sh` locally - it tries to load model checkpoints
  and the GPU stack is on the remote host.

### Mask & feather baseline is locked
- The mask/feather pipeline has been iterated on and reverted several
  times. **Baseline: `ef3903f` with `latentsync/utils/mask.png`.**
- Do not toggle masks, mask width, or feather without an explicit ask.
  Previous revert cycles were explicitly rejected.

### Side-face detection is delicate
- Current implementation in `_estimate_yaw_degrees`
  (`latentsync/pipelines/lipsync_pipeline.py`) uses a conservative
  multi-signal fusion with noise floors:
  - Nose offset (signed, *60 mapping)
  - Eye-width asymmetry (gated at ratio > 1.5)
  - Mouth-corner asymmetry (gated at diff > 0.2)
- Threshold tuning alone (`yaw_skip_threshold`) is a band-aid. Real
  fixes require improving the underlying signals. Use plan mode before
  changing the detection logic.

### Naturalness defaults are tuned (do not silently revert)
- `LATENTSYNC_GUIDANCE_SCALE=1.5` (env), 1.4 frontend
- `LATENTSYNC_INFERENCE_STEPS=40` (env), 35 frontend
- `LATENTSYNC_ENABLE_DEEPCACHE=1`
- `mouth_temporal_stabilization_strength=0.15` (default)
- `mouth_audio_motion_min_scale=0.75` (default)
- See commits `dc8d869` and `37afbc7` for the trade-off reasoning.
  Adjusting these has visible quality vs. speed trade-offs; ask first.

### Two-repo project
- `LatentSync` (this repo) - server + pipeline
- `~/workspace/others/dubbing` (separate git repo) - frontend
- **Frontend form fields must use the `*_override` suffix** to take
  effect on the server. Sending `guidance_scale` is silently ignored;
  the server only reads `guidance_scale_override`. The frontend already
  sends the suffixed names; if you add a new tunable, follow the same
  convention.

### Push policy
- After committing, run `git push origin main` directly. Don't ask
  before pushing unless the change is large or risky.

## Project Map

### Directory structure
```
LatentSync/
├── api.py                          # FastAPI server, /api/lipsync + /api/faces + /health + /api/download
├── predict.py                      # Replicate / Cog entry point
├── gradio_app.py                   # Gradio UI (alternative to ~/workspace/others/dubbing)
├── inference.sh                    # CLI inference (calls scripts/inference.py)
├── setup_env.sh                    # Python env + ffmpeg setup
│
├── latentsync/
│   ├── pipelines/
│   │   └── lipsync_pipeline.py     # ⭐ core: LipsyncPipeline.__call__ + ~30 _helper methods
│   ├── models/                     # UNet3DConditionModel + SyncNet + ResNet + motion module
│   │   ├── unet.py                 # 3D UNet with audio cross-attention
│   │   ├── unet_blocks.py
│   │   ├── motion_module.py        # temporal attention
│   │   ├── attention.py
│   │   ├── resnet.py
│   │   ├── stable_syncnet.py       # SyncNet (lip-sync confidence, training only)
│   │   ├── wav2lip_syncnet.py      # alternate syncnet impl
│   │   └── utils.py
│   ├── utils/
│   │   ├── image_processor.py      # affine transform, mask loading, restorer glue
│   │   ├── face_detector.py        # InsightFace wrapper, pose yaw captured to last_pose_yaw
│   │   ├── affine_transform.py     # AlignRestore — warp + paste face back into frame
│   │   ├── codeformer_restorer.py  # optional face-restoration postprocess
│   │   ├── audio.py                # read_audio / write_video via ffmpeg
│   │   ├── av_reader.py            # video decode (cv2 or decord)
│   │   ├── util.py
│   │   └── mask.png mask2-5.png    # inpaint masks; baseline = mask.png
│   ├── whisper/
│   │   └── audio2feature.py        # whisper-tiny → per-frame features
│   ├── data/                       # training datasets
│   │   ├── unet_dataset.py
│   │   └── syncnet_dataset.py
│   └── trepa/loss.py               # TREPA perceptual loss (training only)
│
├── configs/
│   ├── unet/                       # stage1 / stage2 / stage2_512 / stage2_efficient yaml
│   └── syncnet/                    # syncnet training configs
│
├── assets/                         # demo1/2/3 video+audio pairs
├── tests/                          # pytest (test_codeformer_integration, test_temporal_continuity)
├── preprocess/                     # offline data prep: filter, segment, sync, etc.
├── scripts/                        # train_syncnet / train_unet / inference CLI
├── tools/                          # ad-hoc utilities
├── eval/                           # evaluation: hyper_iqa, syncnet_detect, inference_videos
│
├── checkpoints/                    # ⛔ NOT in repo. Loaded at server startup from disk.
└── data/                           # outputs/inputs/job dirs (created at runtime)
```

### Architecture: one request through the pipeline

```
Client
  │ POST /api/lipsync { video_url, audio_url, [avatar_url], ... }
  ▼
api.py::create_lipsync
  │
  ├─ _download_to_file (×2-3)       # URL → local /inputs/<job_id>/, retries on 5xx
  ├─ [optional] load avatar face embedding if avatar_url provided
  │
  ▼
LatentSyncApiRuntime.synthesize
  │ (singleton; model loaded once at first request)
  ▼
LipsyncPipeline.__call__(video_path, audio_path, video_out_path, ...)
  │
  ├─ audio_encoder.audio2feat(audio)  # whisper-tiny → per-frame features
  ├─ read_video(video_path)           # cv2 or decord
  │
  ▼ loop_video (per batch of num_frames=16 frames)
  │
  ├─ affine_transform_video(video_frames_chunk)
  │   per frame:
  │     ├─ face_detector(frame)              # InsightFace: bbox + 106-pt lmk + pose[1]=yaw
  │     ├─ (refuse if yaw > 30° or yaw-rate > 28°/frame or other prefilter tripped)
  │     ├─ affine warp + crop 512×512
  │     ├─ face embedder (if reference) → identity similarity check
  │     └─ stack aligned faces + per-frame skip_mask + continuity_break_mask
  │
  ├─ pipeline(...)                          # DDIM diffusion in 3D UNet
  │   ├─ color match to reference (optional)
  │   ├─ mouth detail recovery (optional, default on)
  │   ├─ mouth sharpen (optional, default off)
  │   ├─ temporal EMA smoothing (3-tap kernel)
  │   ├─ mouth motion preserve (audio-adaptive)
  │   ├─ mouth temporal stabilization
  │   └─ quality postfilter (default off)
  │
  └─ restore_video
      └─ restorer.restore_img(frame, face, affine)  # paste face back into full frame
  ▼
write_video (ffmpeg + audio mux) → /outputs/<job_id>/result.mp4
  ▼
api.py returns { video_url, download_url, job_id, run_stats, ... }
```

### Prefilters (can skip a frame → fall back to original)
1. **Face detection fail** → zero-placeholder, no inpaint
2. **yaw > yaw_skip_threshold** (multi-signal, default 30°)
3. **yaw-rate > yaw_rate_skip_threshold** (deg/frame, default 28)
4. **face_jump** (landmark center/scale jump)
5. **lipsync_continuity** (geometry break from prev valid frame)
6. **lipsync_mouth_diff_break** (mouth-region pixel diff break)
7. **mouth_occlusion** (default disabled, threshold 1.0)
8. **motion_blur** (Laplacian variance, default 0.08)
9. **identity_similarity** (vs avatar embedding, default 0.5)
10. **adaptive_quality_fallback** (opt-in; composite score after diffusion)

All filter counts log under `[FaceMatch]` — first place to look when output looks wrong. When `adaptive_quality_fallback_enabled=True`, also check `[LipSync]` for `adaptive_quality_fallback=N`.

### Post-processes (applied to generated mouth)
- color match → mouth detail recovery → mouth sharpen → temporal EMA → motion preserve → stabilization
- All have request-level on/off knobs and strength floats
- `codeformer_enabled` is opt-in and runs in a separate pass

### FastAPI routes
| Path | Method | Purpose |
|---|---|---|
| `/health` | GET | liveness + detector/codeformer load state |
| `/api/faces` | POST | list distinct faces in a video (uses `_download_to_file` only) |
| `/api/lipsync` | POST | full pipeline; returns video_url + download_url + run stats |
| `/api/download` | GET | serves `/outputs/<job_id>/<file>` (local) or streams remote URL |

## Lessons Learned

1. **Side-face detection is the recurring pain point.** `yaw_skip_threshold`
   has been tuned 45° → 22° → 30° → 45° → 30° across many commits. Pure
   threshold tuning is a band-aid — fix the underlying signal quality
   (multi-signal with noise floors) instead of widening the threshold.

2. **Mask & feather baseline is locked at `ef3903f` + `mask.png`.** Three
   prior revert cycles were explicitly rejected by the user. Adding
   `mask2`/`mask3` is fine for experiments; do not promote them to
   defaults without an explicit ask.

3. **Frontend field-name bug class: un-suffixed vs `*_override`.** The
   server only reads `guidance_scale_override`, `inference_steps_override`,
   `seed_override`. The frontend was sending the un-suffixed names and
   silently getting the server defaults. New per-request tunables MUST
   follow the `*_override` convention.

4. **DeepCache trade-off is real but worth it.** `cache_interval=3`
   skips 2/3 of UNet forwards (~2× speedup) with mild detail softening.
   Mouth motion stays correct, which is what the user actually judges.
   Re-enabled by default in `37afbc7` after user complained about speed.

5. **EMA smoothing on landmarks reduces mask-boundary jitter.** Added
   in `82e07f6` with `alpha=0.7` on `mouth_info` center/half-width. Cheap
   fix, large visible win when landmark detection is noisy.

6. **Settings are loaded once at startup.** Env vars
   (`LATENTSYNC_GUIDANCE_SCALE`, `LATENTSYNC_INFERENCE_STEPS`,
   `LATENTSYNC_ENABLE_DEEPCACHE`) only take effect on server restart.
   `enable_deepcache_override` field is a hint that logs a warning if
   it differs — DeepCache wiring happens at pipeline load time, not
   per-request.

7. **The naturalness/quality/speed triangle is real.** Each axis can
   be tuned independently but they interact:
   - guidance ↓ → more natural motion, less sync to audio
   - steps ↑ → smoother output, slower
   - DeepCache off → ~2× slower, slightly sharper
   - mouth_temporal_stab ↑ → smoother frame-to-frame, can blur motion
   - mouth_audio_motion_min ↑ → preserves more on weak audio, less "frozen"
   Current tuned defaults balance all three; large changes need plan mode.

8. **Always check `[FaceMatch]` log to diagnose output issues.** It
   reports per-filter skip counts (yaw_skip, yaw_rate_skip,
   identity_skip, mouth_occlusion_skip, motion_blur_skip,
   face_jump_skip, detect_fail). If a sample looks wrong, this is the
   first place to look to know WHICH filter is firing.

9. **Two-repo coupling: LatentSync + ~/workspace/others/dubbing.** Tunable
   defaults need coordinated commits in both repos. Adding a new
   field to `LipSyncRequest` without frontend support means it can
   only be set via direct API call.

10. **Use plan mode for any non-trivial change to the pipeline.** The
    `LipsyncPipeline.__call__` is 1100+ lines with 30+ helper methods
    and many interacting prefilters. Threshold tweaks can have
    non-obvious cascade effects. Plan → verify checklist → execute.

11. **`audio_sync_offset_seconds` semantics are fixed.** Positive means
    the provided audio is **ahead** of the video; the pipeline uses
    earlier audio features for each frame and delays the output audio
    by padding zeros at the start. If your frontend uses the opposite
    convention, negate the value before sending it.

12. **Adaptive quality fallback is opt-in.** The composite score
    (`adaptive_quality_fallback_enabled=False` by default) combines
    mouth sharpness, mouth-region diff, identity similarity, yaw,
    audio confidence and temporal stability. It is capped by
    `adaptive_quality_fallback_max_ratio` and passed through a
    hysteresis filter to suppress isolated-frame flicker. Enable it
    for short-drama content where a bad generated mouth is worse than
    keeping the original.

---

**These guidelines are working if:** fewer unnecessary changes in diffs,
fewer rewrites due to overcomplication, and clarifying questions come
before implementation rather than after mistakes.
