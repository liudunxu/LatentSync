# CodeFormer face-restoration postprocess

LatentSync ships with an optional
[CodeFormer](https://github.com/sczhou/CodeFormer) postprocess that
sharpens the synthesized mouth and recovers identity/edge detail
that the diffusion inpainter tends to soften. It runs on the
**aligned 512×512 face crops** that the pipeline already produces,
then the pipeline pastes the enhanced crops back into the full
video using the existing `restore_img` math. Background, clothing
and the rest of the frame are not touched.

## Why postprocess at the aligned-face stage

CodeFormer is trained on aligned faces. Running it on the full
frame produces visible edge artefacts and also "restores" the
background, which the user can perceive as a different look
between the face and the body. By applying it to the 512×512
aligned crops and then pasting back through the existing affine
transform, the seam is invisible.

## How to enable

### 1. Get the checkpoint

```bash
mkdir -p checkpoints/codeformer
# Either download manually from the upstream v0.1.0 release
curl -L -o checkpoints/codeformer/codeformer.pth \
  https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth
```

The expected file size is ~360 MB. The vendored loader also
accepts `download_if_missing=True` (see `CodeFormer.load_codeformer`
in `latentsync/utils/codeformer/loader.py`).

### 2. Server-side defaults (optional)

| Env var                              | Default                                         | Meaning |
| ------------------------------------ | ----------------------------------------------- | ------- |
| `LATENTSYNC_CODEFORMER_CKPT`         | `checkpoints/codeformer/codeformer.pth`         | Path to the `.pth` file. |
| `LATENTSYNC_CODEFORMER_PRELOAD`      | `0`                                             | Load the model at server startup. `1` = eager. |
| `LATENTSYNC_CODEFORMER_BATCH_SIZE`   | `16`                                            | Faces per forward pass. 16 is safe on 24 GB GPUs. |
| `LATENTSYNC_CODEFORMER_REQUIRED`     | `0`                                             | If `1`, requests with `codeformer_enabled=True` fail loudly when the checkpoint is missing. |

CLI flags `--codeformer_checkpoint_path`, `--codeformer_preload`
and `--codeformer_batch_size` override the same values.

If `LATENTSYNC_CODEFORMER_PRELOAD=0` (the default), the model is
loaded lazily on the first request that asks for it, so unused
installations pay no GPU memory.

### 3. Per-request fields on `POST /api/lipsync`

| Field                          | Type   | Default | Meaning |
| ------------------------------ | ------ | ------- | ------- |
| `codeformer_enabled`           | bool   | `false` | Run CodeFormer on the aligned face crops before paste-back. |
| `codeformer_fidelity_weight`   | float  | `0.5`   | `w` parameter. `0.0` = sharpest, `1.0` = closest to input. `0.5` is the upstream balanced default. |
| `codeformer_required`          | bool   | `false` | If `true`, fail with HTTP 503 when the model is unavailable. |

Example:

```json
{
  "video_url": "https://example.com/face.mp4",
  "audio_url": "https://example.com/voice.wav",
  "codeformer_enabled": true,
  "codeformer_fidelity_weight": 0.5
}
```

### 4. Telemetry in the response

Every `/api/lipsync` response now includes a `codeformer` block:

```json
{
  "codeformer": {
    "requested": true,
    "fidelity_weight": 0.5,
    "required": false,
    "runtime_available": true,
    "runtime_load_error": "",
    "checkpoint_path": "checkpoints/codeformer/codeformer.pth",
    "frames_total": 750,
    "frames_enhanced": 700,
    "frames_skipped_by_pipeline": 50,
    "elapsed_seconds": 12.4,
    "error": ""
  }
}
```

`frames_skipped_by_pipeline` is the count of frames that LatentSync
*already* decided to fall back to the source video for (side
profile, motion blur, occluded mouth, etc.). Those frames bypass
CodeFormer so the source face is never re-sharpened on top of
itself.

## Health endpoint

`GET /health` reports the load state of the model:

```json
{
  "status": "ok",
  "codeformer": {
    "checkpoint_path": "checkpoints/codeformer/codeformer.pth",
    "loaded": false,
    "preload_requested": false,
    "load_error": ""
  }
}
```

`loaded: false` with empty `load_error` means CodeFormer hasn't
been used yet on this server (lazy load). A non-empty
`load_error` means the checkpoint is missing or corrupt.

## Vendoring rationale

We vendor `vqgan_arch.py` and `codeformer_arch.py` from
[`sczhou/CodeFormer`](https://github.com/sczhou/CodeFormer)
(NTU S-Lab License 1.0) instead of depending on `basicsr`.
The reasoning is documented at the top of
`latentsync/utils/codeformer/vqgan_arch.py`; in short,
`basicsr` pulls in `facexlib`, `lmdb`, `tb-nightly`, `yapf`
and a fragile `setup.py develop` install that doesn't
co-exist cleanly with the lean LatentSync runtime.

The vendored module is inference-only: no `ARCH_REGISTRY`
dependency, no training-only escape hatches, no logging
imports. Loading the published `codeformer.pth` works
without key remapping.

## Tests

`tests/test_codeformer_integration.py` covers the offline
contract: vendored model constructs and runs a dummy forward
pass with the right tensor shape, the restorer fails safely
on missing checkpoints and malformed inputs, and stats
serialize to JSON for the API response. Run with:

```bash
python -m unittest tests.test_codeformer_integration -v
```

The full happy-path (load `.pth` and run inference on real
images) is left to the inference host where the GPU and the
checkpoint are both available.
