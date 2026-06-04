# CodeFormer face-restoration postprocess

The synthesized mouth and lip area are sometimes softer than the
surrounding face, because the diffusion inpainter favours plausible
shapes over crisp edges. As an opt-in postprocess, we now run
[CodeFormer](https://github.com/sczhou/CodeFormer) on the aligned
512×512 face crops just before they are pasted back to the full
video, then re-apply the same affine transform the pipeline already
uses. Background, body, and clothing are not touched, so the seam
between the restored face and the unchanged frame is invisible.

## What ships

* **Vendored CodeFormer.** A minimal, self-contained subset of
  CodeFormer's VQGAN + transformer architecture is vendored under
  `latentsync/utils/codeformer/`. We do not depend on `basicsr`:
  installing the upstream framework pulls in `facexlib`, `lmdb`,
  `tb-nightly`, and a `setup.py develop` install that doesn't
  compose well with the lean LatentSync runtime. The vendored
  module is inference-only and loads the published
  `codeformer.pth` checkpoint without key remapping.
* **Lazy, singleton restorer.** `CodeFormerRestorer` builds the
  model on the first request that asks for it and reuses it
  across calls. `LATENTSYNC_CODEFORMER_PRELOAD=1` opts in to
  eager load at server startup; the default is lazy so unused
  installations pay no GPU memory.
* **Per-request API fields.** `POST /api/lipsync` gains
  `codeformer_enabled`, `codeformer_fidelity_weight` (the `w`
  parameter; `0.5` matches the upstream default) and
  `codeformer_required` (when true, fail with HTTP 503 if the
  checkpoint is missing instead of skipping silently).
* **Per-frame passthrough.** Frames that the pipeline already
  decided to fall back to the source video for (side profile,
  motion blur, occluded mouth, etc.) are not re-sharpened, so
  the source face is never enhanced on top of itself.
* **Telemetry.** Every response now carries a `codeformer` block
  with `frames_total`, `frames_enhanced`, `frames_skipped_by_pipeline`,
  `elapsed_seconds`, and `error` if any. `GET /health` reports
  the load state and any load error.

## How to enable

1. Drop the released checkpoint at
   `checkpoints/codeformer/codeformer.pth` (or set
   `LATENTSYNC_CODEFORMER_CKPT`).
2. Set `codeformer_enabled: true` in the request body, or run
   the server with `LATENTSYNC_CODEFORMER_PRELOAD=1` to load
   eagerly.

See [docs/codeformer_integration.md](codeformer_integration.md)
for the full list of env vars, CLI flags, and request fields.

## Tests

`tests/test_codeformer_integration.py` covers the offline
contract: the vendored model constructs and runs a dummy forward
pass with the right tensor shape, the restorer fails safely on
missing checkpoints and malformed inputs, and the stats payload
round-trips through JSON. Twelve tests, all pass without GPU
or network access.
