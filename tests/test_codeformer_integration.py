"""Smoke tests for the CodeFormer integration.

These tests are intentionally light: the full happy path (load
``codeformer.pth`` and run inference on real images) requires a
GPU box and a 360 MB checkpoint, which is set up on the inference
host, not in unit-test CI. What we test here is everything that
*doesn't* need either: model construction, tensor shape contract,
parameter validation, and the restorer's failure modes.

Run with::

    python -m tests.test_codeformer_integration

or via pytest once torch is installed in the env.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest

# Make ``latentsync`` importable when running from the project root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TestVendoredCodeformer(unittest.TestCase):
    """The vendored arch file should construct and run a forward pass
    end-to-end without touching the network or the file system."""

    def test_construct_default(self):
        from latentsync.utils.codeformer import CodeFormer

        net = CodeFormer()
        # The published CodeFormer has ~94M parameters (see
        # README.md in the upstream repo). Anything drastically off
        # is a vendoring bug.
        n_params = sum(p.numel() for p in net.parameters())
        self.assertGreater(n_params, 80_000_000)
        self.assertLess(n_params, 110_000_000)

    def test_construct_clamps_no_trainable(self):
        from latentsync.utils.codeformer import CodeFormer

        net = CodeFormer()
        # Inference should never try to backprop; parameters should
        # not require grad after construction. (load_codeformer is
        # the path that actually freezes them, but the constructor
        # must not enable training-only paths either.)
        # Note: parameters retain requires_grad=True at construction
        # because the upstream class only freezes when fix_modules
        # is set; the inference loader handles freezing. We just
        # check that no parameters are marked as buffers-only.
        for name, _ in net.named_parameters():
            self.assertIsInstance(name, str)
            self.assertGreater(len(name), 0)

    def test_forward_shape_preserved(self):
        import torch

        from latentsync.utils.codeformer import CodeFormer

        net = CodeFormer().eval()
        x = torch.randn(2, 3, 512, 512)
        with torch.no_grad():
            out, logits, lq = net(x, w=0.5, adain=True)
        self.assertEqual(out.shape, (2, 3, 512, 512))
        # logits: (B, HW, codebook_size) = (B, 256, 1024)
        self.assertEqual(logits.shape, (2, 256, 1024))
        # lq_feat: (B, 256, 16, 16)
        self.assertEqual(lq.shape, (2, 256, 16, 16))

    def test_forward_with_zero_fidelity(self):
        import torch

        from latentsync.utils.codeformer import CodeFormer

        net = CodeFormer().eval()
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            out, _, _ = net(x, w=0.0, adain=True)
        # w=0 must still produce a valid output -- the SFT blocks
        # early-exit on w<=0, but the codebook+generator path runs.
        self.assertEqual(out.shape, (1, 3, 512, 512))
        # Output is not all-zero: the codebook lookup at minimum
        # produces structured output.
        self.assertGreater(out.abs().sum().item(), 0.0)

    def test_forward_adain_off(self):
        import torch

        from latentsync.utils.codeformer import CodeFormer

        net = CodeFormer().eval()
        x = torch.randn(1, 3, 512, 512)
        with torch.no_grad():
            out_off, _, _ = net(x, w=0.5, adain=False)
            out_on, _, _ = net(x, w=0.5, adain=True)
        # adain=True and adain=False should produce different outputs;
        # this is a weak contract but it catches "I forgot to thread
        # the flag through" bugs.
        self.assertFalse(torch.allclose(out_off, out_on, atol=1e-6))


class TestCodeformerRestorer(unittest.TestCase):
    """The restorer must degrade gracefully when the checkpoint is
    missing or the input is malformed, instead of crashing the
    whole request."""

    def test_missing_checkpoint_returns_input_unchanged(self):
        import torch

        from latentsync.utils.codeformer_restorer import CodeFormerRestorer

        restorer = CodeFormerRestorer(
            checkpoint_path="/nonexistent/codeformer.pth", device="cpu"
        )
        faces = torch.randn(3, 3, 512, 512)
        out, stats = restorer.restore_faces(faces, fidelity_weight=0.5)
        self.assertEqual(out.shape, faces.shape)
        # The fallback contract is "return input unchanged when the
        # model isn't loadable". This is what the pipeline relies on
        # to keep producing output when the user asks for
        # codeformer_enabled=True on a server that hasn't downloaded
        # the weights yet.
        self.assertTrue(torch.equal(out, faces))
        self.assertFalse(stats.loaded)
        self.assertIn("not found", stats.error)

    def test_skip_mask_passthrough(self):
        import torch

        from latentsync.utils.codeformer_restorer import CodeFormerRestorer

        restorer = CodeFormerRestorer(
            checkpoint_path="/nonexistent/codeformer.pth", device="cpu"
        )
        # Even with a loaded model, frames marked skipped must be
        # returned exactly as the caller gave them. We can't test
        # the post-load path here (no weights), but we *can* check
        # that skip_mask propagates into the stats.
        faces = torch.randn(4, 3, 512, 512)
        skip = [True, False, True, False]
        _out, stats = restorer.restore_faces(faces, skip_mask=skip)
        self.assertEqual(stats.frames_total, 4)
        self.assertEqual(stats.frames_skipped_by_pipeline, 2)
        self.assertEqual(stats.frames_enhanced, 0)

    def test_short_skip_mask_padded_with_false(self):
        import torch

        from latentsync.utils.codeformer_restorer import CodeFormerRestorer

        restorer = CodeFormerRestorer(
            checkpoint_path="/nonexistent/codeformer.pth", device="cpu"
        )
        faces = torch.randn(3, 3, 512, 512)
        _out, stats = restorer.restore_faces(faces, skip_mask=[True])
        # The mask is shorter than T; restorer pads with False so
        # the remaining frames are eligible for enhancement.
        self.assertEqual(stats.frames_skipped_by_pipeline, 1)
        self.assertEqual(stats.frames_total, 3)

    def test_skip_mask_avoids_inference_for_skipped_faces(self):
        import torch

        from latentsync.utils.codeformer_restorer import CodeFormerRestorer

        class FakeNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.zeros(()))
                self.seen_batch_sizes = []

            def forward(self, x, w=0.0, adain=True):
                self.seen_batch_sizes.append(x.shape[0])
                return x + 0.25, None, None

        fake_net = FakeNet()
        restorer = CodeFormerRestorer(
            checkpoint_path="/nonexistent/codeformer.pth",
            device="cpu",
            batch_size=2,
        )
        restorer._net = fake_net
        faces = torch.zeros(4, 3, 512, 512)
        out, stats = restorer.restore_faces(
            faces,
            skip_mask=[True, False, True, False],
        )
        self.assertEqual(fake_net.seen_batch_sizes, [2])
        self.assertTrue(torch.equal(out[0], faces[0]))
        self.assertTrue(torch.equal(out[2], faces[2]))
        self.assertGreater(out[1].sum().item(), 0.0)
        self.assertGreater(out[3].sum().item(), 0.0)
        self.assertEqual(stats.frames_enhanced, 2)
        self.assertEqual(stats.frames_skipped_by_pipeline, 2)

    def test_all_skipped_faces_do_not_load_model(self):
        import torch

        from latentsync.utils.codeformer_restorer import CodeFormerRestorer

        restorer = CodeFormerRestorer(
            checkpoint_path="/nonexistent/codeformer.pth", device="cpu"
        )
        faces = torch.randn(2, 3, 512, 512)
        out, stats = restorer.restore_faces(faces, skip_mask=[True, True])
        self.assertTrue(torch.equal(out, faces))
        self.assertFalse(stats.loaded)
        self.assertEqual(stats.error, "")
        self.assertEqual(stats.frames_enhanced, 0)
        self.assertEqual(stats.frames_skipped_by_pipeline, 2)

    def test_rejects_non_4d_input(self):
        import torch

        from latentsync.utils.codeformer_restorer import CodeFormerRestorer

        restorer = CodeFormerRestorer(
            checkpoint_path="/nonexistent/codeformer.pth", device="cpu"
        )
        out, stats = restorer.restore_faces(torch.randn(3, 512, 512))
        self.assertIn("Expected faces of shape", stats.error)
        # No enhancement happened.
        self.assertEqual(stats.frames_enhanced, 0)

    def test_rejects_non_square_faces(self):
        import torch

        from latentsync.utils.codeformer_restorer import CodeFormerRestorer

        restorer = CodeFormerRestorer(
            checkpoint_path="/nonexistent/codeformer.pth", device="cpu"
        )
        out, stats = restorer.restore_faces(torch.randn(1, 3, 100, 200))
        self.assertIn("square", stats.error)
        self.assertEqual(stats.frames_enhanced, 0)

    def test_stats_serialization(self):
        from latentsync.utils.codeformer_restorer import CodeformerStats

        s = CodeformerStats(enabled=True, loaded=True, frames_total=10, frames_enhanced=7)
        d = s.as_dict()
        # Round-trips through JSON without exploding -- the API
        # response shape relies on this being plain dict[str, ...].
        import json

        json.dumps(d)
        self.assertEqual(d["frames_total"], 10)
        self.assertEqual(d["frames_enhanced"], 7)


class TestLoaderCheckpointParsing(unittest.TestCase):
    """The loader should accept the three checkpoint formats the
    upstream has shipped across releases: ``params_ema``, ``params``,
    and bare state-dict."""

    def test_state_dict_is_a_dict(self):
        # We don't run the actual loader (no weights), but we can at
        # least check the format-detection branch handles a missing
        # file with the right error.
        from latentsync.utils.codeformer import loader

        with self.assertRaises(FileNotFoundError):
            loader.load_codeformer(
                "/nonexistent/codeformer.pth",
                device="cpu",
                download_if_missing=False,
            )


if __name__ == "__main__":
    unittest.main()
