"""Unit tests for ImageProcessor batch preparation."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch


class TestPrepareMasksAndMaskedImages(unittest.TestCase):
    def _make_processor(self):
        try:
            from latentsync.utils.image_processor import ImageProcessor
        except ModuleNotFoundError as exc:
            self.skipTest(f"optional dependency missing: {exc.name}")
        return ImageProcessor(resolution=512, device="cpu")

    def _prepare_loop_reference(self, processor, images, per_frame_masks=None):
        """Reference implementation using the old per-frame loop."""
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        results = []
        for i, image in enumerate(images):
            mask_override = per_frame_masks[i] if per_frame_masks is not None else None
            results.append(processor.preprocess_fixed_mask_image(image, affine_transform=False, mask_override=mask_override))
        pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
        return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), torch.stack(masks_list)

    def test_batch_matches_loop_reference(self):
        processor = self._make_processor()
        images = torch.randint(0, 255, (4, 3, 512, 512), dtype=torch.uint8)
        pv, mpv, masks = processor.prepare_masks_and_masked_images(images, affine_transform=False)
        pv_ref, mpv_ref, masks_ref = self._prepare_loop_reference(processor, images)
        self.assertTrue(torch.allclose(pv, pv_ref, atol=1e-5))
        self.assertTrue(torch.allclose(mpv, mpv_ref, atol=1e-5))
        self.assertTrue(torch.allclose(masks, masks_ref, atol=1e-5))

    def test_numpy_hwc_input(self):
        processor = self._make_processor()
        images = np.random.randint(0, 255, (2, 512, 512, 3), dtype=np.uint8)
        pv, mpv, masks = processor.prepare_masks_and_masked_images(images, affine_transform=False)
        self.assertEqual(pv.shape, (2, 3, 256, 256))
        self.assertEqual(masks.shape, (2, 1, 256, 256))

    def test_per_frame_masks(self):
        processor = self._make_processor()
        images = torch.randint(0, 255, (3, 3, 512, 512), dtype=torch.uint8)
        per_frame_masks = torch.ones((3, 256, 256), dtype=torch.float32) * 0.5
        pv, mpv, masks = processor.prepare_masks_and_masked_images(
            images, affine_transform=False, per_frame_masks=per_frame_masks
        )
        self.assertEqual(masks.shape, (3, 1, 256, 256))
        self.assertTrue(torch.allclose(masks.squeeze(1), per_frame_masks, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
