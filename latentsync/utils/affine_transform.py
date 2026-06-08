# Adapted from https://github.com/guanjz20/StyleSync/blob/main/utils.py

import numpy as np
import cv2
import torch
from einops import rearrange
import kornia


class AlignRestore(object):
    def __init__(self, align_points=3, resolution=256, device="cpu", dtype=torch.float16):
        if align_points == 3:
            self.upscale_factor = 1
            ratio = resolution / 256 * 2.8
            self.crop_ratio = (ratio, ratio)
            self.face_template = np.array([[19 - 2, 30 - 10], [56 + 2, 30 - 10], [37.5, 45 - 5]])
            self.face_template = self.face_template * ratio
            self.face_size = (int(75 * self.crop_ratio[0]), int(100 * self.crop_ratio[1]))
            self.p_bias = None
            self.device = device
            self.dtype = dtype
            self.fill_value = torch.tensor([127, 127, 127], device=device, dtype=dtype)
            self.mask = torch.ones((1, 1, self.face_size[1], self.face_size[0]), device=device, dtype=dtype)

    def reset_p_bias(self) -> None:
        """Reset the cross-frame affine translation smoothing state.

        Call this at the start of any batch where the previous
        batch's end-of-stream bias would not be representative of
        the incoming frames (e.g. a scene cut, a different camera
        distance, or simply the start of a new video). Without a
        reset, ``p_bias`` EMA converges to a value that drifts from
        the per-frame true bias and corrupts ``M[:, 2]`` for the
        first few frames of the new batch.
        """
        self.p_bias = None

    def align_warp_face(self, img, landmarks3, smooth=True):
        affine_matrix, self.p_bias = self.transformation_from_points(
            landmarks3, self.face_template, smooth, self.p_bias
        )

        img = rearrange(torch.from_numpy(img).to(device=self.device, dtype=self.dtype), "h w c -> c h w").unsqueeze(0)
        affine_matrix = torch.from_numpy(affine_matrix).to(device=self.device, dtype=self.dtype).unsqueeze(0)

        cropped_face = kornia.geometry.transform.warp_affine(
            img,
            affine_matrix,
            (self.face_size[1], self.face_size[0]),
            mode="bilinear",
            padding_mode="fill",
            fill_value=self.fill_value,
        )
        cropped_face = rearrange(cropped_face.squeeze(0), "c h w -> h w c").cpu().numpy().astype(np.uint8)
        return cropped_face, affine_matrix

    def restore_img(self, input_img, face, affine_matrix, paste_mask_512=None):
        h, w, _ = input_img.shape

        if isinstance(affine_matrix, np.ndarray):
            affine_matrix = torch.from_numpy(affine_matrix).to(device=self.device, dtype=self.dtype).unsqueeze(0)

        inv_affine_matrix = kornia.geometry.transform.invert_affine_transform(affine_matrix)
        face = face.to(dtype=self.dtype).unsqueeze(0)

        inv_face = kornia.geometry.transform.warp_affine(
            face, inv_affine_matrix, (h, w), mode="bilinear", padding_mode="fill", fill_value=self.fill_value
        ).squeeze(0)
        inv_face = (inv_face / 2 + 0.5).clamp(0, 1) * 255

        input_img = rearrange(torch.from_numpy(input_img).to(device=self.device, dtype=self.dtype), "h w c -> c h w")

        if paste_mask_512 is not None:
            # Per-frame inpaint-mask path. The mask lives in 512x512
            # aligned-face space; warp it back into the original frame
            # coordinates so the paste region exactly covers the area
            # the inpainter touched (mouth + small feather band). A
            # small Gaussian feather softens the seam against the
            # surrounding unmodified source pixels -- much smaller than
            # the legacy cv2.erode(60x60) + gaussian_blur(91, sigma=37)
            # band, which produced a ~440x540 ellipse covering the
            # entire lower half of the face.
            inv_soft_mask = kornia.geometry.transform.warp_affine(
                paste_mask_512.to(device=self.device, dtype=self.dtype),
                inv_affine_matrix, (h, w), padding_mode="zeros",
            ).squeeze(0)  # (1, h, w)
            inv_soft_mask = kornia.filters.gaussian_blur2d(
                inv_soft_mask.unsqueeze(0), (21, 21), (4.0, 4.0)
            ).squeeze(0)
            inv_soft_mask_3d = inv_soft_mask.expand_as(inv_face)
            img_back = inv_soft_mask_3d * inv_face + (1 - inv_soft_mask_3d) * input_img
        else:
            # Legacy full-face paste path. Kept as a fallback for
            # callers that don't have a per-frame inpaint mask.
            inv_mask = kornia.geometry.transform.warp_affine(
                self.mask, inv_affine_matrix, (h, w), padding_mode="zeros"
            )  # (1, 1, h_up, w_up)

            inv_mask_erosion = kornia.morphology.erosion(
                inv_mask,
                torch.ones(
                    (int(2 * self.upscale_factor), int(2 * self.upscale_factor)), device=self.device, dtype=self.dtype
                ),
            )

            inv_mask_erosion_t = inv_mask_erosion.squeeze(0).expand_as(inv_face)
            pasted_face = inv_mask_erosion_t * inv_face
            total_face_area = torch.sum(inv_mask_erosion.float())
            w_edge = max(int(total_face_area**0.5) // 16, 6)
            erosion_radius = w_edge * 2

            # This step will consume a large amount of GPU memory.
            # inv_mask_center = kornia.morphology.erosion(
            #     inv_mask_erosion, torch.ones((erosion_radius, erosion_radius), device=self.device, dtype=self.dtype)
            # )

            # Run on CPU to avoid consuming a large amount of GPU memory.
            inv_mask_erosion = inv_mask_erosion.squeeze().cpu().numpy().astype(np.float32)
            inv_mask_center = cv2.erode(inv_mask_erosion, np.ones((erosion_radius, erosion_radius), np.uint8))
            inv_mask_center = torch.from_numpy(inv_mask_center).to(device=self.device, dtype=self.dtype)[None, None, ...]

            blur_size = w_edge * 3 // 2 * 2 + 1
            sigma = 0.4 * ((blur_size - 1) * 0.5 - 1) + 0.9
            inv_soft_mask = kornia.filters.gaussian_blur2d(
                inv_mask_center, (blur_size, blur_size), (sigma, sigma)
            ).squeeze(0)
            inv_soft_mask_3d = inv_soft_mask.expand_as(inv_face)
            img_back = inv_soft_mask_3d * pasted_face + (1 - inv_soft_mask_3d) * input_img

        img_back = rearrange(img_back, "c h w -> h w c").contiguous().to(dtype=torch.uint8)
        img_back = img_back.cpu().numpy()
        return img_back

    def transformation_from_points(self, points1: torch.Tensor, points0: torch.Tensor, smooth=True, p_bias=None):
        if isinstance(points0, np.ndarray):
            points2 = torch.tensor(points0, device=self.device, dtype=torch.float32)
        else:
            points2 = points0.clone()

        if isinstance(points1, np.ndarray):
            points1_tensor = torch.tensor(points1, device=self.device, dtype=torch.float32)
        else:
            points1_tensor = points1.clone()

        c1 = torch.mean(points1_tensor, dim=0)
        c2 = torch.mean(points2, dim=0)

        points1_centered = points1_tensor - c1
        points2_centered = points2 - c2

        s1 = torch.std(points1_centered)
        s2 = torch.std(points2_centered)

        points1_normalized = points1_centered / s1
        points2_normalized = points2_centered / s2

        covariance = torch.matmul(points1_normalized.T, points2_normalized)
        U, S, V = torch.svd(covariance.float())

        R = torch.matmul(V, U.T)

        det = torch.det(R.float())
        if det < 0:
            V[:, -1] = -V[:, -1]
            R = torch.matmul(V, U.T)

        sR = (s2 / s1) * R
        T = c2.reshape(2, 1) - (s2 / s1) * torch.matmul(R, c1.reshape(2, 1))

        M = torch.cat((sR, T), dim=1)

        if smooth:
            bias = points2_normalized[2] - points1_normalized[2]
            if p_bias is None:
                p_bias = bias
            else:
                bias = p_bias * 0.2 + bias * 0.8
            p_bias = bias
            M[:, 2] = M[:, 2] + bias

        return M.cpu().numpy(), p_bias
