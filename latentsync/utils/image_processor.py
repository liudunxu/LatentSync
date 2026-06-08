# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from latentsync.utils.util import read_video, write_video
from torchvision import transforms
import cv2
from einops import rearrange
import torch
import numpy as np
from typing import Union
import logging

logger = logging.getLogger(__name__)
from .affine_transform import AlignRestore
from .face_detector import FaceDetector


def load_fixed_mask(resolution: int, mask_image_path="latentsync/utils/mask.png") -> torch.Tensor:
    mask_image = cv2.imread(mask_image_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.resize(mask_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4) / 255.0
    mask_image = rearrange(torch.from_numpy(mask_image), "h w c -> c h w")
    return mask_image


class ImageProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu", mask_image=None):
        self.resolution = resolution
        self.resize = transforms.Resize(
            (resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        )
        self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)

        self.restorer = AlignRestore(resolution=resolution, device=device)

        if mask_image is None:
            self.mask_image = load_fixed_mask(resolution)
        else:
            self.mask_image = mask_image

        if device == "cpu":
            self.face_detector = None
            self.face_embedder = None
        else:
            # Let the lipsync pipeline decide whether a detected face is too
            # side-on to synthesize. If the detector rejects it here, the
            # pipeline can only count it as a generic detect failure.
            self.face_detector = FaceDetector(device=device, skip_side_face_threshold=None)
            self.face_embedder = None

    def set_face_embedder(self, embedder):
        self.face_embedder = embedder

    @staticmethod
    def _bbox_iou(left, right) -> float:
        lx1, ly1, lx2, ly2 = [float(v) for v in left]
        rx1, ry1, rx2, ry2 = [float(v) for v in right]
        ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
        ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
        right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
        union = left_area + right_area - inter
        return float(inter / union) if union > 0 else 0.0

    def affine_transform(self, image: torch.Tensor) -> np.ndarray:
        if self.face_detector is None:
            raise NotImplementedError("Using the CPU for face detection is not supported")
        bbox, landmark_2d_106 = self.face_detector(image)
        if bbox is None:
            raise RuntimeError("Face not detected")

        pt_left_eye = np.mean(landmark_2d_106[[43, 48, 49, 51, 50]], axis=0)  # left eyebrow center
        pt_right_eye = np.mean(landmark_2d_106[101:106], axis=0)  # right eyebrow center
        pt_nose = np.mean(landmark_2d_106[[74, 77, 83, 86]], axis=0)  # nose center

        # Pass through the 3 align points as float -- the affine
        # transform in AlignRestore.transformation_from_points is
        # already float, so rounding upstream only adds alignment
        # jitter on small / distant faces.
        landmarks3 = np.array([pt_left_eye, pt_right_eye, pt_nose])

        face, affine_matrix = self.restorer.align_warp_face(image.copy(), landmarks3=landmarks3, smooth=True)
        box = [0, 0, face.shape[1], face.shape[0]]  # x1, y1, x2, y2
        face = cv2.resize(face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
        face = rearrange(torch.from_numpy(face), "h w c -> c h w")
        return face, box, affine_matrix

    def affine_transform_with_embedding(self, image: torch.Tensor):
        if self.face_detector is None:
            raise NotImplementedError("Using the CPU for face detection is not supported")
        source_bbox, landmark_2d_106 = self.face_detector(image)
        if source_bbox is None:
            return None, None, None, None

        pt_left_eye = np.mean(landmark_2d_106[[43, 48, 49, 51, 50]], axis=0)
        pt_right_eye = np.mean(landmark_2d_106[101:106], axis=0)
        pt_nose = np.mean(landmark_2d_106[[74, 77, 83, 86]], axis=0)

        # Keep align points as float so AlignRestore can use them at
        # sub-pixel precision (see affine_transform for the rationale).
        landmarks3 = np.array([pt_left_eye, pt_right_eye, pt_nose])

        face, affine_matrix = self.restorer.align_warp_face(image.copy(), landmarks3=landmarks3, smooth=True)
        box = [0, 0, face.shape[1], face.shape[0]]
        face = cv2.resize(face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
        face = rearrange(torch.from_numpy(face), "h w c -> c h w")

        embedding = None
        if self.face_embedder is not None:
            detected_faces = self.face_embedder.get(image.astype(np.uint8) if image.dtype != np.uint8 else image)
            best_face = None
            best_iou = -1.0
            for detected_face in detected_faces:
                detected_bbox = getattr(detected_face, "bbox", None)
                if detected_bbox is None:
                    continue
                iou = self._bbox_iou(source_bbox, detected_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_face = detected_face
            if best_face is None and detected_faces:
                best_face = detected_faces[0]
            if best_face is not None:
                emb = getattr(best_face, "normed_embedding", None)
                if emb is not None:
                    embedding = np.asarray(emb, dtype=np.float32)

        return face, box, affine_matrix, embedding, landmark_2d_106

    def preprocess_fixed_mask_image(self, image: torch.Tensor, affine_transform=False, mask_override=None):
        if affine_transform:
            image, _, _ = self.affine_transform(image)
        else:
            image = self.resize(image)
        pixel_values = self.normalize(image / 255.0)
        if mask_override is not None:
            mask_to_use = mask_override  # (1, H, W), 1=keep, 0=inpaint
        else:
            mask_to_use = self.mask_image[0:1]
        masked_pixel_values = pixel_values * mask_to_use
        return pixel_values, masked_pixel_values, mask_to_use

    def prepare_masks_and_masked_images(self, images: Union[torch.Tensor, np.ndarray], affine_transform=False, per_frame_masks=None):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")
        logger.info(f"[ImageProcessor] prepare_masks_and_masked_images: input shape={images.shape}, affine_transform={affine_transform}, per_frame_masks={'provided' if per_frame_masks is not None else 'none'}")

        results = []
        for i, image in enumerate(images):
            mask_override = per_frame_masks[i] if per_frame_masks is not None else None
            results.append(self.preprocess_fixed_mask_image(image, affine_transform=affine_transform, mask_override=mask_override))

        pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
        return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), torch.stack(masks_list)

    def process_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")
        images = self.resize(images)
        pixel_values = self.normalize(images / 255.0)
        return pixel_values


class VideoProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu"):
        self.image_processor = ImageProcessor(resolution, device)

    def affine_transform_video(self, video_path):
        video_frames = read_video(video_path, change_fps=False)
        results = []
        for frame in video_frames:
            frame, _, _ = self.image_processor.affine_transform(frame)
            results.append(frame)
        results = torch.stack(results)

        results = rearrange(results, "f c h w -> f h w c").numpy()
        return results


if __name__ == "__main__":
    video_processor = VideoProcessor(256, "cuda")
    video_frames = video_processor.affine_transform_video("assets/demo2_video.mp4")
    write_video("output.mp4", video_frames, fps=25)
