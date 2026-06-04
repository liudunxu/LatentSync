# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import math
import os
import shutil
from typing import Callable, List, Optional, Union
import subprocess

import numpy as np
import torch
import torchvision
from torchvision import transforms

from packaging import version

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL
from diffusers.pipelines import DiffusionPipeline
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import deprecate

import logging

from einops import rearrange
import cv2

from ..models.unet import UNet3DConditionModel
from ..utils.util import read_video, read_audio, write_video, check_ffmpeg_installed
from ..utils.image_processor import ImageProcessor, load_fixed_mask
from ..whisper.audio2feature import Audio2Feature
import tqdm
import soundfile as sf

logger = logging.getLogger(__name__)


class LipsyncPipeline(DiffusionPipeline):
    _optional_components = []

    def __init__(
        self,
        vae: AutoencoderKL,
        audio_encoder: Audio2Feature,
        unet: UNet3DConditionModel,
        scheduler: Union[
            DDIMScheduler,
            PNDMScheduler,
            LMSDiscreteScheduler,
            EulerDiscreteScheduler,
            EulerAncestralDiscreteScheduler,
            DPMSolverMultistepScheduler,
        ],
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        is_unet_version_less_0_9_0 = hasattr(unet.config, "_diffusers_version") and version.parse(
            version.parse(unet.config._diffusers_version).base_version
        ) < version.parse("0.9.0.dev0")
        is_unet_sample_size_less_64 = hasattr(unet.config, "sample_size") and unet.config.sample_size < 64
        if is_unet_version_less_0_9_0 and is_unet_sample_size_less_64:
            deprecation_message = (
                "The configuration file of the unet has set the default `sample_size` to smaller than"
                " 64 which seems highly unlikely. If your checkpoint is a fine-tuned version of any of the"
                " following: \n- CompVis/stable-diffusion-v1-4 \n- CompVis/stable-diffusion-v1-3 \n-"
                " CompVis/stable-diffusion-v1-2 \n- CompVis/stable-diffusion-v1-1 \n- runwayml/stable-diffusion-v1-5"
                " \n- runwayml/stable-diffusion-inpainting \n you should change 'sample_size' to 64 in the"
                " configuration file. Please make sure to update the config accordingly as leaving `sample_size=32`"
                " in the config might lead to incorrect results in future versions. If you have downloaded this"
                " checkpoint from the Hugging Face Hub, it would be very nice if you could open a Pull request for"
                " the `unet/config.json` file"
            )
            deprecate("sample_size<64", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(unet.config)
            new_config["sample_size"] = 64
            unet._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        self.set_progress_bar_config(desc="Steps")

    def enable_vae_slicing(self):
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        self.vae.disable_slicing()

    @property
    def _execution_device(self):
        if self.device != torch.device("meta") or not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def decode_latents(self, latents):
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
        latents = rearrange(latents, "b c f h w -> (b f) c h w")
        decoded_latents = self.vae.decode(latents).sample
        return decoded_latents

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(self, height, width, callback_steps):
        assert height == width, "Height and width must be equal"

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

    def prepare_latents(self, num_frames, num_channels_latents, height, width, dtype, device, generator):
        shape = (
            1,
            num_channels_latents,
            1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )  # (b, c, f, h, w)
        rand_device = "cpu" if device.type == "mps" else device
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype).to(device)
        latents = latents.repeat(1, 1, num_frames, 1, 1)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def prepare_mask_latents(
        self, mask, masked_image, height, width, dtype, device, generator, do_classifier_free_guidance
    ):
        # resize the mask to latents shape as we concatenate the mask to the latents
        # we do that before converting to dtype to avoid breaking in case we're using cpu_offload
        # and half precision
        mask = torch.nn.functional.interpolate(
            mask, size=(height // self.vae_scale_factor, width // self.vae_scale_factor)
        )
        masked_image = masked_image.to(device=device, dtype=dtype)

        # encode the mask image into latents space so we can concatenate it to the latents
        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample(generator=generator)
        masked_image_latents = (masked_image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # aligning device to prevent device errors when concating it with the latent model input
        masked_image_latents = masked_image_latents.to(device=device, dtype=dtype)
        mask = mask.to(device=device, dtype=dtype)

        # assume batch size = 1
        mask = rearrange(mask, "f c h w -> 1 c f h w")
        masked_image_latents = rearrange(masked_image_latents, "f c h w -> 1 c f h w")

        mask = torch.cat([mask] * 2) if do_classifier_free_guidance else mask
        masked_image_latents = (
            torch.cat([masked_image_latents] * 2) if do_classifier_free_guidance else masked_image_latents
        )
        return mask, masked_image_latents

    def prepare_image_latents(self, images, device, dtype, generator, do_classifier_free_guidance):
        images = images.to(device=device, dtype=dtype)
        image_latents = self.vae.encode(images).latent_dist.sample(generator=generator)
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        image_latents = rearrange(image_latents, "f c h w -> 1 c f h w")
        image_latents = torch.cat([image_latents] * 2) if do_classifier_free_guidance else image_latents

        return image_latents

    def set_progress_bar_config(self, **kwargs):
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(kwargs)

    @staticmethod
    def paste_surrounding_pixels_back(decoded_latents, pixel_values, masks, device, weight_dtype):
        # Paste the surrounding pixels back, because we only want to change the mouth region
        pixel_values = pixel_values.to(device=device, dtype=weight_dtype)
        masks = masks.to(device=device, dtype=weight_dtype)
        combined_pixel_values = decoded_latents * masks + pixel_values * (1 - masks)
        return combined_pixel_values

    @staticmethod
    def pixel_values_to_images(pixel_values: torch.Tensor):
        pixel_values = rearrange(pixel_values, "f c h w -> f h w c")
        pixel_values = (pixel_values / 2 + 0.5).clamp(0, 1)
        images = (pixel_values * 255).to(torch.uint8)
        images = images.cpu().numpy()
        return images

    @staticmethod
    def _mouth_open_ratio(lmk: np.ndarray, face_size: float) -> float:
        if lmk is None or face_size <= 0:
            return 0.0
        inner_mouth = [52, 58, 67, 61]
        outer_mouth = [48, 54, 51, 57]
        try:
            inner = np.mean([lmk[i] for i in inner_mouth], axis=0)
            outer = np.mean([lmk[i] for i in outer_mouth], axis=0)
            return float(np.linalg.norm(inner - outer) / face_size)
        except (IndexError, TypeError):
            return 0.0

    @staticmethod
    def _estimate_yaw_degrees(lmk: np.ndarray) -> float:
        """Rough yaw estimation in degrees from 106-point landmarks.
        0 ≈ frontal; positive means the face is turned so the nose appears
        shifted to the subject's right in the image. Used as a coarse
        prefilter to skip lip-sync on heavily side-on faces where the
        affine alignment becomes unreliable.
        """
        if lmk is None or len(lmk) < 106:
            return 0.0
        try:
            pt_left_eye = np.mean(lmk[[43, 48, 49, 51, 50]], axis=0)
            pt_right_eye = np.mean(lmk[101:106], axis=0)
            pt_nose = np.mean(lmk[[74, 77, 83, 86]], axis=0)
        except (IndexError, TypeError):
            return 0.0
        inter_ocular = float(pt_right_eye[0] - pt_left_eye[0])
        if abs(inter_ocular) < 1e-3:
            return 0.0
        expected = inter_ocular / 2.0
        nose_to_left = float(abs(pt_nose[0] - pt_left_eye[0]))
        # Positive when the nose is left of the eye midpoint (subject's right turn)
        delta = (expected - nose_to_left) / expected
        # Empirically: delta=0.5 ≈ 30°, delta=1.0 ≈ 60°. Keep linear, conservative.
        return float(delta * 60.0)

    @staticmethod
    def _match_color_to_reference(
        face: torch.Tensor,
        ref_face: torch.Tensor,
        mask: torch.Tensor,
        strength: float = 0.6,
    ) -> torch.Tensor:
        """Per-frame mean+std color transfer from `face` to match `ref_face`,
        applied only inside the mask region of `face`.

        Why: the inpainter outputs the full 512x512 face, but only the masked
        region (lower face) is "new" -- the rest is supposed to be a no-op
        pass-through. In practice the diffusion inpainter slightly drifts
        the whole face's color statistics, and the soft-mask boundary in
        restore_img feathers that drift into the visible mouth. Aligning
        the generated face's per-channel mean/std to the original (where
        the original is sharp and unmasked) makes the seam invisible.

        Args:
            face: (3, H, W) generated face in [-1, 1] (after paste step).
            ref_face: (3, H, W) original face in [-1, 1] -- same shape, no paste.
            mask: (1, H, W) in [0, 1] (1 = generated region).
            strength: 0 = no transfer, 1 = full transfer.

        Returns:
            face with color statistics transferred; shape unchanged.
        """
        if strength <= 0 or face is None or ref_face is None or mask is None:
            return face
        if face.shape != ref_face.shape or face.shape[0] != 3:
            return face
        try:
            x = face.detach().to(torch.float32)
            r = ref_face.detach().to(torch.float32)
            m1 = mask.detach().to(torch.float32)
            # Normalize mask to (3, H, W) for elementwise mul, and a per-
            # channel weight tensor (1, H, W) for stats so the per-channel
            # mean/std isn't diluted by a 3x broadcast.
            if m1.dim() == 3:
                if m1.shape[0] == 3:
                    m3 = m1
                    m_w = m1[0:1]
                else:  # (1, H, W)
                    m3 = m1.expand(3, -1, -1)
                    m_w = m1
            else:  # (H, W)
                m3 = m1.unsqueeze(0).expand(3, -1, -1)
                m_w = m1.unsqueeze(0)
            weight_sum = m_w.sum(dim=(1, 2)).clamp_min(1.0).expand(3)
            src_pixels = x * m3
            tgt_pixels = r * m3
            src_mean = src_pixels.sum(dim=(1, 2)) / weight_sum
            tgt_mean = tgt_pixels.sum(dim=(1, 2)) / weight_sum
            src_var = ((x - src_mean.view(3, 1, 1)) ** 2 * m3).sum(dim=(1, 2)) / weight_sum
            tgt_var = ((r - tgt_mean.view(3, 1, 1)) ** 2 * m3).sum(dim=(1, 2)) / weight_sum
            src_std = src_var.clamp_min(1e-6).sqrt()
            tgt_std = tgt_var.clamp_min(1e-6).sqrt()
            scale = tgt_std / src_std
            shift = tgt_mean - src_mean * scale
            # Apply ONLY inside the mask -- never touch the unmasked region
            # (which is the original face and is already correct).
            adjusted = x * scale.view(3, 1, 1) + shift.view(3, 1, 1)
            mixed = x + m3 * strength * (adjusted - x)
            return mixed.to(face.dtype)
        except Exception:
            return face

    @staticmethod
    def _unsharp_mask(face: torch.Tensor, mask: torch.Tensor, amount: float = 0.35, radius: int = 3) -> torch.Tensor:
        """Light unsharp mask applied only inside `mask` (1 = sharpen, 0 = leave).

        Standard unsharp: original + amount * (original - blur(original)).
        amount=0 disables, 0.3-0.5 is typical, >0.7 looks "crunchy".

        We use a separable Gaussian via 2 convs (faster than full kernel)
        and cast to fp32 for the conv since the [-1, 1] deltas would
        underflow in fp16.
        """
        if amount <= 0 or face is None or mask is None:
            return face
        if face.shape[0] != 3:
            return face
        try:
            x = face.detach().to(torch.float32)
            m = mask.detach().to(torch.float32)
            if m.shape[0] == 1:
                m = m.expand(3, -1, -1)
            # Build a small Gaussian kernel -- radius 3 ≈ sigma 1.0
            k = 2 * radius + 1
            sigma = max(1.0, (k - 1) / 6.0)
            ax = torch.arange(k, dtype=torch.float32, device=x.device) - radius
            g1 = torch.exp(-(ax ** 2) / (2 * sigma * sigma))
            g1 = g1 / g1.sum()
            # Per-channel separable 1D convs (groups=3 so each input channel
            # gets its own filter, which is just the 1D Gaussian repeated).
            kx = g1.view(1, 1, 1, k).expand(3, 1, 1, k)
            ky = g1.view(1, 1, k, 1).expand(3, 1, k, 1)
            inp = x.unsqueeze(0)
            inp_pad = torch.nn.functional.pad(inp, (radius, radius, 0, 0), mode="reflect")
            inp_pad = torch.nn.functional.pad(inp_pad, (0, 0, radius, radius), mode="reflect")
            tmp = torch.nn.functional.conv2d(inp_pad, kx, groups=3)
            blurred = torch.nn.functional.conv2d(tmp, ky, groups=3)
            sharpened = x + amount * (x - blurred.squeeze(0))
            out = x + m * (sharpened - x)
            return out.to(face.dtype)
        except Exception:
            return face

    @staticmethod
    def _mouth_occlusion_score(face: torch.Tensor) -> float:
        """Heuristic mouth-visibility score in [0, 1].

        0.0 = clearly visible mouth (dark mouth interior present)
        1.0 = likely occluded (hand, mic, phone, mask covering the mouth)

        Aligned face is in [-1, 1] after `Normalize([0.5],[0.5])`. A real
        open mouth contains a dark interior (gray < ~-0.2); a hand or
        microphone covering the mouth is uniform skin/object color with
        no dark interior. We crop a fixed mouth ROI and count the
        fraction of dark pixels.

        We deliberately use a low-tech pixel ratio rather than a learned
        detector so this stays in the prefilter tier (cheap, no model
        load, runs on every face in the batch).
        """
        if face is None or face.numel() == 0:
            return 0.0
        try:
            x = face.detach().to(torch.float32)
            H, W = x.shape[1], x.shape[2]
            # Mouth ROI on the aligned 512x512 face. Affine aligner
            # centers eyes ~y=200, nose ~y=290, mouth ~y=370-440, chin
            # ~y=480. Rows [55%, 72%] / cols [32%, 68%] covers the lip
            # line plus mouth interior on both 256 and 512 alignments.
            y0, y1 = int(H * 0.55), int(H * 0.72)
            x0, x1 = int(W * 0.32), int(W * 0.68)
            roi = x.mean(dim=0)[y0:y1, x0:x1]
            if roi.numel() == 0:
                return 0.0
            dark_ratio = float((roi < -0.2).float().mean().item())
            # Linear map: dark_ratio >= 0.15 -> score 0 (visible)
            #             dark_ratio <= 0.00 -> score 1 (occluded)
            score = max(0.0, min(1.0, 1.0 - dark_ratio / 0.15))
            return score
        except Exception:
            return 0.0

    @staticmethod
    def _face_sharpness(face: torch.Tensor) -> float:
        """Laplacian variance as a sharpness proxy. face shape (3, H, W), any dtype/range.
        Returns 0.0 on failure.

        Implementation note: we always cast to float32 before computing. The previous
        version ran the conv in fp16 (the dtype of the paste step output) and the
        squared-laplacian underflowed to 0 for every face, which caused the postfilter
        to flag 100% of frames and fall back to the original video. fp32 is ~free at
        512x512 and gives stable variance values in the ~0.5–20 range.
        """
        if face is None or face.numel() == 0:
            return 0.0
        try:
            x = face.detach().to(torch.float32)
            kernel = torch.tensor(
                [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
                dtype=torch.float32,
                device=x.device,
            ).view(1, 1, 3, 3)
            gray = x.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, H, W)
            lap = torch.nn.functional.conv2d(gray, kernel, padding=1)
            return float(lap.pow(2).mean().item())
        except Exception:
            return 0.0

    @staticmethod
    def _smooth_face_sequence(
        face_crops: torch.Tensor,
        prev_face: Optional[torch.Tensor],
        prev_valid: bool,
        inference_skip_mask,
        weights=(0.25, 0.5, 0.25),
    ):
        """3-tap temporal EMA across face crops. Returns
        (smoothed, last_face, last_valid).

        - Skips any frame where inference_skip_mask[k] is True and resets the
          carry state so the next valid frame doesn't blend in a zero placeholder.
        - prev_face is only used when prev_valid is True.
        - Triangular kernel by default (weights = prev, cur, next) so the
          middle frame keeps 50% weight and neighbours each contribute 25%.
        """
        B = face_crops.shape[0]
        if B == 0:
            return face_crops, prev_face, prev_valid
        w_prev, w_cur, w_next = weights
        smoothed = face_crops.clone()
        last_valid = prev_valid
        last_face = prev_face

        for k in range(B):
            if inference_skip_mask[k]:
                # Zero-placeholder face from affine_transform_video: don't
                # pollute neighbours; reset carry.
                last_face = None
                last_valid = False
                continue

            if k == 0 and last_valid and last_face is not None:
                if B > 1:
                    smoothed[0] = w_prev * last_face + w_cur * face_crops[0] + w_next * face_crops[1]
                else:
                    smoothed[0] = 0.5 * last_face + 0.5 * face_crops[0]
            elif 0 < k < B - 1:
                smoothed[k] = w_prev * face_crops[k - 1] + w_cur * face_crops[k] + w_next * face_crops[k + 1]
            elif k == B - 1 and B > 1:
                # Last frame: blend with its predecessor (no next frame in batch)
                smoothed[k] = 0.5 * face_crops[k - 1] + 0.5 * face_crops[k]

            last_face = face_crops[k]
            last_valid = True

        return smoothed, last_face, last_valid

    def detect_main_speaker_embedding(self, video_frames: np.ndarray, face_embedder) -> Optional[np.ndarray]:
        if face_embedder is None or len(video_frames) == 0:
            return None
        best_frame_idx = None
        best_ratio = -1.0
        best_emb = None
        sample_indices = list(range(0, len(video_frames), max(1, len(video_frames) // 10)))[:20]
        for idx in sample_indices:
            frame = video_frames[idx]
            bbox, lmk = self.image_processor.face_detector(frame)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            face_size = max(x2 - x1, y2 - y1)
            mouth_ratio = self._mouth_open_ratio(lmk, face_size)
            if mouth_ratio > best_ratio:
                best_ratio = mouth_ratio
                best_frame_idx = idx
        if best_frame_idx is None:
            return None
        frame = video_frames[best_frame_idx]
        faces = face_embedder.get(frame.astype(np.uint8))
        if faces:
            emb = getattr(faces[0], "normed_embedding", None)
            if emb is not None:
                best_emb = np.asarray(emb, dtype=np.float32)
        logger.info(f"[LipSync] Main speaker from frame {best_frame_idx} with mouth_ratio={best_ratio:.4f}")
        return best_emb

    def affine_transform_video(
        self,
        video_frames: np.ndarray,
        reference_embedding=None,
        yaw_skip_threshold: float = 20.0,
        yaw_rate_skip_threshold: float = 8.0,
        mouth_occlusion_skip_threshold: float = 0.85,
        motion_blur_skip_threshold: float = 0.20,
        apply_identity_filter: bool = True,
        side_face_episode_pre_pad: int = 4,
        side_face_episode_post_pad: int = 4,
        yaw_warn_threshold_ratio: float = 0.5,
    ):
        logger.info(
            f"[FaceMatch] Starting: reference_embedding={'loaded' if reference_embedding is not None else 'None'}, "
            f"frames={len(video_frames)}, yaw_skip_threshold={yaw_skip_threshold}, "
            f"yaw_rate_skip_threshold={yaw_rate_skip_threshold}, "
            f"mouth_occlusion_skip_threshold={mouth_occlusion_skip_threshold}, "
            f"motion_blur_skip_threshold={motion_blur_skip_threshold}, "
            f"apply_identity_filter={apply_identity_filter}"
        )
        faces = []
        boxes = []
        affine_matrices = []
        skip_mask = []
        # Parallel arrays used by the episode-level side-face filter below:
        #   yaws[k] is the per-frame yaw in degrees (None for detect-fail frames)
        #   yaw_skip_reasons[k] is True iff THIS frame was skipped for yaw alone
        #     (not for identity / occlusion / blur / detect-fail -- those don't
        #     represent a "side face" and shouldn't trigger the episode pad).
        yaws: List[Optional[float]] = []
        yaw_skip_reasons: List[bool] = []
        yaw_skip_count = 0
        yaw_rate_skip_count = 0
        mouth_occlusion_skip_count = 0
        motion_blur_skip_count = 0
        identity_skip_count = 0
        detect_fail_count = 0
        prev_yaw: Optional[float] = None
        print(f"Affine transforming {len(video_frames)} faces...")
        for idx, frame in enumerate(tqdm.tqdm(video_frames)):
            face, box, affine_matrix, face_emb, lmk = self.image_processor.affine_transform_with_embedding(frame)
            if face is None:
                detect_fail_count += 1
                skip_mask.append(True)
                faces.append(torch.zeros(3, self.image_processor.resolution, self.image_processor.resolution))
                boxes.append([0, 0, 0, 0])
                affine_matrices.append(np.eye(3))
                yaws.append(None)
                yaw_skip_reasons.append(False)
                prev_yaw = None  # reset so we don't carry a stale yaw across a detect-fail gap
                continue
            should_skip = False
            if apply_identity_filter and reference_embedding is not None and face_emb is not None:
                similarity = float(np.dot(face_emb, reference_embedding))
                if similarity < 0.8:
                    should_skip = True
                    identity_skip_count += 1
            yaw_deg = 0.0
            yaw_was_skipped = False  # tracks the absolute yaw threshold (not yaw_rate)
            if yaw_skip_threshold > 0 and lmk is not None:
                yaw_deg = self._estimate_yaw_degrees(lmk)
                if abs(yaw_deg) > yaw_skip_threshold:
                    should_skip = True
                    yaw_was_skipped = True
                    yaw_skip_count += 1
            # Yaw-rate (deg/frame) catches the mid-turn frames where the face
            # hasn't crossed the absolute threshold yet but is rotating fast
            # enough that affine alignment is unreliable. Threshold is per
            # frame, not per second, so 8°/frame ≈ 200°/sec at 25fps.
            if (
                not should_skip
                and yaw_rate_skip_threshold > 0
                and prev_yaw is not None
                and yaw_skip_threshold > 0
                and lmk is not None
            ):
                rate = abs(yaw_deg - prev_yaw)
                if rate > yaw_rate_skip_threshold:
                    should_skip = True
                    yaw_rate_skip_count += 1
            # Mouth occlusion: skip frames where the mouth is covered by a
            # hand, microphone, phone, mask, etc. Cheap pixel-ratio check on
            # the aligned face -- no model load.
            if not should_skip and mouth_occlusion_skip_threshold > 0:
                occ = self._mouth_occlusion_score(face)
                if occ > mouth_occlusion_skip_threshold:
                    should_skip = True
                    mouth_occlusion_skip_count += 1
            # Motion blur: if the whole input face is already smeared (e.g.
            # fast head turn or camera shake), the inpainter cannot recover
            # a clean lip -- the output is at best as blurry as the input.
            # Reuse _face_sharpness on the aligned face; a sharp still face
            # scores ~5-20, a motion-blurred one scores well below 1.0.
            if not should_skip and motion_blur_skip_threshold > 0:
                face_sharp = self._face_sharpness(face)
                if face_sharp < motion_blur_skip_threshold:
                    should_skip = True
                    motion_blur_skip_count += 1
            skip_mask.append(should_skip)
            yaws.append(yaw_deg if (yaw_skip_threshold > 0 and lmk is not None) else None)
            yaw_skip_reasons.append(yaw_was_skipped)
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)
            prev_yaw = yaw_deg if (yaw_skip_threshold > 0 and lmk is not None) else None
        logger.info(
            f"[FaceMatch] detect_fail={detect_fail_count}, identity_skip={identity_skip_count}, "
            f"yaw_skip={yaw_skip_count}, yaw_rate_skip={yaw_rate_skip_count}, "
            f"mouth_occlusion_skip={mouth_occlusion_skip_count}, "
            f"motion_blur_skip={motion_blur_skip_count}"
        )
        self._last_yaw_skip_count = yaw_skip_count
        self._last_yaw_rate_skip_count = yaw_rate_skip_count
        self._last_mouth_occlusion_skip_count = mouth_occlusion_skip_count
        self._last_motion_blur_skip_count = motion_blur_skip_count

        # Episode-level side-face filter: a contiguous run of yaw-skipped
        # frames represents a single turning motion. The frames immediately
        # before/after the run typically have yaw in the warn band (e.g.
        # 10-20° for the default 20° threshold) where affine alignment is
        # still unreliable; we extend the skip_mask to include those
        # transition frames so the whole turn becomes a single side-face
        # episode (instead of a fragmentary skip that lets blur sneak in
        # at the boundaries).
        yaw_warn_threshold = yaw_skip_threshold * yaw_warn_threshold_ratio
        side_face_episode_extra_skip_count = 0
        if yaw_warn_threshold > 0 and (
            side_face_episode_pre_pad > 0 or side_face_episode_post_pad > 0
        ):
            n = len(skip_mask)
            i = 0
            while i < n:
                if not yaw_skip_reasons[i]:
                    i += 1
                    continue
                # find run end (contiguous yaw-skipped frames)
                j = i
                while j < n and yaw_skip_reasons[j]:
                    j += 1
                # expand left into pre_pad window
                for k in range(max(0, i - side_face_episode_pre_pad), i):
                    if (
                        not skip_mask[k]
                        and yaws[k] is not None
                        and abs(yaws[k]) > yaw_warn_threshold
                    ):
                        skip_mask[k] = True
                        side_face_episode_extra_skip_count += 1
                # expand right into post_pad window
                for k in range(j, min(n, j + side_face_episode_post_pad)):
                    if (
                        not skip_mask[k]
                        and yaws[k] is not None
                        and abs(yaws[k]) > yaw_warn_threshold
                    ):
                        skip_mask[k] = True
                        side_face_episode_extra_skip_count += 1
                i = j
        self._last_side_face_episode_extra_skip_count = side_face_episode_extra_skip_count
        if side_face_episode_extra_skip_count:
            logger.info(
                f"[FaceMatch] side_face_episode_extra_skip={side_face_episode_extra_skip_count} "
                f"(pre_pad={side_face_episode_pre_pad}, post_pad={side_face_episode_post_pad}, "
                f"warn_threshold={yaw_warn_threshold:.1f}°)"
            )

        faces_tensor = torch.stack(faces)
        return faces_tensor, boxes, affine_matrices, skip_mask

    def restore_video(self, faces: torch.Tensor, video_frames: np.ndarray, boxes: list, affine_matrices: list, skip_mask=None):
        video_frames = video_frames[: len(faces)]
        out_frames = []
        print(f"Restoring {len(faces)} faces...")
        for index, face in enumerate(tqdm.tqdm(faces)):
            x1, y1, x2, y2 = boxes[index]
            height = int(y2 - y1)
            width = int(x2 - x1)
            should_skip = skip_mask[index] if skip_mask and index < len(skip_mask) else False
            if should_skip or height <= 0 or width <= 0:
                out_frames.append(video_frames[index])
            else:
                face_resized = torchvision.transforms.functional.resize(
                    face, size=(height, width), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
                )
                out_frame = self.image_processor.restorer.restore_img(video_frames[index], face_resized, affine_matrices[index])
                out_frames.append(out_frame)
        return np.stack(out_frames, axis=0)

    def loop_video(
        self,
        whisper_chunks: list,
        video_frames: np.ndarray,
        reference_embedding=None,
        face_embedder=None,
        skip_mask=None,
        yaw_skip_threshold: float = 20.0,
        yaw_rate_skip_threshold: float = 8.0,
        mouth_occlusion_skip_threshold: float = 0.85,
        motion_blur_skip_threshold: float = 0.20,
        apply_identity_filter: bool = True,
    ):
        logger.info(
            f"[LipSync] loop_video: reference_embedding={'loaded' if reference_embedding is not None else 'None'}, "
            f"frames={len(video_frames)}, yaw_skip_threshold={yaw_skip_threshold}, "
            f"yaw_rate_skip_threshold={yaw_rate_skip_threshold}, "
            f"mouth_occlusion_skip_threshold={mouth_occlusion_skip_threshold}, "
            f"motion_blur_skip_threshold={motion_blur_skip_threshold}, apply_identity_filter={apply_identity_filter}"
        )
        if reference_embedding is None and face_embedder is not None:
            reference_embedding = self.detect_main_speaker_embedding(video_frames, face_embedder)
            logger.info(f"[LipSync] Auto-detected main speaker embedding: {'loaded' if reference_embedding is not None else 'None'}")
        if len(whisper_chunks) > len(video_frames):
            faces, boxes, affine_matrices, frame_skip_mask = self.affine_transform_video(
                video_frames,
                reference_embedding,
                yaw_skip_threshold=yaw_skip_threshold,
                yaw_rate_skip_threshold=yaw_rate_skip_threshold,
                mouth_occlusion_skip_threshold=mouth_occlusion_skip_threshold,
                motion_blur_skip_threshold=motion_blur_skip_threshold,
                apply_identity_filter=apply_identity_filter,
            )
            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            loop_video_frames = []
            loop_faces = []
            loop_boxes = []
            loop_affine_matrices = []
            loop_skip_mask = []
            for i in range(num_loops):
                if i % 2 == 0:
                    loop_video_frames.append(video_frames)
                    loop_faces.append(faces)
                    loop_boxes += boxes
                    loop_affine_matrices += affine_matrices
                    loop_skip_mask += frame_skip_mask
                else:
                    loop_video_frames.append(video_frames[::-1])
                    loop_faces.append(faces.flip(0))
                    loop_boxes += boxes[::-1]
                    loop_affine_matrices += affine_matrices[::-1]
                    loop_skip_mask += frame_skip_mask[::-1]

            video_frames = np.concatenate(loop_video_frames, axis=0)[: len(whisper_chunks)]
            faces = torch.cat(loop_faces, dim=0)[: len(whisper_chunks)]
            boxes = loop_boxes[: len(whisper_chunks)]
            affine_matrices = loop_affine_matrices[: len(whisper_chunks)]
            skip_mask = loop_skip_mask[: len(whisper_chunks)]
        else:
            video_frames = video_frames[: len(whisper_chunks)]
            faces, boxes, affine_matrices, frame_skip_mask = self.affine_transform_video(
                video_frames,
                reference_embedding,
                yaw_skip_threshold=yaw_skip_threshold,
                yaw_rate_skip_threshold=yaw_rate_skip_threshold,
                mouth_occlusion_skip_threshold=mouth_occlusion_skip_threshold,
                motion_blur_skip_threshold=motion_blur_skip_threshold,
                apply_identity_filter=apply_identity_filter,
            )
            skip_mask = frame_skip_mask

        return video_frames, faces, boxes, affine_matrices, skip_mask

    @torch.no_grad()
    def __call__(
        self,
        video_path: str,
        audio_path: str,
        video_out_path: str,
        num_frames: int = 16,
        video_fps: int = 25,
        audio_sample_rate: int = 16000,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 40,
        guidance_scale: float = 1.5,
        weight_dtype: Optional[torch.dtype] = torch.float16,
        eta: float = 0.0,
        mask_image_path: str = "latentsync/utils/mask.png",
        temp_dir: str = "temp",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        reference_embedding=None,
        face_embedder=None,
        # --- quality / temporal gating (added 2026-06) ---
        temporal_smoothing_enabled: bool = True,
        # Postfilter: skip frames where the generated face is unsharp
        # (Laplacian variance below threshold) or much softer than the
        # original. Catches the "blurry block" failure mode where the
        # inpainter produces a smeared mouth. See _face_sharpness for
        # why we cast to fp32 (avoids the underflow that used to make
        # this misfire on every frame).
        quality_gate_enabled: bool = False,
        # Default 0.5 because aligned 512x512 faces in [-1, 1] typically
        # score 0.5..20.0 on a real mouth; a normal mouth is well above
        # this; a smeared/blurred mouth sits below. The previous 15.0
        # default was tuned for a different scale and skipped everything.
        quality_min_laplacian: float = 0.5,
        quality_min_sharpness_ratio: float = 0.20,
        # Yaw-based prefilters for side faces / fast head turns. 20°/8°/frame
        # is the empirical sweet spot: 30° lets too many "mostly-side" frames
        # through (they render as visible blur blocks around the mouth);
        # turning >8°/frame also produces blur because the affine alignment
        # can't keep up. Both fall back to the original frame in restore_video.
        yaw_skip_threshold: float = 20.0,
        yaw_rate_skip_threshold: float = 8.0,
        # Episode-level side-face filter: when contiguous frames exceed
        # yaw_skip_threshold, also skip pre_pad/post_pad transition frames
        # around the episode (whose yaw is in the warn band between
        # yaw_skip_threshold * yaw_warn_threshold_ratio and yaw_skip_threshold).
        # Set pre_pad/post_pad to 0 to disable the padding.
        side_face_episode_pre_pad: int = 4,
        side_face_episode_post_pad: int = 4,
        yaw_warn_threshold_ratio: float = 0.5,
        # Mouth-occlusion prefilter: skip frames where the mouth is covered
        # by a hand, microphone, phone, mask, etc. Score 0..1; above the
        # threshold (default 0.85 -- was 0.7, too sensitive on multi-speaker
        # / side-face clips) the frame is treated as not-lip-syncable and
        # the original frame is used. Set to 1.0 to disable.
        mouth_occlusion_skip_threshold: float = 0.85,
        # Motion-blur input filter: skip frames whose aligned face is too
        # smeared to inpaint cleanly. Default 0.20 (Laplacian variance in
        # the [-1, 1] face space; a sharp face scores ~5-20, a motion-blurred
        # one <1.0). Set to 0 to disable.
        motion_blur_skip_threshold: float = 0.20,
        # Per-frame color transfer from generated to original (inside the
        # mask). 0 = off, 1 = full mean+std match. Default 0.60.
        color_match_strength: float = 0.60,
        # Unsharp-mask amount applied to the generated mouth region.
        # 0 = off, 1 = strong sharpen. Default 0.35.
        mouth_sharpen_strength: float = 0.35,
        **kwargs,
    ):
        is_train = self.unet.training
        self.unet.eval()

        check_ffmpeg_installed()

        # 0. Define call parameters
        device = self._execution_device
        mask_image = load_fixed_mask(height, mask_image_path)
        self.image_processor = ImageProcessor(height, device="cuda", mask_image=mask_image)
        if face_embedder is not None:
            self.image_processor.set_face_embedder(face_embedder)
            logger.info(f"[LipSync] Set face_embedder on ImageProcessor for face matching")
        self.set_progress_bar_config(desc=f"Sample frames: {num_frames}")

        # 1. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 2. Check inputs
        self.check_inputs(height, width, callback_steps)

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 4. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        whisper_feature = self.audio_encoder.audio2feat(audio_path)
        whisper_chunks = self.audio_encoder.feature2chunks(feature_array=whisper_feature, fps=video_fps)
        logger.info(f"[LipSync] audio: whisper_chunks={len(whisper_chunks)}, video_fps={video_fps}")

        audio_samples = read_audio(audio_path)
        video_frames = read_video(video_path, use_decord=False)
        logger.info(f"[LipSync] video_frames shape={video_frames.shape}")

        # Only apply identity filtering when the user explicitly provided an
        # avatar (reference_embedding). When the user didn't supply one,
        # loop_video will auto-detect a "main speaker" — that detection is
        # not reliable enough to reject other frames (false positives on
        # busy/occluded faces), so we keep all detected faces.
        apply_identity_filter = reference_embedding is not None
        video_frames, faces, boxes, affine_matrices, skip_mask = self.loop_video(
            whisper_chunks,
            video_frames,
            reference_embedding=reference_embedding,
            face_embedder=face_embedder,
            yaw_skip_threshold=yaw_skip_threshold,
            yaw_rate_skip_threshold=yaw_rate_skip_threshold,
            mouth_occlusion_skip_threshold=mouth_occlusion_skip_threshold,
            motion_blur_skip_threshold=motion_blur_skip_threshold,
            apply_identity_filter=apply_identity_filter,
        )
        logger.info(f"[LipSync] after loop_video: faces={faces.shape}, boxes={len(boxes)}, affine_matrices={len(affine_matrices)}, apply_identity_filter={apply_identity_filter}, skip_true={sum(skip_mask)}/{len(skip_mask)}")

        # State carried across batches for temporal EMA smoothing
        prev_face: Optional[torch.Tensor] = None
        prev_valid: bool = False
        quality_fallback_count: int = 0
        quality_skip_mask: List[bool] = [False] * len(skip_mask)

        synced_video_frames = []

        num_channels_latents = self.vae.config.latent_channels

        # Prepare latent variables
        all_latents = self.prepare_latents(
            len(whisper_chunks),
            num_channels_latents,
            height,
            width,
            weight_dtype,
            device,
            generator,
        )

        num_inferences = math.ceil(len(whisper_chunks) / num_frames)
        logger.info(f"[LipSync] num_inferences={num_inferences}, num_frames={num_frames}, add_audio_layer={self.unet.add_audio_layer}")
        for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
            if self.unet.add_audio_layer:
                audio_embeds = torch.stack(whisper_chunks[i * num_frames : (i + 1) * num_frames])
                audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
            else:
                audio_embeds = None
            inference_faces = faces[i * num_frames : (i + 1) * num_frames]
            latents = all_latents[:, :, i * num_frames : (i + 1) * num_frames]
            ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                inference_faces, affine_transform=False
            )

            # 7. Prepare mask latent variables
            mask_latents, masked_image_latents = self.prepare_mask_latents(
                masks,
                masked_pixel_values,
                height,
                width,
                weight_dtype,
                device,
                generator,
                do_classifier_free_guidance,
            )

            # 8. Prepare image latents
            ref_latents = self.prepare_image_latents(
                ref_pixel_values,
                device,
                weight_dtype,
                generator,
                do_classifier_free_guidance,
            )

            # 9. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for j, t in enumerate(timesteps):
                    # expand the latents if we are doing classifier free guidance
                    unet_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                    unet_input = self.scheduler.scale_model_input(unet_input, t)

                    # concat latents, mask, masked_image_latents in the channel dimension
                    unet_input = torch.cat([unet_input, mask_latents, masked_image_latents, ref_latents], dim=1)

                    # predict the noise residual
                    noise_pred = self.unet(unet_input, t, encoder_hidden_states=audio_embeds).sample
                    if j == 0:
                        logger.info(f"[LipSync] inference {i}: audio_embeds shape={audio_embeds.shape if audio_embeds is not None else None}, unet_input shape={unet_input.shape}, noise_pred shape={noise_pred.shape}")

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_audio = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_audio - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # call the callback, if provided
                    if j == len(timesteps) - 1 or ((j + 1) > num_warmup_steps and (j + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and j % callback_steps == 0:
                            callback(j, t, latents)

            # Recover the pixel values
            decoded_latents = self.decode_latents(latents)
            # Diagnostic: show the mask convention and how much the model is
            # actually deviating from the input on the first batch.
            if i == 0:
                with torch.no_grad():
                    mask_first = masks[0, 0]  # (H, W) in [0, 1]
                    logger.info(
                        f"[Diag] batch0 mask: min={mask_first.min().item():.3f} "
                        f"max={mask_first.max().item():.3f} mean={mask_first.mean().item():.3f} "
                        f"frac_inpaint={float((mask_first < 0.5).float().mean().item()):.3f}"
                    )
                    # Use .cpu() to avoid cuda/cpu device mismatch: decoded_latents
                    # lives on the pipeline device but ref_pixel_values comes from
                    # image_processor on cpu.
                    decoded_first = decoded_latents[0].detach().cpu()
                    ref_first = ref_pixel_values[0].detach().cpu()
                    diff = (decoded_first - ref_first).abs().mean().item()
                    logger.info(
                        f"[Diag] batch0 frame0: decoded range [{decoded_first.min().item():.2f}, {decoded_first.max().item():.2f}] "
                        f"ref range [{ref_first.min().item():.2f}, {ref_first.max().item():.2f}] "
                        f"mean|decoded-ref|={diff:.4f}"
                    )
            decoded_latents = self.paste_surrounding_pixels_back(
                decoded_latents, ref_pixel_values, 1 - masks, device, weight_dtype
            )
            # Per-frame color match: align generated face stats to original
            # so the soft-mask boundary in restore_img doesn't reveal a
            # tone drift. Applied inside the mask region only.
            if color_match_strength > 0:
                decoded_latents = self._match_color_to_reference(
                    decoded_latents, ref_pixel_values, masks, strength=color_match_strength
                )
            # Mouth-region unsharp: recover high-frequency detail in the
            # generated mouth. Inpainter outputs tend to be slightly soft
            # because the prompt encourages plausible-but-not-sharp.
            if mouth_sharpen_strength > 0:
                decoded_latents = self._unsharp_mask(
                    decoded_latents, masks, amount=mouth_sharpen_strength
                )
            if i == 0:
                with torch.no_grad():
                    combined_first = decoded_latents[0].detach().cpu()
                    ref_first = ref_pixel_values[0].detach().cpu()
                    diff = (combined_first - ref_first).abs().mean().item()
                    logger.info(
                        f"[Diag] batch0 frame0 after paste: mean|combined-ref|={diff:.4f} "
                        f"(~0 means model output was overwritten by ref)"
                    )

            # Temporal EMA across face crops (cross-batch state via prev_face)
            inference_skip_mask = skip_mask[i * num_frames : (i + 1) * num_frames]
            if temporal_smoothing_enabled:
                decoded_latents, prev_face, prev_valid = self._smooth_face_sequence(
                    decoded_latents,
                    prev_face=prev_face,
                    prev_valid=prev_valid,
                    inference_skip_mask=inference_skip_mask,
                )

            # Quality postfilter: flag frames whose generated face is too blurry
            # to be worth showing. Checked AFTER paste so the value domain is [-1, 1].
            if quality_gate_enabled:
                B = decoded_latents.shape[0]
                base = i * num_frames
                gen_laps = []
                ref_laps = []
                for k in range(B):
                    if inference_skip_mask[k]:
                        continue  # already going to fall back to original
                    gen_lap = self._face_sharpness(decoded_latents[k])
                    ref_lap = self._face_sharpness(ref_pixel_values[k])
                    gen_laps.append(gen_lap)
                    ref_laps.append(ref_lap)
                    if gen_lap < quality_min_laplacian:
                        quality_skip_mask[base + k] = True
                        quality_fallback_count += 1
                        logger.info(
                            f"[Diag] postfilter fallback batch{i} k{k}: gen_lap={gen_lap:.2f} < {quality_min_laplacian}"
                        )
                        continue
                    if ref_lap > 0 and (gen_lap / ref_lap) < quality_min_sharpness_ratio:
                        quality_skip_mask[base + k] = True
                        quality_fallback_count += 1
                        logger.info(
                            f"[Diag] postfilter fallback batch{i} k{k}: gen_lap={gen_lap:.2f} / ref_lap={ref_lap:.2f} = {gen_lap/ref_lap:.3f} < {quality_min_sharpness_ratio}"
                        )
                if i == 0 and gen_laps:
                    import statistics
                    logger.info(
                        f"[Diag] batch0 laplacian: gen min={min(gen_laps):.2f} max={max(gen_laps):.2f} median={statistics.median(gen_laps):.2f} "
                        f"ref min={min(ref_laps):.2f} max={max(ref_laps):.2f} median={statistics.median(ref_laps):.2f}"
                    )

            synced_video_frames.append(decoded_latents)

        logger.info(f"[LipSync] decoded {len(synced_video_frames)} batches, restoring video...")
        # OR-merge the quality postfilter with the original skip_mask
        effective_skip_mask = [a or b for a, b in zip(skip_mask, quality_skip_mask)]
        pre_skip = sum(skip_mask)
        quality_skip = sum(quality_skip_mask)
        effective_skip = sum(effective_skip_mask)
        logger.info(
            f"[Diag] skip summary: pre(loop_video)={pre_skip} quality_postfilter={quality_skip} "
            f"effective_total={effective_skip} / {len(skip_mask)}"
        )
        if quality_fallback_count:
            logger.info(f"[LipSync] quality_fallback_frames={quality_fallback_count} / {len(skip_mask)}")
        synced_video_frames = self.restore_video(torch.cat(synced_video_frames), video_frames, boxes, affine_matrices, effective_skip_mask)
        logger.info(f"[LipSync] restored video frames shape={synced_video_frames.shape}")

        audio_samples_remain_length = int(synced_video_frames.shape[0] / video_fps * audio_sample_rate)
        audio_samples = audio_samples[:audio_samples_remain_length].cpu().numpy()

        if is_train:
            self.unet.train()

        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

        write_video(os.path.join(temp_dir, "video.mp4"), synced_video_frames, fps=video_fps)

        # Stash stats for the API layer to read (synthesize() consumes this).
        self._last_run_stats = {
            "quality_fallback_frames": quality_fallback_count,
            "yaw_skip_count": getattr(self, "_last_yaw_skip_count", 0),
            "yaw_rate_skip_count": getattr(self, "_last_yaw_rate_skip_count", 0),
            "mouth_occlusion_skip_count": getattr(self, "_last_mouth_occlusion_skip_count", 0),
            "motion_blur_skip_count": getattr(self, "_last_motion_blur_skip_count", 0),
            "side_face_episode_extra_skip_count": getattr(
                self, "_last_side_face_episode_extra_skip_count", 0
            ),
            "yaw_skip_threshold": yaw_skip_threshold,
            "yaw_rate_skip_threshold": yaw_rate_skip_threshold,
            "mouth_occlusion_skip_threshold": mouth_occlusion_skip_threshold,
            "motion_blur_skip_threshold": motion_blur_skip_threshold,
            "temporal_smoothing_enabled": temporal_smoothing_enabled,
            "quality_gate_enabled": quality_gate_enabled,
        }

        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -crf 18 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)
