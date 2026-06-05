# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/pipelines/pipeline_animation.py

import inspect
import math
import os
import shutil
import statistics
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
from typing import Dict, List, Optional, Tuple

MOUTH_OUTER_LANDMARKS = [48, 54, 51, 57]

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
        """Yaw estimation (degrees) from 106-point landmarks, conservative multi-signal.

        Three signals, all noise-gated so landmark jitter on a frontal face
        can't trip a skip:
        1. Nose offset (signed, original *60 mapping) - the low-noise baseline.
        2. Eye-width asymmetry (unsigned) - fires only when ratio > 1.5;
           ratio 1.5 -> 0 deg, 2.0 -> 15, 2.5 -> 30, 3.0 -> 45.
        3. Mouth-corner asymmetry (unsigned) - fires only when diff > 0.2;
           0.2 -> 0 deg, 0.3 -> 10, 0.4 -> 20, 0.5 -> 30.

        Returns the max absolute yaw across signals; sign from the nose
        offset. Compared to the earlier v2 (zero-floor eye/mouth + *75 nose)
        this is far less aggressive: each signal has to clear its noise
        floor before it can contribute, so noisy frontal faces stay near 0.
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

        # Signal 1: nose offset (signed; positive = subject's right turn)
        nose_to_left = float(abs(pt_nose[0] - pt_left_eye[0]))
        delta = (expected - nose_to_left) / expected
        # Original *60 mapping: 0.5 -> 30 deg, 1.0 -> 60 deg
        nose_yaw = delta * 60.0

        # Signal 2: eye-width asymmetry (unsigned, noise-gated)
        left_eye_x_range = float(np.ptp(lmk[[43, 48, 49, 51, 50], 0]))
        right_eye_x_range = float(np.ptp(lmk[101:106, 0]))
        eye_yaw = 0.0
        if min(left_eye_x_range, right_eye_x_range) > 1e-3:
            eye_asym = max(left_eye_x_range, right_eye_x_range) / min(left_eye_x_range, right_eye_x_range)
            if eye_asym > 1.5:
                eye_yaw = (eye_asym - 1.5) * 30.0

        # Signal 3: mouth-corner asymmetry (unsigned, noise-gated)
        d_left = float(np.linalg.norm(lmk[48] - pt_nose))
        d_right = float(np.linalg.norm(lmk[54] - pt_nose))
        mouth_yaw = 0.0
        if min(d_left, d_right) > 1e-3:
            mouth_asym = abs(d_left - d_right) / max(d_left, d_right)
            if mouth_asym > 0.2:
                mouth_yaw = (mouth_asym - 0.2) * 100.0

        sign = 1.0 if nose_yaw >= 0 else -1.0
        return float(sign * max(abs(nose_yaw), eye_yaw, mouth_yaw))

    @staticmethod
    def _select_yaw_degrees(landmark_yaw: float, pose_yaw: Optional[float]) -> float:
        """Choose the strongest available side-face signal.

        InsightFace pose yaw is usually more reliable for profile detection,
        while landmark yaw remains a useful fallback when pose is absent. Pose
        yaw is absolute, so we keep the landmark sign for continuity-rate
        checks when both values are present.
        """
        if pose_yaw is None:
            return landmark_yaw
        try:
            pose_abs = abs(float(pose_yaw))
        except (TypeError, ValueError):
            return landmark_yaw
        if abs(landmark_yaw) >= pose_abs:
            return landmark_yaw
        sign = -1.0 if landmark_yaw < 0 else 1.0
        return float(sign * pose_abs)

    @staticmethod
    def _landmark_motion_state(lmk: np.ndarray):
        if lmk is None or len(lmk) == 0:
            return None
        try:
            pts = np.asarray(lmk, dtype=np.float32)
            x0, y0 = np.min(pts[:, 0]), np.min(pts[:, 1])
            x1, y1 = np.max(pts[:, 0]), np.max(pts[:, 1])
            size = float(max(x1 - x0, y1 - y0))
            if size <= 1e-3:
                return None
            center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float32)
            return center, size
        except Exception:
            return None

    @staticmethod
    def _match_color_to_reference(
        face: torch.Tensor,
        ref_face: torch.Tensor,
        mask: torch.Tensor,
        strength: float = 0.6,
    ) -> torch.Tensor:
        """Mean+std color transfer from `face` to match `ref_face`,
        applied only inside `mask` (1 = generated region).

        Why: the inpainter outputs the full 512x512 face, but only the masked
        region (lower face) is "new" -- the rest is supposed to be a no-op
        pass-through. In practice the diffusion inpainter slightly drifts
        the whole face's color statistics, and the soft-mask boundary in
        restore_img feathers that drift into the visible mouth. Aligning
        the generated face's per-channel mean/std to the original (where
        the original is sharp and unmasked) makes the seam invisible.

        Args:
            face: (3, H, W) or (B, 3, H, W) generated face in [-1, 1].
            ref_face: same shape as face.
            mask: (1, H, W), (B, 1, H, W), or broadcastable mask in [0, 1].
            strength: 0 = no transfer, 1 = full transfer.

        Returns:
            face with color statistics transferred; shape unchanged.
        """
        if strength <= 0 or face is None or ref_face is None or mask is None:
            return face
        if face.shape != ref_face.shape:
            return face
        try:
            squeeze = False
            x = face.detach().to(torch.float32)
            r = ref_face.detach().to(device=x.device, dtype=torch.float32)
            m1 = mask.detach().to(device=x.device, dtype=torch.float32)
            if x.dim() == 3:
                x = x.unsqueeze(0)
                r = r.unsqueeze(0)
                squeeze = True
            if x.dim() != 4 or x.shape[1] != 3:
                return face
            if m1.dim() == 2:
                m1 = m1.unsqueeze(0).unsqueeze(0)
            elif m1.dim() == 3:
                m1 = m1.unsqueeze(0) if m1.shape[0] in (1, 3) else m1.unsqueeze(1)
            if m1.dim() != 4:
                return face
            if m1.shape[1] == 3:
                m_w = m1[:, 0:1]
                m3 = m1
            else:
                m_w = m1[:, 0:1]
                m3 = m_w.expand(-1, 3, -1, -1)
            if m3.shape[0] == 1 and x.shape[0] > 1:
                m3 = m3.expand(x.shape[0], -1, -1, -1)
                m_w = m_w.expand(x.shape[0], -1, -1, -1)
            if m3.shape[0] != x.shape[0]:
                return face
            weight_sum = m_w.sum(dim=(2, 3)).clamp_min(1.0).expand(-1, 3)
            src_pixels = x * m3
            tgt_pixels = r * m3
            src_mean = src_pixels.sum(dim=(2, 3)) / weight_sum
            tgt_mean = tgt_pixels.sum(dim=(2, 3)) / weight_sum
            src_var = ((x - src_mean[:, :, None, None]) ** 2 * m3).sum(dim=(2, 3)) / weight_sum
            tgt_var = ((r - tgt_mean[:, :, None, None]) ** 2 * m3).sum(dim=(2, 3)) / weight_sum
            src_std = src_var.clamp_min(1e-6).sqrt()
            tgt_std = tgt_var.clamp_min(1e-6).sqrt()
            scale = tgt_std / src_std
            shift = tgt_mean - src_mean * scale
            adjusted = x * scale[:, :, None, None] + shift[:, :, None, None]
            mixed = x + m3 * strength * (adjusted - x)
            if squeeze:
                mixed = mixed.squeeze(0)
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
        try:
            squeeze = False
            x = face.detach().to(torch.float32)
            if x.dim() == 3:
                x = x.unsqueeze(0)
                squeeze = True
            if x.dim() != 4 or x.shape[1] != 3:
                return face
            m = mask.detach().to(device=x.device, dtype=torch.float32)
            if m.dim() == 2:
                m = m.unsqueeze(0).unsqueeze(0)
            elif m.dim() == 3:
                m = m.unsqueeze(0) if m.shape[0] in (1, 3) else m.unsqueeze(1)
            if m.dim() != 4:
                return face
            if m.shape[1] == 1:
                m = m.expand(-1, 3, -1, -1)
            if m.shape[0] == 1 and x.shape[0] > 1:
                m = m.expand(x.shape[0], -1, -1, -1)
            if m.shape[0] != x.shape[0]:
                return face
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
            inp_pad = torch.nn.functional.pad(x, (radius, radius, 0, 0), mode="reflect")
            inp_pad = torch.nn.functional.pad(inp_pad, (0, 0, radius, radius), mode="reflect")
            tmp = torch.nn.functional.conv2d(inp_pad, kx, groups=3)
            blurred = torch.nn.functional.conv2d(tmp, ky, groups=3)
            sharpened = x + amount * (x - blurred)
            out = x + m * (sharpened - x)
            if squeeze:
                out = out.squeeze(0)
            return out.to(face.dtype)
        except Exception:
            return face

    @staticmethod
    def _mouth_core_mask(mask: torch.Tensor, mouth_keep: float = 0.78, mouth_center_norm: Optional[Tuple[float, float]] = None) -> torch.Tensor:
        """Return the central mouth-motion area inside an inpaint mask.

        `mask` is 1 where the model-generated lower face is visible. The
        returned mask keeps the lip aperture and lip contour protected from
        detail restoration, while allowing cheeks/chin around it to recover
        reference texture.

        `mouth_center_norm` is an optional (cx, cy) tuple in [0,1]
        normalized coordinates specifying the actual mouth center in the
        aligned face. When None, falls back to the default (0.50, 0.66).
        """
        if mask is None:
            return mask
        try:
            cx_norm, cy_norm = mouth_center_norm if mouth_center_norm is not None else (0.50, 0.66)
            m = mask.detach().to(torch.float32)
            squeeze = False
            if m.dim() == 2:
                m = m.unsqueeze(0).unsqueeze(0)
                squeeze = True
            elif m.dim() == 3:
                m = m.unsqueeze(0) if m.shape[0] in (1, 3) else m.unsqueeze(1)
                squeeze = True
            if m.dim() != 4:
                return mask
            base = m[:, 0:1]
            B, _, H, W = base.shape
            yy = torch.linspace(0.0, 1.0, H, dtype=torch.float32, device=base.device).view(1, 1, H, 1)
            xx = torch.linspace(0.0, 1.0, W, dtype=torch.float32, device=base.device).view(1, 1, 1, W)
            ell = ((xx - cx_norm) / 0.22) ** 2 + ((yy - cy_norm) / 0.12) ** 2
            core = (1.0 - (ell - mouth_keep) / max(1e-6, 1.0 - mouth_keep)).clamp(0.0, 1.0)
            core = base * core.expand(B, 1, H, W)
            if mask.dim() == 4 and mask.shape[1] == 3:
                core = core.expand(-1, 3, -1, -1)
            if squeeze:
                core = core.squeeze(0)
            return core.to(mask.dtype)
        except Exception:
            return mask

    @staticmethod
    def _restore_reference_detail(
        face: torch.Tensor,
        ref_face: torch.Tensor,
        mask: torch.Tensor,
        strength: float = 0.65,
        radius: int = 3,
    ) -> torch.Tensor:
        """Blend original high-frequency detail back outside the mouth core.

        This reduces washed cheeks/chin and mask-boundary softness while
        preserving the generated lip opening/closing in the central mouth.
        """
        if strength <= 0 or face is None or ref_face is None or mask is None:
            return face
        if face.shape != ref_face.shape:
            return face
        try:
            squeeze = False
            x = face.detach().to(torch.float32)
            r = ref_face.detach().to(device=x.device, dtype=torch.float32)
            if x.dim() == 3:
                x = x.unsqueeze(0)
                r = r.unsqueeze(0)
                squeeze = True
            if x.dim() != 4 or x.shape[1] != 3:
                return face
            m = mask.detach().to(device=x.device, dtype=torch.float32)
            if m.dim() == 2:
                m = m.unsqueeze(0).unsqueeze(0)
            elif m.dim() == 3:
                m = m.unsqueeze(0) if m.shape[0] in (1, 3) else m.unsqueeze(1)
            if m.dim() != 4:
                return face
            if m.shape[1] == 3:
                m = m[:, 0:1]
            if m.shape[0] == 1 and x.shape[0] > 1:
                m = m.expand(x.shape[0], -1, -1, -1)
            if m.shape[0] != x.shape[0]:
                return face
            mouth_core = LipsyncPipeline._mouth_core_mask(m).to(torch.float32)
            detail_mask = (m * (1.0 - mouth_core)).expand(-1, 3, -1, -1)

            k = 2 * radius + 1
            sigma = max(1.0, (k - 1) / 6.0)
            ax = torch.arange(k, dtype=torch.float32, device=x.device) - radius
            g1 = torch.exp(-(ax ** 2) / (2 * sigma * sigma))
            g1 = g1 / g1.sum()
            kx = g1.view(1, 1, 1, k).expand(3, 1, 1, k)
            ky = g1.view(1, 1, k, 1).expand(3, 1, k, 1)

            def blur(inp: torch.Tensor) -> torch.Tensor:
                padded = torch.nn.functional.pad(inp, (radius, radius, 0, 0), mode="reflect")
                padded = torch.nn.functional.pad(padded, (0, 0, radius, radius), mode="reflect")
                return torch.nn.functional.conv2d(
                    torch.nn.functional.conv2d(padded, kx, groups=3), ky, groups=3
                )

            ref_detail = r - blur(r)
            out = x + detail_mask * strength * ref_detail
            if squeeze:
                out = out.squeeze(0)
            return out.clamp(-1.0, 1.0).to(face.dtype)
        except Exception:
            return face

    @staticmethod
    def compute_aligned_mouth_info(
        lmk: Optional[np.ndarray],
        affine_matrix,  # np.ndarray (2,3) or torch.Tensor (1,2,3) or (2,3)
        resolution: int,
    ) -> Optional[Dict[str, float]]:
        """Compute mouth center, width, and height in aligned face space.

        Returns dict with keys: center_x, center_y, half_width, half_height,
        or None if landmarks are unavailable.
        """
        if lmk is None or len(lmk) < max(MOUTH_OUTER_LANDMARKS) + 1:
            return None
        try:
            M = np.array(affine_matrix, dtype=np.float64)
            if M.ndim == 3:
                M = M.squeeze(0)
            if M.shape != (2, 3):
                return None
            outer_pts = lmk[MOUTH_OUTER_LANDMARKS].astype(np.float64)  # (4, 2)
            ones = np.ones((outer_pts.shape[0], 1), dtype=np.float64)
            pts_h = np.concatenate([outer_pts, ones], axis=1)  # (4, 3)
            aligned_pts = (M @ pts_h.T).T  # (4, 2)
            left_corner = aligned_pts[0]   # landmark 48
            right_corner = aligned_pts[1]  # landmark 54
            top_center = aligned_pts[2]    # landmark 51
            bottom_center = aligned_pts[3] # landmark 57

            center_x = float(np.mean(aligned_pts[:, 0]))
            center_y = float(np.mean(aligned_pts[:, 1]))
            half_width = float(np.linalg.norm(right_corner - left_corner) / 2.0)
            half_height = float(
                max(np.linalg.norm(bottom_center - top_center) / 2.0, 2.0)
            )
            if half_width < 1.0 or half_height < 1.0:
                return None
            center_x = max(0.0, min(float(resolution), center_x))
            center_y = max(0.0, min(float(resolution), center_y))
            return {
                "center_x": center_x,
                "center_y": center_y,
                "half_width": half_width,
                "half_height": half_height,
            }
        except (IndexError, TypeError, ValueError):
            return None

    @staticmethod
    def generate_dynamic_mouth_mask(
        mouth_info: Optional[Dict[str, float]],
        resolution: int,
        fallback_center_x_norm: float = 0.50,
        fallback_center_y_norm: float = 0.66,
        fallback_rx_norm: float = 0.225,
        fallback_ry_norm: float = 0.155,
        pad_width_ratio: float = 1.5,
        pad_height_top_ratio: float = 1.3,
        pad_height_bottom_ratio: float = 2.2,
        chin_extend_norm: float = 0.04,
        feather_sigma_px: float = 7.0,
        min_ry_norm: float = 0.10,
        min_rx_norm: float = 0.12,
        max_ry_norm: float = 0.30,
        max_rx_norm: float = 0.40,
    ) -> torch.Tensor:
        """Generate a per-frame mouth inpainting mask from landmark info.

        Returns a (1, H, W) tensor in [0, 1] where 1 = keep (preserve) and
        0 = inpaint (regenerate). This uses the same convention as the fixed
        mask -- 1 outside the mouth (preserved original) and 0 inside the mouth
        (to be regenerated by the inpainter).

        When ``mouth_info`` is None (face not detected or landmarks unavailable),
        falls back to a default elliptical mask matching the old hard-coded
        ``_mouth_core_mask`` geometry.
        """
        H = W = resolution
        if mouth_info is not None:
            cx = mouth_info["center_x"] / resolution
            cy = mouth_info["center_y"] / resolution
            hw = mouth_info["half_width"] / resolution
            hh = mouth_info["half_height"] / resolution
            rx = hw * pad_width_ratio
            ry_top = hh * pad_height_top_ratio
            ry_bottom = hh * pad_height_bottom_ratio + chin_extend_norm
            ry = (ry_top + ry_bottom) / 2.0
            cy = cy + (ry_bottom - ry_top) / 2.0
            rx = max(min_rx_norm, min(max_rx_norm, rx))
            ry = max(min_ry_norm, min(max_ry_norm, ry))
        else:
            cx = fallback_center_x_norm
            cy = fallback_center_y_norm
            rx = fallback_rx_norm
            ry = fallback_ry_norm

        yy = torch.linspace(0.0, 1.0, H, dtype=torch.float32).view(H, 1)
        xx = torch.linspace(0.0, 1.0, W, dtype=torch.float32).view(1, W)
        ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        inpaint_region = (ellipse <= 1.0).float()

        if feather_sigma_px > 0:
            k = int(2 * round(3 * feather_sigma_px) + 1)
            sigma = feather_sigma_px
            kernel_1d = torch.exp(
                -0.5 * (torch.arange(k, dtype=torch.float32) - k // 2).pow(2) / sigma ** 2
            )
            kernel_1d = kernel_1d / kernel_1d.sum()
            kx = kernel_1d.view(1, 1, 1, k).expand(1, 1, 1, k)
            ky = kernel_1d.view(1, 1, k, 1).expand(1, 1, k, 1)
            inpaint_4d = inpaint_region.unsqueeze(0).unsqueeze(0)
            pad_size = k // 2
            padded = torch.nn.functional.pad(inpaint_4d, (pad_size, pad_size, 0, 0), mode="reflect")
            padded = torch.nn.functional.pad(padded, (0, 0, pad_size, pad_size), mode="reflect")
            smoothed = torch.nn.functional.conv2d(padded, kx, groups=1)
            smoothed = torch.nn.functional.conv2d(smoothed, ky, groups=1)
            inpaint_region = smoothed.squeeze(0).squeeze(0)
            inner_ellipse = ((xx - cx) / (rx * 0.75)) ** 2 + ((yy - cy) / (ry * 0.75)) ** 2
            inpaint_region = torch.where(inner_ellipse <= 1.0, torch.ones_like(inpaint_region), inpaint_region)
            inpaint_region = torch.where(ellipse > 1.0, torch.zeros_like(inpaint_region), inpaint_region)

        keep_mask = 1.0 - inpaint_region
        return keep_mask.unsqueeze(0)

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
    def _mouth_roi(face: torch.Tensor) -> Optional[torch.Tensor]:
        """Crop the aligned-face mouth band. Supports (3,H,W) or (B,3,H,W)."""
        if face is None or face.numel() == 0:
            return None
        try:
            if face.dim() == 3:
                H, W = face.shape[1], face.shape[2]
                y0, y1 = int(H * 0.55), int(H * 0.74)
                x0, x1 = int(W * 0.30), int(W * 0.70)
                return face[:, y0:y1, x0:x1]
            if face.dim() == 4:
                H, W = face.shape[2], face.shape[3]
                y0, y1 = int(H * 0.55), int(H * 0.74)
                x0, x1 = int(W * 0.30), int(W * 0.70)
                return face[:, :, y0:y1, x0:x1]
        except Exception:
            return None
        return None

    @staticmethod
    def _mouth_sharpness(face: torch.Tensor) -> float:
        """Laplacian variance on the aligned mouth ROI."""
        roi = LipsyncPipeline._mouth_roi(face)
        if roi is None or roi.numel() == 0:
            return 0.0
        if roi.dim() == 4:
            scores = [LipsyncPipeline._face_sharpness(roi[k]) for k in range(roi.shape[0])]
            return float(np.mean(scores)) if scores else 0.0
        return LipsyncPipeline._face_sharpness(roi)

    @staticmethod
    def _mouth_region_diff(prev_face: torch.Tensor, curr_face: torch.Tensor) -> float:
        """Mean absolute diff in the mouth band of two aligned face crops.

        Both inputs are (3, H, W) tensors in the same dtype/range. Returns
        the diff normalized to [0, 1] (caller is responsible for the range
        the input lives in -- the affine-transformed crops are uint8 in
        [0, 255], so we divide by 255; if the caller passes already
        normalized [-1, 1] tensors, divide by 2 instead).

        This complements the embedding-similarity continuity break:
        embedding answers "is this the same person", pixel diff answers
        "has the content changed". Two people with similar embeddings
        (cosine >= 0.70) still usually have 0.10-0.30 mouth-region diff,
        so this catches face switches the embedding check misses -- and
        it's robust to side faces where embedding models are unreliable.

        Returns 0.0 on None / shape mismatch (defensive: never raises, so
        a corrupt prior frame can't kill the whole affine pass).
        """
        if prev_face is None or curr_face is None:
            return 0.0
        if prev_face.shape != curr_face.shape:
            return 0.0
        if prev_face.dim() != 3:
            return 0.0
        try:
            H, W = curr_face.shape[-2:]
            y0, y1 = int(H * 0.55), int(H * 0.74)
            x0, x1 = int(W * 0.30), int(W * 0.70)
            prev = prev_face[..., y0:y1, x0:x1].to(torch.float32)
            curr = curr_face[..., y0:y1, x0:x1].to(torch.float32)
            return float((curr - prev).abs().mean().item()) / 255.0
        except Exception:
            return 0.0

    @staticmethod
    def _smooth_face_sequence(
        face_crops: torch.Tensor,
        prev_face: Optional[torch.Tensor],
        prev_valid: bool,
        inference_skip_mask,
        continuity_break_mask=None,
        region_mask: Optional[torch.Tensor] = None,
        weights=(0.25, 0.5, 0.25),
    ):
        """3-tap temporal EMA across face crops. Returns
        (smoothed, last_face, last_valid).

        - Skips any frame where inference_skip_mask[k] is True and resets the
          carry state so the next valid frame doesn't blend in a zero placeholder.
        - continuity_break_mask[k] resets temporal carry before frame k and
          prevents k from blending with k-1. This is used at detected face
          switches so the previous person/shot cannot leak into the new one.
        - prev_face is only used when prev_valid is True.
        - Triangular kernel by default (weights = prev, cur, next) so the
          middle frame keeps 50% weight and neighbours each contribute 25%.

        region_mask (optional): (B, 1, H, W) or (B, 3, H, W) in [0, 1] with
        1 = "apply EMA inside this pixel" and 0 = "keep raw face unchanged".
        When provided, the EMA output is masked so the temporal blend only
        happens inside the inpaint region. Pixels outside the mask get the
        raw `face_crops[k]` (i.e. the un-EMA'd face where the inpainter
        didn't touch it). This prevents EMA from smearing previous frames'
        content into the original-face area when the inpaint mask is wider
        than the mouth, which was the root cause of the "previous frame's
        different face glued in" artifact. Default None = full-face EMA
        (legacy behavior).
        """
        B = face_crops.shape[0]
        if B == 0:
            return face_crops, prev_face, prev_valid
        w_prev, w_cur, w_next = weights
        smoothed = face_crops.clone()
        last_valid = prev_valid
        last_face = prev_face
        continuity_break_mask = list(continuity_break_mask or [])
        if len(continuity_break_mask) < B:
            continuity_break_mask = continuity_break_mask + [False] * (B - len(continuity_break_mask))
        elif len(continuity_break_mask) > B:
            continuity_break_mask = continuity_break_mask[:B]

        # Normalize region_mask to (B, 3, H, W) for broadcasting against
        # face_crops (B, 3, H, W). Accepts (B, H, W), (B, 1, H, W),
        # (B, 3, H, W), or (1, *, H, W) shapes.
        region_mask_3d: Optional[torch.Tensor] = None
        if region_mask is not None:
            rm = region_mask
            if rm.dim() == 3:
                rm = rm.unsqueeze(1)
            if rm.dim() != 4:
                # Fall back to no-mask behavior if shape is unexpected.
                rm = None
            else:
                if rm.shape[0] == 1 and B > 1:
                    rm = rm.expand(B, -1, -1, -1)
                if rm.shape[0] != B:
                    rm = None
                else:
                    if rm.shape[1] == 3:
                        rm = rm[:, 0:1]
                    elif rm.shape[1] != 1:
                        rm = None
                    if rm is not None:
                        region_mask_3d = rm.expand(-1, 3, -1, -1)

        for k in range(B):
            if continuity_break_mask[k]:
                last_face = None
                last_valid = False

            if inference_skip_mask[k]:
                # Zero-placeholder face from affine_transform_video: don't
                # pollute neighbours; reset carry.
                last_face = None
                last_valid = False
                continue

            weighted = w_cur * face_crops[k]
            total_weight = w_cur

            if k == 0:
                if last_valid and last_face is not None:
                    weighted = weighted + w_prev * last_face.to(face_crops.device)
                    total_weight += w_prev
            elif not inference_skip_mask[k - 1] and not continuity_break_mask[k]:
                weighted = weighted + w_prev * face_crops[k - 1]
                total_weight += w_prev

            if k + 1 < B and not inference_skip_mask[k + 1] and not continuity_break_mask[k + 1]:
                weighted = weighted + w_next * face_crops[k + 1]
                total_weight += w_next

            if total_weight > w_cur:
                smoothed[k] = weighted / total_weight

            # Region-mask: outside the inpaint area, use the raw face so EMA
            # can't leak previous-frame content into the original-face area.
            if region_mask_3d is not None:
                mask_k = region_mask_3d[k]
                smoothed[k] = (1.0 - mask_k) * face_crops[k] + mask_k * smoothed[k]

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
        yaw_skip_threshold: float = 30.0,
        yaw_rate_skip_threshold: float = 28.0,
        mouth_occlusion_skip_threshold: float = 1.0,
        motion_blur_skip_threshold: float = 0.08,
        face_jump_center_threshold: float = 0.0,
        face_jump_scale_threshold: float = 0.0,
        lipsync_continuity_max_center_shift: float = 0.35,
        lipsync_continuity_max_scale_change: float = 0.35,
        lipsync_mouth_diff_break_threshold: float = 0.10,
        identity_similarity_threshold: float = 0.5,
        apply_identity_filter: bool = True,
        side_face_episode_pre_pad: int = 0,
        side_face_episode_post_pad: int = 0,
        yaw_warn_threshold_ratio: float = 0.75,
        side_face_warn_min_run_frames: int = 0,
    ):
        logger.info(
            f"[FaceMatch] Starting: reference_embedding={'loaded' if reference_embedding is not None else 'None'}, "
            f"frames={len(video_frames)}, yaw_skip_threshold={yaw_skip_threshold}, "
            f"yaw_rate_skip_threshold={yaw_rate_skip_threshold}, "
            f"mouth_occlusion_skip_threshold={mouth_occlusion_skip_threshold}, "
            f"motion_blur_skip_threshold={motion_blur_skip_threshold}, "
            f"face_jump_center_threshold={face_jump_center_threshold}, "
            f"face_jump_scale_threshold={face_jump_scale_threshold}, "
            f"lipsync_continuity_max_center_shift={lipsync_continuity_max_center_shift}, "
            f"lipsync_continuity_max_scale_change={lipsync_continuity_max_scale_change}, "
            f"lipsync_mouth_diff_break_threshold={lipsync_mouth_diff_break_threshold}, "
            f"identity_similarity_threshold={identity_similarity_threshold}, "
            f"apply_identity_filter={apply_identity_filter}"
        )
        faces = []
        boxes = []
        affine_matrices = []
        skip_mask = []
        aligned_mouth_info: List[Optional[Dict[str, float]]] = []
        # Parallel arrays used by the episode-level side-face filter below:
        #   yaws[k] is the per-frame yaw in degrees (None for detect-fail frames)
        #   yaw_skip_reasons[k] is True iff THIS frame was skipped for yaw alone
        #     (not for identity / occlusion / blur / detect-fail -- those don't
        #     represent a "side face" and shouldn't trigger the episode pad).
        yaws: List[Optional[float]] = []
        yaw_skip_reasons: List[bool] = []
        if video_frames is None or len(video_frames) == 0:
            # Empty input: don't crash with `stack expects a non-empty TensorList`.
            # `restore_video` already returns the empty array as a no-op downstream.
            logger.error("[FaceMatch] empty video_frames (len=0 or None); skipping affine transform")
            empty_zeros = torch.zeros(0, 3, self.image_processor.resolution, self.image_processor.resolution)
            return (
                empty_zeros,
                [],
                [],
                [],
                [],
                [],
            )
        yaw_skip_count = 0
        yaw_rate_skip_count = 0
        mouth_occlusion_skip_count = 0
        motion_blur_skip_count = 0
        face_jump_skip_count = 0
        temporal_identity_break_count = 0
        temporal_geometry_break_count = 0
        temporal_diff_break_count = 0
        identity_skip_count = 0
        detect_fail_count = 0
        identity_similarities: List[float] = []
        prev_yaw: Optional[float] = None
        prev_motion_state = None
        prev_temporal_motion_state = None
        prev_temporal_embedding = None
        prev_temporal_face: Optional[torch.Tensor] = None
        prev_mouth_info: Optional[Dict[str, float]] = None
        continuity_break_mask = []
        print(f"Affine transforming {len(video_frames)} faces...")
        for idx, frame in enumerate(tqdm.tqdm(video_frames)):
            affine_result = self.image_processor.affine_transform_with_embedding(frame)
            if len(affine_result) == 5:
                face, box, affine_matrix, face_emb, lmk = affine_result
            else:
                face, box, affine_matrix, face_emb = affine_result
                lmk = None
            if face is None:
                detect_fail_count += 1
                skip_mask.append(True)
                faces.append(torch.zeros(3, self.image_processor.resolution, self.image_processor.resolution))
                boxes.append([0, 0, 0, 0])
                affine_matrices.append(np.eye(3))
                aligned_mouth_info.append(None)
                yaws.append(None)
                yaw_skip_reasons.append(False)
                prev_yaw = None  # reset so we don't carry a stale yaw across a detect-fail gap
                prev_motion_state = None
                prev_temporal_motion_state = None
                prev_temporal_embedding = None
                prev_temporal_face = None
                prev_mouth_info = None
                continuity_break_mask.append(True)
                continue
            should_skip = False
            if apply_identity_filter and reference_embedding is not None and face_emb is not None:
                similarity = float(np.dot(face_emb, reference_embedding))
                identity_similarities.append(similarity)
                if similarity < identity_similarity_threshold:
                    should_skip = True
                    identity_skip_count += 1
            yaw_deg = 0.0
            yaw_available = False
            yaw_was_skipped = False  # tracks the absolute yaw threshold (not yaw_rate)
            if yaw_skip_threshold > 0:
                landmark_yaw = self._estimate_yaw_degrees(lmk) if lmk is not None else 0.0
                pose_yaw = getattr(self.image_processor.face_detector, "last_pose_yaw", None)
                yaw_available = lmk is not None or pose_yaw is not None
                yaw_deg = self._select_yaw_degrees(landmark_yaw, pose_yaw) if yaw_available else 0.0
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
                and yaw_available
            ):
                rate = abs(yaw_deg - prev_yaw)
                if rate > yaw_rate_skip_threshold:
                    should_skip = True
                    yaw_rate_skip_count += 1
            motion_state = self._landmark_motion_state(lmk)
            if (
                not should_skip
                and motion_state is not None
                and prev_motion_state is not None
                and (face_jump_center_threshold > 0 or face_jump_scale_threshold > 0)
            ):
                center, size = motion_state
                prev_center, prev_size = prev_motion_state
                center_shift = float(np.linalg.norm(center - prev_center) / max(prev_size, 1.0))
                scale_shift = float(abs(size - prev_size) / max(prev_size, 1.0))
                if (
                    face_jump_center_threshold > 0
                    and center_shift > face_jump_center_threshold
                ) or (
                    face_jump_scale_threshold > 0
                    and scale_shift > face_jump_scale_threshold
                ):
                    should_skip = True
                    face_jump_skip_count += 1
            continuity_break = False
            if not should_skip:
                if face_emb is not None and prev_temporal_embedding is not None:
                    continuity_similarity = float(np.dot(face_emb, prev_temporal_embedding))
                    if continuity_similarity < 0.70:
                        continuity_break = True
                        temporal_identity_break_count += 1
                if (
                    motion_state is not None
                    and prev_temporal_motion_state is not None
                    and (
                        lipsync_continuity_max_center_shift > 0
                        or lipsync_continuity_max_scale_change > 0
                    )
                ):
                    center, size = motion_state
                    prev_center, prev_size = prev_temporal_motion_state
                    center_shift = float(np.linalg.norm(center - prev_center) / max(prev_size, 1.0))
                    scale_shift = float(abs(size - prev_size) / max(prev_size, 1.0))
                    geometry_break = (
                        lipsync_continuity_max_center_shift > 0
                        and center_shift > lipsync_continuity_max_center_shift
                    ) or (
                        lipsync_continuity_max_scale_change > 0
                        and scale_shift > lipsync_continuity_max_scale_change
                    )
                    if geometry_break:
                        continuity_break = True
                        temporal_geometry_break_count += 1
                # Mouth-region pixel diff: catches face switches the embedding
                # check misses (similar-looking people, side faces). Strictly
                # complementary to the embedding break: embedding asks "same
                # person?", pixel diff asks "same content?". Cheap (one crop
                # + abs + mean) so it doesn't gate the loop.
                if (
                    not continuity_break
                    and lipsync_mouth_diff_break_threshold > 0
                    and prev_temporal_face is not None
                    and face is not None
                ):
                    mouth_diff = self._mouth_region_diff(prev_temporal_face, face)
                    if mouth_diff > lipsync_mouth_diff_break_threshold:
                        continuity_break = True
                        temporal_diff_break_count += 1
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
                mouth_sharp = self._mouth_sharpness(face)
                if (
                    face_sharp < motion_blur_skip_threshold
                    and mouth_sharp < motion_blur_skip_threshold * 0.5
                ):
                    should_skip = True
                    motion_blur_skip_count += 1
            skip_mask.append(should_skip)
            continuity_break_mask.append(should_skip or continuity_break)
            yaws.append(yaw_deg if (yaw_skip_threshold > 0 and yaw_available) else None)
            yaw_skip_reasons.append(yaw_was_skipped)
            faces.append(face)
            boxes.append(box)
            affine_matrices.append(affine_matrix)
            mouth_info = self.compute_aligned_mouth_info(
                lmk, affine_matrix, self.image_processor.resolution
            )
            # EMA smoothing on mouth_info to reduce mask-boundary jitter
            # from noisy landmark detection across consecutive frames.
            if mouth_info is not None and prev_mouth_info is not None:
                alpha = 0.7
                mouth_info = {
                    "center_x": alpha * mouth_info["center_x"] + (1 - alpha) * prev_mouth_info["center_x"],
                    "center_y": alpha * mouth_info["center_y"] + (1 - alpha) * prev_mouth_info["center_y"],
                    "half_width": alpha * mouth_info["half_width"] + (1 - alpha) * prev_mouth_info["half_width"],
                    "half_height": alpha * mouth_info["half_height"] + (1 - alpha) * prev_mouth_info["half_height"],
                }
            if mouth_info is not None:
                prev_mouth_info = mouth_info
            aligned_mouth_info.append(mouth_info)
            prev_yaw = yaw_deg if (yaw_skip_threshold > 0 and yaw_available) else None
            if not should_skip and motion_state is not None:
                prev_motion_state = motion_state
                prev_temporal_motion_state = motion_state
            if not should_skip and face_emb is not None:
                prev_temporal_embedding = face_emb
            if not should_skip and face is not None:
                prev_temporal_face = face
            if should_skip:
                prev_temporal_motion_state = None
                prev_temporal_embedding = None
                prev_temporal_face = None
                prev_mouth_info = None
        logger.info(
            f"[FaceMatch] detect_fail={detect_fail_count}, identity_skip={identity_skip_count}, "
            f"yaw_skip={yaw_skip_count}, yaw_rate_skip={yaw_rate_skip_count}, "
            f"mouth_occlusion_skip={mouth_occlusion_skip_count}, "
            f"motion_blur_skip={motion_blur_skip_count}, "
            f"face_jump_skip={face_jump_skip_count}, "
            f"temporal_identity_break={temporal_identity_break_count}, "
            f"temporal_geometry_break={temporal_geometry_break_count}, "
            f"temporal_diff_break={temporal_diff_break_count}"
        )
        if identity_similarities:
            logger.info(
                "[FaceMatch] identity similarity: min=%.3f median=%.3f max=%.3f threshold=%.3f",
                min(identity_similarities),
                statistics.median(identity_similarities),
                max(identity_similarities),
                identity_similarity_threshold,
            )
        self._last_yaw_skip_count = yaw_skip_count
        self._last_yaw_rate_skip_count = yaw_rate_skip_count
        self._last_mouth_occlusion_skip_count = mouth_occlusion_skip_count
        self._last_motion_blur_skip_count = motion_blur_skip_count
        self._last_face_jump_skip_count = face_jump_skip_count
        self._last_identity_skip_count = identity_skip_count
        self._last_temporal_diff_break_count = temporal_diff_break_count
        self._last_identity_similarity_stats = {
            "min": float(min(identity_similarities)) if identity_similarities else 0.0,
            "median": float(statistics.median(identity_similarities)) if identity_similarities else 0.0,
            "max": float(max(identity_similarities)) if identity_similarities else 0.0,
        }

        # Episode-level side-face filter: a contiguous run of yaw-skipped
        # frames represents a single turning motion. The frames immediately
        # before/after the run typically have yaw in the warn band (e.g.
        # 16.5-22° for the default 22° threshold) where affine alignment is
        # still unreliable; we extend the skip_mask to include those
        # transition frames so the whole turn becomes a single side-face
        # episode (instead of a fragmentary skip that lets blur sneak in
        # at the boundaries).
        yaw_warn_threshold = yaw_skip_threshold * yaw_warn_threshold_ratio
        side_face_episode_extra_skip_count = 0
        side_face_warn_run_skip_count = 0
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
                        continuity_break_mask[k] = True
                        side_face_episode_extra_skip_count += 1
                # expand right into post_pad window
                for k in range(j, min(n, j + side_face_episode_post_pad)):
                    if (
                        not skip_mask[k]
                        and yaws[k] is not None
                        and abs(yaws[k]) > yaw_warn_threshold
                    ):
                        skip_mask[k] = True
                        continuity_break_mask[k] = True
                        side_face_episode_extra_skip_count += 1
                i = j
        if yaw_warn_threshold > 0 and side_face_warn_min_run_frames > 0:
            n = len(skip_mask)
            i = 0
            while i < n:
                if skip_mask[i] or yaws[i] is None or abs(yaws[i]) <= yaw_warn_threshold:
                    i += 1
                    continue
                j = i
                while (
                    j < n
                    and not skip_mask[j]
                    and yaws[j] is not None
                    and abs(yaws[j]) > yaw_warn_threshold
                ):
                    j += 1
                if j - i >= side_face_warn_min_run_frames:
                    for k in range(i, j):
                        skip_mask[k] = True
                        continuity_break_mask[k] = True
                        side_face_warn_run_skip_count += 1
                i = j
        self._last_side_face_episode_extra_skip_count = side_face_episode_extra_skip_count
        self._last_side_face_warn_run_skip_count = side_face_warn_run_skip_count
        if side_face_episode_extra_skip_count:
            logger.info(
                f"[FaceMatch] side_face_episode_extra_skip={side_face_episode_extra_skip_count} "
                f"(pre_pad={side_face_episode_pre_pad}, post_pad={side_face_episode_post_pad}, "
                f"warn_threshold={yaw_warn_threshold:.1f}°)"
            )
        if side_face_warn_run_skip_count:
            logger.info(
                f"[FaceMatch] side_face_warn_run_skip={side_face_warn_run_skip_count} "
                f"(min_run={side_face_warn_min_run_frames}, warn_threshold={yaw_warn_threshold:.1f}°)"
            )

        faces_tensor = torch.stack(faces)
        return faces_tensor, boxes, affine_matrices, skip_mask, continuity_break_mask, aligned_mouth_info

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

    def _silent_frame_mask(
        self,
        audio_samples: torch.Tensor,
        frame_count: int,
        video_fps: float,
        audio_sample_rate: int,
        rms_threshold: float,
        min_run_frames: int,
        pad_frames: int,
    ) -> List[bool]:
        if frame_count <= 0 or rms_threshold <= 0 or video_fps <= 0:
            return [False] * max(0, frame_count)

        audio_np = audio_samples.detach().float().cpu().numpy()
        if audio_np.ndim > 1:
            audio_np = audio_np.reshape(-1)
        mask = []
        for idx in range(frame_count):
            start = int(idx / video_fps * audio_sample_rate)
            end = int((idx + 1) / video_fps * audio_sample_rate)
            frame_audio = audio_np[start:min(end, len(audio_np))]
            if frame_audio.size == 0:
                mask.append(True)
                continue
            rms = float(np.sqrt(np.mean(np.square(frame_audio), dtype=np.float64)))
            mask.append(rms < rms_threshold)

        if min_run_frames > 1:
            filtered = [False] * len(mask)
            i = 0
            while i < len(mask):
                if not mask[i]:
                    i += 1
                    continue
                j = i
                while j < len(mask) and mask[j]:
                    j += 1
                if j - i >= min_run_frames:
                    left = max(0, i - pad_frames)
                    right = min(len(mask), j + pad_frames)
                    for k in range(left, right):
                        filtered[k] = True
                i = j
            mask = filtered
        elif pad_frames > 0:
            padded = mask[:]
            for idx, value in enumerate(mask):
                if not value:
                    continue
                for k in range(max(0, idx - pad_frames), min(len(mask), idx + pad_frames + 1)):
                    padded[k] = True
            mask = padded

        return mask

    def loop_video(
        self,
        whisper_chunks: list,
        video_frames: np.ndarray,
        reference_embedding=None,
        face_embedder=None,
        skip_mask=None,
        yaw_skip_threshold: float = 30.0,
        yaw_rate_skip_threshold: float = 28.0,
        mouth_occlusion_skip_threshold: float = 1.0,
        motion_blur_skip_threshold: float = 0.08,
        face_jump_center_threshold: float = 0.0,
        face_jump_scale_threshold: float = 0.0,
        lipsync_continuity_max_center_shift: float = 0.35,
        lipsync_continuity_max_scale_change: float = 0.35,
        lipsync_mouth_diff_break_threshold: float = 0.10,
        identity_similarity_threshold: float = 0.5,
        apply_identity_filter: bool = True,
        side_face_episode_pre_pad: int = 0,
        side_face_episode_post_pad: int = 0,
        yaw_warn_threshold_ratio: float = 0.75,
        side_face_warn_min_run_frames: int = 0,
    ):
        logger.info(
            f"[LipSync] loop_video: reference_embedding={'loaded' if reference_embedding is not None else 'None'}, "
            f"frames={len(video_frames)}, yaw_skip_threshold={yaw_skip_threshold}, "
            f"yaw_rate_skip_threshold={yaw_rate_skip_threshold}, "
            f"mouth_occlusion_skip_threshold={mouth_occlusion_skip_threshold}, "
            f"motion_blur_skip_threshold={motion_blur_skip_threshold}, "
            f"face_jump_center_threshold={face_jump_center_threshold}, "
            f"face_jump_scale_threshold={face_jump_scale_threshold}, "
            f"lipsync_continuity_max_center_shift={lipsync_continuity_max_center_shift}, "
            f"lipsync_continuity_max_scale_change={lipsync_continuity_max_scale_change}, "
            f"lipsync_mouth_diff_break_threshold={lipsync_mouth_diff_break_threshold}, "
            f"identity_similarity_threshold={identity_similarity_threshold}, "
            f"apply_identity_filter={apply_identity_filter}"
        )
        if reference_embedding is None and face_embedder is not None:
            reference_embedding = self.detect_main_speaker_embedding(video_frames, face_embedder)
            logger.info(f"[LipSync] Auto-detected main speaker embedding: {'loaded' if reference_embedding is not None else 'None'}")
        if len(whisper_chunks) > len(video_frames):
            faces, boxes, affine_matrices, frame_skip_mask, frame_continuity_break_mask, frame_aligned_mouth_info = self.affine_transform_video(
                video_frames,
                reference_embedding,
                yaw_skip_threshold=yaw_skip_threshold,
                yaw_rate_skip_threshold=yaw_rate_skip_threshold,
                mouth_occlusion_skip_threshold=mouth_occlusion_skip_threshold,
                motion_blur_skip_threshold=motion_blur_skip_threshold,
                face_jump_center_threshold=face_jump_center_threshold,
                face_jump_scale_threshold=face_jump_scale_threshold,
                lipsync_continuity_max_center_shift=lipsync_continuity_max_center_shift,
                lipsync_continuity_max_scale_change=lipsync_continuity_max_scale_change,
                lipsync_mouth_diff_break_threshold=lipsync_mouth_diff_break_threshold,
                identity_similarity_threshold=identity_similarity_threshold,
                apply_identity_filter=apply_identity_filter,
                side_face_episode_pre_pad=side_face_episode_pre_pad,
                side_face_episode_post_pad=side_face_episode_post_pad,
                yaw_warn_threshold_ratio=yaw_warn_threshold_ratio,
                side_face_warn_min_run_frames=side_face_warn_min_run_frames,
            )
            num_loops = math.ceil(len(whisper_chunks) / len(video_frames))
            loop_video_frames = []
            loop_faces = []
            loop_boxes = []
            loop_affine_matrices = []
            loop_skip_mask = []
            loop_continuity_break_mask = []
            loop_aligned_mouth_info = []
            for i in range(num_loops):
                if i % 2 == 0:
                    loop_video_frames.append(video_frames)
                    loop_faces.append(faces)
                    loop_boxes += boxes
                    loop_affine_matrices += affine_matrices
                    loop_skip_mask += frame_skip_mask
                    loop_aligned_mouth_info += frame_aligned_mouth_info
                    loop_continuity_break_mask += [
                        (True if i > 0 and k == 0 else value)
                        for k, value in enumerate(frame_continuity_break_mask)
                    ]
                else:
                    loop_video_frames.append(video_frames[::-1])
                    loop_faces.append(faces.flip(0))
                    loop_boxes += boxes[::-1]
                    loop_affine_matrices += affine_matrices[::-1]
                    loop_skip_mask += frame_skip_mask[::-1]
                    loop_aligned_mouth_info += frame_aligned_mouth_info[::-1]
                    n_breaks = len(frame_continuity_break_mask)
                    reversed_breaks = [
                        True if k == 0 else frame_continuity_break_mask[n_breaks - k]
                        for k in range(n_breaks)
                    ]
                    loop_continuity_break_mask += reversed_breaks

            video_frames = np.concatenate(loop_video_frames, axis=0)[: len(whisper_chunks)]
            faces = torch.cat(loop_faces, dim=0)[: len(whisper_chunks)]
            boxes = loop_boxes[: len(whisper_chunks)]
            affine_matrices = loop_affine_matrices[: len(whisper_chunks)]
            skip_mask = loop_skip_mask[: len(whisper_chunks)]
            continuity_break_mask = loop_continuity_break_mask[: len(whisper_chunks)]
            aligned_mouth_info = loop_aligned_mouth_info[: len(whisper_chunks)]
        else:
            video_frames = video_frames[: len(whisper_chunks)]
            faces, boxes, affine_matrices, frame_skip_mask, continuity_break_mask, frame_aligned_mouth_info = self.affine_transform_video(
                video_frames,
                reference_embedding,
                yaw_skip_threshold=yaw_skip_threshold,
                yaw_rate_skip_threshold=yaw_rate_skip_threshold,
                mouth_occlusion_skip_threshold=mouth_occlusion_skip_threshold,
                motion_blur_skip_threshold=motion_blur_skip_threshold,
                face_jump_center_threshold=face_jump_center_threshold,
                face_jump_scale_threshold=face_jump_scale_threshold,
                lipsync_continuity_max_center_shift=lipsync_continuity_max_center_shift,
                lipsync_continuity_max_scale_change=lipsync_continuity_max_scale_change,
                lipsync_mouth_diff_break_threshold=lipsync_mouth_diff_break_threshold,
                identity_similarity_threshold=identity_similarity_threshold,
                apply_identity_filter=apply_identity_filter,
                side_face_episode_pre_pad=side_face_episode_pre_pad,
                side_face_episode_post_pad=side_face_episode_post_pad,
                yaw_warn_threshold_ratio=yaw_warn_threshold_ratio,
                side_face_warn_min_run_frames=side_face_warn_min_run_frames,
            )
            skip_mask = frame_skip_mask
            aligned_mouth_info = frame_aligned_mouth_info

        return video_frames, faces, boxes, affine_matrices, skip_mask, continuity_break_mask, aligned_mouth_info

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
        identity_similarity_threshold: float = 0.5,
        # --- quality / temporal gating (added 2026-06) ---
        temporal_smoothing_enabled: bool = True,
        # Preserve current-frame mouth-core motion after temporal smoothing.
        # 0 = fully smoothed mouth, 1 = keep generated current-frame mouth.
        mouth_motion_preserve_strength: float = 0.45,
        # Lightly stabilize mouth-core color/detail between consecutive valid
        # generated frames to reduce flicker without freezing lip motion.
        mouth_temporal_stabilization_strength: float = 0.08,
        # If the current mouth core differs too much from the previous
        # stabilized mouth, clear carry state instead of blending. This keeps
        # stabilization from borrowing lips across speaker/shot changes that
        # were not caught by geometry or identity continuity breaks.
        mouth_temporal_stabilization_max_delta: float = 0.12,
        # Audio-adaptive mouth motion: preserve more current generated mouth
        # motion on high-energy speech frames and less on weak/silent frames.
        mouth_audio_adaptive_motion_enabled: bool = True,
        mouth_audio_motion_min_scale: float = 0.65,
        mouth_audio_motion_max_scale: float = 1.15,
        # Postfilter: skip frames where the generated mouth ROI is clearly
        # blurrier than the original mouth ROI. Checked after paste/detail
        # recovery, and conservative enough to keep closed/low-texture mouths.
        quality_gate_enabled: bool = False,
        quality_min_laplacian: float = 0.04,
        quality_min_sharpness_ratio: float = 0.05,
        quality_ref_min_laplacian: float = 1.00,
        quality_max_fallback_ratio: float = 0.80,
        # Yaw-based prefilters for side faces / fast head turns. Defaults are
        # intentionally permissive so clear frontal faces are not filtered out.
        yaw_skip_threshold: float = 30.0,
        yaw_rate_skip_threshold: float = 28.0,
        # Episode-level side-face filter: when contiguous frames exceed
        # yaw_skip_threshold, also skip pre_pad/post_pad transition frames
        # around the episode (whose yaw is in the warn band between
        # yaw_skip_threshold * yaw_warn_threshold_ratio and yaw_skip_threshold).
        # Set pre_pad/post_pad to 0 to disable the padding.
        side_face_episode_pre_pad: int = 0,
        side_face_episode_post_pad: int = 0,
        yaw_warn_threshold_ratio: float = 0.75,
        side_face_warn_min_run_frames: int = 0,
        # Mouth-occlusion prefilter: skip frames where the mouth is covered
        # by a hand, microphone, phone, mask, etc. Score 0..1; above the
        # threshold the frame is treated as not-lip-syncable and the original
        # frame is used. Default 1.0 disables this heuristic because it was
        # too sensitive on side/profile shots and could eat most frames.
        mouth_occlusion_skip_threshold: float = 1.0,
        # Motion-blur input filter: skip frames whose aligned face is too
        # smeared to inpaint cleanly. Default 0.08 (Laplacian variance in
        # the [-1, 1] face space; a sharp face scores ~5-20, a motion-blurred
        # one <1.0). Set to 0 to disable.
        motion_blur_skip_threshold: float = 0.08,
        # Face-jump input filter: skip frames where landmark center/scale
        # changes abruptly, which usually means detection/alignment jumped.
        face_jump_center_threshold: float = 0.0,
        face_jump_scale_threshold: float = 0.0,
        # Temporal continuity break: clear EMA/mouth stabilization state
        # across large landmark jumps without necessarily skipping the frame.
        lipsync_continuity_max_center_shift: float = 0.35,
        lipsync_continuity_max_scale_change: float = 0.35,
        # Mouth-region pixel diff break: complementary to the embedding
        # similarity check above. When the mouth region mean abs diff
        # between consecutive aligned face crops exceeds this fraction,
        # treat the next frame as a continuity break -- this catches
        # face switches the embedding check misses (similar-looking
        # people, side faces). 0 disables. Default 0.10 is well above
        # same-person expression/pose diff (~0.02-0.05) and well below
        # cross-person diff (~0.10-0.30).
        lipsync_mouth_diff_break_threshold: float = 0.10,
        # Audio-energy prefilter: skip sustained silent runs before diffusion.
        silent_skip_enabled: bool = False,
        silent_rms_threshold: float = 0.003,
        silent_min_run_frames: int = 8,
        silent_pad_frames: int = 0,
        # Per-frame color transfer from generated to original (inside the
        # mask). 0 = off, 1 = full mean+std match. Default 0.60.
        color_match_strength: float = 0.60,
        # Unsharp-mask amount applied to the generated mouth region.
        # 0 = off, 1 = strong sharpen. Default 0.0.
        mouth_sharpen_strength: float = 0.0,
        # Original-detail restoration outside the central mouth-motion core.
        # 0 = off, 1 = strong reference detail. Default 0.65.
        mouth_detail_strength: float = 0.65,
        # --- CodeFormer face-restoration postprocess (added 2026-06) ---
        # When ``codeformer_restorer`` is provided and ``codeformer_enabled``
        # is True, the pipeline runs the released CodeFormer model on every
        # non-skipped aligned face crop right before pasting back to the
        # full video. This sharpens the synthesized mouth and helps recover
        # identity/edge detail that the diffusion inpainter tends to soften.
        # Set ``codeformer_enabled=False`` to skip entirely; pass a
        # :class:`CodeFormerRestorer` instance to actually invoke the model.
        # Default fidelity_weight 0.7 (was 0.5): the README's 0.5 is
        # balanced for real-degraded faces, but the inpainter's output
        # is *generated* content -- at w=0.5 the codebook path tends
        # to overwrite the lipsync result with a "more typical" face.
        # 0.7 keeps more of the input, at a small cost in sharpness.
        codeformer_enabled: bool = False,
        codeformer_fidelity_weight: float = 0.7,
        codeformer_adain: bool = True,
        codeformer_restorer=None,
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
        video_frames, faces, boxes, affine_matrices, skip_mask, continuity_break_mask, aligned_mouth_info = self.loop_video(
            whisper_chunks,
            video_frames,
            reference_embedding=reference_embedding,
            face_embedder=face_embedder,
            yaw_skip_threshold=yaw_skip_threshold,
            yaw_rate_skip_threshold=yaw_rate_skip_threshold,
            mouth_occlusion_skip_threshold=mouth_occlusion_skip_threshold,
            motion_blur_skip_threshold=motion_blur_skip_threshold,
            face_jump_center_threshold=face_jump_center_threshold,
            face_jump_scale_threshold=face_jump_scale_threshold,
            lipsync_continuity_max_center_shift=lipsync_continuity_max_center_shift,
            lipsync_continuity_max_scale_change=lipsync_continuity_max_scale_change,
            lipsync_mouth_diff_break_threshold=lipsync_mouth_diff_break_threshold,
            identity_similarity_threshold=identity_similarity_threshold,
            apply_identity_filter=apply_identity_filter,
            side_face_episode_pre_pad=side_face_episode_pre_pad,
            side_face_episode_post_pad=side_face_episode_post_pad,
            yaw_warn_threshold_ratio=yaw_warn_threshold_ratio,
            side_face_warn_min_run_frames=side_face_warn_min_run_frames,
        )
        silent_skip_mask = [False] * len(skip_mask)
        if silent_skip_enabled:
            silent_skip_mask = self._silent_frame_mask(
                audio_samples,
                frame_count=len(skip_mask),
                video_fps=float(video_fps),
                audio_sample_rate=audio_sample_rate,
                rms_threshold=silent_rms_threshold,
                min_run_frames=silent_min_run_frames,
                pad_frames=silent_pad_frames,
            )
            silent_skip_count = sum(silent_skip_mask)
            if silent_skip_count:
                skip_mask = [a or b for a, b in zip(skip_mask, silent_skip_mask)]
                continuity_break_mask = [a or b for a, b in zip(continuity_break_mask, silent_skip_mask)]
            logger.info(
                f"[LipSync] silent_skip={silent_skip_count}/{len(silent_skip_mask)} "
                f"(threshold={silent_rms_threshold}, min_run={silent_min_run_frames}, pad={silent_pad_frames})"
            )
        audio_motion_scales = [1.0] * len(skip_mask)
        if mouth_audio_adaptive_motion_enabled and len(skip_mask) > 0:
            frame_rms = []
            samples_per_frame = max(1, int(round(float(audio_sample_rate) / max(float(video_fps), 1e-6))))
            audio_float = audio_samples.detach().to(torch.float32)
            for frame_index in range(len(skip_mask)):
                start = frame_index * samples_per_frame
                end = min(int(audio_float.shape[0]), start + samples_per_frame)
                if start >= end:
                    frame_rms.append(0.0)
                else:
                    frame_rms.append(float(audio_float[start:end].pow(2).mean().sqrt().item()))
            rms_tensor = torch.tensor(frame_rms, dtype=torch.float32)
            if rms_tensor.numel() > 0 and float(rms_tensor.max().item()) > 0:
                lo = torch.quantile(rms_tensor, 0.20)
                hi = torch.quantile(rms_tensor, 0.90)
                denom = (hi - lo).clamp_min(1e-6)
                norm = ((rms_tensor - lo) / denom).clamp(0.0, 1.0)
                min_scale = min(float(mouth_audio_motion_min_scale), float(mouth_audio_motion_max_scale))
                max_scale = max(float(mouth_audio_motion_min_scale), float(mouth_audio_motion_max_scale))
                scales = min_scale + norm * (max_scale - min_scale)
                for frame_index, is_silent in enumerate(silent_skip_mask[: len(scales)]):
                    if is_silent:
                        scales[frame_index] = min_scale
                audio_motion_scales = [float(v) for v in scales.tolist()]
            logger.info(
                "[LipSync] audio_adaptive_motion=%s scale min=%.3f median=%.3f max=%.3f",
                mouth_audio_adaptive_motion_enabled,
                min(audio_motion_scales) if audio_motion_scales else 1.0,
                statistics.median(audio_motion_scales) if audio_motion_scales else 1.0,
                max(audio_motion_scales) if audio_motion_scales else 1.0,
            )
        logger.info(
            f"[LipSync] after loop_video: faces={faces.shape}, boxes={len(boxes)}, "
            f"affine_matrices={len(affine_matrices)}, apply_identity_filter={apply_identity_filter}, "
            f"skip_true={sum(skip_mask)}/{len(skip_mask)}, "
            f"continuity_break_true={sum(continuity_break_mask)}/{len(continuity_break_mask)}"
        )

        # State carried across batches for temporal EMA smoothing
        prev_face: Optional[torch.Tensor] = None
        prev_valid: bool = False
        prev_mouth_stabilized: Optional[torch.Tensor] = None
        prev_mouth_stabilized_valid: bool = False
        quality_fallback_count: int = 0
        quality_skip_mask: List[bool] = [False] * len(skip_mask)
        mouth_delta_values: List[float] = []
        mouth_stabilization_delta_skip_count = 0
        mouth_stabilization_applied_count = 0

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
        skipped_inference_batches = 0
        skipped_inference_frames = 0
        for i in tqdm.tqdm(range(num_inferences), desc="Doing inference..."):
            batch_start = i * num_frames
            batch_end = min((i + 1) * num_frames, len(whisper_chunks))
            inference_skip_mask = skip_mask[batch_start:batch_end]
            inference_continuity_break_mask = continuity_break_mask[batch_start:batch_end]
            batch_audio_motion_scales = audio_motion_scales[batch_start:batch_end]
            inference_faces = faces[batch_start:batch_end]
            if inference_skip_mask and all(inference_skip_mask):
                skipped_inference_batches += 1
                skipped_inference_frames += len(inference_skip_mask)
                prev_face = None
                prev_valid = False
                prev_mouth_stabilized = None
                prev_mouth_stabilized_valid = False
                synced_video_frames.append(inference_faces.to(device=device, dtype=weight_dtype))
                continue
            if self.unet.add_audio_layer:
                audio_embeds = torch.stack(whisper_chunks[batch_start:batch_end])
                audio_embeds = audio_embeds.to(device, dtype=weight_dtype)
                if do_classifier_free_guidance:
                    null_audio_embeds = torch.zeros_like(audio_embeds)
                    audio_embeds = torch.cat([null_audio_embeds, audio_embeds])
            else:
                audio_embeds = None
            latents = all_latents[:, :, batch_start:batch_end]
            batch_mouth_info = aligned_mouth_info[batch_start:batch_end]
            # Fixed U-shaped mask for the UNet (model was trained on this).
            ref_pixel_values, masked_pixel_values, masks = self.image_processor.prepare_masks_and_masked_images(
                inference_faces, affine_transform=False
            )
            # Dynamic mouth-centered mask for post-processing paste-back.
            # This uses per-frame mouth landmarks to create a tight mask
            # that only covers the mouth area, avoiding identity drift and
            # cheek ghosting while the model still sees the full U-shaped
            # inpainting region during inference.
            dynamic_region_masks = []
            for k_mi, mi in enumerate(batch_mouth_info):
                dm = self.generate_dynamic_mouth_mask(mi, height)
                dynamic_region_masks.append(dm)
            dynamic_region_mask_batch = torch.stack(dynamic_region_masks)  # (B, 1, H, W)

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
            # Use dynamic mouth-centered mask for paste-back so only the mouth
            # region takes generated content; cheeks/chin/forehead stay as
            # the original reference pixels. The model still saw the full
            # U-shaped inpainting region, so its output is coherent, but we
            # only preserve the mouth portion.
            # dynamic_region_mask_batch: 1=keep (reference), 0=inpaint (generated)
            # paste_surrounding_pixels_back expects: 1=generated, 0=reference
            # generated_region_mask: 1=generated region, 0=reference
            generated_region_mask = (1.0 - dynamic_region_mask_batch).to(device=device, dtype=decoded_latents.dtype)
            decoded_latents = self.paste_surrounding_pixels_back(
                decoded_latents, ref_pixel_values, generated_region_mask, device, weight_dtype
            )
            # Per-frame color match: align generated face stats to original
            # so the soft-mask boundary in restore_img doesn't reveal a
            # tone drift. Applied inside the mask region only.
            if color_match_strength > 0:
                decoded_latents = self._match_color_to_reference(
                    decoded_latents, ref_pixel_values, generated_region_mask, strength=color_match_strength
                )
            # Restore original high-frequency skin/detail around the lips
            # while protecting the central mouth aperture/contour where the
            # generated motion must remain dominant.
            if mouth_detail_strength > 0:
                decoded_latents = self._restore_reference_detail(
                    decoded_latents,
                    ref_pixel_values,
                    generated_region_mask,
                    strength=mouth_detail_strength,
                )
            # Mouth-region unsharp: recover high-frequency detail in the
            # generated mouth. Inpainter outputs tend to be slightly soft
            # because the prompt encourages plausible-but-not-sharp.
            if mouth_sharpen_strength > 0:
                decoded_latents = self._unsharp_mask(
                    decoded_latents, generated_region_mask, amount=mouth_sharpen_strength
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

            # Compute per-frame mouth center norm from aligned landmarks for
            # dynamic mask positioning in _mouth_core_mask calls below.
            first_center = None
            for mi in batch_mouth_info:
                if mi is not None:
                    first_center = (mi["center_x"] / height, mi["center_y"] / height)
                    break

            # Temporal EMA across face crops (cross-batch state via prev_face).
            # region_mask restricts the EMA to the inpaint area: outside the
            # mask the face stays as the raw inpainter+paste-back output, so
            # EMA can't smear previous frames' content into the original-face
            # area (root cause of "previous frame's different face glued in"
            # artifacts with the wider inpaint mask).
            if temporal_smoothing_enabled:
                current_mouth_motion = decoded_latents
                decoded_latents, prev_face, prev_valid = self._smooth_face_sequence(
                    decoded_latents,
                    prev_face=prev_face,
                    prev_valid=prev_valid,
                    inference_skip_mask=inference_skip_mask,
                    continuity_break_mask=inference_continuity_break_mask,
                    region_mask=generated_region_mask,
                )
                if mouth_motion_preserve_strength > 0:
                    mouth_motion_mask = self._mouth_core_mask(generated_region_mask, mouth_center_norm=first_center).to(
                        device=decoded_latents.device,
                        dtype=decoded_latents.dtype,
                    )
                    if mouth_motion_mask.dim() == 4 and mouth_motion_mask.shape[1] == 1:
                        mouth_motion_mask = mouth_motion_mask.expand(-1, 3, -1, -1)
                    audio_motion_scale_tensor = torch.tensor(
                        batch_audio_motion_scales,
                        device=decoded_latents.device,
                        dtype=decoded_latents.dtype,
                    ).view(-1, 1, 1, 1)
                    decoded_latents = decoded_latents + (
                        mouth_motion_mask
                        * mouth_motion_preserve_strength
                        * audio_motion_scale_tensor
                        * (current_mouth_motion - decoded_latents)
                    )
            if mouth_temporal_stabilization_strength > 0:
                mouth_stabilize_mask = self._mouth_core_mask(generated_region_mask, mouth_center_norm=first_center).to(
                    device=decoded_latents.device,
                    dtype=decoded_latents.dtype,
                )
                if mouth_stabilize_mask.dim() == 4 and mouth_stabilize_mask.shape[1] == 1:
                    mouth_stabilize_mask = mouth_stabilize_mask.expand(-1, 3, -1, -1)
                for k in range(decoded_latents.shape[0]):
                    if inference_skip_mask[k] or inference_continuity_break_mask[k]:
                        prev_mouth_stabilized = None
                        prev_mouth_stabilized_valid = False
                        if inference_skip_mask[k]:
                            continue
                    current_frame = decoded_latents[k]
                    if prev_mouth_stabilized_valid and prev_mouth_stabilized is not None:
                        prev_frame = prev_mouth_stabilized.to(current_frame.device)
                        effective_stabilization_strength = mouth_temporal_stabilization_strength
                        if mouth_temporal_stabilization_max_delta > 0:
                            mask_k = mouth_stabilize_mask[k]
                            mask_sum = mask_k.sum().clamp_min(1e-6)
                            mouth_delta = (
                                (current_frame - prev_frame).abs() * mask_k
                            ).sum() / mask_sum
                            mouth_delta_values.append(float(mouth_delta.item()))
                            if mouth_delta.item() > mouth_temporal_stabilization_max_delta:
                                mouth_stabilization_delta_skip_count += 1
                                prev_mouth_stabilized = current_frame.detach()
                                prev_mouth_stabilized_valid = True
                                continue
                            continuity = 1.0 - (
                                mouth_delta / mouth_temporal_stabilization_max_delta
                            ).clamp(0.0, 1.0)
                            continuity = continuity * continuity
                            effective_stabilization_strength = (
                                mouth_temporal_stabilization_strength * float(continuity.item())
                            )
                        stabilized = (
                            current_frame
                            + mouth_stabilize_mask[k]
                            * effective_stabilization_strength
                            * (prev_frame - current_frame)
                        )
                        decoded_latents[k] = stabilized
                        prev_mouth_stabilized = stabilized.detach()
                        if effective_stabilization_strength > 0:
                            mouth_stabilization_applied_count += 1
                    else:
                        prev_mouth_stabilized = current_frame.detach()
                    prev_mouth_stabilized_valid = True

            # Quality postfilter: flag frames whose generated mouth ROI is too
            # blurry to be worth showing. Checked AFTER paste/detail recovery.
            if quality_gate_enabled:
                B = decoded_latents.shape[0]
                base = i * num_frames
                gen_laps = []
                ref_laps = []
                for k in range(B):
                    if inference_skip_mask[k]:
                        continue  # already going to fall back to original
                    gen_lap = self._mouth_sharpness(decoded_latents[k])
                    ref_lap = self._mouth_sharpness(ref_pixel_values[k])
                    gen_laps.append(gen_lap)
                    ref_laps.append(ref_lap)
                    ratio = (gen_lap / ref_lap) if ref_lap > 0 else 1.0
                    if gen_lap < quality_min_laplacian * 0.25:
                        quality_skip_mask[base + k] = True
                        quality_fallback_count += 1
                        logger.info(
                            f"[Diag] mouth postfilter fallback batch{i} k{k}: gen_lap={gen_lap:.2f} < {quality_min_laplacian * 0.25:.2f}"
                        )
                        continue
                    if ref_lap >= quality_ref_min_laplacian and ratio < quality_min_sharpness_ratio:
                        quality_skip_mask[base + k] = True
                        quality_fallback_count += 1
                        logger.info(
                            f"[Diag] mouth postfilter fallback batch{i} k{k}: gen_lap={gen_lap:.2f} / ref_lap={ref_lap:.2f} = {ratio:.3f} < {quality_min_sharpness_ratio} (ref_lap >= {quality_ref_min_laplacian})"
                        )
                if i == 0 and gen_laps:
                    logger.info(
                        f"[Diag] batch0 mouth laplacian: gen min={min(gen_laps):.2f} max={max(gen_laps):.2f} median={statistics.median(gen_laps):.2f} "
                        f"ref min={min(ref_laps):.2f} max={max(ref_laps):.2f} median={statistics.median(ref_laps):.2f}"
                    )

            synced_video_frames.append(decoded_latents)

        logger.info(f"[LipSync] decoded {len(synced_video_frames)} batches, restoring video...")
        pre_skip = sum(skip_mask)
        quality_candidate_count = len(skip_mask) - pre_skip
        if (
            quality_gate_enabled
            and quality_max_fallback_ratio > 0
            and quality_candidate_count > 0
            and sum(quality_skip_mask) / quality_candidate_count > quality_max_fallback_ratio
        ):
            logger.warning(
                f"[LipSync] quality gate skipped {sum(quality_skip_mask)}/{quality_candidate_count} "
                f"candidate frames (> {quality_max_fallback_ratio:.2f}); disabling quality fallback for this run"
            )
            quality_skip_mask = [False] * len(quality_skip_mask)
            quality_fallback_count = 0
        # OR-merge the quality postfilter with the original skip_mask
        effective_skip_mask = [a or b for a, b in zip(skip_mask, quality_skip_mask)]
        quality_skip = sum(quality_skip_mask)
        effective_skip = sum(effective_skip_mask)
        effective_generated = len(effective_skip_mask) - effective_skip
        logger.info(
            f"[Diag] skip summary: pre(loop_video)={pre_skip} quality_postfilter={quality_skip} "
            f"effective_total={effective_skip} generated={effective_generated} / {len(skip_mask)} "
            f"inference_short_circuit_batches={skipped_inference_batches} "
            f"inference_short_circuit_frames={skipped_inference_frames}"
        )
        if quality_fallback_count:
            logger.info(f"[LipSync] quality_fallback_frames={quality_fallback_count} / {len(skip_mask)}")
        all_faces = torch.cat(synced_video_frames, dim=0)
        # CodeFormer postprocess. We hand the restorer the *aligned* face
        # crops (512x512, [-1, 1]) rather than the full-frame output, for
        # three reasons:
        #   * CodeFormer is trained on aligned faces; off-aligned inputs
        #     produce visible edge artefacts.
        #   * Background and clothing don't get sharpened, which would
        #     otherwise reveal the postprocess step in the seams between
        #     the restored face and the unchanged body.
        #   * The restored face still goes through ``restore_img`` below,
        #     so the existing paste-back, box-resize and affine math
        #     apply unchanged.
        # Frames marked skipped by the pipeline are passed through
        # untouched (the restorer handles that internally) so the source
        # video is never re-sharpened on top of itself.
        self._last_codeformer_stats = {
            "enabled": bool(codeformer_enabled),
            "loaded": False,
            "frames_total": int(all_faces.shape[0]),
            "frames_enhanced": 0,
            "frames_skipped_by_pipeline": int(sum(effective_skip_mask)),
            "elapsed_seconds": 0.0,
            "fidelity_weight": float(codeformer_fidelity_weight),
            "batch_size": 0,
            "checkpoint_path": "",
            "error": "",
        }
        if codeformer_enabled:
            if codeformer_restorer is None:
                logger.warning(
                    "[LipSync] codeformer_enabled=True but no restorer was passed; "
                    "skipping CodeFormer postprocess"
                )
                self._last_codeformer_stats["error"] = "no restorer"
            else:
                logger.info(
                    f"[LipSync] CodeFormer postprocess starting: faces={all_faces.shape}, "
                    f"fidelity_weight={codeformer_fidelity_weight}, "
                    f"frames_to_enhance={all_faces.shape[0] - int(sum(effective_skip_mask))}"
                )
                all_faces, cf_stats = codeformer_restorer.restore_faces(
                    all_faces,
                    skip_mask=effective_skip_mask,
                    fidelity_weight=codeformer_fidelity_weight,
                    adain=codeformer_adain,
                )
                self._last_codeformer_stats = cf_stats.as_dict()
        synced_video_frames = self.restore_video(all_faces, video_frames, boxes, affine_matrices, effective_skip_mask)
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
            "pre_skip_frames": pre_skip,
            "quality_skip_frames": quality_skip,
            "effective_skip_frames": effective_skip,
            "effective_generated_frames": effective_generated,
            "silent_skip_frames": sum(silent_skip_mask),
            "silent_skip_enabled": silent_skip_enabled,
            "silent_rms_threshold": silent_rms_threshold,
            "silent_min_run_frames": silent_min_run_frames,
            "silent_pad_frames": silent_pad_frames,
            "skipped_inference_batches": skipped_inference_batches,
            "skipped_inference_frames": skipped_inference_frames,
            "identity_skip_count": getattr(self, "_last_identity_skip_count", 0),
            "yaw_skip_count": getattr(self, "_last_yaw_skip_count", 0),
            "yaw_rate_skip_count": getattr(self, "_last_yaw_rate_skip_count", 0),
            "mouth_occlusion_skip_count": getattr(self, "_last_mouth_occlusion_skip_count", 0),
            "motion_blur_skip_count": getattr(self, "_last_motion_blur_skip_count", 0),
            "face_jump_skip_count": getattr(self, "_last_face_jump_skip_count", 0),
            "side_face_episode_extra_skip_count": getattr(
                self, "_last_side_face_episode_extra_skip_count", 0
            ),
            "side_face_warn_run_skip_count": getattr(
                self, "_last_side_face_warn_run_skip_count", 0
            ),
            "yaw_skip_threshold": yaw_skip_threshold,
            "yaw_rate_skip_threshold": yaw_rate_skip_threshold,
            "yaw_warn_threshold_ratio": yaw_warn_threshold_ratio,
            "side_face_warn_min_run_frames": side_face_warn_min_run_frames,
            "mouth_occlusion_skip_threshold": mouth_occlusion_skip_threshold,
            "motion_blur_skip_threshold": motion_blur_skip_threshold,
            "face_jump_center_threshold": face_jump_center_threshold,
            "face_jump_scale_threshold": face_jump_scale_threshold,
            "lipsync_continuity_max_center_shift": lipsync_continuity_max_center_shift,
            "lipsync_continuity_max_scale_change": lipsync_continuity_max_scale_change,
            "lipsync_mouth_diff_break_threshold": lipsync_mouth_diff_break_threshold,
            "temporal_diff_break_count": getattr(self, "_last_temporal_diff_break_count", 0),
            "identity_similarity_threshold": identity_similarity_threshold,
            "identity_similarity": getattr(
                self,
                "_last_identity_similarity_stats",
                {"min": 0.0, "median": 0.0, "max": 0.0},
            ),
            "temporal_smoothing_enabled": temporal_smoothing_enabled,
            "mouth_motion_preserve_strength": mouth_motion_preserve_strength,
            "mouth_temporal_stabilization_strength": mouth_temporal_stabilization_strength,
            "mouth_temporal_stabilization_max_delta": mouth_temporal_stabilization_max_delta,
            "mouth_temporal": {
                "delta_min": float(min(mouth_delta_values)) if mouth_delta_values else 0.0,
                "delta_median": float(statistics.median(mouth_delta_values)) if mouth_delta_values else 0.0,
                "delta_max": float(max(mouth_delta_values)) if mouth_delta_values else 0.0,
                "delta_skip_frames": int(mouth_stabilization_delta_skip_count),
                "stabilized_frames": int(mouth_stabilization_applied_count),
                "audio_motion_min_scale": float(min(audio_motion_scales)) if audio_motion_scales else 1.0,
                "audio_motion_median_scale": float(statistics.median(audio_motion_scales)) if audio_motion_scales else 1.0,
                "audio_motion_max_scale": float(max(audio_motion_scales)) if audio_motion_scales else 1.0,
            },
            "quality_gate_enabled": quality_gate_enabled,
            "quality_ref_min_laplacian": quality_ref_min_laplacian,
            "quality_max_fallback_ratio": quality_max_fallback_ratio,
            "color_match_strength": color_match_strength,
            "mouth_detail_strength": mouth_detail_strength,
            "mouth_sharpen_strength": mouth_sharpen_strength,
            "codeformer": self._last_codeformer_stats,
        }

        sf.write(os.path.join(temp_dir, "audio.wav"), audio_samples, audio_sample_rate)

        command = f"ffmpeg -y -loglevel error -nostdin -i {os.path.join(temp_dir, 'video.mp4')} -i {os.path.join(temp_dir, 'audio.wav')} -c:v libx264 -crf 18 -c:a aac -q:v 0 -q:a 0 {video_out_path}"
        subprocess.run(command, shell=True)
