"""Fine-tuning presets and config generation."""
from typing import Any, Dict, Tuple
from pathlib import Path

from latentsync.finetune import ASSETS_DIR, FINETUNE_BASE_DIR, SYNCNET_CONFIG_DIR

# Default preset used when the Fine-tune Studio first loads. LoRA is the safest
# starting point: low VRAM (~12-15GB), fast iteration, and minimal risk of
# degrading the base model.
DEFAULT_PRESET_NAME = "Stage 2 LoRA (256, 12-15GB)"

PRESETS: Dict[str, Dict[str, Any]] = {
    "Stage 2 LoRA (256, 12-15GB)": {
        "config_file": "configs/unet/stage2.yaml",
        "resume_ckpt": "checkpoints/latentsync_unet.pt",
        "batch_size": 1,
        "num_frames": 16,
        "resolution": 256,
        "learning_rate": 5e-5,
        "use_motion_module": True,
        "pixel_space_supervise": True,
        "use_syncnet": True,
        "sync_loss_weight": 0.05,
        "perceptual_loss_weight": 0.1,
        "recon_loss_weight": 1.0,
        "trepa_loss_weight": 10.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": True,
        "mask_image_path": "latentsync/utils/mask.png",
        "description": (
            "⚪ **通用 baseline / 首次微调首选**。只训练 LoRA adapter (~10MB)，"
            "不改动 base UNet，显存 12-15GB，试错成本最低。"
        ),
        "lora": {
            "enabled": True,
            "rank": 32,
            "alpha": 64,
            "dropout": 0.05,
            "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
            "qlora": False,
        },
        "freeze_attn2": False,
    },
    "🎯 Badcase Fix (侧脸+运动, LoRA, 12-15GB)": {
        "config_file": "configs/unet/stage2.yaml",
        "resume_ckpt": "checkpoints/latentsync_unet.pt",
        "batch_size": 1,
        "num_frames": 16,
        "resolution": 256,
        "learning_rate": 5e-5,
        "use_motion_module": True,
        "pixel_space_supervise": True,
        "use_syncnet": True,
        "sync_loss_weight": 0.02,
        "perceptual_loss_weight": 0.15,
        "recon_loss_weight": 1.0,
        "trepa_loss_weight": 10.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": True,
        "mask_image_path": "latentsync/utils/mask.png",
        "save_ckpt_steps": 500,
        "max_train_steps": 3000,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 500,
        "description": (
            "🟢 **推荐 — 内容型 badcase**\n"
            "resolution=256 显存友好,训练快;"
            "LoRA rank=64, lr=5e-5, sync_loss=0.02, cosine+500 warmup, 每 500 步存 ckpt。\n"
            "适用:嘴型/audio 同步、嘴糊、paste-back 外溢、侧脸唇形不同步。\n"
            "注意:SyncNet 只支持 16 帧,所以不要改 num_frames。\n"
            "freeze_attn2=True 保护基础唇音同步能力。\n"
            "训练前请确认数据集已做人脸对齐(init_finetune_dataset.py --align --align-resolution 256)。"
        ),
        "lora": {
            "enabled": True,
            "rank": 64,
            "alpha": 128,
            "dropout": 0.10,
            "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
            "qlora": False,
        },
        "freeze_attn2": True,
    },
    "🧩 Structural Fix (LoRA + conv, 18-22GB)": {
        "config_file": "configs/unet/stage2.yaml",
        "resume_ckpt": "checkpoints/latentsync_unet.pt",
        "batch_size": 1,
        "num_frames": 16,
        "resolution": 256,
        "learning_rate": 3e-5,
        "use_motion_module": True,
        "pixel_space_supervise": True,
        "use_syncnet": True,
        "sync_loss_weight": 0.05,
        "perceptual_loss_weight": 0.20,
        "recon_loss_weight": 1.0,
        "trepa_loss_weight": 10.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": True,
        "mask_image_path": "latentsync/utils/mask.png",
        "save_ckpt_steps": 1000,
        "max_train_steps": 30000,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 300,
        "description": (
            "🔴 **推荐 — 结构性 badcase**\n"
            "LoRA target 加 conv1/conv2/conv_shortcut/proj_in/proj_out/conv_in/conv_out\n"
            "(11 项,~25-30M params,3x capacity,够 cover 侧脸几何错位)。\n"
            "VRAM 占用 18-22GB(Lower lr 防过拟合; perceptual↑保细节)。\n"
            "内容型嘴错见 🎯 Badcase Fix; 短剧多说话人见 🎬 Short Drama; 通用 baseline 见 ⚪ Stage 2 LoRA。"
        ),
        "lora": {
            "enabled": True,
            "rank": 16,
            "alpha": 32,
            "dropout": 0.10,
            "target_modules": [
                # attention projections (latentsync/models/attention.py)
                "to_q", "to_k", "to_v", "to_out.0",
                # Resnet convs (latentsync/models/resnet.py)
                "conv1", "conv2", "conv_shortcut",
                # Attention 1×1 conv re-mappers (latentsync/models/attention.py)
                "proj_in", "proj_out",
                # UNet input/output gates (latentsync/models/unet.py)
                "conv_in", "conv_out",
            ],
            "qlora": False,
        },
        "freeze_attn2": True,
    },
    "💋 Side-Face Lip Quality (LoRA+conv, 18-22GB)": {
        "config_file": "configs/unet/stage2.yaml",
        "resume_ckpt": "checkpoints/latentsync_unet.pt",
        "batch_size": 1,
        # SyncNet 只支持 16 帧,长时序上下文需用 motion module 补偿
        "num_frames": 16,
        "resolution": 256,
        "learning_rate": 3e-5,
        "use_motion_module": True,
        "pixel_space_supervise": True,
        "use_syncnet": True,
        # 侧脸时唇被遮 ~30-50%,必须强推 audio-driven 想象
        "sync_loss_weight": 0.15,
        # 唇部纹理在侧脸时 shading 不同,提权
        "perceptual_loss_weight": 0.25,
        "recon_loss_weight": 1.0,
        # 稍降 TREPA 让唇部允许更多形状变化(闭嘴 → 张嘴)
        "trepa_loss_weight": 8.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": True,
        "mask_image_path": "latentsync/utils/mask.png",
        "save_ckpt_steps": 500,
        "max_train_steps": 30000,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 300,
        "description": (
            "💋 **推荐 — 侧脸唇形质量 (yaw 15-30°)**\n"
            "针对侧脸时嘴部遮挡 (~30-50%) + 唇纹理变化的双重挑战。\n"
            "LoRA target 加 conv(11 项,同 Structural),rank=48 留更多 capacity 学唇形。\n"
            "sync_loss=0.15 强推唇音同步(并改善张嘴幅度偏小); perceptual=0.25 锐化唇部纹理;\n"
            "num_frames=16 (SyncNet 只支持 16 帧),motion module 补偿时序连贯。\n"
            "数据:用 celebv_hq_head_talk_side_big_mouth recipe(p95≥0.22 大嘴筛选)\n"
            "改善微调后张嘴幅度偏小;侧脸召回用 celebv_hq_head_talk_side。"
        ),
        "lora": {
            "enabled": True,
            "rank": 48,
            "alpha": 96,
            "dropout": 0.10,
            "target_modules": [
                "to_q", "to_k", "to_v", "to_out.0",
                "conv1", "conv2", "conv_shortcut",
                "proj_in", "proj_out",
                "conv_in", "conv_out",
            ],
            "qlora": False,
        },
        "freeze_attn2": True,
    },
    "🎬 Short Drama (LoRA+conv, 多说话人, 18-22GB)": {
        "config_file": "configs/unet/stage2.yaml",
        "resume_ckpt": "checkpoints/latentsync_unet.pt",
        "batch_size": 1,
        # Short drama 单段短(5-15s),长上下文稀释信号 — 回到 16
        "num_frames": 16,
        "resolution": 256,
        "learning_rate": 3e-5,
        "use_motion_module": True,
        "pixel_space_supervise": True,
        "use_syncnet": True,
        # 短剧容错低,口型必须跟得紧
        "sync_loss_weight": 0.12,
        "perceptual_loss_weight": 0.20,
        "recon_loss_weight": 1.0,
        "trepa_loss_weight": 10.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": True,
        "mask_image_path": "latentsync/utils/mask.png",
        # 短剧 ckpt 多 — 频繁切 → 多存点
        "save_ckpt_steps": 500,
        "max_train_steps": 25000,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 300,
        "description": (
            "🟣 **推荐 — 短剧 (多说话人/频繁切场景)**\n"
            "LoRA target 加 conv(同 Structural Fix 11 项),\n"
            "sync_loss=0.12 容错低,save_ckpt_steps=500 短段多存。\n"
            "数据准备:用 tools/preprocess_short_drama.py 把剧按场景切,\n"
            "再走 curate_finetune_samples.py 分桶。\n"
            "通用 baseline 见 ⚪ Stage 2 LoRA,单人 badcase 见 🟢 🎯 Badcase Fix。"
        ),
        "lora": {
            "enabled": True,
            "rank": 16,
            "alpha": 32,
            "dropout": 0.10,
            "target_modules": [
                # attention projections
                "to_q", "to_k", "to_v", "to_out.0",
                # Resnet convs
                "conv1", "conv2", "conv_shortcut",
                # Attention 1×1 conv re-mappers
                "proj_in", "proj_out",
                # UNet input/output gates
                "conv_in", "conv_out",
            ],
            "qlora": False,
        },
        "freeze_attn2": True,
    },
    "🎭 Short Drama 唇形真实感 (LoRA+conv, 18-22GB)": {
        "config_file": "configs/unet/stage2.yaml",
        "resume_ckpt": "checkpoints/latentsync_unet.pt",
        "batch_size": 1,
        # SyncNet 只支持 16 帧
        "num_frames": 16,
        "resolution": 256,
        "learning_rate": 3e-5,
        "use_motion_module": True,
        "pixel_space_supervise": True,
        "use_syncnet": True,
        # 嘴特别小/唇薄:强推 audio-driven 嘴型,减少"开不开都行"的模糊地带
        "sync_loss_weight": 0.15,
        # 唇部质感(唇薄/糊):提权锐化嘴部纹理
        "perceptual_loss_weight": 0.25,
        "recon_loss_weight": 1.0,
        # 快切场景:帧间一致性(降到 8,给嘴部形状变化留空间; v2v 高运动片段
        # 上 TREPA 过高会压出嘴糊/拖影, 同 Side-Face preset 的取舍)
        "trepa_loss_weight": 8.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": True,
        "mask_image_path": "latentsync/utils/mask.png",
        "save_ckpt_steps": 500,
        "max_train_steps": 10000,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 300,
        "description": (
            "🎭 **推荐 — 短剧唇形真实感 (嘴小/唇薄/侧脸糊/快切场景)**\n"
            "嘴特别小+唇薄 → sync_loss=0.15 强推 audio-driven 嘴型 + perceptual=0.25 锐化唇部质感;\n"
            "侧脸糊 → LoRA target 加 conv(11 项, rank=48, 同 Side-Face) 补结构容量;\n"
            "快速切场景 → TREPA=8 保帧间一致又不压嘴部动势(训练数据必须按镜头切, 不跨 cut);\n"
            "数据:短剧 shots(tools/preprocess_short_drama.py, 自动丢无人脸镜头)\n"
            "+ celebv_hq_head_talk_side_big_mouth / huge_mouth 大嘴筛选 recipe 混合;\n"
            "对齐分辨率 256 (init_finetune_dataset.py --align-resolution 256)。\n"
            "通用短剧版见 🎬 Short Drama; 画质优先见 🎨 Lip Forcing。"
        ),
        "lora": {
            "enabled": True,
            "rank": 48,
            "alpha": 96,
            "dropout": 0.10,
            "target_modules": [
                "to_q", "to_k", "to_v", "to_out.0",
                "conv1", "conv2", "conv_shortcut",
                "proj_in", "proj_out",
                "conv_in", "conv_out",
            ],
            "qlora": False,
        },
        "freeze_attn2": True,
    },
    "🎨 Lip Forcing 风格 (保真优先 LoRA, 512)": {
        "config_file": "configs/unet/stage2_512.yaml",
        "resume_ckpt": "checkpoints/latentsync_unet.pt",
        "batch_size": 1,
        "num_frames": 16,
        "resolution": 512,
        "learning_rate": 5e-5,
        "use_motion_module": True,
        "pixel_space_supervise": True,
        "use_syncnet": True,
        # 论文 fidelity–sync tradeoff 的保真一侧:放松同步约束换画质
        "sync_loss_weight": 0.02,
        # 提真实感/抗糊
        "perceptual_loss_weight": 0.25,
        "recon_loss_weight": 1.0,
        # TREPA 保帧间时序一致性
        "trepa_loss_weight": 10.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": True,
        "mask_image_path": "latentsync/utils/mask2.png",
        "save_ckpt_steps": 500,
        "max_train_steps": 10000,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 500,
        "description": (
            "🎨 **论文配方 — 保真优先(放同步换画质, 512)** (参考 Lip Forcing, arXiv:2606.11180)\n"
            "论文 fidelity–sync tradeoff:放松同步约束 → FID/FVD/LPIPS 更优(更真实、更连贯、不糊)。\n"
            "本预设落在 reference-leaning 一侧:sync_loss=0.02 弱同步监督;\n"
            "perceptual=0.25 提真实感抗糊;TREPA=10 保帧间一致性;\n"
            "freeze_attn2 锁住基础唇音通路,防止放松监督后同步能力崩塌。\n"
            "适用:画质/一致性优先、可容忍同步略松的内容"
            "(观众对脸崩/不自然的容忍度远低于毫秒级口型偏差,预算押保真一侧是务实操作点)。\n"
            "512 说明:用 stage2_512 配置 + mask2.png,显存估计 ~30-40GB(512 全量参考 ~55GB);\n"
            "数据对齐分辨率必须匹配:init_finetune_dataset.py --align-resolution 512。\n"
            "数据:配合「Lip Forcing 论文混合数据」预设(VoxCeleb2 多样性 + HDTF/Hallo3 高清干净音)。\n"
            "注意:论文的 DMD 蒸馏 / 因果 student / 两步推理是架构级改动,本预设不含。"
        ),
        "lora": {
            "enabled": True,
            "rank": 64,
            "alpha": 128,
            "dropout": 0.10,
            "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
            "qlora": False,
        },
        "freeze_attn2": True,
    },
    "Stage 2 QLoRA (256, 8-10GB)": {
        "config_file": "configs/unet/stage2.yaml",
        "resume_ckpt": "checkpoints/latentsync_unet.pt",
        "batch_size": 1,
        "num_frames": 16,
        "resolution": 256,
        "learning_rate": 2e-4,
        "use_motion_module": True,
        "pixel_space_supervise": True,
        "use_syncnet": True,
        "sync_loss_weight": 0.05,
        "perceptual_loss_weight": 0.1,
        "recon_loss_weight": 1.0,
        "trepa_loss_weight": 10.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": True,
        "mask_image_path": "latentsync/utils/mask.png",
        "description": "QLoRA：base UNet 4-bit 量化 + LoRA。需 peft + bitsandbytes。",
        "lora": {
            "enabled": True,
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
            "qlora": True,
        },
    },
    "Stage 2 (256, 全量训练)": {
        "config_file": "configs/unet/stage2.yaml",
        "resume_ckpt": "checkpoints/latentsync_unet.pt",
        "batch_size": 1,
        "num_frames": 16,
        "resolution": 256,
        "learning_rate": 1e-5,
        "use_motion_module": True,
        "pixel_space_supervise": True,
        "use_syncnet": True,
        "sync_loss_weight": 0.05,
        "perceptual_loss_weight": 0.1,
        "recon_loss_weight": 1.0,
        "trepa_loss_weight": 10.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": True,
        "mask_image_path": "latentsync/utils/mask.png",
        "description": "全量训练 Stage 2（motion_modules + attentions）。~30GB VRAM。",
    },
    "Stage 2 Efficient (256, 20GB)": {
        "config_file": "configs/unet/stage2_efficient.yaml",
        "resume_ckpt": "checkpoints/latentsync_unet.pt",
        "batch_size": 1,
        "num_frames": 16,
        "resolution": 256,
        "learning_rate": 1e-5,
        "use_motion_module": True,
        "pixel_space_supervise": True,
        "use_syncnet": True,
        "sync_loss_weight": 0.05,
        "perceptual_loss_weight": 0.1,
        "recon_loss_weight": 1.0,
        "trepa_loss_weight": 0.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": True,
        "mask_image_path": "latentsync/utils/mask.png",
        "description": "关 TREPA，只训 motion + attn2。~20GB VRAM。",
    },
    "Stage 2 512 (高分辨率)": {
        "config_file": "configs/unet/stage2_512.yaml",
        "resume_ckpt": "checkpoints/latentsync_unet.pt",
        "batch_size": 1,
        "num_frames": 16,
        "resolution": 512,
        "learning_rate": 1e-5,
        "use_motion_module": True,
        "pixel_space_supervise": True,
        "use_syncnet": True,
        "sync_loss_weight": 0.05,
        "perceptual_loss_weight": 0.1,
        "recon_loss_weight": 1.0,
        "trepa_loss_weight": 10.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": True,
        "mask_image_path": "latentsync/utils/mask2.png",
        "description": "512 分辨率，需对应 mask2.png。~55GB VRAM。",
    },
    "Stage 1 (256, 全量训练)": {
        "config_file": "configs/unet/stage1.yaml",
        "resume_ckpt": "checkpoints/latentsync_unet.pt",
        "batch_size": 1,
        "num_frames": 16,
        "resolution": 256,
        "learning_rate": 1e-5,
        "use_motion_module": False,
        "pixel_space_supervise": False,
        "use_syncnet": False,
        "sync_loss_weight": 0.05,
        "perceptual_loss_weight": 0.1,
        "recon_loss_weight": 1.0,
        "trepa_loss_weight": 10.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": True,
        "mask_image_path": "latentsync/utils/mask.png",
        "description": "Stage 1：学视觉重建，不加 sync / LPIPS / TREPA。~23GB VRAM。",
    },
    "SyncNet 训练": {
        "config_file": "configs/syncnet/syncnet_16_pixel_attn.yaml",
        "resume_ckpt": "",
        "batch_size": 256,
        "num_frames": 16,
        "resolution": 256,
        "learning_rate": 1e-5,
        "use_motion_module": False,
        "pixel_space_supervise": False,
        "use_syncnet": False,
        "sync_loss_weight": 0.0,
        "perceptual_loss_weight": 0.0,
        "recon_loss_weight": 0.0,
        "trepa_loss_weight": 0.0,
        "mixed_precision_training": True,
        "enable_gradient_checkpointing": False,
        "mask_image_path": "latentsync/utils/mask.png",
        "description": "训练 StableSyncNet。batch 建议 ≥256，最好 1024。",
    },
}
DATASET_PRESETS: Dict[str, Dict[str, str]] = {
    "assets 演示数据 (3 videos，可完整跑通)": {
        "train_data_dir": "assets",
        "train_fileslist": "data/demo_fileslist.txt",
        "val_video_path": "assets/demo1_video.mp4",
        "val_audio_path": "assets/demo1_audio.wav",
    },
    "assets 演示数据 (仅目录，自动生成 fileslist)": {
        "train_data_dir": "assets",
        "train_fileslist": "",
        "val_video_path": "assets/demo1_video.mp4",
        "val_audio_path": "assets/demo1_audio.wav",
    },
    "preprocess/high_visual_quality (示例路径)": {
        "train_data_dir": "preprocess/high_visual_quality",
        "train_fileslist": "preprocess/high_visual_quality/fileslist.txt",
        "val_video_path": "assets/demo1_video.mp4",
        "val_audio_path": "assets/demo1_audio.wav",
    },
    "data/train (示例路径)": {
        "train_data_dir": "data/train",
        "train_fileslist": "data/train/fileslist.txt",
        "val_video_path": "assets/demo1_video.mp4",
        "val_audio_path": "assets/demo1_audio.wav",
    },
    "Lip Forcing 论文混合数据 (VoxCeleb2+HDTF+Hallo3, 目录递归示例)": {
        # 把 voxceleb2/ hdtf/ hallo3/ 软链或放入该目录,unet_dataset 递归收集 *.mp4
        "train_data_dir": "data/lip_forcing_mix",
        "train_fileslist": "",
        "val_video_path": "assets/demo1_video.mp4",
        "val_audio_path": "assets/demo1_audio.wav",
    },
    "Lip Forcing 论文混合数据 (VoxCeleb2+HDTF+Hallo3, fileslist 示例)": {
        "train_data_dir": "",
        "train_fileslist": "data/lip_forcing_mix_fileslist.txt",
        "val_video_path": "assets/demo1_video.mp4",
        "val_audio_path": "assets/demo1_audio.wav",
    },
    "短剧唇形混合数据 (short_drama shots+大嘴 recipe, fileslist 示例)": {
        # 短剧 shots 对齐目录 + celebv_hq_head_talk_*_big_mouth 对齐目录,
        # 两份 fileslist 直接 cat 合并即可(UNetDataset 按行读绝对路径)
        "train_data_dir": "",
        "train_fileslist": "data/short_drama_lip_v1_fileslist.txt",
        "val_video_path": "assets/demo1_video.mp4",
        "val_audio_path": "assets/demo1_audio.wav",
    },
}
def build_config_from_form(
    preset_name: str,
    train_data_dir: str,
    train_fileslist: str,
    val_video_path: str,
    val_audio_path: str,
    resume_ckpt: str,
    batch_size: int,
    num_frames: int,
    resolution: int,
    learning_rate: float,
    use_motion_module: bool,
    pixel_space_supervise: bool,
    use_syncnet: bool,
    sync_loss_weight: float,
    perceptual_loss_weight: float,
    recon_loss_weight: float,
    trepa_loss_weight: float,
    mixed_precision_training: bool,
    enable_gradient_checkpointing: bool,
    mask_image_path: str,
    save_ckpt_steps: int,
    max_train_steps: int,
    num_workers: int,
    train_output_dir: str,
    freeze_attn2: bool,
    val_inference_steps: int,
    val_guidance_scale: float,
    val_seed: int,
    lr_scheduler: str,
    lr_warmup_steps: int,
) -> Dict[str, Any]:
    """Merge user-form values with the chosen preset's defaults."""
    preset = PRESETS[preset_name]

    # Resolve train_output_dir relative to FINETUNE_BASE_DIR so the training
    # script (which runs with cwd=REPO_ROOT) puts outputs where the UI expects.
    train_output_dir = (train_output_dir or "").strip()
    if train_output_dir:
        p = Path(train_output_dir)
        if not p.is_absolute():
            train_output_dir = str(FINETUNE_BASE_DIR / p)
    else:
        train_output_dir = str(FINETUNE_BASE_DIR / "unet")

    cfg: Dict[str, Any] = {
        "data": {
            "train_data_dir": train_data_dir or "",
            "train_fileslist": train_fileslist or "",
            "val_video_path": val_video_path or str(ASSETS_DIR / "demo1_video.mp4"),
            "val_audio_path": val_audio_path or str(ASSETS_DIR / "demo1_audio.wav"),
            "audio_embeds_cache_dir": str(FINETUNE_BASE_DIR / "audio_embeds_cache"),
            "audio_mel_cache_dir": str(FINETUNE_BASE_DIR / "audio_mel_cache"),
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "num_frames": int(num_frames),
            "resolution": int(resolution),
            "val_resolution": 512,
            "mask_image_path": mask_image_path,
            "audio_sample_rate": 16000,
            "video_fps": 25,
            "audio_feat_length": [2, 2],
            "train_output_dir": train_output_dir,
            # train_unet.py loads this to get the StableSyncNet checkpoint path.
            # It must point to a syncnet config, not the UNet config.
            "syncnet_config_path": str(SYNCNET_CONFIG_DIR / "syncnet_16_pixel_attn.yaml"),
        },
        "ckpt": {
            "resume_ckpt_path": resume_ckpt or preset["resume_ckpt"],
            "save_ckpt_steps": int(save_ckpt_steps),
        },
        "run": {
            "pixel_space_supervise": bool(pixel_space_supervise),
            "use_syncnet": bool(use_syncnet),
            "sync_loss_weight": float(sync_loss_weight),
            "perceptual_loss_weight": float(perceptual_loss_weight),
            "recon_loss_weight": float(recon_loss_weight),
            "trepa_loss_weight": float(trepa_loss_weight),
            "guidance_scale": float(val_guidance_scale),
            "inference_steps": int(val_inference_steps),
            "seed": int(val_seed),
            "use_mixed_noise": True,
            "mixed_noise_alpha": 1,
            "mixed_precision_training": bool(mixed_precision_training),
            "enable_gradient_checkpointing": bool(enable_gradient_checkpointing),
            "max_train_steps": int(max_train_steps),
            "max_train_epochs": -1,
            "trainable_modules": ["motion_modules.", "attentions."] if use_motion_module else [],
        },
        "optimizer": {
            "lr": float(learning_rate),
            "scale_lr": False,
            "max_grad_norm": 1.0,
            "lr_scheduler": lr_scheduler,
            "lr_warmup_steps": int(lr_warmup_steps),
        },
        "model": {
            "act_fn": "silu",
            "add_audio_layer": True,
            "attention_head_dim": 8,
            "block_out_channels": [320, 640, 1280, 1280],
            "center_input_sample": False,
            "cross_attention_dim": 384,
            "down_block_types": [
                "CrossAttnDownBlock3D",
                "CrossAttnDownBlock3D",
                "CrossAttnDownBlock3D",
                "DownBlock3D",
            ],
            "mid_block_type": "UNetMidBlock3DCrossAttn",
            "up_block_types": [
                "UpBlock3D",
                "CrossAttnUpBlock3D",
                "CrossAttnUpBlock3D",
                "CrossAttnUpBlock3D",
            ],
            "downsample_padding": 1,
            "flip_sin_to_cos": True,
            "freq_shift": 0,
            "in_channels": 13,
            "layers_per_block": 2,
            "mid_block_scale_factor": 1,
            "norm_eps": 1e-5,
            "norm_num_groups": 32,
            "out_channels": 4,
            "sample_size": 64,
            "resnet_time_scale_shift": "default",
            "use_motion_module": bool(use_motion_module),
            "motion_module_resolutions": [1, 2, 4, 8],
            "motion_module_mid_block": False,
            "motion_module_decoder_only": False,
            "motion_module_type": "Vanilla",
            "motion_module_kwargs": {
                "num_attention_heads": 8,
                "num_transformer_block": 1,
                "attention_block_types": ["Temporal_Self", "Temporal_Self"],
                "temporal_position_encoding": True,
                "temporal_position_encoding_max_len": 24,
                "temporal_attention_dim_div": 1,
                "zero_initialize": True,
            },
        },
    }

    # If the preset carries a LoRA block, propagate it into the generated
    # config. train_unet_lora.py will pick it up; train_unet.py will
    # simply ignore it (it has its own trainable_modules logic).
    if "lora" in preset:
        cfg["lora"] = dict(preset["lora"])
    else:
        # Default-off block so users can hand-edit the generated yaml
        cfg["lora"] = {
            "enabled": False,
            "rank": 16,
            "alpha": 32,
            "dropout": 0.05,
            "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
            "qlora": False,
            "freeze_attn2": False,
        }
    cfg["lora"]["freeze_attn2"] = bool(freeze_attn2)

    # Validate syncnet / num_frames compatibility early so the user gets a
    # clear message instead of a conv2d channel mismatch deep in training.
    if cfg["run"]["use_syncnet"] and cfg["run"]["pixel_space_supervise"]:
        import yaml as _yaml
        syncnet_cfg_path = Path(cfg["data"]["syncnet_config_path"])
        syncnet_cfg = _yaml.safe_load(syncnet_cfg_path.read_text())
        syncnet_num_frames = syncnet_cfg.get("data", {}).get("num_frames")
        if syncnet_num_frames is not None and syncnet_num_frames != cfg["data"]["num_frames"]:
            raise ValueError(
                f"SyncNet config `{syncnet_cfg_path.name}` expects num_frames={syncnet_num_frames}, "
                f"but training is configured with num_frames={cfg['data']['num_frames']}. "
                f"Please set num_frames to {syncnet_num_frames} (or disable use_syncnet)."
            )
        syncnet_ckpt = syncnet_cfg.get("ckpt", {}).get("inference_ckpt_path", "")
        if not syncnet_ckpt:
            raise ValueError(
                f"SyncNet config `{syncnet_cfg_path.name}` has no inference_ckpt_path. "
                "Please use a syncnet config that points to a valid checkpoint."
            )

    return cfg
def on_preset_change(preset_name: str) -> Tuple[Any, ...]:
    """When user picks a preset, fill the form fields with preset defaults."""
    preset = PRESETS[preset_name]
    # Fallback for keys older presets don't carry — current form-value semantics
    # for save_ckpt_steps / max_train_steps / lr_*. Existing Stage 1 / Stage 2
    # presets still match the form's default behavior.
    return (
        preset["batch_size"],
        preset["num_frames"],
        preset["resolution"],
        preset["learning_rate"],
        preset["use_motion_module"],
        preset["pixel_space_supervise"],
        preset["use_syncnet"],
        preset["sync_loss_weight"],
        preset["perceptual_loss_weight"],
        preset["recon_loss_weight"],
        preset["trepa_loss_weight"],
        preset["mixed_precision_training"],
        preset["enable_gradient_checkpointing"],
        preset["mask_image_path"],
        preset["resume_ckpt"],
        preset.get("save_ckpt_steps", 10000),
        preset.get("max_train_steps", 10000),
        preset.get("lr_scheduler", "constant"),
        preset.get("lr_warmup_steps", 0),
        preset["description"],
        preset.get("freeze_attn2", False),
    )
def apply_dataset_preset(preset_name: str) -> Tuple[str, str, str, str]:
    """Fill train_data_dir / train_fileslist / val paths from a dataset preset."""
    preset = DATASET_PRESETS.get(preset_name, {})
    return (
        preset.get("train_data_dir", ""),
        preset.get("train_fileslist", ""),
        preset.get("val_video_path", "assets/demo1_video.mp4"),
        preset.get("val_audio_path", "assets/demo1_audio.wav"),
    )

