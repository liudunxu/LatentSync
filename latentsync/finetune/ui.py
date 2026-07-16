"""Assembles the Gradio Fine-tune Studio UI from per-tab modules."""
from pathlib import Path
from typing import Any, Dict, List

import gradio as gr

from latentsync.finetune import REPO_ROOT, FINETUNE_BASE_DIR
from latentsync.finetune.config import DATASET_PRESETS, PRESETS, DEFAULT_PRESET_NAME, apply_dataset_preset, on_preset_change
from latentsync.finetune.utils import (
    _analyze_training_video_yaw,
    _format_preview_info,
    _list_training_videos,
    _load_preview_cache,
    _preview_cache_path,
    _safe_video_update,
    _save_preview_cache,
    list_checkpoints,
    list_checkpoints_in_run,
    list_datasets,
    list_validation_videos,
    parse_loss_chart,
    parse_sync_conf_chart,
    read_loss_from_checkpoint,
    refresh_runs,
)
from latentsync.finetune.ui_launch import (
    debug_all_inputs,
    launch_training,
    one_click_launch,
    ping_backend,
    refresh_training_log,
    stop_training,
    _prebuilt_choices,
    _run_curate_finetune,
    _run_init_prebuilt,
    _run_merge_lora,
)
from latentsync.finetune.ui_monitor import (
    _on_page_load,
    monitor_refresh,
)
from latentsync.finetune.ui_inference import (
    _poll_compare_state,
    _poll_validate_state,
    run_compare,
    run_validation,
    stop_inference,
)
from latentsync.finetune.ui_tools import (
    _recommend_finetune_preset,
    evaluate_dataset_quality,
    generate_identity_kit,
    reset_identity_defaults,
    run_badcase_checklist,
    _diagnose_short_drama,
)

# _TRAINER and _INFERENCE live in process.py so every UI module shares the same singletons.

_SYSTEM_FONT_CSS = """
* {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial,
                 "Noto Sans", "PingFang SC", "Microsoft YaHei", sans-serif !important;
}
"""
def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="LatentSync Fine-tune Studio",
        theme=gr.themes.Soft(),
        css=_SYSTEM_FONT_CSS,
    ) as demo:
        # Compute defaults from the default preset so the form is consistent
        # on first load (no manual preset click required).
        _default_preset_values = on_preset_change(DEFAULT_PRESET_NAME)
        (
            _default_batch_size, _default_num_frames, _default_resolution,
            _default_lr, _default_use_mm, _default_pixel, _default_use_sync,
            _default_sync_w, _default_lpips_w, _default_recon_w, _default_trepa_w,
            _default_mp, _default_gc, _default_mask, _default_resume,
            _default_save_steps, _default_max_steps, _default_lr_scheduler,
            _default_warmup, _default_preset_desc, _default_freeze_attn2,
        ) = _default_preset_values

        gr.Markdown(
            """
# 🎛 LatentSync Fine-tune Studio

> **核心目的**：**以微调 UNet 为主**的端到端工作台。覆盖**训练 → 监控 → 验证 → 调参**全流程。
> 也支持 **SyncNet 单独训练**（Tab 1 选 SyncNet preset）。

## 推荐工作流

```
[Tab 1] 配数据集 + 选 preset (Stage 2 / LoRA / QLoRA) → 启动训练
   ↓
[Tab 2] 看 loss 曲线 / validation 视频 / 日志 / sync_conf
   ↓
[Tab 3.5] 单 ckpt 推理 + 质量自检（首次验证）
   ↓
[Tab 3]  base vs fine-tuned 并排对比
   ↓
[Tab 4 / 6] 调 Identity 保护 / 查 Badcase
```

## 6 个 Tab 一览

| Tab | 阶段 | 核心作用 |
|---|---|---|
| **1. 配置 & 启动** | 训练 | 选 preset、调超参、启动 `torchrun` |
| **2. 训练监控** | 训练 | loss 曲线、val 视频、日志、auto refresh 15s |
| **3. 推理对比** | 推理 | base vs fine-tuned 并排跑 |
| **3.5. 验证 (单 ckpt)** | 推理 | **新** 选 1 个 ckpt 跑 + 自动质量自检（嘴糊 / 闪烁 / 人脸检测）|
| **4. Identity 保护** | 推理 | 调 4 层身份保持参数，生成推理 kwargs |
| **5. 数据集质量评估** | 数据 | HyperIQA + SyncNet_conf 分布 |
| **6. Badcase 检查** | 推理 | 量化"嘴糊不糊"等指标 |

> ⚠️ **GPU 提示**：本页会本地拉起 `torchrun`。如果没有 CUDA，会启动失败。
> 推荐在带 GPU 的机器上启动；如果在 CPU 机器上启动，至少能在 **Tab 1** 配置并保存 yaml 供远程训练使用。
            """
        )

        # =========================================================
        # Tab 1: Configure & Launch
        # =========================================================
        with gr.Tab("1️⃣ 配置 & 启动"):
            with gr.Accordion("📖 常见问题 & 默认值说明", open=False):
                gr.Markdown(
                    """
### 默认值为什么这样填？
- **默认 Preset = Stage 2 LoRA (256)**：首次微调最稳妥，只训一个 ~10MB 的 LoRA adapter，不破坏 base UNet，显存 12-15GB。
- **resolution=256**：与默认 `mask.png` 配对；改成 512 时务必切到 `mask2.png`，否则输出会花。
- **num_frames=16**：StableSyncNet 目前只支持 16 帧；只要开 `use_syncnet`，就不要改这个数字。
- **freeze_attn2**：对侧脸/大嘴等容易破坏唇音同步的 badcase，建议勾上；通用微调可不开。

### 训练前 checklist
1. 数据集已做人脸对齐（`init_finetune_dataset.py --align --align-resolution 256`）。
2. `train_fileslist.txt` 存在且每行是一个 mp4 路径。
3. 显存不够时：切 QLoRA preset 或降 resolution；不要只降 batch_size（当前只支持 1）。
                    """
                )

            # ──────────────────────────────────────────────────────────
            # 🚀 一键启动训练:顶部大按钮,自动填默认 + 启动
            # ──────────────────────────────────────────────────────────
            gr.Markdown(
                "> **懒人入口**:点一下这个按钮,自动填默认字段 + 用当前 preset 启训练。"
                "想细调的字段下面表单手动改。"
            )
            with gr.Row():
                one_click_btn = gr.Button(
                    "🚀 一键启动训练 (自动填默认 + 启 torchrun)",
                    variant="primary",
                    scale=4,
                )
            one_click_status = gr.Textbox(
                label="一键启动状态", interactive=False, lines=3,
                value="👆 点上面按钮开始。空字段会用 preset 默认值 / prebuilt 默认数据 / assets demo 视频。",
            )

            with gr.Row():
                preset_dd = gr.Dropdown(
                    choices=list(PRESETS.keys()),
                    value=DEFAULT_PRESET_NAME,
                    label="预设 (Preset)",
                    scale=2,
                )
                preset_desc = gr.Textbox(
                    label="预设说明",
                    value=_default_preset_desc,
                    interactive=False,
                    scale=3,
                )

            with gr.Row():
                freeze_attn2 = gr.Checkbox(
                    label="LoRA: 冻结 attn2 (audio cross-attn) — 防 sync 退化",
                    value=_default_freeze_attn2,
                    info="仅 LoRA 生效；勾上后 attn2 的 LoRA 参数冻结,牺牲一点灵活性换取 sync_conf 稳定",
                )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📂 数据集")
                    dataset_preset_dd = gr.Dropdown(
                        choices=list(DATASET_PRESETS.keys()),
                        value="assets 演示数据 (3 videos，可完整跑通)",
                        label="数据集预设 (Dataset Preset)",
                    )
                    train_data_dir = gr.Textbox(
                        label="train_data_dir (目录)",
                        placeholder="data/my_high_quality_videos",
                        value="assets",
                    )
                    train_fileslist = gr.Textbox(
                        label="train_fileslist (文件列表，一行一个 mp4)",
                        placeholder="data/my_high_quality_videos/fileslist.txt",
                        value="data/demo_fileslist.txt",
                    )
                    val_video_path = gr.Textbox(
                        label="val_video_path",
                        value="assets/demo1_video.mp4",
                    )
                    val_audio_path = gr.Textbox(
                        label="val_audio_path",
                        value="assets/demo1_audio.wav",
                    )

                    # Quick-pick presets so the user can verify the full
                    # launch path before plugging in their own data.
                    gr.Markdown("#### 📌 常用路径示例（点击填入）")
                    gr.Examples(
                        examples=[
                            ["assets", "data/demo_fileslist.txt"],
                            ["preprocess/high_visual_quality", "preprocess/high_visual_quality/fileslist.txt"],
                            ["data/train", "data/train/fileslist.txt"],
                            ["data/my_avatar", "data/my_avatar/fileslist.txt"],
                            ["data/multilingual", "data/multilingual/fileslist.txt"],
                        ],
                        inputs=[train_data_dir, train_fileslist],
                        label=None,
                    )

                    dataset_choices = gr.Dropdown(
                        choices=list_datasets(),
                        label="或从已有数据集选 (click 后填到 train_data_dir)",
                        value=None,
                    )
                    dataset_choices.change(
                        lambda x: x,
                        inputs=dataset_choices,
                        outputs=train_data_dir,
                    )

                    dataset_preset_dd.change(
                        fn=apply_dataset_preset,
                        inputs=dataset_preset_dd,
                        outputs=[train_data_dir, train_fileslist, val_video_path, val_audio_path],
                    )

                with gr.Column():
                    gr.Markdown("### 🏗 模型 & 训练")
                    resume_ckpt = gr.Textbox(
                        label="resume_ckpt (base UNet .pt 或 LoRA adapter 目录)",
                        value=_default_resume,
                        info="传 .pt 表示从 base/pretrained 开始训；传 LoRA adapter 目录（含 adapter_config.json）表示从该 checkpoint 继续训练。",
                    )
                    batch_size = gr.Slider(1, 64, value=_default_batch_size, step=1, label="batch_size", info="当前实现只支持 1")
                    num_frames = gr.Slider(
                        8, 32, value=_default_num_frames, step=1, label="num_frames",
                        info="StableSyncNet 只支持 16 帧。开 use_syncnet 时请勿修改。",
                    )
                    resolution = gr.Radio(
                        [256, 512], value=_default_resolution, label="resolution",
                        info="256 配 mask.png；512 必须配 mask2.png。",
                    )
                    learning_rate = gr.Number(value=_default_lr, label="learning_rate", precision=8, info="LoRA 通常 3e-5~5e-5；全量训练 1e-5。")

                    use_motion_module = gr.Checkbox(
                        value=_default_use_mm,
                        label="use_motion_module (Stage 2 必开)",
                        info="关闭后时序连贯性显著下降。",
                    )
                    pixel_space_supervise = gr.Checkbox(
                        value=_default_pixel,
                        label="pixel_space_supervise (Stage 2 必开)",
                        info="用原始像素监督，改善嘴部清晰度。",
                    )
                    use_syncnet = gr.Checkbox(
                        value=_default_use_sync,
                        label="use_syncnet (Stage 2 必开)",
                        info="唇音同步监督。开启后 num_frames 必须等于 SyncNet config 的 16 帧。",
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ⚖️ 损失权重")
                    sync_loss_weight = gr.Slider(
                        0.0, 1.0, value=_default_sync_w, step=0.01,
                        label="sync_loss_weight",
                        info="唇音同步权重。侧脸/大嘴 badcase 可提到 0.10-0.15；通用 0.02-0.05。",
                    )
                    perceptual_loss_weight = gr.Slider(
                        0.0, 1.0, value=_default_lpips_w, step=0.01,
                        label="perceptual_loss_weight (LPIPS)",
                        info="细节/纹理保真。嘴糊或侧脸阴影变化大时可提到 0.15-0.25。",
                    )
                    recon_loss_weight = gr.Slider(
                        0.0, 5.0, value=_default_recon_w, step=0.1,
                        label="recon_loss_weight",
                        info="像素级重建，通常保持 1.0。",
                    )
                    trepa_loss_weight = gr.Slider(
                        0.0, 50.0, value=_default_trepa_w, step=1.0,
                        label="trepa_loss_weight (0=关闭)",
                        info="时序连贯性。设 0 可省显存，但可能闪烁。",
                    )

                with gr.Column():
                    gr.Markdown("### 🖼 Validation 推理参数")
                    gr.Markdown(
                        "<small>每 save_ckpt_steps 步生成一段验证视频用的推理质量/速度。"
                        "自然度默认 steps=40,guidance=1.5;快速试训可降到 20/1.5。"
                        "推到 HF 后别人用同样的 ckpt + 这俩默认值就能复现。</small>"
                    )
                    val_inference_steps = gr.Slider(
                        5, 80, value=40, step=5, label="validation inference_steps (20=快, 40=自然)"
                    )
                    val_guidance_scale = gr.Slider(
                        1.0, 4.0, value=1.5, step=0.1, label="validation guidance_scale"
                    )
                    val_seed = gr.Number(
                        value=1247, label="validation seed (随机种子)", precision=0
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### ⚙️ 训练设置")
                    mixed_precision_training = gr.Checkbox(value=_default_mp, label="mixed_precision_training (fp16)")
                    enable_gradient_checkpointing = gr.Checkbox(value=_default_gc, label="enable_gradient_checkpointing")
                    mask_image_path = gr.Textbox(
                        label="mask_image_path",
                        value=_default_mask,
                    )
                    save_ckpt_steps = gr.Slider(
                        500, 50000, value=_default_save_steps, step=500, label="save_ckpt_steps",
                        info="每 N 步存 ckpt 并跑一段 validation 视频。试训建议 500，正式训练 5000-10000。",
                    )
                    max_train_steps = gr.Slider(
                        1000, 100000, value=_default_max_steps, step=1000,
                        label="max_train_steps (cap: 100k ≈ 1.7 days @1.5s/step)",
                        info="badcase 修复通常 3000-5000 步即可，避免过拟合。",
                    )
                    num_workers = gr.Slider(0, 32, value=12, step=1, label="num_workers")
                    lr_scheduler = gr.Dropdown(
                        choices=["constant", "cosine", "cosine_with_restarts", "linear", "polynomial"],
                        value=_default_lr_scheduler,
                        label="lr_scheduler",
                    )
                    lr_warmup_steps = gr.Slider(
                        0, 2000, value=_default_warmup, step=50,
                        label="lr_warmup_steps (推荐 100-300 for cosine)",
                        info="cosine 建议 300-500；constant 可设 0。",
                    )
                    train_output_dir = gr.Textbox(
                        label=f"train_output_dir (相对于 {FINETUNE_BASE_DIR.name})",
                        value="unet",
                    )
                    gr.Markdown(
                        f"> 📂 微调中间产物根目录：`{FINETUNE_BASE_DIR}`\n"
                        f"> 可通过环境变量 `LATENTSYNC_FINETUNE_DIR` 修改"
                    )
                    nproc_per_node = gr.Slider(1, 8, value=1, step=1, label="torchrun nproc_per_node")
                    master_port = gr.Slider(20000, 30000, value=25679, step=1, label="torchrun master_port")
                    extra_env = gr.Textbox(
                        label="额外环境变量 (可选，每行 KEY=VALUE)",
                        placeholder="LATENTSYNC_GUIDANCE_SCALE=1.5\nHF_TOKEN=...",
                        lines=3,
                    )

            with gr.Row():
                launch_btn = gr.Button("🚀 启动训练", variant="primary", scale=2)
                stop_btn = gr.Button("⏹ 停止训练", variant="stop", scale=1)
                ping_btn = gr.Button("🔍 Ping 后端", scale=1)
                debug_btn = gr.Button("🐛 Debug 输入", scale=1)

            launch_status = gr.Textbox(label="启动状态", lines=10)
            log_path_state = gr.State(value="")
            launch_btn.click(
                fn=launch_training,
                inputs=[
                    preset_dd, train_data_dir, train_fileslist, val_video_path, val_audio_path,
                    resume_ckpt, batch_size, num_frames, resolution, learning_rate,
                    use_motion_module, pixel_space_supervise, use_syncnet,
                    sync_loss_weight, perceptual_loss_weight, recon_loss_weight, trepa_loss_weight,
                    mixed_precision_training, enable_gradient_checkpointing, mask_image_path,
                    save_ckpt_steps, max_train_steps, num_workers, train_output_dir,
                    freeze_attn2,
                    val_inference_steps, val_guidance_scale, val_seed,
                    lr_scheduler, lr_warmup_steps,
                    nproc_per_node, master_port, extra_env,
                ],
                outputs=[launch_status, log_path_state],
            )
            stop_btn.click(fn=stop_training, outputs=launch_status)

            # The big top-of-tab "一键启动" button delegates to the
            # standard launch_training flow with auto-filled defaults.
            one_click_btn.click(
                fn=one_click_launch,
                inputs=[
                    preset_dd, train_data_dir, train_fileslist,
                    val_video_path, val_audio_path, resume_ckpt,
                    batch_size, num_frames, resolution, learning_rate,
                    use_motion_module, pixel_space_supervise, use_syncnet,
                    sync_loss_weight, perceptual_loss_weight, recon_loss_weight, trepa_loss_weight,
                    mixed_precision_training, enable_gradient_checkpointing, mask_image_path,
                    save_ckpt_steps, max_train_steps, num_workers, train_output_dir,
                    freeze_attn2, val_inference_steps, val_guidance_scale, val_seed,
                    lr_scheduler, lr_warmup_steps,
                    nproc_per_node, master_port, extra_env,
                ],
                outputs=one_click_status,
            )

            # ---- 预制数据集 (HF Hub 自动下载 + curate) ----
            with gr.Accordion("📚 预制数据集 (HF Hub 自动下 + curate)", open=False):
                gr.Markdown(
                    "**最省事的入口** — 从 `tools/prebuilt_datasets.yaml` 选一个,"
                    "一键下载 + face 检测 + 按 yaw/motion 分桶 + 写 fileslist.txt,"
                    "直接拿来训练。无需自己找数据源。"
                )
                with gr.Row():
                    prebuilt_dd = gr.Dropdown(
                        choices=_prebuilt_choices(),
                        label="预制数据集 (HF Hub)",
                        value=None,
                        scale=2,
                    )
                    prebuilt_target = gr.Textbox(
                        label=f"输出目录 (默认 {FINETUNE_BASE_DIR}/init_finetune)",
                        placeholder=str(FINETUNE_BASE_DIR / "init_finetune"),
                        value=str(FINETUNE_BASE_DIR / "init_finetune"),
                        scale=2,
                    )
                prebuilt_hf_token = gr.Textbox(
                    label="HF Token (可选,用于 gated 数据集,留空走 HF_TOKEN 环境变量)",
                    placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
                    type="password",
                )
                prebuilt_btn = gr.Button("⬇ 下载 + Curate", variant="primary")
                prebuilt_log = gr.Textbox(label="输出", lines=18, interactive=False)
                prebuilt_btn.click(
                    fn=_run_init_prebuilt,
                    inputs=[prebuilt_dd, prebuilt_target, prebuilt_hf_token],
                    outputs=[prebuilt_log, train_data_dir, train_fileslist],
                )

            # ---- 数据集一键准备 (download + curate) ----
            with gr.Accordion("📥 数据集一键准备 (download_curated_finetune_set)", open=False):
                gr.Markdown(
                    "端到端跑 `tools/download_curated_finetune_set.py`:给一批 URL 或本地视频,"
                    "自动按 yaw/motion 分桶,产出可直接填进上面表单的 fileslist。"
                )
                with gr.Row():
                    curate_urls = gr.Textbox(
                        label="URL 列表文件 (可空,用 --source-dir)",
                        placeholder="tools/finetune_starter_urls.example.txt",
                        scale=2,
                    )
                    curate_source_dir = gr.Textbox(
                        label="本地源目录 (可空,用 --urls)",
                        placeholder="/data/my_raw_videos",
                        scale=2,
                    )
                with gr.Row():
                    curate_output_dir = gr.Textbox(
                        label="curated 输出目录",
                        value=str(FINETUNE_BASE_DIR / "finetune_samples_v1"),
                        scale=3,
                    )
                    curate_scale = gr.Dropdown(
                        choices=["small", "medium", "large"],
                        value="small",
                        label="scale (small=200, medium=1000, large=5000)",
                        scale=2,
                    )
                with gr.Row():
                    curate_btn = gr.Button("📥 跑 download_curated_finetune_set", variant="primary")
                curate_log = gr.Textbox(label="输出", lines=18, interactive=False)
                curate_btn.click(
                    fn=_run_curate_finetune,
                    inputs=[curate_urls, curate_source_dir, curate_output_dir, curate_scale],
                    outputs=curate_log,
                )

            debug_status = gr.Textbox(label="诊断信息", lines=8, interactive=False)
            ping_btn.click(fn=ping_backend, outputs=debug_status)
            debug_btn.click(
                fn=debug_all_inputs,
                inputs=[
                    preset_dd, train_data_dir, train_fileslist, val_video_path, val_audio_path,
                    resume_ckpt, batch_size, num_frames, resolution, learning_rate,
                    use_motion_module, pixel_space_supervise, use_syncnet,
                    sync_loss_weight, perceptual_loss_weight, recon_loss_weight, trepa_loss_weight,
                    mixed_precision_training, enable_gradient_checkpointing, mask_image_path,
                    save_ckpt_steps, max_train_steps, num_workers, train_output_dir,
                    nproc_per_node, master_port, extra_env,
                ],
                outputs=debug_status,
            )

            with gr.Row():
                training_log_box = gr.Textbox(
                    label="训练日志 (实时，尾部 80 行)",
                    lines=20,
                    interactive=False,
                    value="(训练日志会在这里显示)",
                )

            refresh_log_btn = gr.Button("🔄 手动刷新训练日志", variant="secondary")
            refresh_log_btn.click(
                fn=refresh_training_log,
                outputs=[training_log_box, launch_status],
            )
            log_timer = gr.Timer(value=3)
            log_timer.tick(
                fn=refresh_training_log,
                outputs=[training_log_box, launch_status],
            )

            # preset → fill defaults
            preset_dd.change(
                fn=on_preset_change,
                inputs=preset_dd,
                outputs=[
                    batch_size, num_frames, resolution, learning_rate,
                    use_motion_module, pixel_space_supervise, use_syncnet,
                    sync_loss_weight, perceptual_loss_weight, recon_loss_weight, trepa_loss_weight,
                    mixed_precision_training, enable_gradient_checkpointing, mask_image_path,
                    resume_ckpt,
                    save_ckpt_steps, max_train_steps, lr_scheduler, lr_warmup_steps,
                    preset_desc, freeze_attn2,
                ],
            )

        # =========================================================
        # Tab 2: Monitor
        # =========================================================
        with gr.Tab("2️⃣ 训练监控"):
            with gr.Row():
                monitor_output_dir = gr.Textbox(
                    label=f"train_output_dir (相对于 {FINETUNE_BASE_DIR.name})",
                    value="unet",
                )
                refresh_runs_btn = gr.Button("🔄 刷新 run 列表")

            run_dd = gr.Dropdown(label="run 目录", choices=[], value=None, allow_custom_value=True)
            refresh_runs_btn.click(
                fn=refresh_runs,
                inputs=monitor_output_dir,
                outputs=run_dd,
            )

            trainer_status = gr.Textbox(label="Trainer 状态", interactive=False)
            log_box = gr.Textbox(label="最新日志 (尾部 80 行)", lines=20, interactive=False)
            ckpt_dd = gr.Dropdown(label="Checkpoint", choices=[], value=None)
            ckpt_info_box = gr.Textbox(label="Checkpoint 信息", lines=10, interactive=False)

            with gr.Row():
                with gr.Column():
                    loss_chart_img = gr.Image(label="Loss 曲线 (lr + total + recon + lpips + sync)", type="filepath")
                    sync_conf_img = gr.Image(label="Sync_conf 曲线 (finetune 核心信号)", type="filepath")
                with gr.Column():
                    val_video_dd = gr.Dropdown(label="Validation 视频", choices=[])
                    val_video_player = gr.Video(label="预览", interactive=False)

            def _on_run_change(run_path):
                chart = parse_loss_chart(run_path)
                sync_chart = parse_sync_conf_chart(run_path)
                vids = list_validation_videos(run_path)
                ckpts = list_checkpoints_in_run(run_path)
                ckpt_update = gr.update(choices=ckpts, value=ckpts[-1] if ckpts else None)
                if ckpts:
                    ckpt_path = Path(ckpts[-1])
                    if not ckpt_path.is_absolute():
                        ckpt_path = REPO_ROOT / ckpt_path
                    ck_info = read_loss_from_checkpoint(str(ckpt_path))
                else:
                    ck_info = "(no checkpoint yet)"
                return (
                    chart,
                    sync_chart,
                    gr.update(choices=vids, value=vids[0] if vids else None),
                    ckpt_update,
                    ck_info,
                )

            def _on_ckpt_change(run_path, ckpt_path):
                if not ckpt_path:
                    return "(no checkpoint selected)"
                p = Path(ckpt_path)
                if not p.is_absolute():
                    p = REPO_ROOT / p
                return read_loss_from_checkpoint(str(p))

            run_dd.change(
                fn=_on_run_change,
                inputs=run_dd,
                outputs=[loss_chart_img, sync_conf_img, val_video_dd, ckpt_dd, ckpt_info_box],
            )
            ckpt_dd.change(
                fn=_on_ckpt_change,
                inputs=[run_dd, ckpt_dd],
                outputs=ckpt_info_box,
            )
            val_video_dd.change(
                fn=_safe_video_update,
                inputs=val_video_dd,
                outputs=val_video_player,
            )

            monitor_btn = gr.Button("🔄 手动刷新", variant="primary")
            run_dir_hidden = gr.Textbox(visible=False)
            with gr.Row():
                progress_bar = gr.Slider(
                    0, 100, value=0, step=0.1, interactive=False,
                    label="📈 训练进度 (step / max_step, %)",
                )
                progress_text = gr.Textbox(
                    label="⏱ 耗时 / 速度 / ETA",
                    interactive=False,
                    scale=2,
                )
            monitor_btn.click(
                fn=monitor_refresh,
                inputs=[monitor_output_dir, run_dd, log_path_state],
                outputs=[
                    run_dd, run_dir_hidden, loss_chart_img, sync_conf_img,
                    val_video_dd, ckpt_dd, log_box, ckpt_info_box, trainer_status,
                    progress_bar, progress_text,
                ],
            )

            timer = gr.Timer(value=15)
            timer.tick(
                fn=monitor_refresh,
                inputs=[monitor_output_dir, run_dd, log_path_state],
                outputs=[
                    run_dd, run_dir_hidden, loss_chart_img, sync_conf_img,
                    val_video_dd, ckpt_dd, log_box, ckpt_info_box, trainer_status,
                    progress_bar, progress_text,
                ],
            )

            # ---- 合并 LoRA 子表(训练完自然接这一步) ----
            with gr.Accordion("🔀 合并 LoRA adapter (merge_lora)", open=False):
                gr.Markdown(
                    "把训练好的 LoRA adapter (~10MB) 折回 base UNet,产出可独立部署的 "
                    "`latentsync_unet.pt`。可选同时 push 到 HuggingFace Hub。"
                )
                with gr.Row():
                    merge_base_ckpt = gr.Textbox(
                        label="base UNet ckpt",
                        value="checkpoints/latentsync_unet.pt",
                        scale=2,
                    )
                    merge_adapter_dir = gr.Textbox(
                        label="adapter 目录 (训练产物, 含 adapter_config.json)",
                        placeholder="debug/unet_lora/train_lora-2025.../checkpoints/checkpoint-5000",
                        scale=3,
                    )
                with gr.Row():
                    merge_out_ckpt = gr.Textbox(
                        label="合并输出路径",
                        value=str(FINETUNE_BASE_DIR / "unet" / "merged.pt"),
                        scale=3,
                    )
                    merge_push_repo = gr.Textbox(
                        label="(可选) HF Hub repo_id,留空不 push",
                        placeholder="username/latentsync-lora-finetune-v1",
                        scale=3,
                    )
                merge_btn = gr.Button("🔀 合并 LoRA → merged.pt (可同时 push)", variant="primary")
                merge_log = gr.Textbox(label="merge 输出", lines=10, interactive=False)
                merge_btn.click(
                    fn=_run_merge_lora,
                    inputs=[merge_base_ckpt, merge_adapter_dir, merge_out_ckpt, merge_push_repo],
                    outputs=merge_log,
                )

        # =========================================================
        # Tab 3: Compare
        # =========================================================
        with gr.Tab("3️⃣ 推理对比 (base vs fine-tuned)"):
            gr.Markdown(
                "上传同一段视频 + 音频，分别用 base 和微调后的 checkpoint 跑推理，并排对比效果。"
                "LoRA adapter 目录会被自动合并，无需先跑 merge_lora.py。"
            )
            with gr.Row():
                with gr.Column():
                    cmp_video = gr.Video(label="Input Video")
                    cmp_audio = gr.Audio(label="Input Audio", type="filepath")
                with gr.Column():
                    cmp_include_lora = gr.Checkbox(value=True, label="包含 LoRA adapter 目录")
                    cmp_base = gr.Dropdown(
                        choices=list_checkpoints(include_lora=True),
                        label="Base checkpoint",
                        allow_custom_value=True,
                        info="通常是 checkpoints/latentsync_unet.pt",
                    )
                    cmp_ft = gr.Dropdown(
                        choices=list_checkpoints(include_lora=True),
                        label="Fine-tuned checkpoint",
                        allow_custom_value=True,
                        info="可以是 .pt、LoRA adapter 目录或 merge 后的 .pt",
                    )
                    cmp_resolution = gr.Radio([256, 512], value=512, label="resolution")

            with gr.Row():
                cmp_steps = gr.Slider(10, 50, value=20, step=1, label="inference_steps")
                cmp_guidance = gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="guidance_scale")
                cmp_seed = gr.Number(value=1247, label="seed", precision=0)
                cmp_baseline = gr.Checkbox(value=False, label="基线模式（禁用所有质量优化）")

            with gr.Row():
                cmp_btn = gr.Button("🎬 生成对比", variant="primary", scale=3)
                cmp_cancel_btn = gr.Button("⏹ 取消当前推理", variant="stop", scale=1)
            cmp_status = gr.Textbox(label="状态", interactive=False, visible=True)

            with gr.Row():
                cmp_out_base = gr.Video(label="Base 输出")
                cmp_out_ft = gr.Video(label="Fine-tuned 输出")

            cmp_btn.click(
                fn=run_compare,
                inputs=[
                    cmp_video, cmp_audio, cmp_base, cmp_ft,
                    cmp_steps, cmp_guidance, cmp_seed, cmp_resolution,
                    cmp_baseline,
                ],
                outputs=[cmp_out_base, cmp_out_ft, cmp_status],
            )
            cmp_include_lora.change(
                fn=lambda incl: gr.update(choices=list_checkpoints(include_lora=incl)),
                inputs=cmp_include_lora,
                outputs=[cmp_base, cmp_ft],
            )
            cmp_cancel_btn.click(fn=stop_inference, outputs=cmp_status)

            cmp_timer = gr.Timer(value=1, active=True)
            cmp_timer.tick(
                fn=_poll_compare_state,
                outputs=[cmp_out_base, cmp_out_ft, cmp_status],
            )

        # =========================================================
        # Tab 3.5: Validation - run inference with a single ckpt
        # =========================================================
        with gr.Tab("🧪 验证 (单 ckpt 推理)"):
            gr.Markdown(
                """
选一个 checkpoint（base / fine-tuned / LoRA adapter 目录），上传视频和音频，
跑一次推理并自动做 **质量自检**（嘴糊比例 / 闪烁 / 人脸检测率）。

> 比 Tab 3 简单：只跑一个 ckpt，更快。  
> LoRA adapter 会被自动合并；输出视频 + 质量报告。  
> 默认 `guidance_scale=1.5, steps=40` 偏自然度；想快测可降到 `steps=20`。
                """
            )
            with gr.Row():
                with gr.Column():
                    val_video = gr.Video(label="Input Video", scale=2)
                    val_audio = gr.Audio(label="Input Audio", type="filepath", scale=2)
                with gr.Column():
                    val_include_lora = gr.Checkbox(value=True, label="包含 LoRA adapter 目录")
                    val_ckpt = gr.Dropdown(
                        choices=list_checkpoints(include_lora=True),
                        label="Checkpoint（base / fine-tuned / LoRA adapter）",
                        value="checkpoints/latentsync_unet.pt" if (REPO_ROOT / "checkpoints/latentsync_unet.pt").exists() else None,
                        allow_custom_value=True,
                        info="LoRA adapter 目录（含 adapter_config.json）可直接选，会自动合并。",
                    )
                    val_config = gr.Dropdown(
                        choices=[
                            "configs/unet/stage2.yaml",
                            "configs/unet/stage2_512.yaml",
                            "configs/unet/stage2_efficient.yaml",
                            "configs/unet/stage2_lora.yaml",
                        ],
                        value="configs/unet/stage2.yaml",
                        label="UNet config（必须和 ckpt 匹配）",
                        info="256 分辨率用 stage2.yaml；512 分辨率用 stage2_512.yaml。",
                    )
                    val_resolution = gr.Radio([256, 512], value=512, label="resolution")

            with gr.Row():
                val_steps = gr.Slider(10, 50, value=20, step=1, label="inference_steps")
                val_guidance = gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="guidance_scale")
                val_seed = gr.Number(value=1247, label="seed", precision=0)
                val_deepcache = gr.Checkbox(value=True, label="enable_deepcache (快 2x)")
                val_skip_qc = gr.Checkbox(value=True, label="跳过质量自检（更快）")
                val_baseline = gr.Checkbox(value=False, label="基线模式（禁用所有质量优化）")

            with gr.Row():
                val_btn = gr.Button("🚀 推理 + 质量自检", variant="primary", scale=3)
                val_cancel_btn = gr.Button("⏹ 取消当前推理", variant="stop", scale=1)

            val_compat = gr.Textbox(label="ckpt 兼容性检查", lines=4, interactive=False)
            val_output = gr.Video(label="生成结果", interactive=False)
            val_report = gr.Textbox(label="质量报告", lines=18, interactive=False)
            val_saved = gr.Textbox(label="保存路径", interactive=False)

            val_btn.click(
                fn=run_validation,
                inputs=[
                    val_video, val_audio, val_ckpt, val_config,
                    val_steps, val_guidance, val_seed, val_resolution,
                    val_deepcache, val_skip_qc, val_baseline,
                ],
                outputs=[val_output, val_compat, val_report, val_saved, val_btn],
            )
            val_include_lora.change(
                fn=lambda incl: gr.update(choices=list_checkpoints(include_lora=incl)),
                inputs=val_include_lora,
                outputs=val_ckpt,
            )
            val_cancel_btn.click(fn=stop_inference, outputs=[val_report, val_btn])

            val_timer = gr.Timer(value=1, active=True)
            val_timer.tick(
                fn=_poll_validate_state,
                inputs=[val_skip_qc],
                outputs=[val_output, val_compat, val_report, val_saved, val_btn],
            )

        # =========================================================
        # Tab 4: Identity Protection Strategy
        # =========================================================
        with gr.Tab("🛡️ Identity 保护策略"):
            gr.Markdown(
                """
LatentSync 用 **4 层防御** 保证只改嘴部、不改脸：

| 层 | 机制 | 在哪控制 |
|---|---|---|
| L1 | `ref_pixel_values` 提供 identity | 训练时 `UNetDataset` + 推理时实时 ref |
| L2 | UNet 训练学到"看着 ref 还原 identity" | 训练分布自动学习 |
| L3 | `paste_surrounding_pixels_back` mask 截断 | 推理时 `dynamic_region_mask` |
| L4 | `_restore_reference_detail` 高频细节贴回 | 推理时 `mouth_detail_strength` |

下面三个区块分别调 L1 / L3 / L4 的关键参数。改完点 **生成推理 yaml** 即可在 `gradio_app.py` / `api.py` 里复用。
                """
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### L1: ref 窗口策略（影响训练数据采样 + 推理 ref 选择）")
                    ref_strategy = gr.Radio(
                        choices=["random", "adjacent", "fixed_first_frame"],
                        value="random",
                        label="ref 窗口选择策略",
                        info=(
                            "random: 随机抽远端帧（论文 baseline）\n"
                            "adjacent: 抽相邻帧（identity 更稳但极端表情少）\n"
                            "fixed_first_frame: 固定用第 1 帧（一致性最强但多样性差）"
                        ),
                    )
                    ref_window_distance = gr.Slider(
                        minimum=0,
                        maximum=64,
                        value=16,
                        step=1,
                        label="ref 与 gt 的最小距离（帧）",
                        info="论文 baseline = 16 帧（约 0.64 秒）。太小容易把同一段当 ref。",
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### L3: dynamic mask 大小（控制 paste-back 范围）")
                    dynamic_mask_mode = gr.Radio(
                        choices=["conservative", "standard", "aggressive"],
                        value="standard",
                        label="dynamic mask 大小策略",
                        info=(
                            "conservative: 椭圆更小（默认 pad_width×0.8），更保守\n"
                            "standard: 论文默认（pad_width×1.5）\n"
                            "aggressive: 椭圆更大（pad_width×2.0），覆盖大笑嘴"
                        ),
                    )
                    paste_back_blur_sigma = gr.Slider(
                        minimum=0.0,
                        maximum=15.0,
                        value=7.0,
                        step=0.5,
                        label="paste back 边缘模糊 sigma (像素)",
                        info="越大 paste-back 边界越平滑，但嘴部边缘会糊",
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### L4: detail / color post-processing")
                    detail_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.05,
                        label="mouth_detail_strength (L4 detail restore)",
                        info="越大越贴原图皮肤纹理（痣、皱纹）。>0.85 会盖掉生成的嘴型",
                    )
                    color_match_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.05,
                        label="color_match_strength",
                        info="越大颜色越平滑（避免 mask 边界色差）。>0.9 可能过度",
                    )

            with gr.Row():
                identity_generate_btn = gr.Button("📝 生成推理 kwargs / yaml", variant="primary")
                identity_clear_btn = gr.Button("🧹 重置为默认值")

            identity_output = gr.Code(
                label="生成的推理 kwargs (Python) 和 yaml (Config)",
                language="python",
                lines=20,
            )

            identity_generate_btn.click(
                fn=generate_identity_kit,
                inputs=[
                    ref_strategy, ref_window_distance,
                    dynamic_mask_mode, paste_back_blur_sigma,
                    detail_strength, color_match_strength,
                ],
                outputs=identity_output,
            )

            identity_clear_btn.click(
                fn=reset_identity_defaults,
                outputs=[
                    ref_strategy, ref_window_distance,
                    dynamic_mask_mode, paste_back_blur_sigma,
                    detail_strength, color_match_strength,
                ],
            )

            gr.Markdown(
                """
### 使用方法

生成的 `kwargs` 可以直接传给 `LipsyncPipeline.__call__(..., **kwargs)`：

```python
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
pipe = LipsyncPipeline(...)

pipe(
    video_path=...,
    audio_path=...,
    video_out_path=...,
    inference_ckpt_path=...,
    **identity_kwargs,  # ← 把上方生成的字典展开
)
```
                """
            )

        # =========================================================
        # Tab 5: Dataset Quality Evaluation
        # =========================================================
        with gr.Tab("📊 数据集质量评估"):
            gr.Markdown(
                """
训练前先评估数据，避免训完才发现质量问题。

会跑：
1. **HyperIQA 分数**（每视频取 3 帧，看视觉质量分布）
2. **SyncNet confidence**（每视频算 AV 同步质量）
3. **文件完整性**（损坏 / 缺失 / 时长不足）
4. **统计摘要** + **潜在问题列表**
                """
            )

            with gr.Row():
                ds_dir_input = gr.Textbox(
                    label="high_visual_quality 目录",
                    placeholder="/data/voxceleb2/high_visual_quality",
                    scale=3,
                )
                ds_max_videos = gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=100,
                    step=10,
                    label="最多评估视频数（采样）",
                    scale=1,
                )
                ds_eval_btn = gr.Button("🔍 开始评估", variant="primary", scale=1)

            with gr.Row():
                with gr.Column():
                    ds_stats = gr.Textbox(label="统计摘要", lines=15)
                with gr.Column():
                    ds_issues = gr.Textbox(label="潜在问题", lines=15)

            ds_chart = gr.Plot(label="HyperIQA / Sync_conf 分布")

            ds_eval_btn.click(
                fn=evaluate_dataset_quality,
                inputs=[ds_dir_input, ds_max_videos],
                outputs=[ds_stats, ds_issues, ds_chart],
            )

        # =========================================================
        # Tab 6: Badcase Checklist
        # =========================================================
        with gr.Tab("⚠️ Badcase 检查清单"):
            gr.Markdown(
                """
对生成的视频跑全套质量检查（对应 §13）：

| 检查项 | 目标 | 含义 |
|---|---|---|
| 嘴糊比例 | < 30% | Laplacian 方差低于阈值的帧占比 |
| 闪烁评分 | < 8 | 嘴部帧间平均像素差 |
| 唇音同步 | > 7 | SyncNet confidence |
| 身份保持 | > 0.8 | Face embedding 余弦相似度 |
| 张嘴幅度 p95 | > 0.18 | 95 分位 嘴高/嘴宽；偏小→大嘴数据重训 |

### Badcase → 推荐操作 速查

| 现象 | 数值信号 | 首选排查方向 | 备选:微调 |
|---|---|---|---|
| **侧脸同步弱 / 嘴唇不动** | `唇音同步 < 5` 且 yaw 大的帧 | Tab 4 把 `adaptive_quality_fallback` 打开 / 放宽 `yaw_skip_threshold` (30°→40°) | Stage 2 LoRA,feed ≥ 50 条 yaw 15-30° 样本,`freeze_attn2=True` |
| **嘴糊** (整嘴一片糊) | `嘴糊比例 > 40%` | 检查 `mask_image_path` 是否被改成 mask2/3 (baseline 是 `mask.png`) | Stage 2 LoRA rank=32 |
| **嘴唇外也糊** (paste-back 边界外溢) | 闪烁评分高 + 边缘像素突跳 | Tab 4 `dynamic_mask_mode` 切 `aggressive`,`paste_back_blur_sigma` 降到 5.0 | Stage 2 LoRA + 混合 clip / distance samples |
| **人脸快速移动时嘴糊** | 闪烁评分 > 12 + sync 抖动 | 调 `mouth_temporal_stabilization_strength`↑ 到 0.25 / `mouth_audio_motion_min_scale`↑ 到 0.9 | Stage 2 LoRA + 含 motion-blur 样本 |
| **身份漂移 (像别人了)** | `身份保持 < 0.7` | Tab 4 `ref_strategy` 改 `fixed_first_frame`,`color_match_strength`↑ 到 0.75 | LoRA 不要碰 attn1 self-attn (默认就只 wrap to_q/k/v/out.0) |
| **训练后 sync_conf 退化** | 训练前后 sync 下降 > 1 | (训练配置)勾 `freeze_attn2` 重训 | — |
| **显存 OOM** | torch OOM 日志 | 切 Stage 2 QLoRA | — |
| **生成的嘴没动但能听到声音** | `唇音同步 ≈ 1` (没生成) | 看训练日志 `[FaceMatch]` 哪个 filter 跳了 — 通常是 `yaw_skip` 或 `face_jump` | Stage 2 LoRA + 多角度训练数据 |
| **张嘴幅度小 (嘴张不开)** | `张嘴幅度 p95 < 0.18` | 用 📚 `celebv_hq_head_talk_side_big_mouth` (p95≥0.22) 数据重训 | 💋 Side-Face Lip Quality (sync_loss=0.15) |

**数据先行原则**:finetune 只能缓解,不能根治。`Tab 5` 先跑一遍数据集质量评估,HyperIQA 分布去掉 < 40 的样本后再训,事半功倍。
                """
            )

            with gr.Row():
                bc_video = gr.Video(label="生成结果视频", scale=2)
                bc_reference = gr.Video(label="原始参考视频（可选，用于 identity sim）", scale=2)

            bc_check_btn = gr.Button("🔍 跑 Badcase 检测", variant="primary")

            with gr.Row():
                with gr.Column():
                    bc_blurry = gr.Number(label="嘴糊比例 (目标 < 30%)")
                    bc_flicker = gr.Number(label="闪烁评分 (目标 < 8)")
                    bc_sync = gr.Number(label="唇音同步 (目标 > 7)")
                    bc_identity = gr.Number(label="身份保持 (目标 > 0.8)")
                    bc_yaw = gr.Number(
                        label="平均 yaw (°; 0=正面, ≥15°=侧脸, ≥25°=重度侧脸)",
                        precision=1,
                    )
                    bc_mouth = gr.Number(
                        label="张嘴幅度 p95 (目标 > 0.18; 偏小→用大嘴数据重训)",
                        precision=2,
                    )
                with gr.Column():
                    bc_report = gr.Textbox(label="诊断报告", lines=20)
                    bc_recommendation = gr.Textbox(
                        label="🎯 finetune preset 推荐 (基于上面 6 个数字自动判定)",
                        lines=4,
                        interactive=False,
                        value="跑完上方 🔍 检测后,这里会自动出推荐 preset。",
                    )

            bc_check_btn.click(
                fn=run_badcase_checklist,
                inputs=[bc_video, bc_reference],
                outputs=[bc_blurry, bc_flicker, bc_sync, bc_identity, bc_yaw, bc_mouth, bc_report],
            )

            # Whenever any of the 6 metric numbers change, refresh the
            # preset recommendation. Tab 6 re-run fills them all in one
            # .click event, so the user sees the recommendation update
            # immediately after the numbers settle.
            for bc_metric in (bc_blurry, bc_flicker, bc_sync, bc_identity, bc_yaw, bc_mouth):
                bc_metric.change(
                    fn=_recommend_finetune_preset,
                    inputs=[bc_blurry, bc_flicker, bc_sync, bc_identity, bc_yaw, bc_mouth],
                    outputs=bc_recommendation,
                )

            with gr.Accordion("🎬 短剧专项诊断 (多说话人 / 频繁切场景)", open=False):
                gr.Markdown(
                    "对短剧类输入做额外的场景/说话人数量估计。"
                    "基于 HSV histogram(场景切点)+ face bbox 聚类(说话人数),"
                    "无新依赖;若有 face_recognition,会进一步用 embedding 精确聚类。"
                )
                drama_btn = gr.Button("🎬 跑短剧场景诊断", variant="primary")
                drama_report = gr.Textbox(label="短剧诊断", lines=10, interactive=False)
                drama_btn.click(
                    fn=_diagnose_short_drama,
                    inputs=[bc_video],
                    outputs=drama_report,
                )

        # =========================================================
        # Tab 7: Training-set preview
        # =========================================================
        with gr.Tab("📁 训练集预览"):
            gr.Markdown(
                """
浏览并播放训练集中的原始视频样本。
支持从 `train_fileslist` 读取（优先），或扫描 `train_data_dir` 下的 `.mp4` 文件。
可先快速加载列表，再按需分析人脸 yaw 并筛选正脸 / 侧脸。
                """
            )
            with gr.Row():
                preview_data_dir = gr.Textbox(
                    label="train_data_dir",
                    value="",
                    scale=2,
                )
                preview_fileslist = gr.Textbox(
                    label="train_fileslist（优先使用）",
                    value="",
                    scale=2,
                )
                preview_load_btn = gr.Button("🔄 加载视频列表", variant="primary", scale=1)
                preview_analyze_btn = gr.Button("🔍 分析 yaw", variant="secondary", scale=1)

            with gr.Row():
                preview_filter = gr.Dropdown(
                    label="筛选",
                    choices=["全部", "正脸", "侧脸"],
                    value="全部",
                    scale=1,
                )
                preview_threshold = gr.Slider(
                    label="yaw 阈值 (°)，≥ 为侧脸",
                    minimum=0,
                    maximum=45,
                    value=15,
                    step=1,
                    scale=2,
                )
                preview_count = gr.Textbox(
                    label="统计",
                    value="",
                    interactive=False,
                    scale=2,
                )

            with gr.Row():
                preview_video_dd = gr.Dropdown(
                    label="选择视频",
                    choices=[],
                    value=None,
                    scale=3,
                )
                preview_yaw_info = gr.Textbox(
                    label="人脸 / yaw 信息",
                    value="",
                    lines=4,
                    interactive=False,
                    scale=1,
                )

            preview_video_player = gr.Video(label="预览", interactive=False)
            preview_videos_state = gr.State([])
            preview_analysis_state = gr.State({})

            def _load_preview_videos(data_dir: str, fileslist: str, threshold: float):
                videos, status = _list_training_videos(data_dir, fileslist)
                cache_path = _preview_cache_path(data_dir, fileslist)
                cached_analysis = _load_preview_cache(cache_path, videos)
                if cached_analysis is not None:
                    status += " | 已使用缓存"
                    return (
                        gr.update(choices=videos, value=videos[0] if videos else None),
                        status,
                        gr.update(value=None),
                        videos,
                        cached_analysis,
                        _format_preview_info(videos[0] if videos else "", cached_analysis),
                    )
                return (
                    gr.update(choices=videos, value=videos[0] if videos else None),
                    status,
                    gr.update(value=None),
                    videos,
                    {},
                    "",
                )

            def _analyze_preview_videos(
                videos: List[str],
                threshold: float,
                data_dir: str,
                fileslist: str,
            ):
                if not videos:
                    return (
                        gr.update(choices=[], value=None),
                        "⚠️ 没有视频可分析",
                        {},
                        "",
                    )
                analysis: Dict[str, Any] = {}
                frontal = 0
                side = 0
                unknown = 0
                for v in videos:
                    info = _analyze_training_video_yaw(v, n_frames=5)
                    analysis[v] = info
                    ft = info.get("face_type", "unknown")
                    if ft == "frontal":
                        frontal += 1
                    elif ft == "side":
                        side += 1
                    else:
                        unknown += 1
                status = f"已分析 {len(videos)} 个 | 正脸 {frontal} | 侧脸 {side}"
                if unknown:
                    status += f" | 未检测 {unknown}"
                _save_preview_cache(
                    _preview_cache_path(data_dir, fileslist),
                    videos,
                    analysis,
                    threshold,
                )
                return (
                    gr.update(choices=videos, value=videos[0] if videos else None),
                    status,
                    analysis,
                    _format_preview_info(videos[0] if videos else "", analysis),
                )

            def _apply_preview_filter(
                filter_type: str,
                threshold: float,
                analysis: Dict[str, Any],
                videos: List[str],
            ):
                if filter_type == "全部":
                    return (
                        gr.update(choices=videos, value=videos[0] if videos else None),
                        f"共 {len(videos)} 个视频",
                    )
                if not analysis:
                    return (
                        gr.update(choices=[], value=None),
                        "⚠️ 请先点击 🔍 分析 yaw",
                    )
                filtered: List[str] = []
                for path, info in analysis.items():
                    yaw_mean = info.get("yaw_mean")
                    is_side = yaw_mean is not None and yaw_mean >= threshold
                    if filter_type == "侧脸" and is_side:
                        filtered.append(path)
                    elif filter_type == "正脸" and not is_side and yaw_mean is not None:
                        filtered.append(path)
                return (
                    gr.update(choices=filtered, value=filtered[0] if filtered else None),
                    f"筛选后: {len(filtered)} 个视频",
                )

            def _on_preview_video_change(video_path: str, analysis: Dict[str, Any]):
                return _format_preview_info(video_path, analysis)

            preview_load_btn.click(
                fn=_load_preview_videos,
                inputs=[preview_data_dir, preview_fileslist, preview_threshold],
                outputs=[
                    preview_video_dd, preview_count, preview_video_player,
                    preview_videos_state, preview_analysis_state, preview_yaw_info,
                ],
            )
            preview_analyze_btn.click(
                fn=_analyze_preview_videos,
                inputs=[preview_videos_state, preview_threshold, preview_data_dir, preview_fileslist],
                outputs=[
                    preview_video_dd, preview_count,
                    preview_analysis_state, preview_yaw_info,
                ],
            )
            preview_filter.change(
                fn=_apply_preview_filter,
                inputs=[preview_filter, preview_threshold, preview_analysis_state, preview_videos_state],
                outputs=[preview_video_dd, preview_count],
            )
            preview_video_dd.change(
                fn=_safe_video_update,
                inputs=preview_video_dd,
                outputs=preview_video_player,
            )
            preview_video_dd.change(
                fn=_on_preview_video_change,
                inputs=[preview_video_dd, preview_analysis_state],
                outputs=preview_yaw_info,
            )

            # On page (re)load: repopulate trainer status + run dropdown
            # from the in-process _TRAINER singleton. This survives browser
            # refreshes — the Python process keeps the training subprocess
            # alive even if the user's tab disconnects, so we re-detect here.
            demo.load(
                fn=_on_page_load,
                inputs=[monitor_output_dir],
                outputs=[
                    trainer_status, launch_status, log_path_state, run_dd, monitor_btn,
                    run_dir_hidden, loss_chart_img, sync_conf_img,
                    val_video_dd, ckpt_dd, log_box, ckpt_info_box,
                    progress_bar, progress_text,
                ],
            )

    return demo
