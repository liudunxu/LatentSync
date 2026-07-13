# Changelog

本仓库所有重要变更的集中记录。**新版**在顶部。

格式参考 [Keep a Changelog](https://keepachangelog.com/)。每个条目标注影响范围：
- 🎨 docs：纯文档
- ✨ feat：新功能
- 🐛 fix：bug 修复
- ⚡ perf：性能优化
- 🔧 chore：杂项 / 配置

---

## [Unreleased] — 微调工具与文档体系

### ✨ Added
- **PEFT / LoRA / QLoRA 微调**（`scripts/train_unet_lora.py` + `merge_lora.py` + `configs/unet/stage2_lora.yaml` + `train_unet_lora.sh`）
  - `Stage 2 LoRA (256, 12-15GB)` 和 `Stage 2 QLoRA (256, 8-10GB)` presets
  - 推理侧合并工具（LoRA adapter → 标准 .pt）
  - `freeze_attn2` 选项保护 audio cross-attn
- **Gradio Fine-tune Studio**（`gradio_finetune.py`，6 个 Tab + 验证 Tab）
  - Tab 1 配置 & 启动（5 个 UNet preset + 2 个 LoRA preset + 1 个 SyncNet preset）
  - Tab 2 训练监控（15s 自动刷新）
  - Tab 3 推理对比（base vs fine-tuned）
  - Tab 3.5 验证（单 ckpt 推理 + 自动质量自检）
  - Tab 4 Identity 保护策略（4 层参数 + 推理 kwargs 生成）
  - Tab 5 数据集质量评估（HyperIQA + SyncNet 分布）
  - Tab 6 Badcase 检查清单（4 项指标）
- **训练恢复 / 质量加权 / 追踪**
  - `preprocess/manifest.py` — preprocess 断点续传 + 增量处理 + observability
  - `latentsync/utils/training_state.py` — 完整训练 state 保存（optimizer / scaler / scheduler / RNG）
  - `latentsync/utils/quality_sampler.py` — 按 HyperIQA 分数加权的 WeightedRandomSampler
  - `latentsync/utils/tracker.py` — WandB / TensorBoard 统一 logger（`LATENTSYNC_TRACKER` 环境变量）
- **评估工具**
  - `scripts/evaluate_checkpoint.py` — 端到端 checkpoint 评估（自动跑推理 + 算指标 + 出 JSON）
  - `eval/generate_report.py` — 自包含 HTML 报告（嵌入视频 data-URL）

### 🎨 Documentation
- `docs/finetune_studio_guide.md` — Fine-tune Studio 完整用户指南（11 节）
- `docs/training_pipeline.md` — 从 §1 扩充到 §26（架构 / 训练 / 评估 / 成本 / PEFT / HeyGen 对比 / MuseTalk 对比 / v1.5 vs v1.6 / Scene detection 等）
- 重点章节：
  - §5.8 TREPA 详解（输入输出 / 推理不参与 / 4 项损失 vs SyncNet）
  - §5.9 LPIPS & pixel-space supervision
  - §13 UNet 生成质量评估
  - §14 Identity 4 层防御
  - §15 训练 UI 优化建议
  - §16 训练 vs 推理对比
  - §17 训练成本与数据集规模
  - §18 PEFT / LoRA 微调
  - §19 LPIPS & koniq 完整应用
  - §20 Badcase 驱动的微调策略
  - §21 数据集格式支持
  - §22 为什么所有微调都是 Stage 2
  - §23 侧脸 / 大角度深度专题
  - §24 LatentSync 1.5 vs 1.6
  - §25 与 MuseTalk 的对比
  - §26 Scene Detection（训练 vs 推理）

### 🐛 Fixed
- `latentsync/data/unet_dataset.py` — `train_data_dir` 从 `os.listdir` 改为 `Path.rglob("*.mp4")`，**支持嵌套目录**（如 `data/train/<speaker>/<video>.mp4`）

### 🔧 Changed
- `gradio_finetune.py` 默认端口 `7861` → `6006`（与 TensorBoard 默认端口对齐）
- `requirements.txt` — 加上 `peft` / `bitsandbytes` / `wandb` / `matplotlib` / `scikit-image` 等可选依赖
- `docs/changelog_v1.5.md` 和 `docs/changelog_v1.6.md` — 保留（官方版本记录）

---

## 之前的历史

更早的 commit 通过 `git log` 可见（按时间倒序）：

```bash
git log --oneline -20
```

主要 commit：
- `2c2baf3` docs: add §19 LPIPS & koniq_pretrained.pkl complete usage guide
- `9d7dc92` docs: add §18 PEFT / LoRA fine-tuning guide
- `6036e23` docs: add §17 training cost + dataset size guide
- `42659e6` docs: explain why all finetune presets are Stage 2 + side-face deep dive
- `42659e6` feat: tier-1+2 training/eval optimizations + gradio validation tab
- `b0e037d` feat: implement LoRA / QLoRA training script and merge utility
- `81f0cd6` feat(gradio): add LoRA/QLoRA presets + 3 new tabs
- `24d4897` docs: add §4.7 StableSyncNet vs SyncNetEval comparison
- `21d1f2b` docs: add §5.8.15 'why do we need these besides SyncNet?'

---

## 历史版本（来自 `docs/changelog_v1.5.md` / `changelog_v1.6.md`）

### LatentSync 1.6（2025-06-11 发布）
- 训练数据从 256 升到 512
- 缓解牙齿 / 嘴唇模糊问题
- 模型结构 / 训练策略**不变**

### LatentSync 1.5（2025-03-14 发布）
- 加入 Temporal Layer（Motion Module）— 论文修正后实现
- 改进中文视频表现
- Stage 2 VRAM 优化到 20 GB（4 项优化：gradient checkpointing / FlashAttention-2 / CUDA cache 清理 / 只训 temporal+audio）
- 移除 xFormers / Triton 依赖
- 升级 diffusers 到 0.32.2
- 可在单张 RTX 3090 上跑 Stage 2 训练

### LatentSync 1.0（论文版，2024-12-12）
- SD1.5 UNet + Audio Cross-Attention
- StableSyncNet（94% HDTF 准确率）
- TREPA 时序对齐损失
- 5 项 SOTA（HDTF FID 7.22 / SSIM 0.79 / Sync_conf 8.9 / LMD 0.30 / FVD 162.74）
