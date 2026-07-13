# LatentSync 文档索引

> **从哪开始?** 90% 的 finetune 用户只需要 §6(presets)+ §7(faq)。

## 按使用场景分

### 🎯 我想 finetune 改善 badcase

| 文档 | 用途 | 大小 |
|---|---|---|
| [finetune_studio_guide.md](finetune_studio_guide.md) | **完整 finetune 操作手册** — 启动 / 6 个 Tab 详解 / 数据集评估 / badcase 速查 / 短剧专项 / 预制数据集 / 进阶 CLI | 1000+ 行 |
| [short_drama_workflow.md](short_drama_workflow.md) | **短剧 1 页速查卡** — 6 步流程 + preset 参数表 + 故障排查 | 79 行 |
| [tools/finetune_starter_urls.example.txt](../tools/finetune_starter_urls.example.txt) | 给 curate_finetune_samples 的 URL 模板 | 65 行 |

### 🛠️ 我想理解系统架构 / 改 model

| 文档 | 用途 |
|---|---|
| [training_pipeline.md](training_pipeline.md) | 训练全流程技术细节 (5600+ 行, deep reference) |
| [lipsync_strategy.md](lipsync_strategy.md) | 唇同步策略技术分析 |
| [lipsync_optimization_roadmap.md](lipsync_optimization_roadmap.md) | 优化 roadmap + 已知瓶颈 |
| [syncnet_arch.md](syncnet_arch.md) | SyncNet 模型架构 |

### 📝 我想看 changelog / 发布说明

| 文档 | 用途 |
|---|---|
| [changelog_v1.6.md](changelog_v1.6.md) | v1.6 (2025-06-11) — 512×512 + blurriness fix |
| [changelog_v1.5.md](changelog_v1.5.md) | v1.5 (2025-03-14) — temporal + Chinese + 20GB VRAM |
| [changelog_codeformer.md](changelog_codeformer.md) | CodeFormer 后处理集成 |
| [codeformer_integration.md](codeformer_integration.md) | CodeFormer 集成技术文档 |

## 按目录树

```
docs/
├── README.md                            ← 本文件
├── finetune_studio_guide.md             ← ★ 操作手册入口
├── short_drama_workflow.md              ← 短剧速查卡
├── training_pipeline.md                 ← ★ 训练 deep reference
├── lipsync_strategy.md
├── lipsync_optimization_roadmap.md
├── syncnet_arch.md
├── codeformer_integration.md
├── changelog_v1.5.md
├── changelog_v1.6.md
└── changelog_codeformer.md

tools/
├── finetune_starter_urls.example.txt     ← ★ 数据源 URL 模板
├── preprocess_short_drama.py           ← ★ 短剧场景切分
├── curate_finetune_samples.py           ← ★ 按 yaw/motion 分桶
├── download_curated_finetune_set.py     ← URL → 下载 → curate 一条龙
├── init_finetune_dataset.py             ← ★ HF Hub 预制数据集
├── prebuilt_datasets.yaml               ← 4 个预制数据集配方
├── merge_lora.py                        ← (含 --push_to_hub)
├── download_checkpoints.py
├── download_web_videos.py
└── ...
```

## 快速链接

- 启动 gradio: `python gradio_finetune.py --port 6006`
- 启动 fine-tune UI: 同样命令,UI 在 Tab 1「📚 预制数据集」accordion 一键下数据
- CLI 数据集初始化: `python tools/init_finetune_dataset.py --list`
- Merge + push HF Hub: `python -m scripts.merge_lora --push_to_hub username/repo`