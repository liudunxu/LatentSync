# 短剧微调快速参考卡 🎬

> **场景**: 短剧/有声漫画/对谈类视频(2-4 个说话人,频繁切场景)。
> LatentSync 默认 pipeline 假设单人,套到短剧会有:第 2 张脸被 paste-back 糊掉、audio 跟错人、切场第一帧触发 face_jump 跳回原图。
>
> 本卡: 6 步端到端流程, **不改 model**,全用现有工具拼。

---

## 6 步流程

| 步 | 命令 / 操作 | 产物 |
|---|---|---|
| **1. 切场景** | `python tools/preprocess_short_drama.py --input data/raw/ep01.mp4 --output-dir data/drama/ep01 --threshold 0.35` | `data/drama/ep01/shots/*.mp4` + `.wav` + `fileslist.txt` + `shots.json` |
| **2. 分桶** | `python tools/curate_finetune_samples.py --source-dir data/drama/ep01 --output-dir data/drama/ep01_curated --scale medium` | `data/drama/ep01_curated/{frontal,side_face,fast_motion}/` + `fileslist.txt` + `curation_report.json` |
| **3. 自检** | `cat data/drama/ep01_curated/curation_report.json \| python -m json.tool \| head -40` | 期望: kept ≥ 800, 三桶都不空 |
| **4. Tab 1 配** | preset 选 `🎬 Short Drama (LoRA+conv, 多说话人, 18-22GB)`,填 train_data_dir + train_fileslist,🚀 启动 | 后台训练 (单卡 A100 ~6-12h) |
| **5. Tab 2 选 ckpt** | 看 sync_conf 整体水平 + 每 1k 步 val_video 快放,挑 2-3 个候选 | 候选 ckpt 路径 |
| **6. 验证 + merge** | Tab 3.5/6 验证最佳 ckpt → Tab 2 「🔀 合并 LoRA」→ `latentsync_unet.pt` 可部署 | 最终 ckpt |

---

## 🎬 Short Drama preset 关键参数

| 参数 | 值 | 为什么 |
|---|---|---|
| `target_modules` | 11 项 (att + conv1/conv2/conv_shortcut/proj_in/proj_out/conv_in/conv_out) | 切点附近脸几何漂需要 conv capacity |
| `sync_loss_weight` | **0.18** | 短剧容错低,口型必须跟紧 |
| `num_frames` | **16** | 短段(5-15s)长上下文稀释信号 |
| `save_ckpt_steps` | **500** | 短段多,多存点方便挑 |
| `max_train_steps` | 25000 | scale=medium 默认 |
| `lr_scheduler` / `lr_warmup_steps` | cosine / 300 | 晚段不卡 |

---

## 故障排查速查

| 症状 | 修法 |
|---|---|
| Step 1 切出大量 1-2 帧 micro-shot | `--threshold` 提到 0.40-0.50 |
| Step 1 漏掉大段切换 | `--threshold` 降到 0.25 |
| Step 2 `fast_motion=0` | 数据太静,不是真短剧 |
| 训练 sync_conf 退化 | 确认 `freeze_attn2=True`(默认) |
| 推理第二张脸被驱动 | **已知局限**:Step 4 (model-side routing) 未实现 |
| 切场后第一帧跳原图 | 推理参数 `face_jump_threshold` 调高 |

---

## 数据量速查

| scale | target | max_candidates | min_frames | 训练时长 (单 A100) |
|---|---|---|---|---|
| small | 200 | 2,000 | 30 | ~2-4h |
| medium | 1000 | 10,000 | 60 | ~6-12h |
| large | 5000 | 50,000 | 120 | ~2-4 天 |

短剧推荐 **medium** (1000 条)起步,不够再升 large。

---

## 关键文件

| 文件 | 作用 |
|---|---|
| `tools/preprocess_short_drama.py` | Step 1 切场景 + 提单人 |
| `tools/curate_finetune_samples.py` | Step 2 分桶 + score_cache |
| `tools/download_curated_finetune_set.py` | URL → download → curate 一条龙 |
| `docs/finetune_studio_guide.md` §6.4 | 详细版(故障排查 / 局限 / 完整参数表) |
| `gradio_finetune.py` | UI, Tab 1 preset 下拉 + Tab 6 🎬 accordion |

---

## 已知局限(诚实声明)

当前流程**没动 model**:
- 多说话人同时开口 → fallback 选最大脸
- 密集场景(>2 人) → face_detector 跟踪容易丢
- 复杂切场(快速 zoom) → 直方图切场景可能误判

完整 multi-speaker audio routing 是 **Step 4** (model 改动),需要 2-speaker 合成测试集才能 regression-test, **当前未实现**。