# LatentSync Lipsync 链路优化路线图

> 文档目的：系统梳理当前 lipsync 链路的策略现状、性能/质量/badcase 优化空间，为后续迭代提供决策依据。
> 基线版本：当前 `main`（默认 `mask.png`、guidance=1.5、steps=40、DeepCache interval=3、num_frames=16）
> 核心文件：`latentsync/pipelines/lipsync_pipeline.py`、`api.py`、`latentsync/utils/*`

---

## 1. 整体链路流程

```
Client POST /api/lipsync
    │
    ▼
api.py: create_lipsync
    ├─ 下载视频/音频/avatar 到 /inputs/<job_id>/
    ├─ 加载 avatar face embedding（若提供）
    ▼
LatentSyncApiRuntime.synthesize（单例，run_lock 串行）
    │
    ▼
LipsyncPipeline.__call__(video_path, audio_path, video_out_path, ...)
    │
    ├─ audio_encoder.audio2feat(audio)         # Whisper tiny → 50fps 特征
    ├─ feature2chunks(..., offset_seconds)     # 映射到 25fps，支持音频同步偏移
    ├─ read_video(..., use_decord=True)        # 整盘解码到内存
    ├─ 场景切分 / 合并
    │
    ▼ 每个 scene 调用 _process_clip
    │
    ├─ loop_video / affine_transform_video     # 人脸检测 + 对齐 + 预过滤
    │     - InsightFace: bbox + 106 landmark + embedding + pose yaw
    │     - AlignRestore.align_warp_face → 512×512
    │     - 生成 skip_mask / continuity_break_mask / dynamic masks / mouth_info
    │
    ├─ shot_passthrough_guard（可选）
    ├─ silent_frame_mask（可选，默认关闭）
    ├─ audio_motion_scales（音频能量自适应）
    │
    ▼ diffusion 推理循环（batch = 16 frames）
    │     - 全 skip batch 短路跳过 UNet
    │     - prepare masks / latents → DDIM 40 steps → decode latents
    │
    ▼ 后处理链（按顺序）
    │     1. paste_surrounding_pixels_back（动态 mask 贴回）
    │     2. _match_color_to_reference（颜色匹配）
    │     3. _restore_reference_detail（嘴周细节恢复）
    │     4. _unsharp_mask（嘴部锐化）
    │     5. _smooth_face_sequence（3-tap 时序 EMA）
    │     6. mouth motion preserve（音频自适应运动保持）
    │     7. mouth temporal stabilization（时序稳定）
    │     8. quality gate（质量门，默认关闭）
    │     9. adaptive quality fallback（自适应质量回退，API 默认开启）
    │     10. CodeFormer 人脸修复（可选，默认关闭）
    │
    ├─ restore_video → AlignRestore.restore_img  # 512×512 贴回原帧
    │
    ▼
write_video + soundfile + ffmpeg mux → /outputs/<job_id>/result.mp4
```

关键阶段代码位置：

| 阶段 | 文件 | 行号 |
|---|---|---|
| 入口 / 场景切分 | `latentsync/pipelines/lipsync_pipeline.py` | 4408–4864 |
| 单 scene 处理 | `latentsync/pipelines/lipsync_pipeline.py` | 3391–4406 |
| 人脸检测/对齐/预过滤 | `latentsync/pipelines/lipsync_pipeline.py` | 2228–2785 |
| yaw 估计 | `latentsync/pipelines/lipsync_pipeline.py` | 274–395 |
| episode pad / warn-run skip | `latentsync/pipelines/lipsync_pipeline.py` | 412–525 |
| shot-level guard | `latentsync/pipelines/lipsync_pipeline.py` | 714–804 |
| segment consistency | `latentsync/pipelines/lipsync_pipeline.py` | 918–1066 |
| 动态嘴部 mask | `latentsync/pipelines/lipsync_pipeline.py` | 1613–1705 |
| 嘴部遮挡 / 锐度检测 | `latentsync/pipelines/lipsync_pipeline.py` | 1706–1801 |
| 时序 EMA | `latentsync/pipelines/lipsync_pipeline.py` | 1839–1947 |
| 自适应质量评分 | `latentsync/pipelines/lipsync_pipeline.py` | 1196–1252 |
| 自适应阈值 / hysteresis | `latentsync/pipelines/lipsync_pipeline.py` | 1254–1323 |
| 颜色匹配 | `latentsync/pipelines/lipsync_pipeline.py` | 1325、3871 |
| 细节恢复 | `latentsync/pipelines/lipsync_pipeline.py` | 1501、3890 |
| unsharp mask | `latentsync/pipelines/lipsync_pipeline.py` | 1403、3900 |
| 运动保持 / 时序稳定 | `latentsync/pipelines/lipsync_pipeline.py` | 3940–4016 |
| adaptive quality fallback | `latentsync/pipelines/lipsync_pipeline.py` | 4056–4114 |
| CodeFormer 后处理 | `latentsync/pipelines/lipsync_pipeline.py` | 4220–4259 |
| 对齐还原贴回 | `latentsync/utils/affine_transform.py` | 57–91 |
| API 默认参数 / 路由 | `api.py` | 222–677、1640–1760、2002 |
| 音频特征提取 | `latentsync/whisper/audio2feature.py` | 88–130 |
| 视频解码 | `latentsync/utils/util.py` | 58–90 |
| 视频编码 | `latentsync/utils/util.py` | 135–146 |

---

## 2. 现状策略总览

### 2.1 质量优化策略

| 策略 | 说明 | 默认参数 / 开关 |
|---|---|---|
| 动态嘴部 mask | 基于 106 点 landmark 生成嘴部椭圆 mask；用 `fixed_keep_mask` 限制生成区不超出 `mask.png` 下脸区 | `generate_dynamic_mouth_mask`；`aligned_mouth_ema_alpha=0.85` |
| 颜色匹配 | 生成脸向参考脸做 per-channel mean/std 迁移，mask 经 max_pool 膨胀覆盖羽化带 | `color_match_strength=0.60` |
| 细节恢复 | 在嘴部 core 外恢复参考脸高频皮肤细节，core 内保护生成嘴型 | `mouth_detail_strength=0.65` |
| 嘴部锐化 | 对生成区域做 unsharp mask | `mouth_sharpen_strength=0.30` |
| 时序 EMA 平滑 | 3-tap 三角核 `(0.25,0.5,0.25)` 跨帧平滑，限制在生成区内 | `_smooth_face_sequence` |
| 嘴部运动保持 | 在 mouth core 内将平滑结果按 `audio_motion_scale` 拉回当前帧 | `mouth_motion_preserve_strength=0.45`；`mouth_audio_motion_min_scale=0.85`，`max_scale=1.60` |
| 嘴部时序稳定 | 1-order EMA 向上一帧稳定嘴部 blend；delta 过大时断链 | `mouth_temporal_stabilization_strength=0.15`；`max_delta=0.12` |
| CodeFormer 人脸修复 | 对非 skip 对齐脸做修复，支持 Tier1/2/3（自适应 w、retry、嘴部 ROI paste） | 默认关闭 |
| 侧脸边界渐变 | skip ↔ generate 边界做线性 cross-fade | `side_face_blend_fade_frames=3` |

### 2.2 性能优化策略

| 策略 | 说明 | 默认参数 / 开关 |
|---|---|---|
| DeepCache | 给 UNet 包 DeepCacheHelper，cache_interval=3，跳过 2/3 UNet forwards | 默认开启（load 时固定） |
| batch 推理 | 每 16 帧一个 batch 跑 diffusion | `num_frames=16` |
| batch 短路 | 整 batch 都被 skip 时直接跳过 UNet，复用源帧 | `_process_clip:3723` |
| 场景切分/合并 | 按硬切切分 scene，过短 scene 合并，减少状态污染 | `scene_split_enabled=True` |
| 静音 skip | 长静音段直接原帧 passthrough | 默认关闭 |
| 全串行 run_lock | 单 GPU 上所有请求通过 `runtime.run_lock` 串行 | `api.py:1077`、`api.py:1556` |

### 2.3 Badcase 防御策略

| 策略 | 触发条件 | 生产默认值 |
|---|---|---|
| 检测失败 fallback | `face is None` | 无条件 skip |
| yaw 侧脸过滤 | `abs(yaw) > yaw_skip_threshold` | 40° |
| yaw rate 过滤 | 相邻帧 yaw 差 > threshold | 10°/frame |
| aggressive 侧脸 passthrough | yaw 在 (threshold, yaw_skip_threshold) 区间 | 0（关闭） |
| episode pad / warn-run skip | 连续 yaw 进入 warn band 前后扩展 / 持续 warn | pre_pad=3, post_pad=3, ratio=0.75 |
| face jump | landmark 中心/尺度突变 | 默认 0（关闭） |
| 时序 continuity break | embedding 相似度 <0.70、geometry shift >35%、嘴部 diff >0.10 | 硬编码/可配置 |
| mouth occlusion | 嘴部 ROI 暗像素比例 | 阈值 1.0（关闭） |
| motion blur | 拉普拉斯方差过低 | 0.08 |
| identity 过滤 | 与 avatar/reference embedding 点积 < threshold | 默认关闭；threshold=0.5 |
| small face | 人脸面积 / 帧面积 < 0.015 | 0.015 |
| scene cut break | 相邻源帧场景切分得分 >0.45 | 0.45 |
| shot passthrough guard | shot 内 prefilter skip 比例过高 | 默认关闭 |
| adaptive quality fallback | 生成后综合质量分 <0.35，且比例受 max_ratio=0.35 限制 | API 默认开启 |

---

## 3. 待优化点

### 3.1 性能优化

#### 🔥 小改动、高收益（建议优先）

| # | 优化点 | 预期收益 | 风险 | 代码位置 |
|---|---|---|---|---|
| 1 | **视频解码按需分片**：用 decord `[start:end]` 按 scene/batch 读，不一次性 `vr[:]` | 内存 ↓ 50–80%，启动快 | 低 | `util.py:86` |
| 2 | **`prepare_masks_and_masked_images` batch 化** | 预处理 ↓ 30–50% | 低 | `image_processor.py:196` |
| 3 | **人脸检测 key-frame + embedding 缓存**：每 N 帧跑一次 InsightFace，中间帧用 landmark EMA/光流插值；同一身份 embedding 缓存 | 人脸阶段 ↓ 40–60% | 中（跟踪质量） | `affine_transform_video:2387` |
| 4 | **修复 audio embedding 缓存 key**：当前 `replace(".mp4", ...)` 对 wav/mp3 不生效，且无内容哈希 | 重复音频请求 ↓ 大量 CPU | 低 | `audio2feature.py:129` |
| 5 | **`_smooth_face_sequence` 向量化**：用 1-D conv / cumsum 替代 Python loop | 后处理 ↓ 20–30% | 低 | `lipsync_pipeline.py:1839` |
| 6 | **`/api/faces` 不拿 `run_lock`**：它只做人脸检测，可与 lipsync 并行 | 并发提升 | 低 | `api.py:2002` |
| 7 | **ffmpeg 直接 pipe 写，避免二次编码**：当前临时视频 `-crf 13` + 最终 `-crf 18`，有损两次 | IO/编码 ↓ 20% | 低 | `util.py:135`、`__call__:4855` |
| 8 | **`torch.compile` UNet/VAE** | inference ↓ 10–30% | 中（编译时间） | `api.py` load 后 |
| 9 | **合并后处理重复 Gaussian blur**：detail restore 和 unsharp 各做一次 blur，可复用 | kernel launch ↓ | 低 | `lipsync_pipeline.py:1403、1501` |
| 10 | **`prepare_latents` 直接按 batch 采样**：当前 1 frame repeat 到 16 帧 | 减少 repeat 开销 | 低 | `_process_clip:3694` |

#### ⚙️ 中等投入、中等收益

| # | 优化点 | 预期收益 | 风险 | 代码位置 |
|---|---|---|---|---|
| 11 | **batch affine warp / restore**：多张图 + 多个仿射矩阵一次 `kornia.warp_affine` | 对齐阶段 ↓ 40% | 中 | `affine_transform.py` |
| 12 | **sliding-window batch overlap**：相邻 batch 重叠 4–8 帧，边界 cross-fade | 质量 + 边界一致性 | 中 | `_process_clip` |
| 13 | **DeepCache 动态 interval / 热切换**：静态头加大 interval，快速嘴动降级到 1 | 灵活度 + 部分质量回退 | 中 | `api.py:1184` |
| 14 | **CPU 阶段移出 `run_lock`**：decode、face detect、whisper 与 GPU inference 并行 | 并发提升明显 | 中 | `runtime.synthesize` |
| 15 | **NVENC / 硬件编码** | 编码 ↓ 50%+ | 中（兼容性） | `util.py:135` |
| 16 | **Whisper encoder 直接 forward，跳过完整 transcribe** | audio 特征 ↓ 30% | 中 | `audio2feature.py:110` |
| 17 | **VAE slicing/tiling**：长视频/高分辨率启用 | 显存 ↓ | 中 | `lipsync_pipeline.py:129` |

#### 🏗️ 大投入、架构级

| # | 优化点 | 预期收益 | 风险 |
|---|---|---|---|
| 18 | UNet TensorRT / ONNX 导出 | inference ↓ 30–50% | 高（调试/精度） |
| 19 | UNet 蒸馏到 4–8 步（LCM / 专用蒸馏） | 整体 ↓ 5–10× | 高（训练/质量） |
| 20 | 多 GPU worker / 真正异步队列（Celery / rq） | 吞吐线性扩展 | 高（架构） |
| 21 | GPU decode + end-to-end GPU pipeline | decode + preprocess ↓ | 高（兼容性） |
| 22 | 专用 lightweight face tracker 替代 InsightFace | 人脸阶段 ↓ 70% | 高（精度/ROI） |

### 3.2 质量优化

#### 🔥 快速收益、低风险

| # | 优化点 | 问题 | 建议方案 | 代码位置 |
|---|---|---|---|---|
| 1 | **质量档位封装** | `color_match/detail/sharpen` 强度对前端不直观 | 提供 `quality_preset_override = {natural, balanced, sharp}`，映射到具体参数 | `api.py:385–405` |
| 2 | **restore_img 羽化 σ 自适应分辨率** | 当前 σ=4 对 1080p 边界硬 | `σ = max(4, min(h,w)/256)` | `affine_transform.py:73–91` |
| 3 | **adaptive quality hysteresis 加长** | 当前 2 帧对 25fps 仅 80ms，孤立闪烁抑制不足 | 提到 4–5 帧，或加最小 run-length 过滤 | `lipsync_pipeline.py:1254–1323` |
| 4 | **color match 口腔暗部保护** | 全局 mean/std 会把口腔暗部拉向皮肤均值，发灰 | 在 `_mouth_core_mask` 内按亮度分桶，暗部降低 transfer 强度 | `lipsync_pipeline.py:1325` |
| 5 | **CodeFormer mouth-only ROI 动态化** | 固定矩形 `(0.55–0.74, 0.30–0.70)` 对大嘴/小嘴一刀切 | 改为基于 `aligned_mouth_info` 的椭圆 | `codeformer_restorer.py:617` |

#### ⚙️ 中等收益、需要验证

| # | 优化点 | 问题 | 建议方案 | 代码位置 |
|---|---|---|---|---|
| 6 | **音频驱动的 EMA 权重** | 固定 3-tap 核不区分语音内容，静音段过度平滑 | 高能量帧增大当前帧权重，静音帧降低权重 | `lipsync_pipeline.py:1839` |
| 7 | **自适应 temporal stabilization delta** | `max_delta=0.12` 对深色口红/胡须/大笑易误断 | 按嘴部开口度动态调整：闭嘴帧阈值低，大笑帧阈值高 | `lipsync_pipeline.py:3958` |
| 8 | **局部光照感知的 color match** | 全局 mean/std 在侧光/斑驳光影下像贴图 | 按 landmark 分区（上唇/下唇/嘴角/下巴）分别 transfer 再羽化 | `lipsync_pipeline.py:3871` |
| 9 | **mask 边界 temporal smooth** | landmark 噪声导致 mask 边界逐帧抖动 | 对 `paste_mask_512` 序列做时序高斯/EMA 平滑 | `lipsync_pipeline.py:2835` |
| 10 | **牙齿/舌头细节增强** | diffusion 无显式约束，/th/、/r/、大笑易牙齿粘连/舌头缺失 | 对 landmark 牙齿区单独增强高频；口腔暗部保护 color match | `lipsync_pipeline.py:1459、1501、3900` |

#### 🏗️ 高潜力、需训练/数据

| # | 优化点 | 说明 |
|---|---|---|
| 11 | **Gradient-domain blending（Poisson / Laplacian pyramid）** | 替代高斯羽化贴回，显著减少光照接缝 |
| 12 | **基于音频摩擦音的后处理强度动态调整** | /s/、/sh/、/th/ 能量高时降低 stabilization，保留细节 |
| 13 | **牙齿/舌头的显式生成约束或后处理模块** | 需要数据训练或专用 post-process 网络 |
| 14 | **按 scene 统计 reference 的 running mean/std** | 使 color match 在同场景内更稳定 |

### 3.3 Badcase 降低

#### 侧脸 / 快速转头

| 现状问题 | 优化建议 | 代码位置 |
|---|---|---|
| `yaw_skip_threshold=40°` 较宽，部分大侧脸进入生成 | 提供按内容自适应的侧脸策略：连续 3 帧 yaw>22° 即整条 run passthrough | `_estimate_yaw_degrees`、`api.py` |
| yaw 估计对低头/抬头（pitch）无区分，mouth aspect/area 信号易误判 | 加入 pitch/roll 简易估计；当鼻子-眼睛垂直关系异常时降低 mouth geometry 信号权重 | `_estimate_yaw_degrees:274` |
| yaw-rate 仅一阶差分，快速转头边界帧仍可能漏检 | 引入二阶加速度 / 中值滤波；连续单调变化时提前 skip | `_stabilize_yaw_for_rate:1068` |
| affine 对齐 3 点假设在 profile 下失效 | 大侧脸直接 skip，不强行对齐 | `affine_transform_video` |
| 缺少 landmark 可见性信号 | 当侧脸导致一半 landmark 不可见时强制 skip | `face_detector.py` |

#### 遮挡（手、麦克风、口罩）

| 现状问题 | 优化建议 | 代码位置 |
|---|---|---|
| `mouth_occlusion_skip_threshold=1.0` 默认关闭 | 默认启用并配合更鲁棒的遮挡检测 | `api.py` |
| 仅统计嘴部 ROI 暗像素，对肤色物体/金属麦克风误判高 | 引入 landmark 可见性 + 肤色一致性 + 边缘强度 + 多帧 mouth texture consistency | `_mouth_occlusion_score:1707` |
| 无 mask/麦克风专用 heuristic | 嘴部 ROI 梯度异常低且颜色均匀时触发 | `_mouth_occlusion_score:1707` |

#### 弱音频 / 静音 / 气声

| 现状问题 | 优化建议 | 代码位置 |
|---|---|---|
| `speech_gate_enabled=False` 默认关闭 | 默认启用 silent skip，并加 `silent_pad_frames` 平滑过渡 | `api.py` |
| `mouth_audio_adaptive_motion` 用纯 RMS，噪声 RMS 易被误判为语音 | 用 Whisper chunk L2 norm 或 VAD 替代纯 RMS | `audio2feature.py` |
| 弱音频段 stabilization 过强导致“冻嘴” | 弱音频时降低 `mouth_temporal_stabilization_strength` | `_process_clip:3958` |

#### 大表情 / 大笑 / 大喊

| 现状问题 | 优化建议 | 代码位置 |
|---|---|---|
| `fixed_keep_mask` 可能截断大嘴型 | 评估按 `mouth_open_ratio` 动态外扩生成区；或前端可选更宽松 mask（需 A/B） | `generate_dynamic_mouth_mask:1614` |
| `mouth_audio_motion_max_scale=1.60` 会进一步放大极端嘴型 | 当开口度接近训练分布上限时压回 1.0 | `_process_clip:3947` |
| `max_delta=0.12` 对大笑易断链 | 按开口度动态调整 delta | `_process_clip:3958` |
| 缺少“嘴型与音频能量一致性”校验 | 在 adaptive quality score 中加入该项 | `_compute_frame_quality_score:1196` |

#### 多人 / 身份切换

| 现状问题 | 优化建议 | 代码位置 |
|---|---|---|
| `apply_identity_filter=False` 默认关闭，多人场景易跟错 | 默认启用或至少多脸请求给出警告 | `api.py` |
| `identity_similarity_threshold=0.5` 过松 | 短剧场景提供 `strict_identity_mode`（0.65–0.70） | `api.py` |
| detect-fail 帧继承 prev_track_id，可能跨人传播 | 检测失败时重新验证 embedding 一致性再决定 track_id | `affine_transform_video` |
| 无显式 speaker diarization | 长视频可结合音频/embedding 聚类做轻量说话人切换检测 | 架构层 |

#### 边界帧 / 首帧 / 尾帧 / Batch 边界

| 现状问题 | 优化建议 | 代码位置 |
|---|---|---|
| batch 首帧 `reset_p_bias()`，时序 EMA 在 batch 开头重置 | 当相邻 batch 连续且未跨 scene cut 时保留时序状态 | `_process_clip` |
| hysteresis 保留边界 run，但首/尾帧本身质量差时仍保留 | 对边界 run 也加最小长度限制 | `_apply_quality_hysteresis` |
| loop_video 反向 loop 改变 source_indices，边界 continuity_break 处理简单 | 在 loop 边界处显式重置 temporal state 并加 blend | `loop_video` |

#### 硬切 / Shot 边界

| 现状问题 | 优化建议 | 代码位置 |
|---|---|---|
| scene_cut_break 只重置状态，不 skip；切镜头瞬间仍可能用旧脸 | shot 边界处强制短暂 passthrough 或状态隔离 | `_process_clip` |
| shot_passthrough 只依赖 prefilter skip 比例 | 新增 shot 时若 embedding/光照 histogram 突变，强制全 passthrough | `_apply_shot_passthrough_guard:714` |

---

## 4. 优化路线图

### Phase 1：低风险快速收益（1–2 周）

1. 视频解码按需分片，不整盘加载。
2. `prepare_masks_and_masked_images` batch 化。
3. 修复 audio embedding 缓存 key（扩展名 + 内容哈希）。
4. `/api/faces` 释放 `run_lock`。
5. `_smooth_face_sequence` 向量化。
6. ffmpeg 直接 pipe 写，避免二次编码。
7. 合并后处理重复 Gaussian blur。
8. 质量档位封装（natural / balanced / sharp）。
9. restore_img 羽化 σ 自适应分辨率。
10. adaptive quality hysteresis 加长到 4–5 帧。

### Phase 2：质量与 badcase 重点攻坚（2–4 周）

1. 人脸检测 key-frame + embedding 缓存。
2. 引入 pitch/roll 估计，改善低头/抬头 yaw 误判。
3. yaw-rate 加入二阶加速度检测。
4. 嘴部遮挡检测升级（landmark 可见性 + 肤色一致性 + 边缘）。
5. 默认启用 speech gate / silent skip，并用 Whisper L2 norm 替代 RMS。
6. 音频驱动的 EMA 权重。
7. 自适应 temporal stabilization delta。
8. CodeFormer mouth-only ROI 动态化。
9. 局部光照感知的 color match。
10. mask 边界 temporal smooth。

### Phase 3：架构级优化（1–3 个月）

1. CPU 预处理阶段完全移出 `run_lock`。
2. sliding-window batch overlap。
3. DeepCache interval 可 per-request / 自适应。
4. `torch.compile` / TensorRT / ONNX 模型优化。
5. UNet 蒸馏到 4–8 步。
6. 多 GPU worker / 异步队列。
7. 端到端 golden-sample 回归测试覆盖典型 badcase。

---

## 5. 测试缺口

| 缺口 | 说明 | 建议 |
|---|---|---|
| 无端到端 GPU 推理测试 | diffusion、VAE、UNet 在单元测试中未覆盖 | 在远程 GPU 上建立 smoke test，固定 seed + 短样本 |
| 无 `affine_transform_video` 集成测试 | InsightFace/face_embedder 是 mock 的 | 增加真实 landmark 抖动、漏检、多脸选择测试 |
| 无 `_process_clip` orchestration 测试 | skip mask 与 diffusion 短路、后处理链顺序影响未验证 | 构造固定 skip mask 验证各分支行为 |
| 无性能/并发基准测试 | DeepCache、batch size、scene 并行均无数据 | 增加 perf regression test，记录各阶段耗时 |
| 无 badcase 样本回归测试 | 侧脸、遮挡、快速转头、弱音频、大表情等缺少 golden output | 建立固定样本集 + 指标（sync confidence、identity、sharpness） |
| API 默认值与 pipeline 默认值不一致 | 如 `min_merged_lipsync_seconds` API=0.4s，pipeline=1.5s | 统一默认值并加测试锁定 |
| CodeFormer 真实权重路径未覆盖 | 测试只在缺权重 fallback | 远程真实权重下跑推理回归 |

---

## 6. 诊断抓手

每次请求已有日志标签：

- `[FaceMatch]`：per-filter skip 计数（detect_fail、yaw_skip、yaw_rate_skip、identity_skip、mouth_occlusion_skip、motion_blur_skip、face_jump_skip 等）。
- `[LipSync]`：adaptive_quality_fallback 计数、delta_skip_frames 等。
- `[Diag]` / `[ShotGuard]` / `[CodeFormer]`：诊断与后处理统计。

建议补充到 `run_stats` 返回前端的字段：

- `mouth_temporal.delta_median`
- `audio_motion_median_scale`
- `adaptive_quality_fallback_frames`
- 各阶段耗时（decode / face detect / whisper / denoise / postprocess / encode）

---

## 7. 关键原则（来自 AGENTS.md，需遵守）

1. **Mask & feather 基线锁定**：默认 `latentsync/utils/mask.png` 不要无明确请求切换为 mask2/3/4/5；动态 mask 可 clamp 到 fixed_keep_mask。
2. **自然度默认参数锁定**：`LATENTSYNC_GUIDANCE_SCALE=1.5`、`LATENTSYNC_INFERENCE_STEPS=40`、`mouth_temporal_stabilization_strength=0.15`、`mouth_audio_motion_min_scale=0.85` 不要静默回退。
3. **侧脸检测逻辑**：阈值调整是 band-aid，真正修复需改善底层信号质量；修改前使用 plan mode。
4. **前端字段后缀**：新增可配参数必须使用 `*_override` 后缀才能被 server 读取。
5. **Push policy**：提交后可直接 `git push origin main`。

---

*文档生成时间：2026-06-22。后续每轮迭代后应更新本文档，标记已完成项。*
