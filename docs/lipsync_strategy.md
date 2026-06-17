# LatentSync 唇形同步策略文档

> 本文档整理 LatentSync 推理 pipeline 中的核心策略，分为**模型推理**与**策略控制**两大部分，用于指导参数调优、问题排查和新功能接入。

---

## 1. 模型推理部分

### 1.1 基础推理参数

| 参数 | 服务端默认值 | API 字段 | 说明 |
|------|-------------|----------|------|
| `num_inference_steps` | 40 | `inference_steps_override` | DDIM 去噪步数。步数越多越平滑、越慢；当前默认值在质量与速度间取平衡。 |
| `guidance_scale` | 1.5 | `guidance_scale_override` | Classifier-free guidance 权重。越大嘴型越跟随音频，但容易不自然；越小越自然，但同步感减弱。 |
| `eta` | 0.0 | - | DDIM 随机性。0 为确定性采样。 |
| `weight_dtype` | `torch.float16` | - | 推理精度。fp16 省显存、快；fp32 更稳但慢。 |
| `generator` / `seed` | 随机 | `seed_override` | 随机种子，控制可复现性。 |
| `num_frames` | 16 | - | 每次 UNet 前向处理的连续帧数。受显存限制，一般不动。 |
| `video_fps` | 25 | - | 输入视频帧率，决定 whisper feature 与视频帧对齐关系。 |
| `audio_sample_rate` | 16000 | - | 音频采样率，whisper 特征提取用。 |
| `height` / `width` | `sample_size * vae_scale_factor` | - | 推理分辨率，默认 512×512。 |

### 1.2 推理速度与 DeepCache

- `LATENTSYNC_ENABLE_DEEPCACHE=1` 默认开启，缓存 UNet 中间特征，跳过 2/3 的 UNet forward，约 2× 加速，细节轻微变软。
- `enable_deepcache_override` 仅作为提示，因为 DeepCache 在 pipeline 加载时固定，需重启服务生效。

### 1.3 Mask 策略

#### 固定 Mask
- `mask_image_path` 默认 `"latentsync/utils/mask.png"`，是 baseline，**已锁定**，不可随意切换为 `mask2-5.png`。
- 固定 mask 定义了 UNet 需要 inpaint 的下半脸区域；mask 外部保持原图不变。

#### 动态嘴部 Mask
- `generate_dynamic_mouth_mask` 根据每帧 106 点 landmarks 计算嘴部椭圆区域，再与固定 mask 取 `maximum`，防止大笑/大表情把生成区域扩展到固定 mask 之外。
- 关键参数：
  - `pad_width_ratio=1.5`、`pad_height_top_ratio=1.3`、`pad_height_bottom_ratio=2.2`：控制动态 mask 相对嘴部包围盒的扩展。
  - `feather_sigma_px=7.0`：边缘羽化，减少硬边。
  - `fallback_*`：landmark 缺失时的默认椭圆。
- `aligned_mouth_ema_alpha=0.85`：嘴部中心/半宽在半帧间做 EMA，抑制 landmark 抖动导致的 mask 边界跳动。

### 1.4 音频特征

- 使用 `whisper-tiny` 提取音频特征，按 50 FPS 输出。
- `audio_sync_offset_seconds`：正数表示音频超前视频；pipeline 会用更早的音频特征驱动每帧，并在输出音频开头补零延迟。
- `feature2chunks` 按 `video_fps` 和 `offset_seconds` 将 whisper 特征切分成与视频帧对齐的 chunks。

---

## 2. 策略部分

### 2.1 Filter 策略

Filter 在 `affine_transform_video` / `loop_video` 阶段决定哪些帧需要走 diffusion 生成，哪些帧直接 passthrough（保留原图）。所有 filter 的命中情况汇总在 `[FaceMatch]` 日志中。

#### 2.1.1 人脸检测失败
- 检测不到人脸或 landmark 不完整时，该帧直接 passthrough，记为 `detect_fail`。
- 小 face filter：`lipsync_min_face_area_ratio=0.015`，人脸占画面比例过小则跳过。

#### 2.1.2 侧脸 / 转头 Filter（Yaw）
- `yaw_skip_threshold=40.0`：单帧 yaw 绝对值超过则 passthrough。
- `yaw_rate_skip_threshold=10.0`：相邻帧 yaw 变化超过则 passthrough（防止快速转头）。
- `side_face_passthrough_yaw_threshold=0.0`：在 `(threshold, yaw_skip_threshold)` 区间内的帧也 passthrough；设为 22.5 可让所有非正面脸都不生成。
- Episode 级侧脸 filter：连续 yaw 超过阈值时，对前后 `side_face_episode_pre_pad/post_pad` 帧的过渡区也 passthrough。
- `yaw_warn_threshold_ratio=0.75`：warn 带阈值 = `yaw_skip_threshold * ratio`。
- `side_face_warn_min_run_frames/seconds`：持续在 warn 带超过该时长则整段 passthrough。
- `side_face_blend_fade_frames=3`：在生成/跳过的边界做 cross-fade，柔化硬切。

> 侧脸检测使用多信号融合：鼻偏移、眼宽不对称、嘴角不对称，带 noise floor。

#### 2.1.3 嘴部遮挡 Filter
- `mouth_occlusion_skip_threshold=1.0`：默认关闭（1.0 表示不触发）。
- 检测嘴部 ROI 内是否被手、麦克风、口罩等遮挡，得分超过阈值则 passthrough。

#### 2.1.4 运动模糊 Filter
- `motion_blur_skip_threshold=0.08`：对对齐后人脸和嘴部分别计算 Laplacian 方差，若都低于阈值则 passthrough。

#### 2.1.5 人脸跳动 Filter
- `face_jump_center_threshold=0.0`：landmark 中心相对人脸尺寸跳变超过阈值则 passthrough。
- `face_jump_scale_threshold=0.0`：人脸尺寸跳变超过阈值则 passthrough。

#### 2.1.6 时序连续性 Break
- `lipsync_continuity_max_center_shift=0.35` / `lipsync_continuity_max_scale_change=0.35`：几何跳变不直接跳过帧，而是标记为 `continuity_break`，清空 EMA/稳定化状态。
- `lipsync_mouth_diff_break_threshold=0.10`：嘴部像素平均差超过阈值也触发 continuity break，用于捕捉 embedding 检查漏掉的人脸切换。

#### 2.1.7 场景切分 Break
- `scene_cut_break_enabled=True` / `scene_cut_break_threshold=0.45`：相邻帧直方图距离超过阈值时，重置时序/affine 状态，防止上一镜头内容泄漏。

#### 2.1.8 静音 Filter
- `silent_skip_enabled=False`：默认关闭。
- `silent_rms_threshold=0.003` / `silent_min_run_frames=8` / `silent_pad_frames=0`：对持续静音段 passthrough，节省算力。

#### 2.1.9 自适应质量 Filter
- `adaptive_quality_fallback_enabled=False`：默认关闭，短剧场景可开启。
- 综合嘴部清晰度、嘴部差异、identity 相似度、yaw、音频能量、时序稳定性给出每帧质量分，低于 `adaptive_quality_fallback_threshold=0.35` 则回退原图。
- `adaptive_quality_fallback_max_ratio=0.35`：限制最大回退比例。
- `adaptive_quality_fallback_hysteresis_frames=2`：滞后滤波，抑制单帧闪烁。

#### 2.1.10 质量门 Filter
- `quality_gate_enabled=False`：默认关闭。
- 生成后人脸嘴部 ROI 的 Laplacian 方差若明显低于原图（`quality_min_laplacian`、`quality_min_sharpness_ratio`、`quality_ref_min_laplacian`），则回退原图。

#### 2.1.11 Shot 级 Passthrough
- `shot_passthrough_enabled=False`：默认关闭。
- 当一个 shot 内预 filter 跳过比例超过 `shot_passthrough_skip_ratio_threshold=0.45`，且 shot 长度满足 `shot_passthrough_min_frames=8`、坏帧数满足 `shot_passthrough_min_bad_frames=3` 时，整个 shot 保持原图，避免生成/原图闪烁。

#### 2.1.12 Segment Consistency
- `segment_consistency_hard_cut_enabled=True`：合并相邻有效段时，若中间间隙包含硬切则不合并。
- `segment_consistency_track_aware=True`：track_id 不同的有效段不合并。
- `min_merged_lipsync_seconds=1.5` / `lipsync_min_segment_frames=5`：合并后仍过短的段强制 passthrough。

#### 2.1.13 场景级切分
- `scene_split_enabled=True`：按直方图检测场景边界，每个场景独立处理、独立时序状态，最后拼接。
- `scene_split_threshold=0.45`：场景切分阈值。
- `min_scene_duration_seconds=0.5`：小于该时长的场景会与相邻场景合并，避免过短场景带来的 face detection / 时序状态重置开销。0 表示不合并。

---

### 2.2 人脸选择策略

#### 2.2.1 Reference Embedding（ avatar 模式）
- 请求传入 `avatar_url` 时，服务端提取该 avatar 的人脸 embedding 作为 `reference_embedding`。
- 每帧选择与 reference_embedding 最相似的人脸进行唇形同步。
- `identity_similarity_threshold=0.5`：相似度低于阈值时该帧 passthrough。

#### 2.2.2 Identity Filter（主发言人模式）
- `apply_identity_filter=False`：默认关闭。
- 开启且无 avatar 时，`detect_main_speaker_embedding` 从视频中采样最多 48 帧，聚类出最主要的人脸 embedding，后续只同步该人脸。
- 聚类阈值 `identity_cluster_threshold=0.78`（API 参数）。
- 主发言人选择综合考虑：嘴张开程度、人脸面积、画面中心位置、yaw 绝对值。

#### 2.2.3 默认模式
- `apply_identity_filter=False` 且无 avatar 时，每帧选择最大/最居中的人脸进行同步，不限制身份。

---

### 2.3 前处理 / 后处理策略

#### 2.3.1 前处理

1. **视频读取**：`read_video(video_path, use_decord=False)` 使用 cv2 读取。
2. **音频读取与 offset**：`read_audio` 读取 16kHz 音频；根据 `audio_sync_offset_seconds` 对音频做 zero-pad/crop。
3. **Whisper 特征**：`audio_encoder.audio2feat` + `feature2chunks`。
4. **场景切分**：`scene_split_enabled=True` 时，按直方图切分为多个场景分别处理。
5. **ImageProcessor 创建**：每个请求创建一次，跨场景复用，避免重复加载 InsightFace 人脸检测模型。

#### 2.3.2 对齐与 Warp

- `ImageProcessor.affine_transform` / `affine_transform_with_embedding`：
  - 检测 106 点 landmark。
  - 提取左右眉心、鼻中心作为 3 点对齐基准。
  - `AlignRestore.align_warp_face` 将人脸 crop 到 512×512 并对齐。
- `restore_video`：将生成后的 512×512 人脸通过 affine 矩阵贴回原图。

#### 2.3.3 后处理（按顺序执行）

在每一批 diffusion 输出后依次执行：

1. **Paste surrounding pixels back**  
   用动态嘴部 mask 将非嘴部区域（脸颊、下巴等）贴回原始像素，只保留嘴部生成内容。

2. **Color match**（`color_match_strength=0.60`）  
   在 mask 区域内将生成脸的均值/方差对齐到原图，减少色调跳变。

3. **Mouth detail recovery**（`mouth_detail_strength=0.65`）  
   在嘴部外围恢复原始高频细节，同时保护嘴部 aperture 的生成运动。

4. **Mouth sharpen**（`mouth_sharpen_strength=0.30`）  
   对嘴部 ROI 做 unsharp mask，增强牙齿、唇线等高频细节。

5. **Temporal EMA smoothing**（`temporal_smoothing_enabled=True`）  
   跨帧 EMA 平滑生成脸，仅在 mask 区域内生效，防止上一帧人脸内容粘到当前帧。

6. **Mouth motion preserve**（`mouth_motion_preserve_strength=0.45`）  
   在嘴部核心区域保留当前帧生成运动，避免 EMA 把张嘴帧拉闭合。

7. **Audio-adaptive motion scale**（`mouth_audio_adaptive_motion_enabled=True`）  
   按每帧音频 RMS 在 `mouth_audio_motion_min_scale=0.85` 到 `mouth_audio_motion_max_scale=1.60` 之间插值，高能量语音保留更多当前嘴型。

8. **Mouth temporal stabilization**（`mouth_temporal_stabilization_strength=0.15`）  
   轻量帧间嘴部稳定，抑制闪烁；`mouth_temporal_stabilization_max_delta=0.12` 控制过大运动时中断 carry。

9. **CodeFormer face restoration**（默认关闭）  
   在 aligned 512×512 人脸 crop 上跑 CodeFormer，再贴回；参数见 `docs/codeformer_integration.md`。

10. **Quality gate / Adaptive quality fallback**  
    见 2.1.10 / 2.1.9。

#### 2.3.4 输出

- 最终视频与音频通过 ffmpeg 一次 mux 成 `video_out_path`。
- 日志输出 `input_duration`、`execution_duration`、`realtime_factor`，以及场景数、每场景耗时等性能指标。

---

## 3. 调优速查

| 现象 | 建议调整方向 |
|------|-------------|
| 嘴型不跟随音频 | 增大 `guidance_scale`；增大 `mouth_audio_motion_max_scale`；关闭 `temporal_smoothing_enabled`。 |
| 嘴型不自然 / 过平滑 | 降低 `guidance_scale`；降低 `mouth_temporal_stabilization_strength`；降低 `mouth_motion_preserve_strength`。 |
| 侧脸有鬼影 / 残留 | 开启 `side_face_passthrough_yaw_threshold=22.5`；增大 `side_face_episode_pre_pad/post_pad`。 |
| 生成/原图闪烁严重 | 开启 `shot_passthrough_enabled`；增大 `min_merged_lipsync_seconds`；增大 `lipsync_min_segment_frames`。 |
| 色调不一致 | 增大 `color_match_strength`；检查 mask 边界。 |
| 嘴部模糊 | 增大 `mouth_sharpen_strength`；关闭 DeepCache；增大 `num_inference_steps`；开启 CodeFormer。 |
| 整体速度慢 | 开启 DeepCache；降低 `num_inference_steps`；开启 `silent_skip_enabled`；减少场景数（提高 `scene_split_threshold` 或设置 `min_scene_duration_seconds`）。 |
| 短剧场景希望更保守 | 开启 `adaptive_quality_fallback_enabled`；增大 `adaptive_quality_fallback_max_ratio`。 |

---

## 4. 注意事项

- **Mask baseline 已锁定**：默认固定 mask 为 `latentsync/utils/mask.png`，未经明确允许不要切换为 `mask2-5.png`。
- **Naturalness 默认已调优**：`guidance_scale=1.5`、`inference_steps=40`、DeepCache 开启、`mouth_temporal_stabilization_strength=0.15`、`mouth_audio_motion_min_scale=0.85`。大幅调整需评估质量/速度/自然度三角。
- **前端字段后缀**：API 只识别 `*_override` 字段，`guidance_scale` 等无前缀字段会被忽略。
- **日志诊断入口**：输出异常时优先查看 `[FaceMatch]` 日志，确认是哪种 filter 在命中。
