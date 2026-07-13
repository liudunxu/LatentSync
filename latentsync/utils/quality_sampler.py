"""Weighted sampler that biases training toward high-quality videos.

Loads a per-video quality score (typically from HyperIQA) and returns
a torch WeightedRandomSampler that draws videos with probability
proportional to their quality. Combined with augmentation, this lets
the model "see" bad videos less often without outright throwing them
away (the Wav2Lip-style filter-everything approach).

Output format (quality_scores.json):
{
  "video1.mp4": 73.4,
  "video2.mp4": 41.2,
  ...
}

Usage:
    from latentsync.utils.quality_sampler import make_weighted_sampler
    sampler = make_weighted_sampler(
        train_dataset.video_paths,
        quality_json_path="debug/preprocess_quality.json",
        floor=10.0,           # min weight (avoid zero-prob)
        power=0.5,            # <1 flattens; >1 sharpens
    )
    train_loader = DataLoader(
        train_dataset, batch_size=..., sampler=sampler, ...
    )
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from torch.utils.data import WeightedRandomSampler


def load_quality_scores(
    quality_json_path: str | Path,
    video_paths: Sequence[str],
) -> Dict[str, float]:
    """Read per-video quality scores. Missing entries get `default`.

    Returns a dict keyed by absolute video path -> score in [0, 100].
    """
    p = Path(quality_json_path)
    scores: Dict[str, float] = {}
    if p.exists():
        with open(p) as f:
            raw = json.load(f)
        # tolerate both absolute and basename keys
        scores = {str(k): float(v) for k, v in raw.items()}
    return scores


def make_weights(
    video_paths: Sequence[str],
    quality_scores: Dict[str, float],
    floor: float = 10.0,
    power: float = 0.5,
    default: float = 50.0,
) -> List[float]:
    """Convert quality scores to sampling weights.

    Each video's weight is `max(score, floor) ** power`. Videos not
    present in the scores dict get `default`.

    Args:
        video_paths: absolute paths the dataset will load.
        quality_scores: dict of {video_key: score}.
        floor: minimum weight (prevents zero-prob sampling).
        power: exponent applied to score before sampling.
            - power=1.0: weight = score (linear)
            - power<1.0 (e.g. 0.5): flattens, bad videos get more chance
            - power>1.0: sharpens, good videos dominate
        default: score used when a video is missing from the dict.
    """
    weights: List[float] = []
    for vp in video_paths:
        # try both absolute path and basename lookup
        score = quality_scores.get(vp)
        if score is None:
            score = quality_scores.get(Path(vp).name, default)
        if score is None or score <= 0:
            score = default
        w = max(float(score), float(floor)) ** float(power)
        weights.append(w)
    return weights


def make_weighted_sampler(
    video_paths: Sequence[str],
    quality_json_path: Optional[str | Path] = None,
    quality_scores: Optional[Dict[str, float]] = None,
    floor: float = 10.0,
    power: float = 0.5,
    default: float = 50.0,
    num_samples: Optional[int] = None,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """Build a torch WeightedRandomSampler from video paths + quality scores.

    Provide either `quality_json_path` OR a pre-loaded `quality_scores` dict.
    """
    if quality_scores is None:
        if quality_json_path is None:
            raise ValueError("Provide either quality_json_path or quality_scores")
        quality_scores = load_quality_scores(quality_json_path, video_paths)
    weights = make_weights(
        video_paths, quality_scores,
        floor=floor, power=power, default=default,
    )
    if num_samples is None:
        num_samples = len(video_paths)
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.float),
        num_samples=int(num_samples),
        replacement=replacement,
    )
