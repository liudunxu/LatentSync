"""Unit tests for audio sync offset in Audio2Feature.feature2chunks."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch


class _MockAudio2Feature:
    """Minimal stand-in that exposes the offset-aware chunking logic."""

    def __init__(self, audio_feat_length=(2, 2), embedding_dim=384):
        self.audio_feat_length = list(audio_feat_length)
        self.embedding_dim = embedding_dim

    def get_sliced_feature(self, feature_array, vid_idx, fps=25):
        length = len(feature_array)
        center_idx = int(vid_idx * 50 / fps)
        left_idx = center_idx - self.audio_feat_length[0] * 2
        right_idx = center_idx + (self.audio_feat_length[1] + 1) * 2
        selected_idx = []
        for idx in range(left_idx, right_idx):
            idx = max(0, min(length - 1, idx))
            selected_idx.append(idx)
        # Return a dummy tensor and the selected indices so tests can assert
        # on the audio index window.
        return torch.zeros(1), selected_idx

    def feature2chunks(self, feature_array, fps, offset_seconds=0.0):
        whisper_chunks = []
        whisper_idx_multiplier = 50.0 / fps
        offset_frames = int(round(offset_seconds * fps))
        i = 0
        while True:
            start_idx = int(i * whisper_idx_multiplier)
            vid_idx = max(0, i - offset_frames)
            _feat, selected_idx = self.get_sliced_feature(feature_array=feature_array, vid_idx=vid_idx, fps=fps)
            whisper_chunks.append(selected_idx)
            i += 1
            if start_idx > len(feature_array):
                break
        return whisper_chunks


class TestAudioSyncOffset(unittest.TestCase):
    def _make_feature(self, length: int):
        return [torch.zeros(1) for _ in range(length)]

    def test_zero_offset_matches_no_offset_path(self):
        encoder = _MockAudio2Feature()
        feat = self._make_feature(50)
        chunks = encoder.feature2chunks(feat, fps=25, offset_seconds=0.0)
        self.assertGreater(len(chunks), 0)
        # First chunk should be centered at audio idx 0.
        self.assertEqual(chunks[0][2], 0)

    def test_positive_offset_uses_earlier_audio(self):
        encoder = _MockAudio2Feature()
        feat = self._make_feature(100)
        fps = 25
        offset_seconds = 0.08  # 2 frames at 25fps
        chunks = encoder.feature2chunks(feat, fps=fps, offset_seconds=offset_seconds)
        # Frame 2 should use audio that would have been used by frame 0 without offset.
        no_offset = encoder.feature2chunks(feat, fps=fps, offset_seconds=0.0)
        self.assertEqual(chunks[2], no_offset[0])

    def test_offset_clamps_at_start(self):
        encoder = _MockAudio2Feature()
        feat = self._make_feature(50)
        chunks = encoder.feature2chunks(feat, fps=25, offset_seconds=0.2)
        # Frames 0..4 all map to negative vid_idx, which clamps to 0.
        for i in range(5):
            self.assertEqual(chunks[i][2], 0)

    def test_negative_offset_uses_later_audio(self):
        encoder = _MockAudio2Feature()
        feat = self._make_feature(100)
        fps = 25
        offset_seconds = -0.08  # -2 frames
        chunks = encoder.feature2chunks(feat, fps=fps, offset_seconds=offset_seconds)
        no_offset = encoder.feature2chunks(feat, fps=fps, offset_seconds=0.0)
        # Frame 0 should use audio that would have been used by frame 2 without offset.
        self.assertEqual(chunks[0], no_offset[2])


if __name__ == "__main__":
    unittest.main()
