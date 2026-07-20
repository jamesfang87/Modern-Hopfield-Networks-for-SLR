"""
We follow the same training strategies as "Training Strategies for Isolated Sign Language Recognition"

@misc{kvanchiani2025trainingstrategiesisolatedsign,
      title={Training Strategies for Isolated Sign Language Recognition},
      author={Karina Kvanchiani and Roman Kraynov and Elizaveta Petrova and Petr Surovcev and Aleksandr Nagaev and Alexander Kapitanov},
      year={2025},
      eprint={2412.11553},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.11553},
}
"""

import random

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ASL_Citizen(Dataset):
    def __init__(
        self,
        h5_path: str,
        video_ids: list,
        num_frames: int = 32,
        frame_step: int = 2,
        crop_size: int = 224,
        is_train: bool = True,
    ):
        self.h5_path = h5_path
        self.video_ids = video_ids
        self.num_frames = num_frames
        self.frame_step = frame_step
        self.crop_size = crop_size
        self.is_train = is_train
        self._h5 = None  # opened lazily per-worker, see __getitem__

    def _get_h5(self):
        # h5py file handles don't survive being pickled to DataLoader workers,
        # so open once per worker process on first access.
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, index):
        h5f = self._get_h5()
        video_id = self.video_ids[index]
        grp = h5f[video_id]

        frames = grp["frames"][:]  # decoded once at preprocessing time
        label = int(grp.attrs["label"])

        frames = self._apply_video_augmentation(frames)
        frames = self._sample_frames(frames)
        frames = self._square_pad(frames)
        frames = self._crop(frames)
        frames = self._apply_image_augmentations(frames, label)

        # (T, H, W, C) uint8 -> (C, T, H, W) float tensor, normalized to [0, 1]
        tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float() / 255.0

        return {
            "pixel_values": tensor,
            "label": label,
        }

    def __len__(self):
        return len(self.video_ids)

    # -----------------------------------------------------------------
    # Video-level augmentations
    # -----------------------------------------------------------------
    @staticmethod
    def _speed_up(frames: np.ndarray, factor: int = 2) -> np.ndarray:
        return frames[::factor]

    @staticmethod
    def _slow_down(frames: np.ndarray, factor: int = 2) -> np.ndarray:
        return np.repeat(frames, factor, axis=0)

    @staticmethod
    def _random_drop(frames: np.ndarray, drop_ratio: float = 0.1) -> np.ndarray:
        keep = int(len(frames) * (1 - drop_ratio))
        idx = np.sort(np.random.choice(len(frames), keep, replace=False))
        return frames[idx]

    @staticmethod
    def _random_add(frames: np.ndarray, add_ratio: float = 0.3) -> np.ndarray:
        n_extra = int(len(frames) * add_ratio)
        extra_idx = np.random.choice(len(frames), n_extra, replace=True)
        all_idx = np.sort(np.concatenate([np.arange(len(frames)), extra_idx]))
        return frames[all_idx]

    def _apply_video_augmentation(self, frames: np.ndarray) -> np.ndarray:
        if not self.is_train:
            return frames
        # one of the four is picked at random per sample, matching the paper
        aug_fn = random.choice(
            [self._speed_up, self._slow_down, self._random_drop, self._random_add]
        )
        try:
            return aug_fn(frames)
        except ValueError:
            return frames  # e.g. too few frames for the chosen op

    # -----------------------------------------------------------------
    # Frame sampling
    # -----------------------------------------------------------------
    def _sample_frames(self, frames: np.ndarray) -> np.ndarray:
        total = len(frames)
        needed = self.num_frames * self.frame_step
        if total >= needed:
            start = random.randint(0, total - needed) if self.is_train else 0
            idx = np.arange(start, start + needed, self.frame_step)
        else:
            # not enough frames: sample what we can with the given step,
            # then pad by repeating the last frame (matches paper's approach)
            idx = np.arange(0, total, self.frame_step)
        sampled = frames[idx]
        if len(sampled) < self.num_frames:
            pad = np.repeat(sampled[-1:], self.num_frames - len(sampled), axis=0)
            sampled = np.concatenate([sampled, pad], axis=0)
        return sampled[: self.num_frames]

    # -----------------------------------------------------------------
    # Spatial preprocessing: square pad + crop
    # -----------------------------------------------------------------
    def _square_pad(self, frames: np.ndarray) -> np.ndarray:
        t, h, w, c = frames.shape
        side = max(h, w)
        padded = np.zeros((t, side, side, c), dtype=frames.dtype)
        top = (side - h) // 2
        left = (side - w) // 2
        padded[:, top : top + h, left : left + w, :] = frames
        return padded

    def _crop(self, frames: np.ndarray) -> np.ndarray:
        t, side, _, c = frames.shape
        if self.is_train:
            # random crop, resized to crop_size if padding differs from it
            max_offset = max(side - self.crop_size, 0)
            top = random.randint(0, max_offset) if max_offset > 0 else 0
            left = random.randint(0, max_offset) if max_offset > 0 else 0
        else:
            top = left = (side - self.crop_size) // 2

        if side < self.crop_size:
            # upsample if padded frame is smaller than target crop
            out = np.empty((t, self.crop_size, self.crop_size, c), dtype=frames.dtype)
            for i in range(t):
                out[i] = cv2.resize(frames[i], (self.crop_size, self.crop_size))
            return out

        return frames[:, top : top + self.crop_size, left : left + self.crop_size, :]

    # -----------------------------------------------------------------
    # Image-level augmentations
    # -----------------------------------------------------------------
    def _color_jitter(self, frames: np.ndarray, p=0.5) -> np.ndarray:
        if random.random() > p:
            return frames
        brightness = 1 + random.uniform(-0.1, 0.1)
        contrast = 1 + random.uniform(-0.005, 0.005)
        out = frames.astype(np.float32) * brightness
        mean = out.mean(axis=(1, 2, 3), keepdims=True)
        out = (out - mean) * contrast + mean
        return np.clip(out, 0, 255).astype(np.uint8)

    def _salt_and_pepper(
        self, frames: np.ndarray, p=0.5, amount=(0.001, 0.005)
    ) -> np.ndarray:
        if random.random() > p:
            return frames
        out = frames.copy()
        amt = random.uniform(*amount)
        mask = np.random.rand(*out.shape[:3]) < amt
        salt = np.random.rand(*out.shape[:3]) < 0.5
        out[mask & salt] = 255
        out[mask & ~salt] = 0
        return out

    def _sharpness(
        self, frames: np.ndarray, p=0.35, factor_range=(0.5, 2.0)
    ) -> np.ndarray:
        if random.random() > p:
            return frames
        factor = random.uniform(*factor_range)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        kernel = (
            kernel * factor / 5.0 + np.eye(3, dtype=np.float32).sum() * 0
        )  # keep simple
        out = np.empty_like(frames)
        for i in range(len(frames)):
            out[i] = cv2.filter2D(frames[i], -1, kernel)
        return out

    def _random_erasing(
        self, frames: np.ndarray, p=0.25, area_range=(0.02, 0.33)
    ) -> np.ndarray:
        if random.random() > p:
            return frames
        out = frames.copy()
        t, h, w, _ = out.shape
        area = random.uniform(*area_range) * h * w
        aspect = random.uniform(0.3, 3.3)
        eh = int(round((area * aspect) ** 0.5))
        ew = int(round((area / aspect) ** 0.5))
        if eh < h and ew < w:
            top = random.randint(0, h - eh)
            left = random.randint(0, w - ew)
            out[:, top : top + eh, left : left + ew, :] = 0
        return out

    def _image_compression(
        self, frames: np.ndarray, p=0.15, quality_range=(80, 100)
    ) -> np.ndarray:
        if random.random() > p:
            return frames
        quality = random.randint(*quality_range)
        out = np.empty_like(frames)
        for i, frame in enumerate(frames):
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            out[i] = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
        return out

    def _downscale(
        self, frames: np.ndarray, p=0.15, scale_range=(0.4, 0.8)
    ) -> np.ndarray:
        if random.random() > p:
            return frames
        scale = random.uniform(*scale_range)
        t, h, w, c = frames.shape
        small_h, small_w = max(1, int(h * scale)), max(1, int(w * scale))
        out = np.empty_like(frames)
        for i in range(t):
            small = cv2.resize(frames[i], (small_w, small_h))
            out[i] = cv2.resize(small, (w, h))
        return out

    def _apply_image_augmentations(self, frames: np.ndarray, label: int) -> np.ndarray:
        if not self.is_train:
            return frames
        frames = self._maybe_flip(frames, label)
        frames = self._color_jitter(frames)
        frames = self._salt_and_pepper(frames)
        frames = self._sharpness(frames)
        frames = self._random_erasing(frames)
        frames = self._image_compression(frames)
        frames = self._downscale(frames)
        return frames
