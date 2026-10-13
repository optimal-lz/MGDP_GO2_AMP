"""Validated Go2 trajectory projection and within-clip transition sampling."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


AMP_COLUMNS = np.r_[7:19, 31:34, 34:37, 37:49]


class MotionDataset:
    def __init__(self, root: Path, groups: Dict[str, List[str]],
                 weights: Dict[str, float], dt: float, rng=None):
        self.root = Path(root)
        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("motion step dt must be positive")
        if not groups or set(groups) != set(weights):
            raise ValueError("groups and weights must have matching nonempty keys")
        self.groups = {key: tuple(names) for key, names in groups.items()}
        self.rng = rng if rng is not None else np.random.default_rng()
        self.weights = np.asarray([weights[key] for key in groups], dtype=np.float64)
        if not np.isfinite(self.weights).all() or np.any(self.weights <= 0):
            raise ValueError("group weights must be finite and positive")
        self.weights /= self.weights.sum()
        self.group_names = list(groups)
        self.clips = {}
        for names in self.groups.values():
            if not names:
                raise ValueError("every motion group needs a clip")
            for name in names:
                if name in self.clips:
                    continue
                path = self.root / name
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    raise ValueError(f"{path}: cannot read motion") from exc
                try:
                    frame_dt = float(raw["FrameDuration"])
                    motion_weight = float(raw["MotionWeight"])
                    frames = np.asarray(raw["Frames"], dtype=np.float64)
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"{path}: invalid motion metadata") from exc
                if (frames.ndim != 2 or frames.shape[1] != 61 or len(frames) < 2
                        or not np.isfinite(frames).all() or not np.isfinite(frame_dt)
                        or frame_dt <= 0 or not np.isfinite(motion_weight)
                        or motion_weight <= 0
                        or not np.isclose(frame_dt, self.dt, rtol=0.0, atol=1e-9)):
                    raise ValueError(f"{path}: invalid 61-D trajectory or duration")
                self.clips[name] = (frames[:, AMP_COLUMNS].astype(np.float32), frame_dt,
                                    motion_weight)

    def sample_numpy(self, count: int, rng=None) -> Tuple[np.ndarray, np.ndarray]:
        if count <= 0:
            raise ValueError("count must be positive")
        rng = rng if rng is not None else self.rng
        a = np.empty((count, 30), dtype=np.float32)
        b = np.empty_like(a)
        groups = rng.choice(len(self.group_names), size=count, p=self.weights)
        for group_idx, group_name in enumerate(self.group_names):
            rows = np.flatnonzero(groups == group_idx)
            if not len(rows):
                continue
            self._fill_group(a, b, rows, group_name, rng)
        return a, b

    def sample_group_numpy(self, count: int, group: str, rng=None):
        """Sample transitions only from ``group``, respecting MotionWeight."""
        if count <= 0:
            raise ValueError("count must be positive")
        if group not in self.groups:
            raise ValueError(f"unknown motion group: {group}")
        rng = rng if rng is not None else self.rng
        a = np.empty((count, 30), dtype=np.float32)
        b = np.empty_like(a)
        self._fill_group(a, b, np.arange(count), group, rng)
        return a, b

    def _fill_group(self, a, b, rows, group, rng):
        names = self.groups[group]
        probabilities = np.asarray([self.clips[name][2] for name in names],
                                   dtype=np.float64)
        probabilities /= probabilities.sum()
        choices = rng.choice(len(names), size=len(rows), p=probabilities)
        for clip_idx, name in enumerate(names):
            selected = rows[choices == clip_idx]
            if not len(selected):
                continue
            states, frame_dt, _ = self.clips[name]
            maximum = (len(states) - 1) * frame_dt - self.dt
            t = (rng.uniform(0.0, maximum, size=len(selected))
                 if maximum > 0 else np.zeros(len(selected)))
            for output, position in ((a, t / frame_dt),
                                     (b, (t + self.dt) / frame_dt)):
                low = np.floor(position).astype(np.int64)
                high = np.minimum(low + 1, len(states) - 1)
                fraction = (position - low)[:, None]
                output[selected] = ((1 - fraction) * states[low]
                                    + fraction * states[high])

    def sample(self, count: int, device: str):
        import torch
        a, b = self.sample_numpy(count)
        return torch.as_tensor(a, device=device), torch.as_tensor(b, device=device)

    def sample_group(self, count: int, group: str, device: str):
        import torch
        a, b = self.sample_group_numpy(count, group)
        return torch.as_tensor(a, device=device), torch.as_tensor(b, device=device)
