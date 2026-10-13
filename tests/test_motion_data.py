import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legged_gym"))
from rl.MGDP.amp.motions import MotionDataset


def write_clip(path, value, frames=3, dt=0.02):
    row = [0.0] * 61
    row[7:19] = [value] * 12
    path.write_text(json.dumps({"FrameDuration": dt, "MotionWeight": 1,
                                "Frames": [row] * frames}), encoding="utf-8")


class TestMotionData(unittest.TestCase):
    def test_all_copied_clips_and_active_groups(self):
        import ast
        config_path = ROOT / "legged_gym/legged_gym/envs/go2_amp/config.py"
        assignments = {}
        for node in ast.parse(config_path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id.startswith("AMP_EXPERT"):
                    assignments[target.id] = ast.literal_eval(node.value)
        motion_files = {p.name for p in (ROOT / "datasets/go2_motion").glob("*.txt")}
        self.assertEqual(len(motion_files), 17)
        self.assertEqual(assignments["AMP_EXPERT_GROUP_WEIGHTS"], {"forward": 0.8, "turn": 0.2})
        enabled = {n for group in assignments["AMP_EXPERT_GROUPS"].values() for n in group}
        self.assertEqual(len(enabled), 8)
        self.assertNotIn("go2_stance.txt", enabled)
        self.assertNotIn("go2_forward_faster.txt", enabled)
        MotionDataset(ROOT / "datasets/go2_motion", assignments["AMP_EXPERT_GROUPS"],
                      assignments["AMP_EXPERT_GROUP_WEIGHTS"], 0.02)
        for motion_file in sorted(motion_files):
            MotionDataset(ROOT / "datasets/go2_motion", {"clip": [motion_file]}, {"clip": 1.0}, 0.02)

    def test_projection_and_real_files(self):
        data = MotionDataset(ROOT / "datasets/go2_motion",
                             {"stance": ["go2_stance.txt"]}, {"stance": 1.0}, 0.02)
        a, b = data.sample_numpy(64, np.random.default_rng(4))
        self.assertEqual(a.shape, (64, 30))
        self.assertEqual(b.shape, (64, 30))
        self.assertTrue(np.isfinite(b).all())

    def test_never_crosses_clips(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_clip(root / "a.txt", 1)
            write_clip(root / "b.txt", 101)
            data = MotionDataset(root, {"x": ["a.txt", "b.txt"]}, {"x": 1}, 0.02)
            a, b = data.sample_numpy(100, np.random.default_rng(1))
            np.testing.assert_array_equal(a[:, 0], b[:, 0])
            self.assertEqual(set(a[:, 0]), {1.0, 101.0})

    def test_group_sampling_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_clip(root / "stance.txt", 1)
            write_clip(root / "walk.txt", 2)
            write_clip(root / "turn.txt", 3)
            data = MotionDataset(root, {"stance": ["stance.txt"],
                                        "walk": ["walk.txt"],
                                        "turn": ["turn.txt"]},
                                 {"stance": 0.1, "walk": 0.7, "turn": 0.2}, 0.02)
            a, _ = data.sample_numpy(10000, np.random.default_rng(19))
            counts = np.bincount(a[:, 0].astype(int), minlength=4) / len(a)
            np.testing.assert_allclose(counts[1:], [0.1, 0.7, 0.2], atol=0.02)

    def test_invalid_file_names_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_clip(root / "bad.txt", 1, frames=1)
            with self.assertRaisesRegex(ValueError, "bad.txt"):
                MotionDataset(root, {"x": ["bad.txt"]}, {"x": 1}, 0.02)
            for metadata in (
                {"FrameDuration": 0.0, "MotionWeight": 1, "Frames": [[0] * 61] * 3},
                {"FrameDuration": 0.001, "MotionWeight": 1, "Frames": [[0] * 61] * 3},
                {"FrameDuration": 0.02, "MotionWeight": 1,
                 "Frames": [[float("nan")] * 61] * 3},
            ):
                (root / "bad.txt").write_text(json.dumps(metadata), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "bad.txt"):
                    MotionDataset(root, {"x": ["bad.txt"]}, {"x": 1}, 0.02)
            with self.assertRaisesRegex(ValueError, "missing.txt"):
                MotionDataset(root, {"x": ["missing.txt"]}, {"x": 1}, 0.02)


if __name__ == "__main__":
    unittest.main()
