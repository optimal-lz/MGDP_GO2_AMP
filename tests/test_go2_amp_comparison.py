import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "legged_gym/scripts/evaluate_go2_amp_comparison.py"


def load_module():
    spec = importlib.util.spec_from_file_location("go2_amp_comparison", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestComparisonCases(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_exact_72_case_matrix(self):
        cases = self.module.build_cases()
        self.assertEqual(len(cases), 72)
        forward = [case for case in cases if case["vx"] > 0]
        turns = [case for case in cases if case["vx"] == 0]
        self.assertEqual(len(forward), 60)
        self.assertEqual(len(turns), 12)
        self.assertEqual({case["terrain"] for case in forward},
                         set(self.module.TERRAIN_NAMES))
        self.assertEqual({case["level"] for case in cases}, {0, 9, 19})
        self.assertEqual({case["vx"] for case in forward}, {0.4, 0.8})
        self.assertEqual({case["terrain"] for case in turns},
                         {"flat", "rough_flat"})
        self.assertEqual({case["yaw"] for case in turns}, {-0.5, 0.5})

    def test_episode_schedule_is_identical_for_both_policies(self):
        seen = []

        def evaluator(policy, manifest, schedule, shared_style_manifest):
            seen.append((policy, list(schedule), shared_style_manifest))
            rows = []
            for item in schedule:
                row = dict(item)
                row.update({name: 0.0 for name in self.module.METRIC_FIELDS})
                rows.append(row)
            return rows

        manifests = self.module.ValidatedPair(
            amp_manifest=Path("amp.json"),
            no_amp_manifest=Path("no_amp.json"),
            iteration=1000,
        )
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            rows = self.module.evaluate_pair(
                manifests, Path(temporary), episodes_per_case=20,
                base_seed=1, evaluator=evaluator)
        self.assertEqual(len(rows), 72 * 20 * 2)
        self.assertEqual(seen[0][1], seen[1][1])
        self.assertEqual(seen[0][2], Path("amp.json"))
        self.assertEqual(seen[1][2], Path("amp.json"))
        self.assertEqual(seen[0][0], "amp")
        self.assertEqual(seen[1][0], "no_amp")


class TestComparisonOutputs(unittest.TestCase):
    def setUp(self):
        self.module = load_module()

    def test_only_frozen_csv_and_png_outputs_are_written(self):
        rows = []
        for policy in ("amp", "no_amp"):
            row = {
                "policy": policy, "case_id": "flat_l0_v0p4",
                "terrain": "flat", "level": 0,
                "vx_command": 0.4, "yaw_command": 0.0,
                "episode": 0, "seed": 1,
            }
            row.update({name: 0.5 for name in self.module.METRIC_FIELDS})
            rows.append(row)
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            output = Path(temporary)
            self.module.write_outputs(rows, output)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"summary.csv", "cases.csv", "completion.png",
                 "tracking.png", "smoothness.png", "style_score.png"},
            )
            with (output / "cases.csv").open(
                    "r", encoding="utf-8", newline="") as stream:
                fields = set(csv.DictReader(stream).fieldnames)
            self.assertEqual(fields, set(self.module.CASE_FIELDS))
            for field in (
                    "completion", "progress", "vx_rmse", "yaw_rmse",
                    "base_contact_failure", "action_rate", "torque",
                    "episode_length", "wm_depth_mse", "wm_height_mse",
                    "amp_style_score"):
                self.assertIn(field, fields)

    def test_source_freezes_shared_style_and_disables_training_noise(self):
        source = SCRIPT.read_text(encoding="utf-8")
        for text in (
                "shared_style_manifest", "normalizer", "discriminator",
                "torch.inference_mode", "update_wm = False",
                "add_noise = False", "push_robots = False",
                "randomize_friction", "disable_evaluation_stochasticity",
                "optimizer", "replay"):
            self.assertIn(text, source)
        self.assertNotIn(".update()", source)
        self.assertNotIn(".record(", source)
        self.assertNotIn("pass_fail", source)
        self.assertNotIn("report.md", source)


if __name__ == "__main__":
    unittest.main()
