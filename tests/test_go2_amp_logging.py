import importlib.util
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "legged_gym/rl/MGDP/runners/go2_amp_logging.py"


def load_logger():
    spec = importlib.util.spec_from_file_location("go2_amp_logging", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.IterationMetrics


class Writer:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def add_image(self, *args):
        raise AssertionError("images are forbidden")

    def add_histogram(self, *args):
        raise AssertionError("histograms are forbidden")


class TestGo2AmpLogging(unittest.TestCase):
    def test_core_once_per_iteration_and_detail_every_ten(self):
        metrics = load_logger()(Writer(), mode="amp")
        metrics.write(1, {"reward/task": 1.0, "reward/amp_contribution": 0.1},
                      {"terrain/flat_level": 2.0})
        metrics.write(10, {"reward/task": 2.0, "reward/amp_contribution": 0.2},
                      {"terrain/flat_level": 3.0})
        calls = metrics.writer.scalars
        self.assertEqual(sum(tag == "reward/task" for tag, _, _ in calls), 2)
        self.assertEqual(sum(tag == "terrain/flat_level" for tag, _, _ in calls), 1)
        self.assertTrue(all(step in (1, 10) for _, _, step in calls))

    def test_rejects_nonfinite_and_nonmonotonic_steps(self):
        metrics = load_logger()(Writer(), mode="amp", start_step=5)
        with self.assertRaisesRegex(ValueError, "monotonic"):
            metrics.write(5, {"reward/task": 1.0})
        with self.assertRaisesRegex(ValueError, "finite"):
            metrics.write(6, {"reward/task": math.nan})

    def test_noamp_requires_zero_contribution_and_forbids_d_training_tags(self):
        metrics = load_logger()(Writer(), mode="no_amp")
        metrics.write(0, {"reward/amp_contribution": 0.0, "reward/total": 1.0})
        with self.assertRaisesRegex(ValueError, "zero"):
            metrics.write(1, {"reward/amp_contribution": 0.1})
        with self.assertRaisesRegex(ValueError, "discriminator"):
            metrics.write(2, {"amp/discriminator_loss": 1.0,
                              "reward/amp_contribution": 0.0})


if __name__ == "__main__":
    unittest.main()
