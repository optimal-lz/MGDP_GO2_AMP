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


CORE = {
    "reward/task": 1.0,
    "reward/regularization": -0.1,
    "reward/amp_contribution": 0.1,
    "reward/total": 1.0,
    "tracking/vx_error": 0.2,
    "tracking/yaw_error": 0.1,
    "episode/length": 4.0,
    "episode/base_contact_failure": 0.05,
    "terrain/mean_level": 2.0,
    "wm/total_loss": 0.3,
    "wm/depth_loss": 0.1,
    "wm/height_loss": 0.1,
    "wm/contrastive_loss": 0.1,
    "ppo/surrogate_loss": 0.2,
    "ppo/value_loss": 0.3,
    "ppo/entropy": 0.4,
    "ppo/kl": 0.01,
    "policy/action_std": 0.7,
    "performance/iteration_seconds": 1.2,
}

AMP_CORE = {
    "amp/policy_logit": -0.2,
    "amp/expert_logit": 0.4,
    "amp/discriminator_loss": 0.5,
    "amp/gradient_penalty": 0.6,
    "amp/valid_transition_count": 20,
    "amp/replay_fill": 0.5,
}

DETAIL = {
    "timing/simulation_seconds": 0.7,
    "timing/wm_seconds": 0.1,
    "timing/ppo_seconds": 0.3,
    "timing/discriminator_seconds": 0.1,
    "amp/expert_forward_logit": 0.5,
    "amp/expert_turn_logit": 0.3,
    "terrain/flat/mean_level": 2.0,
    "terrain/flat/completion": 0.8,
    "terrain/flat/base_contact_failure": 0.1,
    "terrain/flat/stuck": 0.2,
}


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
        metrics = load_logger()(Writer(), mode="amp", terrain_names=("flat",))
        core = {**CORE, **AMP_CORE}
        metrics.write(1, core, DETAIL)
        metrics.write(10, core, DETAIL)
        calls = metrics.writer.scalars
        self.assertEqual(sum(tag == "reward/task" for tag, _, _ in calls), 2)
        self.assertEqual(
            sum(tag == "terrain/flat/mean_level" for tag, _, _ in calls), 1)
        self.assertTrue(all(step in (1, 10) for _, _, step in calls))
        self.assertEqual(len([1 for _, _, step in calls if step == 1]), 25)
        self.assertEqual(len([1 for _, _, step in calls if step == 10]), 35)

    def test_rejects_nonfinite_and_nonmonotonic_steps(self):
        metrics = load_logger()(
            Writer(), mode="amp", start_step=5, terrain_names=("flat",))
        with self.assertRaisesRegex(ValueError, "monotonic"):
            metrics.write(5, {**CORE, **AMP_CORE})
        with self.assertRaisesRegex(ValueError, "finite"):
            metrics.write(6, {**CORE, **AMP_CORE, "reward/task": math.nan})

    def test_noamp_requires_zero_contribution_and_forbids_d_training_tags(self):
        metrics = load_logger()(Writer(), mode="no_amp", terrain_names=("flat",))
        no_amp = {**CORE, "reward/amp_contribution": 0.0}
        no_amp_detail = {
            key: value for key, value in DETAIL.items()
            if "discriminator" not in key and not key.startswith("amp/")
        }
        metrics.write(0, no_amp, no_amp_detail)
        with self.assertRaisesRegex(ValueError, "zero"):
            metrics.write(1, {**no_amp, "reward/amp_contribution": 0.1})
        with self.assertRaisesRegex(ValueError, "discriminator"):
            metrics.write(2, {**no_amp, "amp/discriminator_loss": 1.0})

    def test_missing_required_metrics_are_rejected(self):
        metrics = load_logger()(Writer(), mode="amp", terrain_names=("flat",))
        with self.assertRaisesRegex(ValueError, "missing core"):
            metrics.write(1, {"reward/task": 1.0})
        with self.assertRaisesRegex(ValueError, "missing detail"):
            metrics.write(10, {**CORE, **AMP_CORE}, {})


if __name__ == "__main__":
    unittest.main()
