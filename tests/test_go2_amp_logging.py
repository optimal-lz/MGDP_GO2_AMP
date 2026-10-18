import importlib.util
import math
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "legged_gym/rl/MGDP/runners/go2_amp_logging.py"
RUNNER_PATH = ROOT / "legged_gym/rl/MGDP/runners/go2_amp_runner.py"
RANDOM_DOG_PATH = ROOT / "legged_gym/legged_gym/envs/random_dog/random_dog.py"


def load_logging_module():
    spec = importlib.util.spec_from_file_location("go2_amp_logging", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_logger():
    return load_logging_module().IterationMetrics


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
    def test_runner_uses_iteration_aggregate_for_all_wm_updates(self):
        runner = RUNNER_PATH.read_text(encoding="utf-8")
        environment = RANDOM_DOG_PATH.read_text(encoding="utf-8")
        self.assertIn("reset_wm_iteration_metrics", runner)
        self.assertIn("get_wm_iteration_metrics", runner)
        self.assertIn("def reset_wm_iteration_metrics", environment)
        self.assertIn("def get_wm_iteration_metrics", environment)
        self.assertIn("visual_mse_per_env", environment)
        self.assertIn("height_mse_per_env", environment)

    def test_core_once_per_iteration_and_detail_every_ten(self):
        metrics = load_logger()(Writer(), mode="amp", terrain_names=("flat",))
        core = {**CORE, **AMP_CORE}
        scalars = metrics.write(1, core, DETAIL)
        metrics.write(10, core, DETAIL)
        self.assertEqual(scalars["reward/task"], 1.0)
        self.assertNotIn("terrain/flat/mean_level", scalars)
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

    def test_amp_console_panel_keeps_mgdp_layout_with_v1_metrics(self):
        formatter = load_logging_module().format_iteration_console
        panel = formatter(
            mode="amp",
            completed_iteration=5,
            target_iteration=20,
            total_timesteps=7680,
            iteration_timesteps=1536,
            elapsed_seconds=4.8,
            eta_seconds=14.4,
            timings={
                "rollout": 0.7,
                "world_model": 0.1,
                "ppo": 0.3,
                "discriminator": 0.1,
            },
            learning_rate=3e-4,
            metrics={**CORE, **AMP_CORE},
        )

        self.assertIn("#" * 88, panel)
        self.assertIn("MGDP GO2 AMP - Learning iteration 5/20", panel)
        self.assertIn("Mode: AMP", panel)
        for label in (
                "Computation:", "Rollout / WM / PPO / D:",
                "Mean task reward/step:", "Mean |vx error|:",
                "World model total loss:", "Value function loss:",
                "AMP policy logit:", "Valid AMP transitions:",
                "Total timesteps:", "ETA:"):
            self.assertIn(label, panel)
        self.assertNotIn("Mean episode rew_", panel)
        self.assertNotIn("\x1b", panel)
        self.assertLessEqual(max(map(len, panel.splitlines())), 88)

    def test_noamp_console_panel_omits_discriminator_training(self):
        formatter = load_logging_module().format_iteration_console
        panel = formatter(
            mode="no_amp",
            completed_iteration=1,
            target_iteration=10,
            total_timesteps=1536,
            iteration_timesteps=1536,
            elapsed_seconds=1.2,
            eta_seconds=10.8,
            timings={"rollout": 0.8, "world_model": 0.1, "ppo": 0.3},
            learning_rate=3e-4,
            metrics={**CORE, "reward/amp_contribution": 0.0},
        )

        self.assertIn("Mode: noAMP", panel)
        self.assertIn("Mean AMP contribution/step:", panel)
        self.assertIn("0.00000", panel)
        self.assertNotIn("Rollout / WM / PPO / D:", panel)
        self.assertNotIn("AMP policy logit:", panel)
        self.assertNotIn("Discriminator loss:", panel)
        self.assertLessEqual(max(map(len, panel.splitlines())), 88)


if __name__ == "__main__":
    unittest.main()
