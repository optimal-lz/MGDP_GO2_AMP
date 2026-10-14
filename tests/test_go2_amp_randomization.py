import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RANDOMIZATION = (
    ROOT / "legged_gym/legged_gym/envs/baseline/randomization.py")
TERRAINS = (
    ROOT / "legged_gym/legged_gym/envs/baseline/legged_robot_terrains.py")
RANDOM_DOG = ROOT / "legged_gym/legged_gym/envs/random_dog/random_dog.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "go2_amp_randomization", RANDOMIZATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def formal_domain(**overrides):
    values = {
        "randomize_friction": True,
        "randomize_action_latency": True,
        "push_robots": True,
        "randomize_base_mass": False,
        "randomize_link_mass": False,
        "randomize_restitution": False,
        "randomize_com": False,
        "randomize_gains": False,
        "randomize_motor_strength": False,
        "randomize_motor_offset": False,
        "randomize_lag_timesteps": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestGo2AmpRandomization(unittest.TestCase):
    def test_training_allowlist_rejects_every_unapproved_randomization(self):
        module = load_module()
        module.validate_training_randomization(formal_domain())
        for name in module.FORBIDDEN_DOMAIN_RANDOMIZATIONS:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, name):
                    module.validate_training_randomization(
                        formal_domain(**{name: True}))
        with self.assertRaisesRegex(ValueError, "randomize_future_field"):
            module.validate_training_randomization(
                formal_domain(randomize_future_field=True))

    def test_action_latency_does_not_generate_motor_strength_or_offset(self):
        module = load_module()
        domain = formal_domain(randomize_action_latency=True)
        domain.added_motor_strength = [0.8, 1.2]
        domain.added_motor_offset = [-0.1, 0.1]
        strengths, offsets = module.sample_motor_parameters(
            domain, 12, np.random.default_rng(7))
        np.testing.assert_array_equal(strengths, np.ones(12))
        np.testing.assert_array_equal(offsets, np.zeros(12))

        domain.randomize_motor_strength = True
        domain.randomize_motor_offset = True
        strengths, offsets = module.sample_motor_parameters(
            domain, 12, np.random.default_rng(7))
        self.assertFalse(np.array_equal(strengths, np.ones(12)))
        self.assertFalse(np.array_equal(offsets, np.zeros(12)))

    def test_evaluation_disables_proprio_depth_mount_and_domain_randomness(self):
        module = load_module()
        cfg = SimpleNamespace(
            noise=SimpleNamespace(
                add_noise=True, add_height_noise=True,
                add_privileged_noise=True),
            camera=SimpleNamespace(
                update_wm=True, noise_gaussian=0.03, noise_dropout=0.1,
                mount_translation_jitter=0.02,
                mount_pitch_jitter_deg=2.0),
            domain_rand=formal_domain(),
        )
        module.disable_evaluation_stochasticity(cfg)
        module.validate_training_randomization(cfg.domain_rand)
        self.assertFalse(cfg.noise.add_noise)
        self.assertFalse(cfg.noise.add_height_noise)
        self.assertFalse(cfg.noise.add_privileged_noise)
        self.assertFalse(cfg.camera.update_wm)
        self.assertIsNone(cfg.camera.noise_gaussian)
        self.assertIsNone(cfg.camera.noise_dropout)
        self.assertEqual(cfg.camera.mount_translation_jitter, 0.0)
        self.assertEqual(cfg.camera.mount_pitch_jitter_deg, 0.0)
        for name in module.ALL_DOMAIN_RANDOMIZATIONS:
            self.assertFalse(getattr(cfg.domain_rand, name), name)

        cfg.domain_rand.randomize_future_field = True
        module.disable_evaluation_stochasticity(cfg)
        self.assertFalse(cfg.domain_rand.randomize_future_field)

    def test_camera_mount_ranges_come_from_formal_configuration(self):
        module = load_module()
        translation, rotation = module.camera_mount_ranges(0.02, 2.0)
        self.assertEqual(translation, ([-0.02] * 3, [0.02] * 3))
        self.assertEqual(rotation, ([0.0, -2.0, 0.0],
                                    [0.0, 2.0, 0.0]))

    def test_runtime_paths_use_independent_motor_flags_and_camera_ranges(self):
        terrain_source = TERRAINS.read_text(encoding="utf-8")
        dog_source = RANDOM_DOG.read_text(encoding="utf-8")
        self.assertIn("sample_motor_parameters", terrain_source)
        self.assertNotIn(
            'control_type == "P_factors" or self.cfg.domain_rand.randomize_action_latency',
            terrain_source,
        )
        self.assertIn("randomize_motor_offset", dog_source)
        self.assertIn("randomize_motor_strength", dog_source)
        self.assertIn("camera_mount_ranges", dog_source)


if __name__ == "__main__":
    unittest.main()
