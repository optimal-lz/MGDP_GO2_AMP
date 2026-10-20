import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legged_gym/legged_gym/envs/go2_amp"))

from commands import (
    amp_transition_mask,
    apply_fixed_evaluation_commands,
    apply_command_deadband,
    apply_terrain_command_policy,
    classify_commands,
)


class TestGo2AmpCommands(unittest.TestCase):
    def test_fixed_evaluation_commands_replace_only_selected_envs(self):
        commands = np.zeros((4, 4), dtype=np.float32)
        fixed = np.asarray([
            [0.4, 0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
        ], dtype=np.float32)

        applied = apply_fixed_evaluation_commands(
            commands, np.asarray([1, 3]), fixed)

        self.assertTrue(applied)
        np.testing.assert_allclose(commands, [
            [0.0, 0.0, 0.0, 0.0],
            [0.4, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 0.0],
        ])

    def test_missing_fixed_commands_leave_training_commands_untouched(self):
        commands = np.asarray([[0.3, 0.0, 0.2]], dtype=np.float32)

        applied = apply_fixed_evaluation_commands(
            commands, np.asarray([0]), None)

        self.assertFalse(applied)
        np.testing.assert_allclose(commands, [[0.3, 0.0, 0.2]])

    def test_deadband_zeros_boundary_values_and_preserves_locomotion(self):
        commands = np.asarray([
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.2],
            [0.2001, 0.0, 0.2001],
            [1.0, 0.0, -0.2],
            [0.0, 0.0, -0.2001],
        ], dtype=np.float32)

        filtered = apply_command_deadband(commands)

        np.testing.assert_allclose(filtered[:, 0], [0.0, 0.0, 0.2001, 1.0, 0.0])
        np.testing.assert_allclose(filtered[:, 2], [0.0, 0.0, 0.2001, 0.0, -0.2001])
        np.testing.assert_allclose(commands[1], [0.2, 0.0, 0.2])

    def test_idle_is_exactly_zero_forward_and_yaw_after_deadband(self):
        commands = apply_command_deadband(np.asarray([
            [0.2, 0.0, -0.2],
            [0.3, 0.0, 0.0],
            [0.0, 0.0, -0.3],
        ], dtype=np.float32))

        idle, locomotion = classify_commands(commands)

        np.testing.assert_array_equal(idle, [True, False, False])
        np.testing.assert_array_equal(locomotion, [False, True, True])

    def test_structured_terrain_disables_yaw(self):
        commands = np.asarray([[0.8, 0.0, 0.5]] * 4, dtype=np.float32)
        terrain_classes = np.asarray([0, 1, 2, 9])

        filtered = apply_terrain_command_policy(
            commands, terrain_classes, turning_terrain_ids=(0, 1))

        np.testing.assert_allclose(filtered[:, 2], [0.5, 0.5, 0.0, 0.0])
        np.testing.assert_allclose(commands[:, 2], 0.5)

    def test_amp_transition_mask_uses_pre_step_mode_and_epoch(self):
        locomotion_before = np.asarray([True, True, True, False, False])
        epoch_before = np.asarray([2, 2, 2, 2, 2])
        epoch_after = np.asarray([2, 3, 3, 2, 3])
        dones = np.asarray([False, False, True, False, True])

        mask = amp_transition_mask(
            locomotion_before, epoch_before, epoch_after, dones)

        np.testing.assert_array_equal(mask, [True, False, True, False, False])


if __name__ == "__main__":
    unittest.main()
