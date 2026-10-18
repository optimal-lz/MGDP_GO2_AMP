import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from tests.ast_source import node_source


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "legged_gym/legged_gym/envs/go2_amp/env.py"


def _load_reward_method(name):
    source = ENV_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    original = next(node for node in tree.body
                    if isinstance(node, ast.ClassDef)
                    and node.name == "Go2AmpRandomDog")
    method = next(node for node in original.body
                  if isinstance(node, ast.FunctionDef) and node.name == name)

    class FakeTorch:
        @staticmethod
        def clamp(value, min=None, max=None):
            return np.clip(value, min, max)

    isolated = ast.ClassDef(
        name="RewardHarness", bases=[], keywords=[], body=[method], decorator_list=[])
    module = ast.Module(body=[isolated], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": FakeTorch}
    exec(compile(module, str(ENV_PATH), "exec"), namespace)
    return namespace["RewardHarness"], node_source(source, method)


class TestAmpReward(unittest.TestCase):
    def test_total_reward_order_is_clamped_base_plus_amp_plus_termination(self):
        harness, _ = _load_reward_method("compose_total_reward")
        env = harness()
        env._base_reward = np.asarray([-2.0, 3.0])
        env._termination_reward = np.asarray([-1.0, 0.0])
        env.rew_buf = np.zeros(2)

        actual = env.compose_total_reward(np.asarray([0.25, 0.5]))

        np.testing.assert_allclose(actual, [-0.75, 3.5])

    def test_air_time_is_capped_and_only_forward_commands_are_eligible(self):
        _, source = _load_reward_method("_reward_feet_air_time")
        self.assertIn("max=self.cfg.rewards.amp_air_time_cap", source)
        self.assertIn("self.commands[:, 0] > self.cfg.commands.vx_deadband", source)
        self.assertNotIn("torch.norm(self.commands", source)

    def test_conditional_penalties_match_the_frozen_scope(self):
        _, dof_error = _load_reward_method("_reward_dof_error")
        _, feet_edge = _load_reward_method("_reward_feet_edge")
        _, cheat = _load_reward_method("_reward_cheat")
        _, stuck = _load_reward_method("_reward_stuck")
        self.assertIn("self.get_idle_mask()", dof_error)
        self.assertIn("self.env_class == TERRAIN_CLASS_IDS['gap']", feet_edge)
        self.assertIn("self.env_class == TERRAIN_CLASS_IDS['pit']", feet_edge)
        self.assertIn("self.terrain_levels > 3", feet_edge)
        self.assertIn("structured", cheat)
        self.assertIn("torch.abs(heading_error) > 1.0", cheat)
        self.assertIn("self.commands[:, 0] > self.cfg.commands.vx_deadband", stuck)
        self.assertIn("self.base_lin_vel[:, 0] < 0.1", stuck)


if __name__ == "__main__":
    unittest.main()
