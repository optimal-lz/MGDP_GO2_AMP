import ast
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legged_gym"))
HAS_TORCH = importlib.util.find_spec("torch") is not None


class TestAmpEnvironmentTransitionSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / "legged_gym/legged_gym/envs/go2_amp/env.py"
        cls.source = cls.path.read_text(encoding="utf-8")
        tree = ast.parse(cls.source)
        class_node = next(node for node in tree.body
                          if isinstance(node, ast.ClassDef)
                          and node.name == "Go2AmpRandomDog")
        cls.methods = {node.name: ast.unparse(node) for node in class_node.body
                       if isinstance(node, ast.FunctionDef)}

    def test_clean_state_order_is_validated_and_terminal_state_is_pre_reset(self):
        self.assertIn("validate_go2_dofs(self.dof_names)", self.source)
        observation = self.methods["get_amp_observations"]
        self.assertIn("pack_amp_state(self.dof_pos, lin, ang, self.dof_vel)", observation)
        reset = self.methods["reset_idx"]
        self.assertLess(reset.index("self.get_amp_observations()"),
                        reset.index("super().reset_idx(env_ids)"))
        self.assertIn("terminal_amp_states", reset)

    def test_step_exposes_epoch_mask_for_live_and_terminal_transitions(self):
        step = self.methods["step"]
        self.assertIn("epoch_before", step)
        self.assertIn("locomotion_before", step)
        self.assertIn("amp_transition_mask", step)
        self.assertIn("dones", step)
        self.assertIn("amp_transition_mask", self.source)


@unittest.skipUnless(HAS_TORCH, "PyTorch is tested on the training host")
class TestAmpTransition(unittest.TestCase):
    def test_done_uses_terminal_state(self):
        import torch
        from rl.MGDP.amp.transition import select_next_amp_state
        after = torch.zeros(3, 30)
        packet = (torch.tensor([1]), torch.ones(1, 30))
        result = select_next_amp_state(after, torch.tensor([0, 1, 0]), packet)
        self.assertEqual(result[1, 0].item(), 1)
        with self.assertRaises(ValueError):
            select_next_amp_state(after, torch.tensor([0, 1, 0]),
                                  (torch.tensor([0]), torch.ones(1, 30)))

    def test_exact_joint_order(self):
        from rl.MGDP.amp.state import GO2_DOF_NAMES, validate_go2_dofs
        validate_go2_dofs(GO2_DOF_NAMES)
        with self.assertRaisesRegex(ValueError, "DOF order"):
            validate_go2_dofs(GO2_DOF_NAMES[3:6] + GO2_DOF_NAMES[:3]
                              + GO2_DOF_NAMES[6:])


if __name__ == "__main__":
    unittest.main()
