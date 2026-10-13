import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "legged_gym"))
HAS_TORCH = importlib.util.find_spec("torch") is not None
ROOT = Path(__file__).resolve().parents[1]


class TestAmpStateSource(unittest.TestCase):
    def test_clean_state_width_and_order_are_fixed(self):
        source = (ROOT / "legged_gym/rl/MGDP/amp/state.py").read_text(encoding="utf-8")
        self.assertIn("joint_pos, base_lin_vel, base_ang_vel, joint_vel", source)
        self.assertIn("shape[1] != 30", source)


@unittest.skipUnless(HAS_TORCH, "PyTorch is tested on the training host")
class TestAmpState(unittest.TestCase):
    def test_exact_projection_order_and_shape(self):
        import torch
        from rl.MGDP.amp.state import pack_amp_state
        joint = torch.arange(12, dtype=torch.float).reshape(1, 12)
        linear = torch.tensor([[12., 13., 14.]])
        angular = torch.tensor([[15., 16., 17.]])
        joint_velocity = torch.arange(18, 30, dtype=torch.float).reshape(1, 12)
        packed = pack_amp_state(joint, linear, angular, joint_velocity)
        self.assertEqual(packed.shape, (1, 30))
        self.assertTrue(torch.equal(packed[0], torch.arange(30, dtype=torch.float)))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            pack_amp_state(joint * float("nan"), linear, angular, joint_velocity)


if __name__ == "__main__":
    unittest.main()
