import ast
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

from tests.ast_source import node_source


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legged_gym"))
HAS_TORCH = importlib.util.find_spec("torch") is not None


class TestSingleAmpSource(unittest.TestCase):
    def test_discriminator_contract_and_zero_target_gradient_penalty(self):
        path = ROOT / "legged_gym/rl/MGDP/amp/discriminator.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        discriminator = next(node for node in tree.body
                             if isinstance(node, ast.ClassDef)
                             and node.name == "AMPDiscriminator")
        methods = {node.name: node_source(source, node) for node in discriminator.body
                   if isinstance(node, ast.FunctionDef)}
        self.assertIn("hidden_dims=(1024, 512)", methods["__init__"])
        self.assertIn("nn.ReLU()", methods["__init__"])
        self.assertIn("gradient.square()", methods["gradient_penalty"])
        self.assertNotIn("- 1", methods["gradient_penalty"])

    def test_session_has_only_formal_single_discriminator_api(self):
        path = ROOT / "legged_gym/rl/MGDP/amp/session.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        session = next(node for node in tree.body
                       if isinstance(node, ast.ClassDef) and node.name == "AMPSession")
        methods = {node.name: node_source(source, node) for node in session.body
                   if isinstance(node, ast.FunctionDef)}
        self.assertIn("amp_seed", methods["__init__"])
        self.assertIn("eligible", methods["reward"])
        self.assertIn("enabled", methods["reward"])
        self.assertIn("self.config['amp_reward_coef'] * raw", methods["reward"])
        self.assertIn("self.replay.insert(state[mask], next_state[mask])", methods["record"])
        self.assertIn("amp_updates_per_iter", methods["update"])
        self.assertIn("sample_with_groups", methods["update"])
        self.assertIn("expert_forward_logit", methods["update"])
        self.assertIn("expert_turn_logit", methods["update"])
        for obsolete in ("stage", "anchor", "course", "stratified", "ramp", "policy_only"):
            self.assertNotIn(obsolete, source.lower())

    def test_capacity_and_complete_state_are_explicit(self):
        source = (ROOT / "legged_gym/rl/MGDP/amp/session.py").read_text(encoding="utf-8")
        self.assertIn("2 * num_envs * num_steps", source)
        for field in ("discriminator", "optimizer", "normalizer", "replay",
                      "update_count", "compatibility", "amp_seed", "dataset_rng_state"):
            self.assertIn(f'"{field}"', source)


@unittest.skipUnless(HAS_TORCH, "PyTorch is tested on the training host")
class TestSingleAmpTensor(unittest.TestCase):
    @staticmethod
    def _config(**overrides):
        config = {
            "amp_reward_coef": 0.01,
            "amp_batch_size": 8,
            "amp_updates_per_iter": 20,
            "amp_gradient_penalty_coef": 10.0,
            "amp_learning_rate": 1e-4,
            "amp_replay_rollouts": 2,
            "num_envs": 4,
            "num_steps_per_env": 3,
            "amp_groups": {"forward": ["forward.txt"], "turn": ["turn.txt"]},
            "amp_group_weights": {"forward": 0.8, "turn": 0.2},
        }
        config.update(overrides)
        return config

    class Dataset:
        def __init__(self, seed=91):
            self.rng = np.random.default_rng(seed)
            self.group_names = ["forward", "turn"]

        def sample(self, count, device):
            import torch
            first = self.rng.normal(size=(count, 30)).astype(np.float32)
            second = self.rng.normal(size=(count, 30)).astype(np.float32)
            return (torch.as_tensor(first, device=device),
                    torch.as_tensor(second, device=device))

        def sample_with_groups(self, count, device):
            import torch
            first, second = self.sample(count, device)
            groups = torch.arange(count, device=device) % 2
            return first, second, groups

    def test_construction_preserves_global_numpy_and_torch_rng(self):
        import torch
        from rl.MGDP.amp.session import AMPSession
        np.random.seed(17)
        torch.manual_seed(19)
        numpy_before = np.random.get_state()
        torch_before = torch.random.get_rng_state().clone()

        AMPSession(self.Dataset(), "cpu", self._config(), amp_seed=23)

        numpy_after = np.random.get_state()
        self.assertTrue(all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
                            for a, b in zip(numpy_before, numpy_after)))
        self.assertTrue(torch.equal(torch_before, torch.random.get_rng_state()))

    def test_reward_is_exactly_masked_contribution_and_never_updates_stats(self):
        import torch
        from rl.MGDP.amp.session import AMPSession
        session = AMPSession(self.Dataset(), "cpu", self._config(), amp_seed=3)

        class Fixed(torch.nn.Module):
            def forward(self, pair):
                return torch.tensor([1.0, 3.0], device=pair.device)[:pair.shape[0]]

        session.discriminator = Fixed()
        state = torch.zeros(2, 30)
        eligible = torch.tensor([True, True])
        count_before = session.normalizer.count
        enabled = session.reward(state, state, eligible, enabled=True)
        disabled = session.reward(state, state, eligible, enabled=False)
        self.assertTrue(torch.allclose(enabled, torch.tensor([0.01, 0.0])))
        self.assertTrue(torch.equal(disabled, torch.zeros(2)))
        idle_masked = session.reward(
            state, state, torch.tensor([True, False]), enabled=True)
        self.assertEqual(idle_masked[1].item(), 0.0)
        self.assertEqual(session.normalizer.count, count_before)

    def test_single_replay_capacity_update_objective_and_gp(self):
        import torch
        from rl.MGDP.amp.session import AMPSession
        session = AMPSession(self.Dataset(), "cpu", self._config(), amp_seed=5)
        self.assertEqual(session.replay.capacity, 24)
        state = torch.randn(12, 30)
        mask = torch.tensor([True, False] * 6)
        session.record(state, state + 0.1, mask)
        self.assertEqual(session.replay.size, 6)
        count_before = session.normalizer.count
        coefficients = []
        original = session.discriminator.gradient_penalty

        def record_gp(pair, coefficient):
            coefficients.append(coefficient)
            return original(pair, coefficient)

        session.discriminator.gradient_penalty = record_gp
        metrics = session.update()
        self.assertEqual(session.update_count, 20)
        self.assertEqual(metrics["amp_updates"], 20)
        self.assertEqual(coefficients, [10.0] * 20)
        self.assertGreater(session.normalizer.count, count_before)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["discriminator_loss"])))

    def test_complete_round_trip_restores_rng_and_rejects_before_mutation(self):
        import torch
        from rl.MGDP.amp.session import AMPSession
        source = AMPSession(self.Dataset(), "cpu", self._config(), amp_seed=7)
        source.record(torch.randn(10, 30), torch.randn(10, 30), torch.ones(10, dtype=torch.bool))
        source.update()
        checkpoint = source.state_dict()

        restored = AMPSession(self.Dataset(), "cpu", self._config(), amp_seed=7)
        restored.load_state_dict(checkpoint)
        self.assertEqual(restored.update_count, source.update_count)
        self.assertEqual(restored.replay.cursor, source.replay.cursor)
        a1, b1 = source.replay.sample(5)
        a2, b2 = restored.replay.sample(5)
        self.assertTrue(torch.equal(a1, a2))
        self.assertTrue(torch.equal(b1, b2))
        e1 = source.dataset.sample(5, "cpu")
        e2 = restored.dataset.sample(5, "cpu")
        self.assertTrue(torch.equal(e1[0], e2[0]))

        incompatible = AMPSession(
            self.Dataset(), "cpu", self._config(amp_reward_coef=0.02), amp_seed=7)
        before = next(incompatible.discriminator.parameters()).detach().clone()
        with self.assertRaisesRegex(ValueError, "amp_reward_coef"):
            incompatible.load_state_dict(checkpoint)
        self.assertTrue(torch.equal(
            before, next(incompatible.discriminator.parameters()).detach()))


if __name__ == "__main__":
    unittest.main()
