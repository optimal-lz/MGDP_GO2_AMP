import ast
import importlib.util
from pathlib import Path
import sys
import unittest

from tests.ast_source import node_source


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legged_gym"))
HAS_TORCH = importlib.util.find_spec("torch") is not None
ACTOR_PATH = ROOT / "legged_gym/rl/MGDP/modules/actor_critic.py"


class TestObservationBoundarySource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = ACTOR_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        actor = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == "ActorCritic")
        cls.methods = {node.name: node_source(source, node) for node in actor.body
                       if isinstance(node, ast.FunctionDef)}
        cls.source = source

    def test_explicit_actor_and_critic_builders_exist(self):
        self.assertIn("build_actor_input", self.methods)
        self.assertIn("build_critic_input", self.methods)
        self.assertIn("obs_dict['image_buf'].detach()",
                      self.methods["build_actor_input"])
        self.assertNotIn("privileged_info", self.methods["build_actor_input"])
        self.assertNotIn("image_buf", self.methods["build_critic_input"])
        self.assertIn("privileged_info", self.methods["build_critic_input"])

    def test_declared_dimensions_are_116_actor_and_239_critic(self):
        init = self.methods["__init__"]
        self.assertIn("num_obs + camera_dim + proprio_dim", init)
        self.assertIn("num_obs + privileged_dim", init)
        self.assertIn("privileged_dim = 194", init)

    def test_actor_owns_refresh_gated_visual_memory(self):
        init = self.methods["__init__"]
        reset = self.methods["reset"]
        self.assertIn("use_visual_memory", init)
        self.assertIn("visual_gru", init)
        self.assertIn("image_update_mask", self.methods["build_actor_input"])
        self.assertIn("visual_gru", self.methods["_visual_memory"])
        self.assertIn("visual_hidden_state", reset)
        self.assertIn("torch.where", reset)
        self.assertNotIn("self.visual_hidden_state.zero_()", reset)
        self.assertNotIn("self.visual_hidden_state[done_mask]", reset)
        self.assertIn("visual_token[..., 0:32]",
                      self.methods["build_actor_input"])
        self.assertIn("visual_token[..., 32:]",
                      self.methods["build_actor_input"])

    def test_environment_publishes_only_detached_wm_tokens(self):
        source = (ROOT / "legged_gym/legged_gym/envs/random_dog/random_dog.py").read_text(
            encoding="utf-8")
        assignments = [line.strip() for line in source.splitlines()
                       if "obs_dict['image_buf'] =" in line]
        self.assertGreaterEqual(len(assignments), 4)
        self.assertTrue(all(".detach()" in line for line in assignments), assignments)


@unittest.skipUnless(HAS_TORCH, "PyTorch is tested on the training host")
class TestObservationBoundaryTensor(unittest.TestCase):
    def _model(self, use_visual_memory=False):
        from rl.MGDP.modules.actor_critic import ActorCritic
        return ActorCritic(
            45, 12,
            actor_hidden_dims=[32, 16], critic_hidden_dims=[32, 16],
            MLPModule_info={"input_dims": 180, "hidden_dims": [32, 16, 7]},
            CNNModule_info={"output_channels": 64},
            use_visual_memory=use_visual_memory,
        )

    def _observations(self, torch, batch=3):
        return {
            "obs": torch.randn(batch, 45),
            "privileged_info": torch.randn(batch, 194),
            "proprio_hist": torch.randn(batch, 180),
            "image_buf": torch.randn(batch, 64),
            "image_update_mask": torch.ones(batch, 1, dtype=torch.bool),
        }

    def test_dimensions_and_cross_boundary_independence(self):
        import torch
        model = self._model()
        observations = self._observations(torch)
        actor = model.build_actor_input(observations)
        critic = model.build_critic_input(observations)
        self.assertEqual(actor.shape, (3, 116))
        self.assertEqual(critic.shape, (3, 239))

        privileged_changed = dict(observations)
        privileged_changed["privileged_info"] = observations["privileged_info"] + 1000
        self.assertTrue(torch.equal(actor, model.build_actor_input(privileged_changed)))

        wm_changed = dict(observations)
        wm_changed["image_buf"] = observations["image_buf"] + 1000
        self.assertTrue(torch.equal(critic, model.build_critic_input(wm_changed)))

    def test_actor_gradient_stops_at_wm_token(self):
        import torch
        model = self._model(use_visual_memory=True)
        observations = self._observations(torch)
        source = torch.randn(3, 64, requires_grad=True)
        observations["image_buf"] = source * 2.0

        model.build_actor_input(observations).sum().backward()

        self.assertIsNone(source.grad)

    def test_ppo_optimizer_has_no_environment_wm_parameter(self):
        import torch
        from rl.MGDP.algorithms.ppo import PPO
        model = self._model()
        ppo = PPO(model)
        environment_wm_parameter = torch.nn.Parameter(torch.ones(1))
        optimized = {id(parameter) for group in ppo.optimizer.param_groups
                     for parameter in group["params"]}
        self.assertNotIn(id(environment_wm_parameter), optimized)

    def test_actor_gru_is_refresh_gated_trainable_and_wm_detached(self):
        import torch
        model = self._model(use_visual_memory=True)
        observations = self._observations(torch)
        source = torch.randn(3, 64, requires_grad=True)
        observations["image_buf"] = source * 2.0
        initial = torch.zeros(3, 64)

        actor_input = model.build_actor_input(
            observations, hidden_states=initial)
        actor_input.sum().backward()

        self.assertIsNone(source.grad)
        self.assertTrue(any(parameter.grad is not None
                            for parameter in model.visual_gru.parameters()))

        observations["image_update_mask"].zero_()
        held = model.build_actor_input(
            observations, hidden_states=initial)
        self.assertTrue(torch.equal(held[:, -64:], initial))

    def test_runtime_hidden_reset_and_sequence_episode_start(self):
        import torch
        model = self._model(use_visual_memory=True)
        observations = self._observations(torch, batch=3)
        with torch.inference_mode():
            model.act_inference(observations)
        before = model.get_hidden_states().clone()
        self.assertFalse(torch.equal(before, torch.zeros_like(before)))
        model.reset(torch.tensor([False, True, False]))
        after = model.get_hidden_states()
        self.assertTrue(torch.equal(after[0], before[0]))
        self.assertTrue(torch.equal(after[1], torch.zeros_like(after[1])))
        self.assertTrue(torch.equal(after[2], before[2]))

        sequence = {
            "obs": torch.randn(3, 2, 45),
            "privileged_info": torch.randn(3, 2, 194),
            "proprio_hist": torch.randn(3, 2, 180),
            "image_buf": torch.randn(3, 2, 64),
            "image_update_mask": torch.ones(3, 2, 1, dtype=torch.bool),
            "episode_starts": torch.zeros(3, 2, 1, dtype=torch.bool),
        }
        sequence["episode_starts"][1, 0] = True
        initial = torch.randn(2, 64)
        actor_input = model.build_actor_input(sequence, hidden_states=initial)
        expected = model.visual_gru(
            sequence["image_buf"][1, 0].detach().unsqueeze(0),
            torch.zeros(1, 64),
        ).squeeze(0)
        self.assertTrue(torch.allclose(actor_input[1, 0, -64:], expected))


if __name__ == "__main__":
    unittest.main()
