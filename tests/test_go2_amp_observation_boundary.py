import ast
import importlib.util
from pathlib import Path
import sys
import unittest


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
        cls.methods = {node.name: ast.unparse(node) for node in actor.body
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

    def test_environment_publishes_only_detached_wm_tokens(self):
        source = (ROOT / "legged_gym/legged_gym/envs/random_dog/random_dog.py").read_text(
            encoding="utf-8")
        assignments = [line.strip() for line in source.splitlines()
                       if "obs_dict['image_buf'] =" in line]
        self.assertGreaterEqual(len(assignments), 4)
        self.assertTrue(all(".detach()" in line for line in assignments), assignments)


@unittest.skipUnless(HAS_TORCH, "PyTorch is tested on the training host")
class TestObservationBoundaryTensor(unittest.TestCase):
    def _model(self):
        from rl.MGDP.modules.actor_critic import ActorCritic
        return ActorCritic(
            45, 12,
            actor_hidden_dims=[32, 16], critic_hidden_dims=[32, 16],
            MLPModule_info={"input_dims": 180, "hidden_dims": [32, 16, 7]},
            CNNModule_info={"output_channels": 64},
        )

    def _observations(self, torch, batch=3):
        return {
            "obs": torch.randn(batch, 45),
            "privileged_info": torch.randn(batch, 194),
            "proprio_hist": torch.randn(batch, 180),
            "image_buf": torch.randn(batch, 64),
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
        model = self._model()
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


if __name__ == "__main__":
    unittest.main()
