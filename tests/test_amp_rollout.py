import ast
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "legged_gym/rl/MGDP/runners/go2_amp_runner.py"
HAS_TORCH = importlib.util.find_spec("torch") is not None


class TestGo2AmpRunnerSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not RUNNER.exists():
            cls.source = ""
            cls.methods = {}
            return
        cls.source = RUNNER.read_text(encoding="utf-8")
        tree = ast.parse(cls.source)
        runner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name == "Go2AmpPolicyRunner")
        cls.methods = {node.name: ast.unparse(node) for node in runner.body
                       if isinstance(node, ast.FunctionDef)}

    def test_formal_runner_api_and_mode_split(self):
        for name in ("__init__", "learn", "state_dict", "load_state_dict",
                     "get_inference_policy"):
            self.assertIn(name, self.methods)
        init = self.methods["__init__"]
        self.assertIn("if self.mode == 'amp'", init)
        self.assertIn("AMPSession", init)
        self.assertIn("MotionDataset", init)
        self.assertIn("self.amp = None", init)

    def test_rollout_zero_disabled_then_fixed_amp(self):
        learn = self.methods["learn"]
        self.assertIn("enabled=iteration > 0", learn)
        self.assertIn("amp_transition_mask", learn)
        self.assertIn("terminal_amp_states", learn)
        self.assertIn("self.amp.record", learn)
        self.assertIn("self.env.compose_total_reward", learn)

    def test_update_order_is_wm4_gae_ppo_d20(self):
        learn = self.methods["learn"]
        extra_wm = learn.index("wm_rollout_steps")
        gae = learn.index("self.alg.compute_returns")
        ppo = learn.index("self.alg.update()")
        discriminator = learn.index("self.amp.update()")
        self.assertLess(extra_wm, gae)
        self.assertLess(gae, ppo)
        self.assertLess(ppo, discriminator)
        self.assertIn("self.env.update_world_model", learn)

    def test_noamp_never_constructs_or_calls_amp(self):
        init = self.methods["__init__"]
        self.assertLess(init.index("self.amp = None"), init.index("if self.mode == 'amp'"))
        learn = self.methods["learn"]
        self.assertIn("if self.amp is not None", learn)
        self.assertIn("amp_contribution = torch.zeros_like", learn)

    def test_registry_uses_configured_runner_with_legacy_fallback(self):
        source = (ROOT / "legged_gym/legged_gym/utils/task_registry.py").read_text(
            encoding="utf-8")
        self.assertIn("runner_class_name", source)
        self.assertIn("Go2AmpPolicyRunner", source)
        self.assertIn("MGDPPolicyRunner", source)
        self.assertNotIn('eval(args.algo + "PolicyRunner")', source)

    def test_ppo_exposes_metrics_without_changing_return_shape(self):
        source = (ROOT / "legged_gym/rl/MGDP/algorithms/ppo.py").read_text(
            encoding="utf-8")
        self.assertIn("self.last_update_metrics", source)
        self.assertIn('"ppo/entropy"', source)
        self.assertIn('"ppo/kl"', source)
        self.assertIn("return mean_value_loss, mean_surrogate_loss, mean_vel_loss, mean_feet_loss, mean_cnn_loss", source)

    def test_ppo_uses_the_contiguous_recurrent_generator(self):
        ppo = (ROOT / "legged_gym/rl/MGDP/algorithms/ppo.py").read_text(
            encoding="utf-8")
        storage = (ROOT / "legged_gym/rl/MGDP/storage/rollout_storage.py").read_text(
            encoding="utf-8")
        self.assertIn("recurrent_mini_batch_generator", ppo)
        self.assertNotIn("reccurent_mini_batch_generator", ppo)
        self.assertIn("def recurrent_mini_batch_generator", storage)
        self.assertIn("image_update_mask", storage)

    def test_runner_populates_the_frozen_lightweight_metric_schema(self):
        source = self.source
        for tag in (
                "reward/regularization", "tracking/vx_error",
                "tracking/yaw_error", "episode/length",
                "episode/base_contact_failure", "wm/total_loss",
                "wm/depth_loss", "wm/height_loss", "wm/contrastive_loss",
                "performance/iteration_seconds", "amp/valid_transition_count",
                "amp/replay_fill", "timing/simulation_seconds",
                "timing/wm_seconds", "timing/ppo_seconds",
                "timing/discriminator_seconds"):
            self.assertIn(tag, source)
        self.assertIn("TERRAIN_NAMES", source)
        self.assertIn("expert_forward_logit", source)
        self.assertIn("expert_turn_logit", source)


@unittest.skipUnless(HAS_TORCH, "PyTorch is tested on the training host")
class TestRecurrentRolloutStorage(unittest.TestCase):
    def test_generator_keeps_complete_ordered_environment_sequences(self):
        import torch
        from rl.MGDP.storage.rollout_storage import RolloutStorage

        storage = RolloutStorage(
            4, 3, [1], [2], [1], [2], 64, device="cpu")
        for step in range(3):
            transition = RolloutStorage.Transition()
            env_values = torch.arange(4, dtype=torch.float).unsqueeze(1)
            transition.observations = env_values + step * 10
            transition.actions = env_values
            transition.rewards = torch.ones(4)
            transition.dones = torch.tensor(
                [False, step == 0, False, False])
            transition.values = torch.zeros(4, 1)
            transition.actions_log_prob = torch.zeros(4)
            transition.action_mean = torch.zeros(4, 1)
            transition.action_sigma = torch.ones(4, 1)
            transition.hidden_states = env_values.repeat(1, 64) + step * 100
            transition.next_observations = transition.observations + 1
            transition.privileged_info = torch.zeros(4, 2)
            transition.proprio_hist = torch.zeros(4, 2)
            transition.image_buf = torch.zeros(4, 64)
            transition.image_update_mask = torch.full(
                (4, 1), step % 2 == 0, dtype=torch.bool)
            storage.add_transitions(transition)

        batches = list(storage.recurrent_mini_batch_generator(
            num_mini_batches=2, num_epochs=1))
        self.assertEqual(len(batches), 2)
        seen = 0
        for batch in batches:
            obs = batch[0]
            hidden = batch[9][0]
            masks = batch[10]
            self.assertEqual(obs.shape, (3, 2, 1))
            self.assertEqual(hidden.shape, (2, 64))
            self.assertTrue(torch.equal(obs[1] - obs[0],
                                        torch.full((2, 1), 10.0)))
            self.assertTrue(torch.equal(obs[2] - obs[1],
                                        torch.full((2, 1), 10.0)))
            self.assertEqual(masks["image_update_mask"].shape, (3, 2, 1))
            self.assertEqual(masks["episode_starts"].shape, (3, 2, 1))
            seen += obs.shape[1]
        self.assertEqual(seen, 4)


if __name__ == "__main__":
    unittest.main()
