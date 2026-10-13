import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "legged_gym/rl/MGDP/runners/go2_amp_runner.py"


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
        self.assertIn("return mean_value_loss, mean_surrogate_loss, mean_vel_loss, mean_feet_loss, mean_cnn_loss", source)


if __name__ == "__main__":
    unittest.main()
