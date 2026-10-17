import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "legged_gym/scripts/vis_go2_amp.py"
RANDOM_DOG = ROOT / "legged_gym/legged_gym/envs/random_dog/random_dog.py"


class TestGo2AmpViewer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT.read_text(encoding="utf-8") if SCRIPT.exists() else ""
        cls.tree = ast.parse(cls.source) if cls.source else ast.Module(body=[])
        cls.functions = {
            node.name: ast.unparse(node) for node in cls.tree.body
            if isinstance(node, ast.FunctionDef)
        }

    def test_default_is_deterministic_live_viewer(self):
        for text in (
                '"--run_dir"', '"--manifest"', '"--terrain"', '"--level"',
                '"--vx"', '"--yaw"', '"--env_id"', '"--follow_camera"'):
            self.assertIn(text, self.source)
        self.assertIn('"default": 4', self.source)
        self.assertIn("env_cfg.noise.add_noise = False", self.source)
        self.assertIn("env_cfg.domain_rand.push_robots = False", self.source)
        self.assertIn("env_cfg.camera.update_wm = False", self.source)
        self.assertIn("disable_evaluation_stochasticity", self.source)
        self.assertIn("torch.inference_mode", self.source)
        self.assertIn("runner.load_pair", self.source)
        self.assertIn("restore_training_state=False", self.source)
        self.assertIn("actor_critic.eval()", self.source)

    def test_ffmpeg_is_touched_only_by_explicit_recording_branch(self):
        self.assertIn('"--record"', self.source)
        self.assertIn('"--record_output"', self.source)
        self.assertIn('"--duration"', self.source)
        self.assertIn('"--fps"', self.source)
        self.assertIn("_prepare_recording", self.functions)
        self.assertIn("shutil.which('ffmpeg')",
                      self.functions["_prepare_recording"])
        main = self.functions.get("main", "")
        self.assertIn("if args.record", main)
        self.assertIn("_prepare_recording", main)
        self.assertLess(main.index("if args.record"),
                        main.index("_prepare_recording"))

    def test_wm_and_video_paths_remain_beneath_selected_run(self):
        for text in (
                '"eval" / "wm"', '"depth_latest.png"',
                '"height_latest.png"', '"eval" / "videos"'):
            self.assertIn(text, self.source)
        self.assertIn('"--numbered_snapshots"', self.source)
        self.assertIn("ensure_within_run", self.source)
        random_dog = RANDOM_DOG.read_text(encoding="utf-8")
        self.assertIn("compare_vis_filename", random_dog)
        self.assertIn("compare_height_vis_filename", random_dog)

    def test_episode_diagnostics_cover_task_style_and_failure(self):
        for text in (
                "command", "actual velocity", "terrain", "level",
                "AMP reward", "failure"):
            self.assertIn(text, self.source)

    def test_loaded_case_refreshes_before_action_and_resets_done_memory(self):
        main = self.functions["main"]
        self.assertIn("env.refresh_visual_observation()", main)
        self.assertLess(main.index("env.refresh_visual_observation()"),
                        main.index("actions = policy(obs)"))
        self.assertIn("runner.alg.actor_critic.reset(dones)", main)


if __name__ == "__main__":
    unittest.main()
