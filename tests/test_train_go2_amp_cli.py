import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
UTIL_PATH = ROOT / "legged_gym/legged_gym/utils/go2_amp_run.py"
SCRIPT_PATH = ROOT / "legged_gym/scripts/train_go2_amp.py"


def load_utils():
    spec = importlib.util.spec_from_file_location("go2_amp_run", UTIL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestGo2AmpRunUtilities(unittest.TestCase):
    def setUp(self):
        self.module = load_utils()

    def test_frozen_names_and_nondefault_env_suffix(self):
        self.assertEqual(
            self.module.format_run_name("amp", 0.01, 1, 4096),
            "amp_coef0p01_seed1",
        )
        self.assertEqual(
            self.module.format_run_name("no_amp", None, 1, 4096),
            "no_amp_seed1",
        )
        self.assertEqual(
            self.module.format_run_name("amp", 0.0005, 3, 64),
            "amp_coef0p0005_seed3_env64",
        )
        with self.assertRaisesRegex(ValueError, "coefficient"):
            self.module.format_run_name("no_amp", 0.01, 1, 4096)

    def test_absolute_target_is_converted_to_remaining_work(self):
        self.assertEqual(self.module.remaining_iterations(2000, 1000), 1000)
        self.assertEqual(self.module.remaining_iterations(10, 0), 10)
        with self.assertRaisesRegex(ValueError, "greater"):
            self.module.remaining_iterations(1000, 1000)
        with self.assertRaisesRegex(ValueError, "greater"):
            self.module.remaining_iterations(999, 1000)

    def test_directory_requires_explicit_resume_and_metadata_is_complete(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            output_root = Path(temporary)
            run_dir = self.module.prepare_run_directory(
                output_root, "amp_coef0p01_seed1")
            for name in ("checkpoints", "tensorboard", "eval"):
                self.assertTrue((run_dir / name).is_dir())
            with self.assertRaisesRegex(FileExistsError, "resume"):
                self.module.prepare_run_directory(
                    output_root, "amp_coef0p01_seed1")

            resumed = self.module.prepare_run_directory(
                output_root, "amp_coef0p01_seed1", resume=True)
            self.assertEqual(resumed, run_dir)
            self.module.write_run_metadata(
                run_dir,
                effective_config={"runner": {"mode": "amp"}},
                command=["train_go2_amp.py", "--mode", "amp"],
                experts=["go2_forward.txt", "go2_turn_left.txt"],
                git_commit="abc123",
                runtime={"python": "3.8", "torch": "2.0", "cuda": "11.8",
                         "isaac_gym": "preview4"},
            )
            config = json.loads((run_dir / "config.json").read_text(
                encoding="utf-8"))
            run = json.loads((run_dir / "run.json").read_text(
                encoding="utf-8"))
            self.assertEqual(config["runner"]["mode"], "amp")
            self.assertEqual(run["command"][0], "train_go2_amp.py")
            self.assertEqual(run["git_commit"], "abc123")
            self.assertEqual(run["runtime"]["cuda"], "11.8")
            self.assertEqual(run["experts"],
                             ["go2_forward.txt", "go2_turn_left.txt"])


class TestOfficialTrainingScript(unittest.TestCase):
    def test_cli_is_narrow_and_uses_absolute_target(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn('"--mode"', source)
        self.assertIn('"required": True', source)
        self.assertIn('"--target_iteration"', source)
        self.assertIn('"--resume_from"', source)
        self.assertIn('"--amp_reward_coef"', source)
        self.assertIn('"default": 4096', source)
        self.assertIn('"default": 1', source)
        self.assertNotIn('"--task"', source)
        self.assertNotIn('"--algo"', source)
        self.assertNotIn("amp_policy_only", source)
        self.assertIn("gymutil.parse_arguments", source)
        self.assertIn('task_registry.make_env(name="go2_amp"', source)
        self.assertIn("Go2AmpPolicyRunner", source)
        self.assertIn("runner.load_pair", source)
        self.assertIn("remaining_iterations", source)
        self.assertIn("runner.learn", source)


if __name__ == "__main__":
    unittest.main()
