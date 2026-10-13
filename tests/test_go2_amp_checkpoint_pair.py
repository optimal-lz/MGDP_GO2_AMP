import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "legged_gym/rl/MGDP/runners/go2_amp_checkpoint.py"
RUNNER_PATH = ROOT / "legged_gym/rl/MGDP/runners/go2_amp_runner.py"


def load_module():
    spec = importlib.util.spec_from_file_location("go2_amp_checkpoint", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dump_pickle(state, path):
    with Path(path).open("wb") as stream:
        pickle.dump(state, stream)


def load_pickle(path, **_kwargs):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)


class TestPairedCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temp.name)
        self.compatibility = self.module.CheckpointCompatibility.from_config(
            mode="amp",
            amp_reward_coef=0.01,
            seed=1,
            num_envs=4096,
            key_config={"steps": 24, "terrain": ["flat", "stairs"]},
        )

    def tearDown(self):
        self.temp.cleanup()

    def manager(self):
        return self.module.PairedCheckpointManager(
            self.root, save_state=dump_pickle, load_state=load_pickle)

    def test_names_hashes_iteration_and_pointer_only_last(self):
        manager = self.manager()
        manifest_path = manager.save_pair(
            1000, {"iteration": 1000, "actor": "a"},
            {"iteration": 1000, "wm": "w"}, self.compatibility)

        self.assertEqual(manifest_path.name, "checkpoint_1000.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["files"]["policy"]["name"],
                         "policy_amp_1000.pt")
        self.assertEqual(manifest["files"]["world_model"]["name"],
                         "wm_1000.pt")
        self.assertEqual(manifest["iteration"], 1000)
        for record in manifest["files"].values():
            payload = (self.root / record["name"]).read_bytes()
            self.assertEqual(record["sha256"],
                             hashlib.sha256(payload).hexdigest())
        self.assertEqual(
            json.loads((self.root / "last.json").read_text(encoding="utf-8")),
            {"manifest": "checkpoint_1000.json"},
        )
        self.assertEqual(manager.resolve_last(), manifest_path)

        no_amp = dataclasses.replace(
            self.compatibility, mode="no_amp", amp_reward_coef=0.0)
        no_amp_path = manager.save_pair(
            10, {"iteration": 10}, {"iteration": 10}, no_amp)
        no_amp_manifest = json.loads(no_amp_path.read_text(encoding="utf-8"))
        self.assertEqual(no_amp_manifest["files"]["policy"]["name"],
                         "policy_no_amp_10.pt")

    def test_interrupted_save_never_advances_last_pointer(self):
        manager = self.manager()
        manager.save_pair(1, {"iteration": 1}, {"iteration": 1},
                          self.compatibility)
        previous = (self.root / "last.json").read_bytes()
        real_replace = manager._replace

        def fail_before_manifest(source, destination):
            if Path(destination).name == "checkpoint_2.json":
                raise OSError("injected interruption")
            real_replace(source, destination)

        manager._replace = fail_before_manifest
        with self.assertRaisesRegex(OSError, "injected"):
            manager.save_pair(2, {"iteration": 2}, {"iteration": 2},
                              self.compatibility)
        self.assertEqual((self.root / "last.json").read_bytes(), previous)
        self.assertEqual(manager.resolve_last().name, "checkpoint_1.json")

    def test_rejects_compatibility_missing_and_hash_before_deserialization(self):
        manager = self.manager()
        manifest = manager.save_pair(
            5, {"iteration": 5}, {"iteration": 5}, self.compatibility)
        loaded = []
        manager._load_state = lambda path, **kwargs: loaded.append(path)
        variants = {
            "mode": dataclasses.replace(
                self.compatibility, mode="no_amp", amp_reward_coef=0.0),
            "amp_reward_coef": dataclasses.replace(
                self.compatibility, amp_reward_coef=0.02),
            "seed": dataclasses.replace(self.compatibility, seed=2),
            "num_envs": dataclasses.replace(self.compatibility, num_envs=64),
            "config_hash": dataclasses.replace(
                self.compatibility, config_hash="0" * 64),
        }
        for field, incompatible in variants.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    manager.load_pair(manifest, incompatible)
                self.assertEqual(loaded, [])

        data = json.loads(manifest.read_text(encoding="utf-8"))
        policy = self.root / data["files"]["policy"]["name"]
        policy.write_bytes(policy.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "sha256"):
            manager.load_pair(manifest, self.compatibility)
        self.assertEqual(loaded, [])

        policy.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            manager.load_pair(manifest, self.compatibility)
        self.assertEqual(loaded, [])

    def test_round_trip_and_runner_continuity_contract(self):
        policy = {
            "iteration": 7,
            "actor": {"weight": [1, 2]},
            "ppo_optimizer": {"step": 9},
            "terrain_levels": [0, 2, 19],
            "rng": {"python": "p", "numpy": "n", "torch": "t"},
            "amp_state": {
                "discriminator": "d", "optimizer": "do",
                "normalizer": "normal", "replay": "replay",
                "rng": "private",
            },
        }
        world_model = {
            "iteration": 7,
            "networks": {"encoder": "e", "decoder": "d"},
            "optimizer": {"step": 3},
            "depth_stats": {"mean": 0.5},
        }
        manager = self.manager()
        manifest = manager.save_pair(
            7, policy, world_model, self.compatibility)
        restored_policy, restored_wm, metadata = manager.load_pair(
            manifest, self.compatibility)
        self.assertEqual(restored_policy, policy)
        self.assertEqual(restored_wm, world_model)
        self.assertEqual(metadata["iteration"], 7)

        source = RUNNER_PATH.read_text(encoding="utf-8")
        for required in (
                "random.getstate", "np.random.get_state",
                "torch.random.get_rng_state", "torch.cuda.get_rng_state_all",
                "wm_optimizer", "depth_stats", "save_pair", "load_pair"):
            self.assertIn(required, source)
        self.assertIn('if self.amp is not None:', source)


if __name__ == "__main__":
    unittest.main()
