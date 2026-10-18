"""Atomic publication and strict loading of policy/world-model pairs."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import uuid


def _canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_config_hash(config):
    payload = _canonical_json(config).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def restore_ppo_learning_rate(algorithm, learning_rate):
    """Restore PPO's adaptive scalar and every optimizer parameter group."""
    learning_rate = float(learning_rate)
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("PPO learning rate must be finite and positive")
    algorithm.learning_rate = learning_rate
    for group in algorithm.optimizer.param_groups:
        group["lr"] = learning_rate


def prepare_torch_rng_restore(torch_module, rng):
    """Normalize mapped RNG tensors before returning their restore action."""
    torch_state = rng["torch"].cpu()
    cuda_states = [state.cpu() for state in rng["torch_cuda"]]

    def restore():
        torch_module.random.set_rng_state(torch_state)
        if cuda_states and torch_module.cuda.is_available():
            torch_module.cuda.set_rng_state_all(cuda_states)

    return restore


def _copy_array(destination, source):
    if hasattr(destination, "copy_"):
        destination.copy_(source)
    else:
        destination[...] = source


def validate_terrain_state(env, terrain_levels):
    """Validate curriculum rows before any checkpoint state is mutated."""
    if tuple(terrain_levels.shape) != tuple(env.terrain_levels.shape):
        raise ValueError("checkpoint terrain_levels shape mismatch")
    invalid = ((terrain_levels < 0)
               | (terrain_levels >= int(env.max_terrain_level)))
    if bool(invalid.any()):
        raise ValueError("checkpoint terrain_levels are outside valid range")


def restore_terrain_state(env, terrain_levels):
    """Restore curriculum rows and synchronize their derived lookup state."""
    validate_terrain_state(env, terrain_levels)
    _copy_array(env.terrain_levels, terrain_levels)
    origins = env.terrain_origins[env.terrain_levels, env.terrain_types]
    classes = env.terrain_class[env.terrain_levels, env.terrain_types]
    _copy_array(env.env_origins, origins)
    _copy_array(env.env_class, classes)


@dataclass(frozen=True)
class CheckpointCompatibility:
    mode: str
    amp_reward_coef: float
    seed: int
    num_envs: int
    config_hash: str

    @classmethod
    def from_config(cls, mode, amp_reward_coef, seed, num_envs, key_config):
        return cls(
            mode=str(mode),
            amp_reward_coef=float(amp_reward_coef),
            seed=int(seed),
            num_envs=int(num_envs),
            config_hash=canonical_config_hash(key_config),
        )

    def validate(self):
        if self.mode not in ("amp", "no_amp"):
            raise ValueError("mode must be 'amp' or 'no_amp'")
        if self.mode == "no_amp" and self.amp_reward_coef != 0.0:
            raise ValueError("no_amp compatibility requires zero amp_reward_coef")
        if self.mode == "amp" and self.amp_reward_coef <= 0.0:
            raise ValueError("amp compatibility requires positive amp_reward_coef")
        if self.num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if (len(self.config_hash) != 64
                or any(c not in "0123456789abcdef" for c in self.config_hash)):
            raise ValueError("config_hash must be a lowercase SHA-256 digest")


def _torch_save(state, path):
    import torch
    torch.save(state, path)


def _torch_load(path, map_location=None):
    import torch
    return torch.load(path, map_location=map_location)


class PairedCheckpointManager:
    """Publish two state files only after both are complete and hashed."""

    def __init__(self, directory, save_state=None, load_state=None,
                 map_location=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._save_state = save_state or _torch_save
        self._load_state = load_state or _torch_load
        self._map_location = map_location
        self._replace = os.replace

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            while True:
                block = stream.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()

    def _temporary(self, final_path):
        return final_path.with_name(
            ".{}.{}.tmp".format(final_path.name, uuid.uuid4().hex))

    def _write_json_atomic(self, final_path, payload):
        temporary = self._temporary(final_path)
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._replace(str(temporary), str(final_path))
        finally:
            if temporary.exists():
                temporary.unlink()

    def save_pair(self, iteration, policy_state, wm_state, compatibility):
        iteration = int(iteration)
        if iteration < 0:
            raise ValueError("iteration must be nonnegative")
        compatibility.validate()
        for label, state in (("policy", policy_state), ("world_model", wm_state)):
            if int(state.get("iteration", -1)) != iteration:
                raise ValueError(
                    f"{label} state iteration does not match pair iteration")

        policy_name = "policy_{}_{}.pt".format(
            compatibility.mode, iteration)
        wm_name = "wm_{}.pt".format(iteration)
        manifest_name = "checkpoint_{}.json".format(iteration)
        policy_path = self.directory / policy_name
        wm_path = self.directory / wm_name
        manifest_path = self.directory / manifest_name
        policy_temp = self._temporary(policy_path)
        wm_temp = self._temporary(wm_path)

        try:
            self._save_state(policy_state, policy_temp)
            self._save_state(wm_state, wm_temp)
            policy_hash = self._sha256(policy_temp)
            wm_hash = self._sha256(wm_temp)

            # Large files become visible first. They remain harmless orphans if
            # publication is interrupted before the manifest/pointer steps.
            self._replace(str(policy_temp), str(policy_path))
            self._replace(str(wm_temp), str(wm_path))
            manifest = {
                "version": 1,
                "iteration": iteration,
                **asdict(compatibility),
                "files": {
                    "policy": {"name": policy_name, "sha256": policy_hash},
                    "world_model": {"name": wm_name, "sha256": wm_hash},
                },
            }
            self._write_json_atomic(manifest_path, manifest)
            self._write_json_atomic(
                self.directory / "last.json", {"manifest": manifest_name})
            return manifest_path
        finally:
            for temporary in (policy_temp, wm_temp):
                if temporary.exists():
                    temporary.unlink()

    def _read_manifest(self, manifest):
        path = Path(manifest)
        if not path.is_absolute():
            path = self.directory / path
        if not path.is_file():
            raise ValueError(f"checkpoint manifest missing: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid checkpoint manifest: {path}") from exc
        return path, data

    @staticmethod
    def _validate_compatibility(manifest, expected):
        expected.validate()
        for field, expected_value in asdict(expected).items():
            actual = manifest.get(field)
            if actual != expected_value:
                raise ValueError(
                    "{} mismatch: checkpoint {!r}, expected {!r}".format(
                        field, actual, expected_value))

    def _validated_file(self, record, label, directory):
        if not isinstance(record, dict) or set(record) != {"name", "sha256"}:
            raise ValueError(f"invalid {label} file record")
        name = record["name"]
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError(f"invalid {label} filename")
        path = directory / name
        if not path.is_file():
            raise ValueError(f"checkpoint file missing: {name}")
        actual_hash = self._sha256(path)
        if actual_hash != record["sha256"]:
            raise ValueError(f"sha256 mismatch for {name}")
        return path

    def load_pair(self, manifest, expected):
        manifest_path, data = self._read_manifest(manifest)
        if data.get("version") != 1:
            raise ValueError("unsupported checkpoint manifest version")
        self._validate_compatibility(data, expected)
        files = data.get("files", {})
        policy_path = self._validated_file(
            files.get("policy"), "policy", manifest_path.parent)
        wm_path = self._validated_file(
            files.get("world_model"), "world_model", manifest_path.parent)

        policy_state = self._load_state(
            policy_path, map_location=self._map_location)
        wm_state = self._load_state(wm_path, map_location=self._map_location)
        iteration = int(data.get("iteration", -1))
        if int(policy_state.get("iteration", -2)) != iteration:
            raise ValueError("policy state iteration mismatch")
        if int(wm_state.get("iteration", -2)) != iteration:
            raise ValueError("world_model state iteration mismatch")
        return policy_state, wm_state, data

    def resolve_last(self):
        pointer = self.directory / "last.json"
        if not pointer.is_file():
            raise ValueError("last.json is missing")
        try:
            data = json.loads(pointer.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("last.json is invalid") from exc
        if set(data) != {"manifest"}:
            raise ValueError("last.json must contain only a manifest pointer")
        name = data["manifest"]
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("last.json manifest pointer is invalid")
        manifest = self.directory / name
        if not manifest.is_file():
            raise ValueError(f"checkpoint manifest missing: {name}")
        return manifest

