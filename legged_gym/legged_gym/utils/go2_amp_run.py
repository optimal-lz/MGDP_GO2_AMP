"""Pure run naming, target, directory, and provenance helpers."""

import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import uuid

import numpy as np


def normalize_effective_config(value):
    """Return one JSON-native representation for run/checkpoint config.

    ``class_to_dict`` preserves several Python and NumPy container/scalar
    types.  Normalizing once at the formal entrypoint prevents the in-memory
    configuration from comparing unequal to the same configuration loaded
    back from JSON during resume.
    """
    if isinstance(value, np.ndarray):
        return normalize_effective_config(value.tolist())
    if isinstance(value, np.generic):
        return normalize_effective_config(value.item())
    if isinstance(value, dict):
        return {
            normalize_effective_config(key): normalize_effective_config(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [normalize_effective_config(item) for item in value]
    return value


def _coefficient_token(value):
    text = format(float(value), ".15g")
    if "e" in text.lower():
        text = format(float(value), ".15f").rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def format_run_name(mode, amp_reward_coef, seed, num_envs):
    if mode == "amp":
        if amp_reward_coef is None or float(amp_reward_coef) <= 0:
            raise ValueError("AMP mode requires a positive coefficient")
        name = "amp_coef{}_seed{}".format(
            _coefficient_token(amp_reward_coef), int(seed))
    elif mode == "no_amp":
        if amp_reward_coef not in (None, 0, 0.0):
            raise ValueError("no_amp mode does not accept an AMP coefficient")
        name = "no_amp_seed{}".format(int(seed))
    else:
        raise ValueError("mode must be 'amp' or 'no_amp'")
    if int(num_envs) != 4096:
        name += "_env{}".format(int(num_envs))
    return name


def remaining_iterations(target_iteration, current_iteration):
    target = int(target_iteration)
    current = int(current_iteration)
    if target <= current:
        raise ValueError(
            "target_iteration must be greater than current iteration")
    return target - current


def prepare_run_directory(output_root, run_name, resume=False):
    run_dir = Path(output_root) / run_name
    if run_dir.exists() and not resume:
        raise FileExistsError(
            "run directory already exists; use explicit --resume_from")
    if resume and not run_dir.is_dir():
        raise FileNotFoundError(
            "resume run directory does not exist: {}".format(run_dir))
    if not resume:
        run_dir.mkdir(parents=True)
    for relative in (
            "checkpoints", "tensorboard", "eval/wm",
            "eval/comparison", "eval/videos"):
        (run_dir / relative).mkdir(parents=True, exist_ok=True)
    return run_dir


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(
        ".{}.{}.tmp".format(path.name, uuid.uuid4().hex))
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def _default_git_commit():
    repository = Path(__file__).resolve().parents[3]
    try:
        return subprocess.check_output(
            ["git", "-c", "safe.directory={}".format(repository),
             "rev-parse", "HEAD"],
            cwd=str(repository), stderr=subprocess.DEVNULL,
            text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _default_runtime():
    runtime = {
        "python": platform.python_version(),
        "torch": "unavailable",
        "cuda": "unavailable",
        "isaac_gym": "unavailable",
    }
    try:
        import torch
        runtime["torch"] = str(torch.__version__)
        runtime["cuda"] = str(torch.version.cuda)
    except ImportError:
        pass
    try:
        import isaacgym
        runtime["isaac_gym"] = str(
            getattr(isaacgym, "__version__", "installed"))
    except ImportError:
        pass
    return runtime


def write_run_metadata(run_dir, effective_config, command=None, experts=None,
                       git_commit=None, runtime=None):
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    existing_config = None
    if config_path.is_file():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config != effective_config:
            raise ValueError("effective config differs from existing run config")
    else:
        _atomic_json(config_path, effective_config)

    command = list(sys.argv if command is None else command)
    run_path = run_dir / "run.json"
    if run_path.is_file():
        provenance = json.loads(run_path.read_text(encoding="utf-8"))
        provenance.setdefault("resume_commands", []).append(command)
    else:
        provenance = {
            "command": command,
            "git_commit": git_commit or _default_git_commit(),
            "runtime": runtime or _default_runtime(),
            "experts": list(experts or []),
        }
    _atomic_json(run_path, provenance)
    return config_path, run_path

