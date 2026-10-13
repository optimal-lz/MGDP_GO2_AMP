"""Official single entrypoint for AMP and matched noAMP training."""

import json
import os
from pathlib import Path
import sys


os.environ["DOG_NAMES"] = "go2"

import isaacgym
from isaacgym import gymutil

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403 - registers go2_amp
from legged_gym.utils import (
    class_to_dict,
    format_run_name,
    prepare_run_directory,
    remaining_iterations,
    task_registry,
    write_run_metadata,
)
from rl.MGDP.runners import Go2AmpPolicyRunner


def parse_args():
    custom_parameters = [
        {"name": "--mode", "type": str, "choices": ["amp", "no_amp"],
         "required": True},
        {"name": "--target_iteration", "type": int, "required": True},
        {"name": "--resume_from", "type": str, "default": None},
        {"name": "--amp_reward_coef", "type": float, "default": None},
        {"name": "--seed", "type": int, "default": 1},
        {"name": "--num_envs", "type": int, "default": 4096},
        {"name": "--rl_device", "type": str, "default": "cuda:0"},
        {"name": "--graphics_device_num", "type": int, "default": None},
    ]
    return gymutil.parse_arguments(
        description="Train the formal Go2 perception locomotion task",
        custom_parameters=custom_parameters,
    )


def _sim_device(args):
    device = args.sim_device_type
    if device == "cuda":
        device += ":{}".format(args.compute_device_id)
    return device


def _resolve_resume_manifest(resume_from, run_dir):
    requested = Path(resume_from).resolve()
    if requested.is_dir():
        source_run = requested
        pointer = source_run / "checkpoints" / "last.json"
    else:
        if requested.parent.name != "checkpoints":
            raise ValueError("resume manifest must be inside a checkpoints directory")
        source_run = requested.parent.parent
        pointer = requested
    if source_run.resolve() != run_dir.resolve():
        raise ValueError("resume_from does not match the derived run directory")
    if not pointer.is_file():
        raise FileNotFoundError("resume checkpoint pointer is missing: {}".format(pointer))
    if pointer.name == "last.json":
        data = json.loads(pointer.read_text(encoding="utf-8"))
        if set(data) != {"manifest"}:
            raise ValueError("last.json must contain only a manifest pointer")
        pointer = pointer.parent / data["manifest"]
    return pointer


def main():
    args = parse_args()
    coefficient = args.amp_reward_coef
    if args.mode == "amp" and coefficient is None:
        coefficient = 0.01
    if args.mode == "no_amp" and coefficient is not None:
        raise ValueError("--amp_reward_coef is valid only with --mode amp")
    if args.target_iteration <= 0:
        raise ValueError("--target_iteration must be positive")

    run_name = format_run_name(
        args.mode, coefficient, args.seed, args.num_envs)
    output_root = Path(LEGGED_GYM_ROOT_DIR) / "outputs" / "go2_amp"
    run_dir = prepare_run_directory(
        output_root, run_name, resume=args.resume_from is not None)
    resume_manifest = (
        _resolve_resume_manifest(args.resume_from, run_dir)
        if args.resume_from is not None else None)

    env_cfg, train_cfg = task_registry.get_cfgs("go2_amp")
    env_cfg.env.num_envs = args.num_envs
    env_cfg.seed = args.seed
    train_cfg.seed = args.seed
    train_cfg.runner.mode = args.mode
    train_cfg.amp.reward_coef = coefficient if args.mode == "amp" else 0.0
    args.sim_device = _sim_device(args)
    args.sim_device_id = args.compute_device_id
    if args.graphics_device_num is None:
        args.graphics_device_num = args.graphics_device_id

    env, env_cfg = task_registry.make_env(name="go2_amp", args=args,
                                           env_cfg=env_cfg)
    train_cfg_dict = class_to_dict(train_cfg)
    environment_config = class_to_dict(env_cfg)
    effective_config = {
        "environment": environment_config,
        "training": train_cfg_dict,
    }
    # The environment config participates in checkpoint compatibility too.
    runner_config = dict(train_cfg_dict)
    runner_config["_environment_config"] = environment_config
    experts = sorted({
        filename
        for filenames in runner_config["runner"]["amp_groups"].values()
        for filename in filenames
    })
    runner = Go2AmpPolicyRunner(
        env, runner_config, str(run_dir), device=args.rl_device)
    if resume_manifest is not None:
        runner.load_pair(resume_manifest)
    remaining = remaining_iterations(
        args.target_iteration, runner.current_learning_iteration_init)
    write_run_metadata(
        run_dir, effective_config, command=sys.argv, experts=experts)
    runner.learn(
        remaining, init_at_random_ep_len=resume_manifest is None)


if __name__ == "__main__":
    main()

