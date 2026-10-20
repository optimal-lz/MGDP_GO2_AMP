"""Open a compact, checkpoint-free Viewer of the frozen Go2 AMP terrains."""

import os


os.environ["DOG_NAMES"] = "go2"

from isaacgym import gymutil
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403 - registers go2_amp
from legged_gym.envs.baseline.randomization import (
    disable_evaluation_stochasticity,
)
from legged_gym.envs.go2_amp.terrain import (
    TERRAIN_NAMES,
    TERRAIN_OVERVIEW_LEVELS,
    TERRAIN_PROPORTIONS,
    assert_terrain_overview,
    configure_evaluation_terrain_grid,
    freeze_terrain_overview,
    terrain_overview_layout,
)
from legged_gym.utils import task_registry


def parse_args():
    custom_parameters = [
        {"name": "--seed", "type": int, "default": 1},
        {"name": "--rl_device", "type": str, "default": "cuda:0"},
        {"name": "--graphics_device_num", "type": int, "default": None},
        {"name": "--headless", "action": "store_true", "default": False},
    ]
    return gymutil.parse_arguments(
        description="Inspect the frozen Go2 AMP terrain set",
        custom_parameters=custom_parameters,
    )


def _sim_device(args):
    device = args.sim_device_type
    if device == "cuda":
        device += ":{}".format(args.compute_device_id)
    return device


def _disable_randomization(env_cfg):
    disable_evaluation_stochasticity(env_cfg)
    env_cfg.noise.add_noise = False
    env_cfg.camera.update_wm = False
    env_cfg.camera.render_compare_pre_vis = False
    env_cfg.camera.render_compare_pre_map = False
    env_cfg.domain_rand.push_robots = False
    for name in (
            "randomize_friction", "randomize_action_latency",
            "randomize_base_mass", "randomize_link_mass",
            "randomize_restitution", "randomize_com",
            "randomize_motor_strength", "randomize_lag_timesteps",
            "randomize_motor_offset", "randomize_gains"):
        if hasattr(env_cfg.domain_rand, name):
            setattr(env_cfg.domain_rand, name, False)


def _print_layout():
    print("\nMGDP GO2 AMP terrain overview")
    print("  world X / display columns: level {}".format(
        " | ".join(str(level) for level in TERRAIN_OVERVIEW_LEVELS)))
    print("  world Y / display rows: formal terrain types")
    print("  display count is equal per type; percentages are training weights")
    for row, (name, proportion) in enumerate(
            zip(TERRAIN_NAMES, TERRAIN_PROPORTIONS)):
        env_ids = tuple(
            row * len(TERRAIN_OVERVIEW_LEVELS) + offset
            for offset in range(len(TERRAIN_OVERVIEW_LEVELS)))
        print("  row {:02d}: {:20s} {:>3.0f}%  envs={}".format(
            row, name, 100.0 * proportion, env_ids))
    print()


def main():
    args = parse_args()
    if args.headless:
        raise ValueError("terrain overview requires the Isaac Gym Viewer")

    layout = terrain_overview_layout()
    env_cfg, _ = task_registry.get_cfgs("go2_amp")
    env_cfg.seed = args.seed
    env_cfg.env.num_envs = len(layout)
    env_cfg.viewer.ref_env = 0
    env_cfg.asset.fix_base_link = True
    configure_evaluation_terrain_grid(
        env_cfg.terrain, TERRAIN_NAMES, levels=TERRAIN_OVERVIEW_LEVELS)
    _disable_randomization(env_cfg)

    args.sim_device = _sim_device(args)
    args.sim_device_id = args.compute_device_id
    args.num_envs = None
    if args.graphics_device_num is None:
        args.graphics_device_num = args.graphics_device_id
    env, _ = task_registry.make_env(
        name="go2_amp", args=args, env_cfg=env_cfg)

    freeze_terrain_overview(env, layout)
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(env_ids)
    assert_terrain_overview(env, layout)
    _print_layout()

    origins = env.env_origins.detach().cpu().numpy()
    target = origins.mean(axis=0)
    span_x = float(np.ptp(origins[:, 0]) + env.terrain.env_length)
    span_y = float(np.ptp(origins[:, 1]) + env.terrain.env_width)
    camera = target + np.asarray([
        -0.65 * span_x, -0.65 * span_y, 0.85 * max(span_x, span_y)])
    env.set_camera(camera, target)

    zero_actions = torch.zeros(
        env.num_envs, env.num_actions, device=env.device)
    while True:
        with torch.inference_mode():
            _, _, dones, _ = env.step(zero_actions)
        reset_ids = dones.nonzero(as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            assert_terrain_overview(env, layout)


if __name__ == "__main__":
    main()
