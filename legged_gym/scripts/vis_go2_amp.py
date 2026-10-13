"""Interactive Viewer and optional recorder for one paired Go2 AMP checkpoint."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


os.environ["DOG_NAMES"] = "go2"

import isaacgym
from isaacgym import gymutil
import numpy as np
import torch

from legged_gym.envs import *  # noqa: F401,F403 - registers go2_amp
from legged_gym.envs.go2_amp.terrain import TERRAIN_NAMES
from legged_gym.utils import task_registry
from rl.MGDP.amp.transition import select_next_amp_state
from rl.MGDP.runners import Go2AmpPolicyRunner


def parse_args():
    custom_parameters = [
        {"name": "--run_dir", "type": str, "required": True},
        {"name": "--manifest", "type": str, "default": "last.json"},
        {"name": "--terrain", "type": str, "choices": list(TERRAIN_NAMES),
         "default": "flat"},
        {"name": "--level", "type": int, "default": 0},
        {"name": "--vx", "type": float, "default": 0.4},
        {"name": "--yaw", "type": float, "default": 0.0},
        {"name": "--env_id", "type": int, "default": 0},
        {"name": "--num_envs", "type": int, "default": 4},
        {"name": "--follow_camera", "action": "store_true"},
        {"name": "--seed", "type": int, "default": 1},
        {"name": "--rl_device", "type": str, "default": "cuda:0"},
        {"name": "--graphics_device_num", "type": int, "default": None},
        {"name": "--record", "action": "store_true"},
        {"name": "--record_output", "type": str, "default": None},
        {"name": "--duration", "type": float, "default": 10.0},
        {"name": "--fps", "type": int, "default": 30},
        {"name": "--numbered_snapshots", "action": "store_true"},
        {"name": "--snapshot_interval", "type": int, "default": 100},
    ]
    return gymutil.parse_arguments(
        description="Visualize a paired Go2 AMP checkpoint",
        custom_parameters=custom_parameters,
    )


def ensure_within_run(run_dir, path):
    run_dir = Path(run_dir).resolve()
    path = Path(path).resolve()
    if os.path.commonpath((str(run_dir), str(path))) != str(run_dir):
        raise ValueError("output must remain beneath the selected run directory")
    return path


def _resolve_manifest(run_dir, requested):
    checkpoint_dir = Path(run_dir) / "checkpoints"
    path = Path(requested)
    if not path.is_absolute():
        path = checkpoint_dir / path
    path = ensure_within_run(run_dir, path)
    if path.name == "last.json":
        pointer = json.loads(path.read_text(encoding="utf-8"))
        if set(pointer) != {"manifest"}:
            raise ValueError("last.json must contain only a manifest pointer")
        path = checkpoint_dir / pointer["manifest"]
    if not path.is_file():
        raise FileNotFoundError("checkpoint manifest is missing: {}".format(path))
    return path


def _sim_device(args):
    device = args.sim_device_type
    if device == "cuda":
        device += ":{}".format(args.compute_device_id)
    return device


def _disable_randomization(env_cfg):
    env_cfg.noise.add_noise = False
    env_cfg.camera.update_wm = False
    env_cfg.domain_rand.push_robots = False
    for name in (
            "randomize_friction", "randomize_action_latency",
            "randomize_base_mass", "randomize_limb_mass", "randomize_com",
            "randomize_motor_strength", "randomize_lag_timesteps",
            "randomize_motor_offset", "randomize_gains"):
        if hasattr(env_cfg.domain_rand, name):
            setattr(env_cfg.domain_rand, name, False)


def _prepare_recording(args, run_dir):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("--record requires FFmpeg on PATH")
    video_dir = Path(run_dir) / "eval" / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    output = (
        Path(args.record_output) if args.record_output is not None
        else video_dir / "{}_level_{}.mp4".format(args.terrain, args.level))
    if not output.is_absolute():
        output = video_dir / output
    output = ensure_within_run(run_dir, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = tempfile.TemporaryDirectory(prefix=".frames-", dir=str(video_dir))
    return ffmpeg, output, frames


def _encode_video(ffmpeg, frame_dir, output, fps):
    subprocess.run([
        ffmpeg, "-y", "-framerate", str(fps),
        "-i", str(Path(frame_dir) / "%06d.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output),
    ], check=True)


def _freeze_networks(runner, env):
    modules = [runner.alg.actor_critic, env.cnn_encoder, env.cnn_decoder]
    if getattr(env, "use_map_decoder", False):
        modules.extend((env.map_encoder, env.map_decoder))
    if getattr(env, "use_memory", False):
        modules.append(env.memory_lstm)
    if runner.amp is not None:
        modules.append(runner.amp.discriminator)
    for module in modules:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    runner.alg.actor_critic.eval()


def _print_episode(infos, env_id, terrain, level, amp_reward):
    packet = infos.get("terminal_diagnostics")
    if packet is None:
        return
    rows = (packet["env_ids"] == env_id).nonzero(as_tuple=False).flatten()
    if rows.numel() == 0:
        return
    row = int(rows[0])
    failure = (
        "timeout" if bool(packet["time_outs"][row]) else "base_contact")
    command = packet["commands"][row, (0, 2)].detach().cpu().tolist()
    actual = [packet["base_lin_vel"][row, 0].item(),
              packet["base_ang_vel"][row, 2].item()]
    print(
        "episode command(vx,yaw)={} actual velocity(vx,yaw)={} "
        "terrain={} level={} AMP reward={:.6f} failure={}".format(
            command, actual, terrain, level, amp_reward, failure))


def main():
    args = parse_args()
    if not 0 <= args.level <= 19:
        raise ValueError("--level must be in [0, 19]")
    if not 0 <= args.env_id < args.num_envs:
        raise ValueError("--env_id must select one of --num_envs")
    if args.terrain not in ("flat", "rough_flat") and args.yaw != 0.0:
        raise ValueError("structured terrain visualization requires yaw=0")
    if args.fps <= 0 or args.duration <= 0:
        raise ValueError("--fps and --duration must be positive")
    if args.snapshot_interval <= 0:
        raise ValueError("--snapshot_interval must be positive")

    run_dir = Path(args.run_dir).resolve()
    manifest = _resolve_manifest(run_dir, args.manifest)
    saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    training_config = saved["training"]
    checkpoint_key_config = dict(training_config)
    checkpoint_key_config["_environment_config"] = saved["environment"]
    checkpoint_num_envs = int(saved["environment"]["env"]["num_envs"])

    env_cfg, _ = task_registry.get_cfgs("go2_amp")
    env_cfg.seed = args.seed
    env_cfg.env.num_envs = args.num_envs
    env_cfg.viewer.ref_env = args.env_id
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.num_cols = 1
    env_cfg.terrain.explicit_terrain_columns = (args.terrain,)
    env_cfg.commands.ranges.lin_vel_x = [args.vx, args.vx]
    env_cfg.commands.ranges.new_lin_vel_x = [args.vx, args.vx]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.new_lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [args.yaw, args.yaw]
    env_cfg.commands.ranges.new_ang_vel_yaw = [args.yaw, args.yaw]
    _disable_randomization(env_cfg)

    checkpoint_label = manifest.stem
    wm_dir = (run_dir / "eval" / "wm" / checkpoint_label
              / "{}_level_{}".format(args.terrain, args.level))
    wm_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.camera.render_compare_pre_vis = False
    env_cfg.camera.render_compare_pre_map = False
    env_cfg.camera.compare_vis_save_dir = str(wm_dir)
    env_cfg.camera.compare_height_vis_save_dir = str(wm_dir)
    env_cfg.camera.compare_vis_filename = "depth_latest.png"
    env_cfg.camera.compare_height_vis_filename = "height_latest.png"

    args.sim_device = _sim_device(args)
    args.sim_device_id = args.compute_device_id
    if args.graphics_device_num is None:
        args.graphics_device_num = args.graphics_device_id
    env, _ = task_registry.make_env(
        name="go2_amp", args=args, env_cfg=env_cfg)

    runner_config = dict(training_config)
    runner_config["_evaluation"] = True
    runner_config["_checkpoint_key_config"] = checkpoint_key_config
    runner_config["_checkpoint_num_envs"] = checkpoint_num_envs
    runner = Go2AmpPolicyRunner(
        env, runner_config, str(run_dir), device=args.rl_device)
    runner.load_pair(manifest, restore_training_state=False)
    _freeze_networks(runner, env)
    env.cfg.camera.render_compare_pre_vis = True
    env.cfg.camera.render_compare_pre_map = True

    env_ids = torch.arange(env.num_envs, device=env.device)
    env.terrain_levels.fill_(args.level)
    env.env_origins[:] = env.terrain_origins[
        env.terrain_levels, env.terrain_types]
    env.env_class[:] = env.terrain_class[
        env.terrain_levels, env.terrain_types]
    env.reset_idx(env_ids)
    obs = env.get_observations()
    policy, _ = runner.get_inference_policy(device=env.device)

    camera_offset = (
        np.asarray(env_cfg.viewer.pos) - np.asarray(env_cfg.viewer.lookat))
    look_ahead = np.asarray([1.5, 0.0, 0.0])
    recording = None
    if args.record:
        if args.headless:
            raise ValueError("--record requires the Isaac Gym Viewer")
        recording = _prepare_recording(args, run_dir)
    frame_index = 0
    frame_stride = max(1, round(1.0 / (args.fps * env.dt)))
    maximum_steps = (
        max(1, round(args.duration / env.dt)) if args.record else None)

    step = 0
    try:
        while maximum_steps is None or step < maximum_steps:
            amp_before = (
                env.get_amp_observations().clone()
                if runner.amp is not None else None)
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, infos = env.step(actions.detach())
                if runner.amp is not None:
                    amp_after = select_next_amp_state(
                        env.get_amp_observations().clone(), dones,
                        infos.get("terminal_amp_states"))
                    amp_reward = runner.amp.reward(
                        amp_before, amp_after, infos["amp_transition_mask"],
                        enabled=True)
                else:
                    amp_reward = torch.zeros_like(dones, dtype=torch.float)

            if args.follow_camera:
                position = env.root_states[args.env_id, :3].detach().cpu().numpy()
                target = position + look_ahead
                env.set_camera(target + camera_offset, target)
            if bool(dones[args.env_id]):
                _print_episode(
                    infos, args.env_id, args.terrain, args.level,
                    float(amp_reward[args.env_id]))

            if args.numbered_snapshots and step % args.snapshot_interval == 0:
                for stem in ("depth", "height"):
                    latest = wm_dir / "{}_latest.png".format(stem)
                    if latest.is_file():
                        shutil.copy2(
                            latest,
                            wm_dir / "{}_{:06d}.png".format(stem, step))
            if recording is not None and step % frame_stride == 0:
                _, _, frame_directory = recording
                filename = Path(frame_directory.name) / "{:06d}.png".format(
                    frame_index)
                env.gym.write_viewer_image_to_file(
                    env.viewer, str(filename))
                frame_index += 1
            step += 1
    finally:
        if recording is not None:
            ffmpeg, output, frame_directory = recording
            try:
                _encode_video(ffmpeg, frame_directory.name, output, args.fps)
                print("video written to {}".format(output))
            finally:
                frame_directory.cleanup()


if __name__ == "__main__":
    main()
