"""Deterministic, matched AMP/noAMP evaluation with CSV and PNG outputs."""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import NamedTuple
import zlib


os.environ["DOG_NAMES"] = "go2"

TERRAIN_NAMES = (
    "flat", "rough_flat", "smooth_slope", "rough_slope", "stairs",
    "discrete_obstacles", "hurdle", "gap", "pit", "narrow_passage",
)
LEVELS = (0, 9, 19)
FORWARD_SPEEDS = (0.4, 0.8)
TURN_RATES = (-0.5, 0.5)

METRIC_FIELDS = (
    "completion", "progress", "vx_rmse", "yaw_rmse",
    "base_contact_failure", "action_rate", "torque", "episode_length",
    "wm_depth_mse", "wm_height_mse", "amp_style_score",
)
CASE_FIELDS = (
    "policy", "case_id", "terrain", "level", "vx_command",
    "yaw_command", "episode", "seed",
) + METRIC_FIELDS
SUMMARY_FIELDS = ("policy", "episodes") + tuple(
    "mean_{}".format(name) for name in METRIC_FIELDS)


class ValidatedPair(NamedTuple):
    amp_manifest: Path
    no_amp_manifest: Path
    iteration: int


def _token(value):
    return str(value).replace("-", "m").replace(".", "p")


def build_cases():
    cases = []
    for terrain in TERRAIN_NAMES:
        for level in LEVELS:
            for vx in FORWARD_SPEEDS:
                cases.append({
                    "case_id": "{}_l{}_v{}".format(
                        terrain, level, _token(vx)),
                    "terrain": terrain, "level": level,
                    "vx": vx, "yaw": 0.0,
                })
    for terrain in ("flat", "rough_flat"):
        for level in LEVELS:
            for yaw in TURN_RATES:
                cases.append({
                    "case_id": "{}_l{}_yaw{}".format(
                        terrain, level, _token(yaw)),
                    "terrain": terrain, "level": level,
                    "vx": 0.0, "yaw": yaw,
                })
    return cases


def build_episode_schedule(episodes_per_case=20, base_seed=1):
    if int(episodes_per_case) <= 0:
        raise ValueError("episodes_per_case must be positive")
    schedule = []
    for case_index, case in enumerate(build_cases()):
        for episode in range(int(episodes_per_case)):
            schedule.append({
                "case_id": case["case_id"],
                "terrain": case["terrain"],
                "level": case["level"],
                "vx_command": case["vx"],
                "yaw_command": case["yaw"],
                "episode": episode,
                "seed": int(base_seed) + case_index * int(episodes_per_case)
                        + episode,
            })
    return schedule


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_manifest(path):
    path = Path(path).resolve()
    if path.is_dir():
        path = path / "checkpoints" / "last.json"
    if path.name == "last.json":
        pointer = json.loads(path.read_text(encoding="utf-8"))
        if set(pointer) != {"manifest"}:
            raise ValueError("last.json must contain only a manifest pointer")
        path = path.parent / pointer["manifest"]
    if not path.is_file():
        raise ValueError("checkpoint manifest is missing: {}".format(path))
    return path


def _validated_manifest(path, expected_mode):
    path = _resolve_manifest(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("mode") != expected_mode:
        raise ValueError("expected {} checkpoint".format(expected_mode))
    files = data.get("files", {})
    for label in ("policy", "world_model"):
        record = files.get(label, {})
        filename = record.get("name")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("invalid {} checkpoint filename".format(label))
        state_path = path.parent / filename
        if not state_path.is_file():
            raise ValueError("checkpoint file is missing: {}".format(filename))
        if _sha256(state_path) != record.get("sha256"):
            raise ValueError("checkpoint SHA-256 mismatch: {}".format(filename))
    return path, data


def _normalized_config(manifest_path, manifest):
    config_path = manifest_path.parent.parent / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint_config = dict(config["training"])
    checkpoint_config["_environment_config"] = config["environment"]
    encoded = json.dumps(
        checkpoint_config, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != manifest.get("config_hash"):
        raise ValueError("manifest config_hash does not match config.json")
    normalized = copy.deepcopy(config)
    normalized["training"]["runner"]["mode"] = "comparison"
    normalized["training"]["amp"]["reward_coef"] = "comparison"
    return config, normalized


def validate_pair_manifests(amp_manifest, no_amp_manifest):
    amp_path, amp_data = _validated_manifest(amp_manifest, "amp")
    no_path, no_data = _validated_manifest(no_amp_manifest, "no_amp")
    for field in ("iteration", "seed", "num_envs"):
        if amp_data.get(field) != no_data.get(field):
            raise ValueError("paired manifests have different {}".format(field))
    _, amp_normalized = _normalized_config(amp_path, amp_data)
    _, no_normalized = _normalized_config(no_path, no_data)
    if amp_normalized != no_normalized:
        raise ValueError("AMP/noAMP effective configs differ outside AMP")
    return ValidatedPair(
        amp_manifest=amp_path,
        no_amp_manifest=no_path,
        iteration=int(amp_data["iteration"]),
    )


def _add_policy(rows, policy):
    return [{"policy": policy, **row} for row in rows]


def evaluate_pair(validated_pair, output_dir, episodes_per_case=20,
                  base_seed=1, evaluator=None, runtime_options=None):
    schedule = build_episode_schedule(episodes_per_case, base_seed)
    if evaluator is None:
        evaluator = _IsaacEvaluator(runtime_options)
    amp_rows = evaluator(
        "amp", validated_pair.amp_manifest, schedule,
        validated_pair.amp_manifest)
    no_amp_rows = evaluator(
        "no_amp", validated_pair.no_amp_manifest, schedule,
        validated_pair.amp_manifest)
    rows = _add_policy(amp_rows, "amp") + _add_policy(no_amp_rows, "no_amp")
    write_outputs(rows, output_dir)
    return rows


def _means(rows):
    summaries = []
    for policy in ("amp", "no_amp"):
        selected = [row for row in rows if row["policy"] == policy]
        summary = {"policy": policy, "episodes": len(selected)}
        for metric in METRIC_FIELDS:
            summary["mean_{}".format(metric)] = (
                sum(float(row[metric]) for row in selected)
                / max(len(selected), 1))
        summaries.append(summary)
    return summaries


def _write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _png_chunk(kind, payload):
    return (struct.pack(">I", len(payload)) + kind + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xffffffff))


def _fallback_bar_png(path, values, title):
    width, height = 480, 320
    pixels = bytearray([245] * width * height * 3)
    colors = ((52, 120, 246), (239, 102, 72))
    maximum = max(max(values), 1e-12)
    for index, value in enumerate(values):
        left = 90 + index * 190
        right = left + 100
        top = height - 40 - int(230 * max(0.0, value) / maximum)
        for y in range(top, height - 40):
            for x in range(left, right):
                offset = (y * width + x) * 3
                pixels[offset:offset + 3] = bytes(colors[index])
    raw = b"".join(
        b"\x00" + bytes(pixels[y * width * 3:(y + 1) * width * 3])
        for y in range(height))
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2,
                                           0, 0, 0))
        + _png_chunk(b"tEXt", b"Title\x00" + title.encode("utf-8"))
        + _png_chunk(b"IDAT", zlib.compress(raw, 9))
        + _png_chunk(b"IEND", b""))
    Path(path).write_bytes(payload)


def _write_plots(summaries, output_dir):
    plots = (
        ("completion.png", ("mean_completion",), "Completion"),
        ("tracking.png", ("mean_vx_rmse", "mean_yaw_rmse"), "Tracking RMSE"),
        ("smoothness.png", ("mean_action_rate", "mean_torque"), "Smoothness"),
        ("style_score.png", ("mean_amp_style_score",), "AMP style score"),
    )
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        for filename, metrics, title in plots:
            values = [
                sum(float(summary[metric]) for metric in metrics)
                / len(metrics) for summary in summaries]
            _fallback_bar_png(Path(output_dir) / filename, values, title)
        return

    for filename, metrics, title in plots:
        figure, axis = plt.subplots(figsize=(6.4, 4.2))
        width = 0.8 / len(metrics)
        for metric_index, metric in enumerate(metrics):
            positions = [
                index - 0.4 + width / 2 + metric_index * width
                for index in range(2)]
            axis.bar(
                positions, [summary[metric] for summary in summaries],
                width=width, label=metric.replace("mean_", ""))
        axis.set_xticks((0, 1), ("AMP", "noAMP"))
        axis.set_title(title)
        if len(metrics) > 1:
            axis.legend()
        figure.tight_layout()
        figure.savefig(Path(output_dir) / filename, dpi=160)
        plt.close(figure)


def write_outputs(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = _means(rows)
    _write_csv(output_dir / "cases.csv", rows, CASE_FIELDS)
    _write_csv(output_dir / "summary.csv", summaries, SUMMARY_FIELDS)
    _write_plots(summaries, output_dir)


class _IsaacEvaluator:
    """Runtime evaluator; never performs optimizer or replay writes."""

    def __init__(self, runtime_options):
        self.runtime_options = runtime_options
        self.shared_amp = None

    def __call__(self, policy, manifest, schedule, shared_style_manifest):
        runner, env = self._create_runtime(policy, manifest, schedule)
        if policy == "amp":
            self.shared_amp = runner.amp
        if self.shared_amp is None:
            raise RuntimeError(
                "AMP policy must be evaluated first to provide shared style")
        return self._run_schedule(runner, env, schedule, self.shared_amp)

    def _create_runtime(self, policy, manifest, schedule):
        import torch
        from legged_gym.envs import task_registry
        from legged_gym.envs.baseline.randomization import (
            disable_evaluation_stochasticity,
        )
        from legged_gym.utils import set_seed
        from rl.MGDP.runners import Go2AmpPolicyRunner

        options = self.runtime_options
        run_dir = Path(manifest).parent.parent
        saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        training = saved["training"]
        checkpoint_config = dict(training)
        checkpoint_config["_environment_config"] = saved["environment"]
        num_episodes = max(int(item["episode"]) for item in schedule) + 1

        env_cfg, _ = task_registry.get_cfgs("go2_amp")
        env_cfg.seed = int(options.base_seed)
        env_cfg.env.num_envs = num_episodes
        env_cfg.terrain.curriculum = False
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

        set_seed(options.base_seed)
        env, _ = task_registry.make_env(
            name="go2_amp", args=options, env_cfg=env_cfg)
        runner_config = dict(training)
        runner_config["_evaluation"] = True
        runner_config["_checkpoint_key_config"] = checkpoint_config
        runner_config["_checkpoint_num_envs"] = int(
            saved["environment"]["env"]["num_envs"])
        runner = Go2AmpPolicyRunner(
            env, runner_config, str(run_dir), device=options.rl_device)
        runner.load_pair(manifest, restore_training_state=False)

        modules = [runner.alg.actor_critic, env.cnn_encoder, env.cnn_decoder]
        if runner.amp is not None:
            modules.append(runner.amp.discriminator)
        for module in modules:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        runner.alg.actor_critic.eval()
        return runner, env

    @staticmethod
    def _style_score(session, state, next_state):
        import torch
        with torch.inference_mode():
            pair = torch.cat((
                session.normalizer.normalize(state),
                session.normalizer.normalize(next_state),
            ), dim=-1)
            logits = session.discriminator(pair)
            return torch.clamp(
                1.0 - 0.25 * (logits - 1.0).square(), min=0.0)

    def _run_schedule(self, runner, env, schedule, shared_amp):
        import torch
        from legged_gym.envs.go2_amp.terrain import TERRAIN_COLUMNS
        from legged_gym.utils import set_seed
        from rl.MGDP.amp.transition import select_next_amp_state

        policy_fn, _ = runner.get_inference_policy(device=env.device)
        rows = []
        episodes_per_case = env.num_envs
        cases = [schedule[index:index + episodes_per_case]
                 for index in range(0, len(schedule), episodes_per_case)]
        column_for = {
            name: TERRAIN_COLUMNS.index(name) for name in TERRAIN_NAMES}

        for case_rows in cases:
            first = case_rows[0]
            column = column_for[first["terrain"]]
            env.terrain_types.fill_(column)
            env.terrain_levels.fill_(int(first["level"]))
            env.env_origins[:] = env.terrain_origins[
                env.terrain_levels, env.terrain_types]
            env.env_class[:] = env.terrain_class[
                env.terrain_levels, env.terrain_types]
            for key, value in (
                    ("lin_vel_x", first["vx_command"]),
                    ("lin_vel_y", 0.0),
                    ("ang_vel_yaw", first["yaw_command"])):
                env.command_ranges[key] = [value, value]

            for env_index, item in enumerate(case_rows):
                set_seed(item["seed"])
                env.reset_idx(torch.tensor(
                    [env_index], device=env.device, dtype=torch.long))
            env.compute_observations()
            env.refresh_visual_observation()
            obs = env.get_observations()
            runner.alg.actor_critic.reset(torch.ones(
                env.num_envs, dtype=torch.bool, device=env.device))
            active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
            done_timeout = torch.zeros_like(active)
            final_progress = torch.zeros(env.num_envs, device=env.device)
            step_count = torch.zeros(env.num_envs, device=env.device)
            vx_sq = torch.zeros(env.num_envs, device=env.device)
            yaw_sq = torch.zeros(env.num_envs, device=env.device)
            action_rate = torch.zeros(env.num_envs, device=env.device)
            torque = torch.zeros(env.num_envs, device=env.device)
            wm_depth = torch.zeros(env.num_envs, device=env.device)
            wm_height = torch.zeros(env.num_envs, device=env.device)
            wm_refresh_count = torch.zeros(
                env.num_envs, device=env.device)
            style = torch.zeros(env.num_envs, device=env.device)
            initial_xy = env.root_states[:, :2].clone()

            for _ in range(int(env.max_episode_length) + 2):
                if not torch.any(active):
                    break
                amp_before = env.get_amp_observations().clone()
                previous_actions = env.last_actions.clone()
                with torch.inference_mode():
                    actions = policy_fn(obs)
                    obs, _, dones, infos = env.step(actions.detach())
                    runner.alg.actor_critic.reset(dones)
                    amp_after = select_next_amp_state(
                        env.get_amp_observations().clone(), dones,
                        infos.get("terminal_amp_states"))
                    style_step = self._style_score(
                        shared_amp, amp_before, amp_after)

                vx_now = env.base_lin_vel[:, 0].clone()
                yaw_now = env.base_ang_vel[:, 2].clone()
                position_now = env.root_states[:, :2].clone()
                terminal = infos.get("terminal_diagnostics")
                if terminal is not None:
                    ids = terminal["env_ids"].long()
                    vx_now[ids] = terminal["base_lin_vel"][:, 0]
                    yaw_now[ids] = terminal["base_ang_vel"][:, 2]
                    position_now[ids] = terminal["root_positions"][:, :2]
                    done_timeout[ids] = terminal["time_outs"].bool()

                mask = active.float()
                step_count += mask
                vx_sq += mask * (vx_now - first["vx_command"]).square()
                yaw_sq += mask * (yaw_now - first["yaw_command"]).square()
                action_rate += mask * (
                    actions - previous_actions).square().mean(dim=1)
                torque += mask * env.torques.abs().mean(dim=1)
                wm_mask = active & env.wm_metric_mask
                wm_float_mask = wm_mask.float()
                wm_depth += wm_float_mask * env.visual_mse_per_env
                wm_height += wm_float_mask * env.height_mse_per_env
                wm_refresh_count += wm_float_mask
                style += mask * style_step
                progress_now = position_now[:, 0] - initial_xy[:, 0]
                final_progress[active] = progress_now[active]
                active &= ~dones.bool()

            count = torch.clamp(step_count, min=1.0)
            wm_count = torch.clamp(wm_refresh_count, min=1.0)
            forward_case = first["vx_command"] > 0
            completion = (
                final_progress >= env.terrain.env_length / 2
                if forward_case else done_timeout)
            for index, item in enumerate(case_rows):
                rows.append({
                    **item,
                    "completion": float(completion[index]),
                    "progress": float(final_progress[index]),
                    "vx_rmse": math.sqrt(float(vx_sq[index] / count[index])),
                    "yaw_rmse": math.sqrt(float(yaw_sq[index] / count[index])),
                    "base_contact_failure": float(not done_timeout[index]),
                    "action_rate": float(action_rate[index] / count[index]),
                    "torque": float(torque[index] / count[index]),
                    "episode_length": float(step_count[index] * env.dt),
                    "wm_depth_mse": float(wm_depth[index] / wm_count[index]),
                    "wm_height_mse": float(wm_height[index] / wm_count[index]),
                    "amp_style_score": float(style[index] / count[index]),
                })
        return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare matched Go2 AMP and noAMP checkpoints")
    parser.add_argument("--amp_manifest", required=True)
    parser.add_argument("--no_amp_manifest", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--episodes_per_case", type=int, default=20)
    parser.add_argument("--base_seed", type=int, default=1)
    parser.add_argument("--rl_device", default="cuda:0")
    parser.add_argument("--sim_device", default="cuda:0")
    parser.add_argument("--graphics_device_num", type=int, default=0)
    parser.add_argument("--physics_engine", default="physx")
    parser.add_argument("--num_threads", type=int, default=0)
    parser.add_argument("--subscenes", type=int, default=0)
    parser.add_argument("--slices", type=int, default=0)
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--use_gpu_pipeline", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    # Import Isaac Gym only after both manifests and hashes have been checked.
    validated = validate_pair_manifests(
        args.amp_manifest, args.no_amp_manifest)
    output = (
        Path(args.output_dir) if args.output_dir is not None
        else validated.amp_manifest.parent.parent / "eval" / "comparison")

    import isaacgym
    from isaacgym import gymapi
    args.headless = True
    args.num_envs = args.episodes_per_case
    args.seed = args.base_seed
    args.compute_device_id = int(args.sim_device.split(":")[-1])
    args.sim_device_type = "cuda" if args.sim_device.startswith("cuda") else "cpu"
    args.sim_device_id = args.compute_device_id
    args.graphics_device_id = args.graphics_device_num
    args.device = args.sim_device_type
    args.physics_engine = (
        gymapi.SIM_PHYSX if args.physics_engine == "physx" else gymapi.SIM_FLEX)
    evaluate_pair(
        validated, output, args.episodes_per_case, args.base_seed,
        runtime_options=args)


if __name__ == "__main__":
    main()
