"""Single-stage runner for the formal Go2 perception locomotion task."""

from collections import deque
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from rl.MGDP.amp.motions import MotionDataset
from rl.MGDP.amp.session import AMPSession
from rl.MGDP.amp.transition import select_next_amp_state
from legged_gym.envs.go2_amp.terrain import TERRAIN_NAMES

from .go2_amp_checkpoint import (
    CheckpointCompatibility,
    PairedCheckpointManager,
    prepare_torch_rng_restore,
    restore_ppo_learning_rate,
    restore_terrain_state,
    validate_terrain_state,
)
from .go2_amp_logging import IterationMetrics, format_iteration_console
from .policy_runner import MGDPPolicyRunner


class Go2AmpPolicyRunner(MGDPPolicyRunner):
    """Coordinate WM, PPO, and one optional AMP discriminator."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        # Reuse MGDP model/storage construction without creating its legacy
        # stage1_nn/stage1_tb output tree.
        super().__init__(env, train_cfg, None, device)
        self.log_dir = log_dir
        self.nn_dir = None
        self.tb_dir = (
            str(Path(log_dir) / "tensorboard") if log_dir is not None else None)
        self.writer = None
        self.train_cfg = train_cfg
        self.evaluation_mode = bool(train_cfg.get("_evaluation", False))
        checkpoint_num_envs = int(
            train_cfg.get("_checkpoint_num_envs", self.env.num_envs))
        checkpoint_key_config = train_cfg.get(
            "_checkpoint_key_config", train_cfg)
        if self.tb_dir is not None and not self.evaluation_mode:
            Path(self.tb_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.tb_dir, flush_secs=10)
        self.mode = self.cfg.get("mode", "amp")
        if self.mode not in ("amp", "no_amp"):
            raise ValueError("runner mode must be 'amp' or 'no_amp'")

        self.amp = None
        if self.mode == 'amp':
            amp_seed = int(train_cfg.get("seed", 1)) + 1000
            motion_dir = Path(__file__).resolve().parents[4] / "datasets" / "go2_motion"
            dataset = MotionDataset(
                motion_dir,
                self.cfg["amp_groups"],
                self.cfg["amp_group_weights"],
                self.env.dt,
                rng=np.random.default_rng(amp_seed + 2),
            )
            amp_cfg = dict(self.cfg)
            amp_cfg.update({
                "amp_reward_coef": train_cfg["amp"]["reward_coef"],
                "num_envs": checkpoint_num_envs,
                "num_steps_per_env": self.num_steps_per_env,
            })
            self.amp = AMPSession(dataset, self.device, amp_cfg, amp_seed)

        reward_coef = (
            train_cfg["amp"]["reward_coef"] if self.mode == "amp" else 0.0)
        self.checkpoint_compatibility = CheckpointCompatibility.from_config(
            mode=self.mode,
            amp_reward_coef=reward_coef,
            seed=train_cfg.get("seed", 1),
            num_envs=checkpoint_num_envs,
            key_config=checkpoint_key_config,
        )
        self.checkpoints = (
            PairedCheckpointManager(
                Path(log_dir) / "checkpoints", map_location=self.device)
            if log_dir is not None else None)
        self.current_learning_iteration = self.current_learning_iteration_init
        self.iteration_metrics = IterationMetrics(
            self.writer, self.mode, self.current_learning_iteration_init - 1,
            terrain_names=TERRAIN_NAMES)

    @staticmethod
    def _mean(total, count):
        return total / max(count, 1)

    def _new_rollout_metrics(self):
        scalar_names = (
            "task_sum", "regularization_sum", "amp_sum", "total_sum",
            "vx_error_sum", "yaw_error_sum", "episode_length_sum",
            "episode_count", "base_failure_count", "valid_amp_count",
        )
        metrics = {
            name: torch.zeros((), device=self.device) for name in scalar_names
        }
        terrain_count = len(TERRAIN_NAMES)
        for name in (
                "terrain_transition_count", "terrain_level_sum",
                "terrain_stuck_sum", "terrain_episode_count",
                "terrain_completion_sum", "terrain_base_failure_sum"):
            metrics[name] = torch.zeros(terrain_count, device=self.device)
        metrics["transition_count"] = 0
        return metrics

    def _accumulate_rollout_metrics(self, metrics, commands, terrain_classes,
                                    terrain_levels, dones, infos,
                                    amp_transition_mask, amp_contribution,
                                    rewards):
        metrics["task_sum"] += self.env._task_reward.detach().sum()
        metrics["regularization_sum"] += (
            self.env._regularization_reward.detach().sum())
        metrics["amp_sum"] += amp_contribution.detach().sum()
        metrics["total_sum"] += rewards.detach().sum()
        metrics["valid_amp_count"] += amp_transition_mask.detach().sum()
        metrics["transition_count"] += rewards.numel()

        vx = self.env.base_lin_vel[:, 0].detach().clone()
        yaw = self.env.base_ang_vel[:, 2].detach().clone()
        terminal = infos.get("terminal_diagnostics")
        if terminal is not None:
            terminal_ids = terminal["env_ids"].long()
            vx[terminal_ids] = terminal["base_lin_vel"][:, 0]
            yaw[terminal_ids] = terminal["base_ang_vel"][:, 2]
            episode_count = terminal_ids.numel()
            metrics["episode_count"] += episode_count
            metrics["episode_length_sum"] += (
                terminal["episode_lengths"].float().sum() * self.env.dt)
            failures = ~terminal["time_outs"].bool()
            metrics["base_failure_count"] += failures.float().sum()
            terminal_classes = terminal["terrain_classes"].long()
            ones = torch.ones(
                episode_count, device=self.device, dtype=torch.float)
            metrics["terrain_episode_count"].scatter_add_(
                0, terminal_classes, ones)
            metrics["terrain_completion_sum"].scatter_add_(
                0, terminal_classes, terminal["time_outs"].float())
            metrics["terrain_base_failure_sum"].scatter_add_(
                0, terminal_classes, failures.float())

        metrics["vx_error_sum"] += torch.abs(
            vx - commands[:, 0]).sum()
        metrics["yaw_error_sum"] += torch.abs(
            yaw - commands[:, 1]).sum()
        ones = torch.ones_like(terrain_levels, dtype=torch.float)
        metrics["terrain_transition_count"].scatter_add_(
            0, terrain_classes, ones)
        metrics["terrain_level_sum"].scatter_add_(
            0, terrain_classes, terrain_levels.float())
        stuck = ((commands[:, 0] > self.env.cfg.commands.vx_deadband)
                 & (vx < 0.1)).float()
        metrics["terrain_stuck_sum"].scatter_add_(
            0, terrain_classes, stuck)

    @staticmethod
    def _tensor_mean(total, count):
        return total / torch.clamp(count, min=1.0)

    def _terrain_detail_metrics(self, metrics):
        detail = {}
        level = self._tensor_mean(
            metrics["terrain_level_sum"],
            metrics["terrain_transition_count"])
        stuck = self._tensor_mean(
            metrics["terrain_stuck_sum"],
            metrics["terrain_transition_count"])
        completion = self._tensor_mean(
            metrics["terrain_completion_sum"],
            metrics["terrain_episode_count"])
        failure = self._tensor_mean(
            metrics["terrain_base_failure_sum"],
            metrics["terrain_episode_count"])
        for index, name in enumerate(TERRAIN_NAMES):
            prefix = "terrain/{}/".format(name)
            detail[prefix + "mean_level"] = level[index]
            detail[prefix + "completion"] = completion[index]
            detail[prefix + "base_contact_failure"] = failure[index]
            detail[prefix + "stuck"] = stuck[index]
        return detail

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        obs_dict = self.env.get_observations()
        self.alg.actor_critic.train()
        start_iteration = self.current_learning_iteration_init
        end_iteration = start_iteration + num_learning_iterations
        reward_history = deque(maxlen=100)
        session_time = 0.0

        for iteration in range(start_iteration, end_iteration):
            iteration_start = time.time()
            collection_start = time.time()
            rollout_metrics = self._new_rollout_metrics()
            if getattr(self.env, "use_world_model", False):
                self.env.reset_wm_iteration_metrics()

            for _ in range(self.num_steps_per_env):
                commands_before = self.env.commands[:, (0, 2)].detach().clone()
                terrain_classes_before = self.env.env_class.long().clone()
                terrain_levels_before = self.env.terrain_levels.clone()
                if self.amp is not None:
                    amp_before = self.env.get_amp_observations().clone()
                with torch.inference_mode():
                    actions = self.alg.act(obs_dict)

                # env.step performs the online WM update and therefore must keep
                # autograd enabled even though policy rollout inference does not.
                obs_dict, _, dones, infos = self.env.step(actions)
                dones = dones.to(self.device)

                amp_contribution = torch.zeros_like(self.env._base_reward)
                amp_transition_mask = infos["amp_transition_mask"]
                if self.amp is not None:
                    amp_after = select_next_amp_state(
                        self.env.get_amp_observations().clone(),
                        dones,
                        infos.get("terminal_amp_states"),
                    )
                    amp_contribution = self.amp.reward(
                        amp_before,
                        amp_after,
                        amp_transition_mask,
                        enabled=iteration > 0,
                    )
                    self.amp.record(
                        amp_before, amp_after, amp_transition_mask)

                rewards = self.env.compose_total_reward(amp_contribution)
                rewards = rewards.to(self.device)
                self.alg.process_env_step(
                    rewards, dones, next_obs=obs_dict["obs"], infos=infos)

                self._accumulate_rollout_metrics(
                    rollout_metrics, commands_before,
                    terrain_classes_before, terrain_levels_before,
                    dones, infos, amp_transition_mask,
                    amp_contribution, rewards)
                reward_history.append(rewards.detach().mean())

            collection_time = time.time() - collection_start

            # Keep the frozen formal order: online WM rollout, four extra WM
            # updates, GAE, PPO, then twenty discriminator updates.
            wm_rollout_steps = int(
                getattr(self.env.cfg.camera, "wm_rollout_steps", 0))
            wm_start = time.time()
            if getattr(self.env, "use_world_model", False):
                for _ in range(wm_rollout_steps):
                    self.env.update_world_model(
                        batch=getattr(
                            self.env, "latest_world_model_batch", None),
                        num_updates=1,
                    )
            wm_time = time.time() - wm_start

            ppo_start = time.time()
            with torch.inference_mode():
                self.alg.compute_returns(obs_dict)
            self.alg.update()
            ppo_time = time.time() - ppo_start
            discriminator_start = time.time()
            amp_update_metrics = (
                self.amp.update() if self.amp is not None else {})
            discriminator_time = time.time() - discriminator_start
            iteration_time = time.time() - iteration_start

            transition_count = rollout_metrics["transition_count"]
            episode_count = rollout_metrics["episode_count"]
            wm_metrics = (
                self.env.get_wm_iteration_metrics()
                if getattr(self.env, "use_world_model", False)
                else {
                    "total_loss": 0.0, "depth_loss": 0.0,
                    "height_loss": 0.0, "contrastive_loss": 0.0,
                })

            core = {
                "reward/task": self._mean(
                    rollout_metrics["task_sum"], transition_count),
                "reward/regularization": self._mean(
                    rollout_metrics["regularization_sum"], transition_count),
                "reward/amp_contribution": self._mean(
                    rollout_metrics["amp_sum"], transition_count),
                "reward/total": self._mean(
                    rollout_metrics["total_sum"], transition_count),
                "tracking/vx_error": self._mean(
                    rollout_metrics["vx_error_sum"], transition_count),
                "tracking/yaw_error": self._mean(
                    rollout_metrics["yaw_error_sum"], transition_count),
                "episode/length": self._tensor_mean(
                    rollout_metrics["episode_length_sum"], episode_count),
                "episode/base_contact_failure": self._tensor_mean(
                    rollout_metrics["base_failure_count"], episode_count),
                "terrain/mean_level": self._tensor_mean(
                    rollout_metrics["terrain_level_sum"].sum(),
                    rollout_metrics["terrain_transition_count"].sum()),
                "wm/total_loss": wm_metrics["total_loss"],
                "wm/depth_loss": wm_metrics["depth_loss"],
                "wm/height_loss": wm_metrics["height_loss"],
                "wm/contrastive_loss": wm_metrics["contrastive_loss"],
                "ppo/surrogate_loss": self.alg.last_update_metrics[
                    "ppo/surrogate_loss"],
                "ppo/value_loss": self.alg.last_update_metrics[
                    "ppo/value_loss"],
                "ppo/entropy": self.alg.last_update_metrics["ppo/entropy"],
                "ppo/kl": self.alg.last_update_metrics["ppo/kl"],
                "policy/action_std": self.alg.actor_critic.std.mean(),
                "performance/iteration_seconds": iteration_time,
            }
            if self.amp is not None:
                core.update({
                    "amp/policy_logit": amp_update_metrics["policy_logit"],
                    "amp/expert_logit": amp_update_metrics["expert_logit"],
                    "amp/discriminator_loss": amp_update_metrics[
                        "discriminator_loss"],
                    "amp/gradient_penalty": amp_update_metrics[
                        "gradient_penalty"],
                    "amp/valid_transition_count": rollout_metrics[
                        "valid_amp_count"],
                    "amp/replay_fill": (
                        self.amp.replay.size / self.amp.replay.capacity),
                })
            detail = self._terrain_detail_metrics(rollout_metrics)
            detail.update({
                "timing/simulation_seconds": collection_time,
                "timing/wm_seconds": wm_time,
                "timing/ppo_seconds": ppo_time,
            })
            if self.amp is not None:
                detail.update({
                    "timing/discriminator_seconds": discriminator_time,
                    "amp/expert_forward_logit": amp_update_metrics[
                        "expert_forward_logit"],
                    "amp/expert_turn_logit": amp_update_metrics[
                        "expert_turn_logit"],
                })
            scalars = self.iteration_metrics.write(iteration, core, detail)
            completed_iteration = iteration + 1
            self.current_learning_iteration = completed_iteration
            iteration_timesteps = self.num_steps_per_env * self.env.num_envs
            self.tot_timesteps += iteration_timesteps
            self.tot_time += iteration_time
            session_time += iteration_time
            session_iterations = completed_iteration - start_iteration
            remaining_iterations = end_iteration - completed_iteration
            eta_seconds = (
                session_time / max(session_iterations, 1)
                * remaining_iterations)
            timings = {
                "rollout": collection_time,
                "world_model": wm_time,
                "ppo": ppo_time,
            }
            if self.amp is not None:
                timings["discriminator"] = discriminator_time
            print(format_iteration_console(
                mode=self.mode,
                completed_iteration=completed_iteration,
                target_iteration=end_iteration,
                total_timesteps=completed_iteration * iteration_timesteps,
                iteration_timesteps=iteration_timesteps,
                elapsed_seconds=self.tot_time,
                eta_seconds=eta_seconds,
                timings=timings,
                learning_rate=self.alg.learning_rate,
                metrics=scalars,
            ), flush=True)
            if (self.checkpoints is not None
                    and completed_iteration % self.save_interval == 0):
                self.save_pair(completed_iteration)

        self.current_learning_iteration_init = end_iteration
        if (self.checkpoints is not None and num_learning_iterations > 0
                and end_iteration % self.save_interval != 0):
            self.save_pair(end_iteration)
        return {
            "iteration": self.current_learning_iteration,
            "mean_recent_reward": (
                torch.stack(list(reward_history)).mean().item()
                if reward_history else 0.0),
        }

    def _rng_state_dict(self):
        cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "torch_cuda": cuda_rng,
        }

    def state_dict(self, iteration=None):
        iteration = (
            self.current_learning_iteration if iteration is None
            else int(iteration))
        state = {
            "mode": self.mode,
            "actor_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "ppo_learning_rate": float(self.alg.learning_rate),
            "iteration": iteration,
            "terrain_levels": self.env.terrain_levels.clone(),
            "rng": self._rng_state_dict(),
        }
        if self.amp is not None:
            state["amp_state"] = self.amp.state_dict()
        return state

    def _validate_policy_state(
            self, state, mode, validate_environment_state=True):
        required = {
            "mode", "actor_state_dict", "optimizer_state_dict",
            "ppo_learning_rate", "iteration", "terrain_levels", "rng",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(
                f"policy checkpoint missing fields: {sorted(missing)}")
        if state.get("mode") != mode:
            raise ValueError("checkpoint runner mode mismatch")
        learning_rate = float(state["ppo_learning_rate"])
        if not np.isfinite(learning_rate) or learning_rate <= 0.0:
            raise ValueError("checkpoint PPO learning rate is invalid")
        if mode == "amp" and "amp_state" not in state:
            raise ValueError("AMP checkpoint state missing")
        if mode == "no_amp" and "amp_state" in state:
            raise ValueError("noAMP checkpoint cannot contain AMP state")
        rng_fields = {"python", "numpy", "torch", "torch_cuda"}
        if set(state["rng"]) != rng_fields:
            raise ValueError("policy checkpoint RNG fields mismatch")
        if validate_environment_state:
            validate_terrain_state(self.env, state["terrain_levels"])

    def load_state_dict(self, state):
        self._validate_policy_state(state, self.mode)
        restore_torch_rng = prepare_torch_rng_restore(torch, state["rng"])
        self.alg.actor_critic.load_state_dict(state["actor_state_dict"])
        self.alg.optimizer.load_state_dict(state["optimizer_state_dict"])
        restore_ppo_learning_rate(self.alg, state["ppo_learning_rate"])
        self.current_learning_iteration = int(state["iteration"])
        self.current_learning_iteration_init = self.current_learning_iteration
        restore_terrain_state(self.env, state["terrain_levels"])
        if self.amp is not None:
            if "amp_state" not in state:
                raise ValueError("AMP checkpoint state missing")
            self.amp.load_state_dict(state["amp_state"])
        elif "amp_state" in state:
            raise ValueError("noAMP checkpoint cannot contain AMP state")
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        restore_torch_rng()
        self.iteration_metrics.last_step = self.current_learning_iteration - 1

    def world_model_state_dict(self, iteration=None):
        iteration = (
            self.current_learning_iteration if iteration is None
            else int(iteration))
        networks = {
            "image_encoder": self.env.cnn_encoder.state_dict(),
            "image_decoder": self.env.cnn_decoder.state_dict(),
        }
        if getattr(self.env, "use_map_decoder", False):
            networks["map_encoder"] = self.env.map_encoder.state_dict()
            networks["map_decoder"] = self.env.map_decoder.state_dict()
        if getattr(self.env, "use_memory", False):
            networks["memory_encoder"] = self.env.memory_lstm.state_dict()
        depth_stats = {
            name: float(getattr(self.env, name))
            for name in (
                "dynamic_min", "dynamic_max", "range_size",
                "depth_mean", "depth_std")
            if hasattr(self.env, name)
        }
        return {
            "iteration": iteration,
            "networks": networks,
            "wm_optimizer": self.env.wm_optimizer.state_dict(),
            "depth_stats": depth_stats,
        }

    def _validate_world_model_state(self, state):
        required = {"iteration", "networks", "wm_optimizer", "depth_stats"}
        missing = required.difference(state)
        if missing:
            raise ValueError(
                f"world-model checkpoint missing fields: {sorted(missing)}")
        required_networks = {"image_encoder", "image_decoder"}
        if not required_networks.issubset(state["networks"]):
            raise ValueError("world-model checkpoint missing image networks")
        if getattr(self.env, "use_map_decoder", False):
            if not {"map_encoder", "map_decoder"}.issubset(state["networks"]):
                raise ValueError("world-model checkpoint missing map networks")
        if (getattr(self.env, "use_memory", False)
                and "memory_encoder" not in state["networks"]):
            raise ValueError("world-model checkpoint missing memory network")

    def load_world_model_state_dict(self, state, load_optimizer=True):
        self._validate_world_model_state(state)
        networks = state["networks"]
        self.env.cnn_encoder.load_state_dict(networks["image_encoder"])
        self.env.cnn_decoder.load_state_dict(networks["image_decoder"])
        if getattr(self.env, "use_map_decoder", False):
            self.env.map_encoder.load_state_dict(networks["map_encoder"])
            self.env.map_decoder.load_state_dict(networks["map_decoder"])
        if getattr(self.env, "use_memory", False):
            self.env.memory_lstm.load_state_dict(networks["memory_encoder"])
        if load_optimizer:
            self.env.wm_optimizer.load_state_dict(state["wm_optimizer"])
        for name, value in state["depth_stats"].items():
            setattr(self.env, name, float(value))

    def save_pair(self, iteration):
        if self.checkpoints is None:
            raise ValueError("checkpoint output is disabled")
        return self.checkpoints.save_pair(
            iteration,
            self.state_dict(iteration),
            self.world_model_state_dict(iteration),
            self.checkpoint_compatibility,
        )

    def load_pair(self, manifest=None, restore_training_state=True):
        if self.checkpoints is None:
            raise ValueError("checkpoint output is disabled")
        manifest = (
            self.checkpoints.resolve_last() if manifest is None else manifest)
        policy_state, wm_state, metadata = self.checkpoints.load_pair(
            manifest, self.checkpoint_compatibility)
        # Validate both halves before mutating either live subsystem.
        self._validate_policy_state(
            policy_state,
            self.mode,
            validate_environment_state=restore_training_state,
        )
        self._validate_world_model_state(wm_state)
        if restore_training_state:
            self.load_state_dict(policy_state)
            self.load_world_model_state_dict(wm_state)
            self.alg.actor_critic.reset()
            self.env.restart_after_checkpoint_load()
        else:
            self.alg.actor_critic.load_state_dict(
                policy_state["actor_state_dict"])
            if self.amp is not None:
                self.amp.load_state_dict(policy_state["amp_state"])
            self.load_world_model_state_dict(
                wm_state, load_optimizer=False)
            self.current_learning_iteration = int(policy_state["iteration"])
            self.current_learning_iteration_init = self.current_learning_iteration
        return metadata

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference, self.alg.actor_critic.evaluate

