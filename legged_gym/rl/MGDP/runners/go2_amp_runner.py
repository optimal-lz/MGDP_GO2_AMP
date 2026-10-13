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

from .go2_amp_checkpoint import (
    CheckpointCompatibility,
    PairedCheckpointManager,
)
from .go2_amp_logging import IterationMetrics
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
        if self.tb_dir is not None:
            Path(self.tb_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.tb_dir, flush_secs=10)
        self.train_cfg = train_cfg
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
                "num_envs": self.env.num_envs,
                "num_steps_per_env": self.num_steps_per_env,
            })
            self.amp = AMPSession(dataset, self.device, amp_cfg, amp_seed)

        reward_coef = (
            train_cfg["amp"]["reward_coef"] if self.mode == "amp" else 0.0)
        self.checkpoint_compatibility = CheckpointCompatibility.from_config(
            mode=self.mode,
            amp_reward_coef=reward_coef,
            seed=train_cfg.get("seed", 1),
            num_envs=self.env.num_envs,
            key_config=train_cfg,
        )
        self.checkpoints = (
            PairedCheckpointManager(
                Path(log_dir) / "checkpoints", map_location=self.device)
            if log_dir is not None else None)
        self.current_learning_iteration = self.current_learning_iteration_init
        self.iteration_metrics = IterationMetrics(
            self.writer, self.mode, self.current_learning_iteration_init - 1)

    @staticmethod
    def _mean(total, count):
        return total / max(count, 1)

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

        for iteration in range(start_iteration, end_iteration):
            collection_start = time.time()
            task_sum = torch.zeros((), device=self.device)
            amp_sum = torch.zeros((), device=self.device)
            total_sum = torch.zeros((), device=self.device)
            transition_count = 0

            for _ in range(self.num_steps_per_env):
                if self.amp is not None:
                    amp_before = self.env.get_amp_observations().clone()
                with torch.inference_mode():
                    actions = self.alg.act(obs_dict)

                # env.step performs the online WM update and therefore must keep
                # autograd enabled even though policy rollout inference does not.
                obs_dict, _, dones, infos = self.env.step(actions)
                dones = dones.to(self.device)

                amp_contribution = torch.zeros_like(self.env._base_reward)
                if self.amp is not None:
                    amp_after = select_next_amp_state(
                        self.env.get_amp_observations().clone(),
                        dones,
                        infos.get("terminal_amp_states"),
                    )
                    amp_transition_mask = infos["amp_transition_mask"]
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

                task_sum += self.env._base_reward.detach().sum()
                amp_sum += amp_contribution.detach().sum()
                total_sum += rewards.detach().sum()
                transition_count += rewards.numel()
                reward_history.append(rewards.detach().mean())

            collection_time = time.time() - collection_start

            # Keep the frozen formal order: online WM rollout, four extra WM
            # updates, GAE, PPO, then twenty discriminator updates.
            wm_rollout_steps = int(
                getattr(self.env.cfg.camera, "wm_rollout_steps", 0))
            if getattr(self.env, "use_world_model", False):
                for _ in range(wm_rollout_steps):
                    self.env.update_world_model(
                        batch=getattr(
                            self.env, "latest_world_model_batch", None),
                        num_updates=1,
                    )

            with torch.inference_mode():
                self.alg.compute_returns(obs_dict)
            learning_start = time.time()
            self.alg.update()
            amp_update_metrics = (
                self.amp.update() if self.amp is not None else {})
            learning_time = time.time() - learning_start

            core = {
                "reward/task": self._mean(task_sum, transition_count),
                "reward/amp_contribution": self._mean(
                    amp_sum, transition_count),
                "reward/total": self._mean(total_sum, transition_count),
                "performance/collection_seconds": collection_time,
                "performance/learning_seconds": learning_time,
            }
            core.update(self.alg.last_update_metrics)
            if self.amp is not None:
                core.update({
                    f"amp/{key}": value
                    for key, value in amp_update_metrics.items()
                })
            detail = {
                "terrain/mean_level": self.env.terrain_levels.float().mean(),
                "policy/action_std": self.alg.actor_critic.std.mean(),
            }
            self.iteration_metrics.write(iteration, core, detail)
            completed_iteration = iteration + 1
            self.current_learning_iteration = completed_iteration
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            self.tot_time += collection_time + learning_time
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
            "iteration": iteration,
            "terrain_levels": self.env.terrain_levels.clone(),
            "rng": self._rng_state_dict(),
        }
        if self.amp is not None:
            state["amp_state"] = self.amp.state_dict()
        return state

    @staticmethod
    def _validate_policy_state(state, mode):
        required = {
            "mode", "actor_state_dict", "optimizer_state_dict",
            "iteration", "terrain_levels", "rng",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(
                f"policy checkpoint missing fields: {sorted(missing)}")
        if state.get("mode") != mode:
            raise ValueError("checkpoint runner mode mismatch")
        if mode == "amp" and "amp_state" not in state:
            raise ValueError("AMP checkpoint state missing")
        if mode == "no_amp" and "amp_state" in state:
            raise ValueError("noAMP checkpoint cannot contain AMP state")
        rng_fields = {"python", "numpy", "torch", "torch_cuda"}
        if set(state["rng"]) != rng_fields:
            raise ValueError("policy checkpoint RNG fields mismatch")

    def load_state_dict(self, state):
        self._validate_policy_state(state, self.mode)
        self.alg.actor_critic.load_state_dict(state["actor_state_dict"])
        self.alg.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.current_learning_iteration = int(state["iteration"])
        self.current_learning_iteration_init = self.current_learning_iteration
        self.env.terrain_levels.copy_(state["terrain_levels"])
        if self.amp is not None:
            if "amp_state" not in state:
                raise ValueError("AMP checkpoint state missing")
            self.amp.load_state_dict(state["amp_state"])
        elif "amp_state" in state:
            raise ValueError("noAMP checkpoint cannot contain AMP state")
        random.setstate(state["rng"]["python"])
        np.random.set_state(state["rng"]["numpy"])
        torch.random.set_rng_state(state["rng"]["torch"])
        if state["rng"]["torch_cuda"] and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["rng"]["torch_cuda"])
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

    def load_world_model_state_dict(self, state):
        self._validate_world_model_state(state)
        networks = state["networks"]
        self.env.cnn_encoder.load_state_dict(networks["image_encoder"])
        self.env.cnn_decoder.load_state_dict(networks["image_decoder"])
        if getattr(self.env, "use_map_decoder", False):
            self.env.map_encoder.load_state_dict(networks["map_encoder"])
            self.env.map_decoder.load_state_dict(networks["map_decoder"])
        if getattr(self.env, "use_memory", False):
            self.env.memory_lstm.load_state_dict(networks["memory_encoder"])
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

    def load_pair(self, manifest=None):
        if self.checkpoints is None:
            raise ValueError("checkpoint output is disabled")
        manifest = (
            self.checkpoints.resolve_last() if manifest is None else manifest)
        policy_state, wm_state, metadata = self.checkpoints.load_pair(
            manifest, self.checkpoint_compatibility)
        # Validate both halves before mutating either live subsystem.
        self._validate_policy_state(policy_state, self.mode)
        self._validate_world_model_state(wm_state)
        self.load_state_dict(policy_state)
        self.load_world_model_state_dict(wm_state)
        return metadata

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference, self.alg.actor_critic.evaluate

