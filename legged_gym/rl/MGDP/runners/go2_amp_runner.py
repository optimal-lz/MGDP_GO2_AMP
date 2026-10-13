"""Single-stage runner for the formal Go2 perception locomotion task."""

from collections import deque
from pathlib import Path
import time

import numpy as np
import torch

from rl.MGDP.amp.motions import MotionDataset
from rl.MGDP.amp.session import AMPSession
from rl.MGDP.amp.transition import select_next_amp_state

from .go2_amp_logging import IterationMetrics
from .policy_runner import MGDPPolicyRunner


class Go2AmpPolicyRunner(MGDPPolicyRunner):
    """Coordinate WM, PPO, and one optional AMP discriminator."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
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
            self.current_learning_iteration = iteration
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            self.tot_time += collection_time + learning_time

        self.current_learning_iteration_init = end_iteration
        return {
            "iteration": self.current_learning_iteration,
            "mean_recent_reward": (
                torch.stack(list(reward_history)).mean().item()
                if reward_history else 0.0),
        }

    def state_dict(self):
        state = {
            "mode": self.mode,
            "actor_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iteration": self.current_learning_iteration,
            "terrain_levels": self.env.terrain_levels.clone(),
        }
        if self.amp is not None:
            state["amp_state"] = self.amp.state_dict()
        return state

    def load_state_dict(self, state):
        if state.get("mode") != self.mode:
            raise ValueError("checkpoint runner mode mismatch")
        self.alg.actor_critic.load_state_dict(state["actor_state_dict"])
        self.alg.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.current_learning_iteration = int(state["iteration"])
        self.current_learning_iteration_init = self.current_learning_iteration + 1
        self.env.terrain_levels.copy_(state["terrain_levels"])
        if self.amp is not None:
            if "amp_state" not in state:
                raise ValueError("AMP checkpoint state missing")
            self.amp.load_state_dict(state["amp_state"])
        elif "amp_state" in state:
            raise ValueError("noAMP checkpoint cannot contain AMP state")

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference, self.alg.actor_critic.evaluate

