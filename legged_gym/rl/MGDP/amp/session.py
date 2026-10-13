"""Single-discriminator AMP reward, update, and complete checkpoint state."""

import copy
import math

import torch

from .discriminator import AMPDiscriminator
from .replay import AMPReplayBuffer


class RunningNormalizer:
    def __init__(self, width, device):
        self.mean = torch.zeros(width, device=device)
        self.var = torch.ones(width, device=device)
        self.count = 1e-4

    def update(self, batch):
        batch = batch.detach()
        count = batch.shape[0]
        if count == 0:
            return
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        total = self.count + count
        delta = batch_mean - self.mean
        self.mean += delta * count / total
        self.var = (
            self.var * self.count
            + batch_var * count
            + delta.square() * self.count * count / total
        ) / total
        self.count = total

    def normalize(self, batch):
        return (batch - self.mean) / torch.sqrt(self.var + 1e-5)

    def state_dict(self):
        return {
            "mean": self.mean.clone(),
            "var": self.var.clone(),
            "count": self.count,
        }

    def validate_state_dict(self, state):
        if set(state) != {"mean", "var", "count"}:
            raise ValueError("AMP normalizer state fields mismatch")
        if state["mean"].shape != self.mean.shape:
            raise ValueError("AMP normalizer mean shape mismatch")
        if state["var"].shape != self.var.shape:
            raise ValueError("AMP normalizer variance shape mismatch")
        if not math.isfinite(float(state["count"])) or float(state["count"]) <= 0:
            raise ValueError("AMP normalizer count is invalid")

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        self.mean.copy_(state["mean"].to(self.mean))
        self.var.copy_(state["var"].to(self.var))
        self.count = float(state["count"])


class AMPSession:
    def __init__(self, dataset, device, config, amp_seed):
        self._validate_config(config)
        self.dataset = dataset
        self.device = device
        self.config = dict(config)
        self.amp_seed = int(amp_seed)

        # Constructing AMP must not perturb policy/environment torch RNG state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.amp_seed)
            discriminator = AMPDiscriminator()
        self.discriminator = discriminator.to(device)
        self.optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config["amp_learning_rate"],
        )
        self.normalizer = RunningNormalizer(30, device)
        num_envs = int(self.config["num_envs"])
        num_steps = int(self.config["num_steps_per_env"])
        capacity = 2 * num_envs * num_steps
        self.replay = AMPReplayBuffer(
            capacity, 30, device, seed=self.amp_seed + 1)
        self.update_count = 0
        self.last_reward_metrics = {}

    @staticmethod
    def _validate_config(config):
        required = (
            "amp_reward_coef", "amp_batch_size", "amp_updates_per_iter",
            "amp_gradient_penalty_coef", "amp_learning_rate",
            "amp_replay_rollouts", "num_envs", "num_steps_per_env",
        )
        missing = [key for key in required if key not in config]
        if missing:
            raise ValueError(f"AMP config missing fields: {missing}")
        for key in ("amp_reward_coef", "amp_gradient_penalty_coef",
                    "amp_learning_rate"):
            value = float(config[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("amp_batch_size", "amp_updates_per_iter", "num_envs",
                    "num_steps_per_env"):
            value = config[key]
            if int(value) != value or int(value) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if int(config["amp_updates_per_iter"]) != 20:
            raise ValueError("amp_updates_per_iter must be 20 for the formal task")
        if int(config["amp_replay_rollouts"]) != 2:
            raise ValueError("amp_replay_rollouts must be 2 for the formal task")

    def _pair(self, state, next_state):
        return torch.cat((
            self.normalizer.normalize(state),
            self.normalizer.normalize(next_state),
        ), dim=-1)

    def reward(self, state, next_state, eligible, enabled):
        """Return only the masked AMP contribution for the current rollout."""
        if state.shape != next_state.shape or state.ndim != 2 or state.shape[1] != 30:
            raise ValueError("AMP reward expects matching [N, 30] states")
        if eligible.ndim != 1 or eligible.shape[0] != state.shape[0]:
            raise ValueError("eligible must contain one value per transition")
        contribution = torch.zeros(
            state.shape[0], dtype=state.dtype, device=state.device)
        mask = eligible.to(device=state.device, dtype=torch.bool)
        if not enabled or not torch.any(mask):
            self.last_reward_metrics = {
                "valid_transition_count": mask.sum(),
                "amp_contribution_sum": contribution.sum(),
                "policy_logit_sum": contribution.sum(),
            }
            return contribution
        with torch.no_grad():
            logits = self.discriminator(self._pair(state[mask], next_state[mask]))
            raw = torch.clamp(
                1.0 - 0.25 * (logits - 1.0).square(), min=0.0)
            contribution[mask] = self.config['amp_reward_coef'] * raw
        self.last_reward_metrics = {
            "valid_transition_count": mask.sum(),
            "amp_contribution_sum": contribution.sum(),
            "policy_logit_sum": logits.sum(),
        }
        return contribution

    def record(self, state, next_state, mask):
        if mask.ndim != 1 or mask.shape[0] != state.shape[0]:
            raise ValueError("mask must contain one value per transition")
        mask = mask.to(device=state.device, dtype=torch.bool)
        if torch.any(mask):
            self.replay.insert(state[mask], next_state[mask])

    def update(self):
        updates_per_iter = int(self.config["amp_updates_per_iter"])
        if self.replay.size == 0:
            return {
                "amp_updates": 0,
                "amp_update_count": self.update_count,
                "replay_size": 0,
            }
        losses = []
        penalties = []
        policy_outputs = []
        expert_outputs = []
        batch_size = int(self.config["amp_batch_size"])
        coefficient = float(self.config["amp_gradient_penalty_coef"])
        for _ in range(updates_per_iter):
            policy_state, policy_next = self.replay.sample(batch_size)
            expert_state, expert_next = self.dataset.sample(batch_size, self.device)
            self.normalizer.update(torch.cat((
                policy_state, policy_next, expert_state, expert_next,
            ), dim=0))
            policy_pair = self._pair(policy_state, policy_next).detach()
            expert_pair = self._pair(expert_state, expert_next).detach()
            policy_output = self.discriminator(policy_pair)
            expert_output = self.discriminator(expert_pair)
            discriminator_loss = 0.5 * (
                (expert_output - 1.0).square().mean()
                + (policy_output + 1.0).square().mean()
            )
            gradient_penalty = self.discriminator.gradient_penalty(
                expert_pair, coefficient=coefficient)
            self.optimizer.zero_grad()
            (discriminator_loss + gradient_penalty).backward()
            self.optimizer.step()
            self.update_count += 1
            losses.append(discriminator_loss.detach())
            penalties.append(gradient_penalty.detach())
            policy_outputs.append(policy_output.detach())
            expert_outputs.append(expert_output.detach())
        return {
            "amp_updates": updates_per_iter,
            "amp_update_count": self.update_count,
            "replay_size": self.replay.size,
            "discriminator_loss": torch.stack(losses).mean().item(),
            "gradient_penalty": torch.stack(penalties).mean().item(),
            "policy_logit": torch.cat(policy_outputs).mean().item(),
            "expert_logit": torch.cat(expert_outputs).mean().item(),
        }

    def _compatibility(self):
        groups = self.config.get("amp_groups", {})
        weights = self.config.get("amp_group_weights", {})
        keys = (
            "amp_reward_coef", "amp_batch_size", "amp_updates_per_iter",
            "amp_gradient_penalty_coef", "amp_learning_rate",
            "amp_replay_rollouts", "num_envs", "num_steps_per_env",
        )
        compatibility = {key: self.config[key] for key in keys}
        compatibility["amp_groups"] = tuple(
            (str(name), tuple(str(item) for item in groups[name]))
            for name in sorted(groups))
        compatibility["amp_group_weights"] = tuple(
            (str(name), float(weights[name])) for name in sorted(weights))
        return compatibility

    def _dataset_rng_state(self):
        rng = getattr(self.dataset, "rng", None)
        if rng is None or not hasattr(rng, "bit_generator"):
            return None
        return copy.deepcopy(rng.bit_generator.state)

    def state_dict(self):
        return {
            "discriminator": self.discriminator.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "normalizer": self.normalizer.state_dict(),
            "replay": self.replay.state_dict(),
            "update_count": self.update_count,
            "compatibility": self._compatibility(),
            "amp_seed": self.amp_seed,
            "dataset_rng_state": self._dataset_rng_state(),
        }

    def _validate_model_state(self, saved):
        current = self.discriminator.state_dict()
        if set(saved) != set(current):
            raise ValueError("AMP discriminator state fields mismatch")
        for name, tensor in current.items():
            if saved[name].shape != tensor.shape:
                raise ValueError(f"AMP discriminator shape mismatch: {name}")

    def load_state_dict(self, state):
        required = {
            "discriminator", "optimizer", "normalizer", "replay",
            "update_count", "compatibility", "amp_seed", "dataset_rng_state",
        }
        missing = required.difference(state)
        if missing:
            raise ValueError(f"AMP checkpoint missing fields: {sorted(missing)}")
        expected = self._compatibility()
        for field, value in expected.items():
            actual = state["compatibility"].get(field)
            if actual != value:
                raise ValueError(
                    f"incompatible AMP checkpoint field {field}: "
                    f"checkpoint {actual!r}, current {value!r}")
        if int(state["amp_seed"]) != self.amp_seed:
            raise ValueError("incompatible AMP checkpoint field amp_seed")
        self._validate_model_state(state["discriminator"])
        self.normalizer.validate_state_dict(state["normalizer"])
        self.replay.validate_state_dict(state["replay"])
        dataset_rng = getattr(self.dataset, "rng", None)
        if state["dataset_rng_state"] is not None and (
                dataset_rng is None or not hasattr(dataset_rng, "bit_generator")):
            raise ValueError("AMP dataset RNG cannot be restored")

        self.discriminator.load_state_dict(state["discriminator"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.normalizer.load_state_dict(state["normalizer"])
        self.replay.load_state_dict(state["replay"])
        self.update_count = int(state["update_count"])
        if state["dataset_rng_state"] is not None:
            dataset_rng.bit_generator.state = copy.deepcopy(
                state["dataset_rng_state"])


def restore_amp_checkpoint(session, checkpoint):
    if "amp_state" not in checkpoint:
        raise ValueError("AMP checkpoint state missing")
    session.load_state_dict(checkpoint["amp_state"])
