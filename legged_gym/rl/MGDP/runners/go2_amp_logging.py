"""Small, strict TensorBoard scalar logger for the formal Go2 AMP task."""

import math


CORE_TAGS = frozenset({
    "reward/task",
    "reward/regularization",
    "reward/amp_contribution",
    "reward/total",
    "tracking/vx_error",
    "tracking/yaw_error",
    "episode/length",
    "episode/base_contact_failure",
    "terrain/mean_level",
    "wm/total_loss",
    "wm/depth_loss",
    "wm/height_loss",
    "wm/contrastive_loss",
    "ppo/surrogate_loss",
    "ppo/value_loss",
    "ppo/entropy",
    "ppo/kl",
    "policy/action_std",
    "performance/iteration_seconds",
})

AMP_CORE_TAGS = frozenset({
    "amp/policy_logit",
    "amp/expert_logit",
    "amp/discriminator_loss",
    "amp/gradient_penalty",
    "amp/valid_transition_count",
    "amp/replay_fill",
})

DETAIL_TAGS = frozenset({
    "timing/simulation_seconds",
    "timing/wm_seconds",
    "timing/ppo_seconds",
})

AMP_DETAIL_TAGS = frozenset({
    "timing/discriminator_seconds",
    "amp/expert_forward_logit",
    "amp/expert_turn_logit",
})


class IterationMetrics:
    """Write one finite scalar set per monotonically increasing iteration."""

    def __init__(self, writer, mode, start_step=-1, terrain_names=()):
        if mode not in ("amp", "no_amp"):
            raise ValueError("mode must be 'amp' or 'no_amp'")
        self.writer = writer
        self.mode = mode
        self.last_step = int(start_step)
        self.terrain_names = tuple(str(name) for name in terrain_names)

    @staticmethod
    def _scalar(tag, value):
        if hasattr(value, "item"):
            value = value.item()
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"metric {tag!r} must be finite")
        return value

    @classmethod
    def _scalarize(cls, values):
        """Convert all GPU scalar tensors with one stacked device transfer."""
        scalars = {}
        tensor_items = []
        try:
            import torch
        except ImportError:
            torch = None
        for tag, value in values.items():
            if torch is not None and torch.is_tensor(value):
                tensor_items.append((tag, value.detach().reshape(())))
            else:
                scalars[tag] = cls._scalar(tag, value)
        if tensor_items:
            device = tensor_items[0][1].device
            stacked = torch.stack([
                value.to(device=device) for _, value in tensor_items
            ]).cpu().tolist()
            for (tag, _), value in zip(tensor_items, stacked):
                scalars[tag] = cls._scalar(tag, value)
        return scalars

    def _required_detail_tags(self):
        required = set(DETAIL_TAGS)
        for terrain in self.terrain_names:
            required.update({
                "terrain/{}/mean_level".format(terrain),
                "terrain/{}/completion".format(terrain),
                "terrain/{}/base_contact_failure".format(terrain),
                "terrain/{}/stuck".format(terrain),
            })
        if self.mode == "amp":
            required.update(AMP_DETAIL_TAGS)
        return required

    def write(self, iteration, core, detail=None):
        iteration = int(iteration)
        if iteration <= self.last_step:
            raise ValueError("metric iterations must be strictly monotonic")

        required_core = set(CORE_TAGS)
        if self.mode == "amp":
            required_core.update(AMP_CORE_TAGS)
        missing_core = required_core.difference(core)
        if missing_core:
            raise ValueError(
                "missing core metrics: {}".format(sorted(missing_core)))

        if iteration % 10 == 0:
            detail = {} if detail is None else detail
            missing_detail = self._required_detail_tags().difference(detail)
            if missing_detail:
                raise ValueError(
                    "missing detail metrics: {}".format(
                        sorted(missing_detail)))

        values = dict(core)
        if detail is not None and iteration % 10 == 0:
            values.update(detail)
        scalars = self._scalarize(values)

        if self.mode == "no_amp":
            contribution = scalars.get("reward/amp_contribution", 0.0)
            if contribution != 0.0:
                raise ValueError("noAMP reward contribution must remain zero")
            if any("discriminator" in tag.lower() for tag in scalars):
                raise ValueError("noAMP cannot log discriminator training metrics")

        if self.writer is not None:
            for tag, value in scalars.items():
                self.writer.add_scalar(tag, value, iteration)
        self.last_step = iteration

