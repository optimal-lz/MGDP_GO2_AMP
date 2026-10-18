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

CONSOLE_WIDTH = 88
CONSOLE_LABEL_WIDTH = 42


def _console_scalar(name, value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("console value {!r} must be finite".format(name))
    return value


def _console_line(label, value):
    return "{label:>{width}} {value}".format(
        label=label + ":", width=CONSOLE_LABEL_WIDTH, value=value)


def _format_duration(seconds):
    seconds = max(0.0, _console_scalar("duration", seconds))
    if seconds < 60.0:
        return "{:.1f}s".format(seconds)
    if seconds < 3600.0:
        minutes = int(seconds // 60.0)
        return "{}m {:.1f}s".format(minutes, seconds - minutes * 60.0)
    hours = int(seconds // 3600.0)
    minutes = int((seconds - hours * 3600.0) // 60.0)
    remaining = seconds - hours * 3600.0 - minutes * 60.0
    return "{}h {:02d}m {:02.0f}s".format(hours, minutes, remaining)


def format_iteration_console(mode, completed_iteration, target_iteration,
                             total_timesteps, iteration_timesteps,
                             elapsed_seconds, eta_seconds, timings,
                             learning_rate, metrics):
    """Format one MGDP-style console panel from validated V1 scalars."""
    if mode not in ("amp", "no_amp"):
        raise ValueError("mode must be 'amp' or 'no_amp'")
    completed_iteration = int(completed_iteration)
    target_iteration = int(target_iteration)
    if not 0 < completed_iteration <= target_iteration:
        raise ValueError("completed iteration must be inside the target range")

    required_timings = {"rollout", "world_model", "ppo"}
    if mode == "amp":
        required_timings.add("discriminator")
    missing_timings = required_timings.difference(timings)
    if missing_timings:
        raise ValueError(
            "missing console timings: {}".format(sorted(missing_timings)))
    timing = {
        name: _console_scalar(name, timings[name])
        for name in required_timings
    }
    learning_rate = _console_scalar("learning_rate", learning_rate)
    iteration_seconds = _console_scalar(
        "performance/iteration_seconds",
        metrics["performance/iteration_seconds"],
    )
    steps_per_second = int(
        int(iteration_timesteps) / iteration_seconds
        if iteration_seconds > 0.0 else 0)

    mode_label = "AMP" if mode == "amp" else "noAMP"
    lines = [
        "#" * CONSOLE_WIDTH,
        ("MGDP GO2 AMP - Learning iteration {}/{}".format(
            completed_iteration, target_iteration)).center(CONSOLE_WIDTH),
        ("Mode: {}".format(mode_label)).center(CONSOLE_WIDTH),
        "",
        _console_line("Computation", "{} steps/s".format(steps_per_second)),
    ]
    timing_values = "{:.3f}s / {:.3f}s / {:.3f}s".format(
        timing["rollout"], timing["world_model"], timing["ppo"])
    timing_label = "Rollout / WM / PPO"
    if mode == "amp":
        timing_label += " / D"
        timing_values += " / {:.3f}s".format(timing["discriminator"])
    lines.extend([
        _console_line(timing_label, timing_values),
        _console_line(
            "Value function loss",
            "{:.5f}".format(metrics["ppo/value_loss"])),
        _console_line(
            "Surrogate loss",
            "{:.5f}".format(metrics["ppo/surrogate_loss"])),
        _console_line("PPO KL", "{:.6f}".format(metrics["ppo/kl"])),
        _console_line(
            "PPO entropy", "{:.5f}".format(metrics["ppo/entropy"])),
        _console_line("Learning rate", "{:.6f}".format(learning_rate)),
        _console_line(
            "Mean action noise std",
            "{:.3f}".format(metrics["policy/action_std"])),
        "",
        _console_line(
            "Mean task reward/step",
            "{:.5f}".format(metrics["reward/task"])),
        _console_line(
            "Mean regularization reward/step",
            "{:.5f}".format(metrics["reward/regularization"])),
        _console_line(
            "Mean AMP contribution/step",
            "{:.5f}".format(metrics["reward/amp_contribution"])),
        _console_line(
            "Mean total reward/step",
            "{:.5f}".format(metrics["reward/total"])),
        _console_line(
            "Mean |vx error|",
            "{:.5f}".format(metrics["tracking/vx_error"])),
        _console_line(
            "Mean |yaw error|",
            "{:.5f}".format(metrics["tracking/yaw_error"])),
        _console_line(
            "Mean episode length",
            "{:.2f}s".format(metrics["episode/length"])),
        _console_line(
            "Base-contact failure rate",
            "{:.5f}".format(metrics["episode/base_contact_failure"])),
        _console_line(
            "Mean terrain level",
            "{:.2f}".format(metrics["terrain/mean_level"])),
        "",
        _console_line(
            "World model total loss",
            "{:.5f}".format(metrics["wm/total_loss"])),
        _console_line(
            "Depth reconstruction loss",
            "{:.5f}".format(metrics["wm/depth_loss"])),
        _console_line(
            "Height reconstruction loss",
            "{:.5f}".format(metrics["wm/height_loss"])),
        _console_line(
            "Contrastive loss",
            "{:.5f}".format(metrics["wm/contrastive_loss"])),
    ])
    if mode == "amp":
        lines.extend([
            "",
            _console_line(
                "AMP policy logit",
                "{:.5f}".format(metrics["amp/policy_logit"])),
            _console_line(
                "AMP expert logit",
                "{:.5f}".format(metrics["amp/expert_logit"])),
            _console_line(
                "Discriminator loss",
                "{:.5f}".format(metrics["amp/discriminator_loss"])),
            _console_line(
                "Gradient penalty",
                "{:.5f}".format(metrics["amp/gradient_penalty"])),
            _console_line(
                "Valid AMP transitions",
                "{:.0f}".format(metrics["amp/valid_transition_count"])),
            _console_line(
                "Replay fill",
                "{:.3f}".format(metrics["amp/replay_fill"])),
        ])
    lines.extend([
        "-" * CONSOLE_WIDTH,
        _console_line("Total timesteps", str(int(total_timesteps))),
        _console_line("Iteration time", "{:.2f}s".format(iteration_seconds)),
        _console_line("Total time", _format_duration(elapsed_seconds)),
        _console_line("ETA", _format_duration(eta_seconds)),
    ])
    return "\n".join(lines)


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
        return scalars

