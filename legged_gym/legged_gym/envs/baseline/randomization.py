"""Explicit randomization contracts shared by the formal Go2 AMP task."""

import numpy as np


ALLOWED_TRAINING_RANDOMIZATIONS = (
    "randomize_friction",
    "randomize_action_latency",
    "push_robots",
)

FORBIDDEN_DOMAIN_RANDOMIZATIONS = (
    "randomize_base_mass",
    "randomize_link_mass",
    "randomize_restitution",
    "randomize_com",
    "randomize_gains",
    "randomize_motor_strength",
    "randomize_motor_offset",
    "randomize_lag_timesteps",
)

ALL_DOMAIN_RANDOMIZATIONS = (
    ALLOWED_TRAINING_RANDOMIZATIONS + FORBIDDEN_DOMAIN_RANDOMIZATIONS)


def _randomization_switches(domain_rand):
    return tuple(
        name for name in dir(domain_rand)
        if name.startswith("randomize_")
        and not name.startswith("__")
        and isinstance(getattr(domain_rand, name), (bool, np.bool_))
    )


def validate_training_randomization(domain_rand):
    """Fail fast if the formal task escapes its frozen allowlist.

    Approved switches may be disabled by deterministic evaluation tools; the
    formal training config separately pins all three to true.
    """
    enabled = [
        name for name in _randomization_switches(domain_rand)
        if name not in ALLOWED_TRAINING_RANDOMIZATIONS
        if bool(getattr(domain_rand, name, False))
    ]
    if enabled:
        raise ValueError(
            "Go2 AMP training enables forbidden domain randomization: {}"
            .format(", ".join(enabled)))


def sample_motor_parameters(domain_rand, num_actions, rng=None):
    """Sample motor parameters only from their dedicated switches.

    Action latency is intentionally absent: it is implemented by delayed
    action selection in ``LeggedRobot.step`` and has no motor side effects.
    """
    rng = np.random if rng is None else rng
    strengths = np.ones(int(num_actions), dtype=np.float32)
    offsets = np.zeros(int(num_actions), dtype=np.float32)
    if bool(getattr(domain_rand, "randomize_motor_strength", False)):
        low, high = domain_rand.added_motor_strength
        strengths.fill(rng.uniform(low, high))
    if bool(getattr(domain_rand, "randomize_motor_offset", False)):
        low, high = domain_rand.added_motor_offset
        offsets.fill(rng.uniform(low, high))
    return strengths, offsets


def camera_mount_ranges(translation_jitter, pitch_jitter_deg):
    """Return symmetric XYZ translation and pitch-only rotation ranges."""
    translation = float(translation_jitter)
    pitch = float(pitch_jitter_deg)
    if translation < 0.0 or pitch < 0.0:
        raise ValueError("camera mount jitter must be nonnegative")
    return (
        ([-translation] * 3, [translation] * 3),
        ([0.0, -pitch, 0.0], [0.0, pitch, 0.0]),
    )


def disable_evaluation_stochasticity(cfg):
    """Make Viewer/comparison rollouts deterministic without changing training."""
    for name in ("add_noise", "add_height_noise", "add_privileged_noise"):
        if hasattr(cfg.noise, name):
            setattr(cfg.noise, name, False)
    cfg.camera.update_wm = False
    # Randomdog checks these fields against None.  Using numeric zero would
    # still allocate random tensors and, for dropout, could alter exact-zero
    # samples while perturbing the evaluation RNG stream.
    cfg.camera.noise_gaussian = None
    cfg.camera.noise_dropout = None
    cfg.camera.mount_translation_jitter = 0.0
    cfg.camera.mount_pitch_jitter_deg = 0.0
    for name in set(_randomization_switches(cfg.domain_rand)).union(
            ALL_DOMAIN_RANDOMIZATIONS):
        if hasattr(cfg.domain_rand, name):
            setattr(cfg.domain_rand, name, False)

