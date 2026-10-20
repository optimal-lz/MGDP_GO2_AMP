"""Pure command-mode helpers shared by the Go2 AMP environment and tests.

The functions deliberately use only array operations supported by both NumPy
arrays and PyTorch tensors.  Importing them therefore does not pull Isaac Gym
into lightweight validation tools.
"""


def _copy(array):
    return array.clone() if hasattr(array, "clone") else array.copy()


def _bool(array):
    return array.bool() if hasattr(array, "bool") else array.astype(bool)


def apply_fixed_evaluation_commands(commands, env_ids, fixed_commands):
    """Restore an evaluator-owned command after every environment reset."""
    if fixed_commands is None:
        return False
    commands[env_ids] = fixed_commands[env_ids]
    return True


def apply_command_deadband(commands, vx_deadband=0.2, yaw_deadband=0.2):
    """Return commands with the frozen forward/yaw deadbands applied.

    Go2 AMP does not train backward commands, so every forward command at or
    below ``vx_deadband`` becomes zero.  Yaw uses a symmetric deadband.
    """
    filtered = _copy(commands)
    filtered[..., 0][filtered[..., 0] <= vx_deadband] = 0.0
    filtered[..., 2][abs(filtered[..., 2]) <= yaw_deadband] = 0.0
    return filtered


def apply_terrain_command_policy(commands, terrain_classes,
                                 turning_terrain_ids=(0, 1)):
    """Allow yaw only on terrain classes explicitly designated for turning."""
    filtered = _copy(commands)
    turning = terrain_classes == turning_terrain_ids[0]
    for terrain_id in turning_terrain_ids[1:]:
        turning = turning | (terrain_classes == terrain_id)
    filtered[..., 2][~turning] = 0.0
    return filtered


def classify_commands(commands):
    """Return the mutually exclusive ``(idle, locomotion)`` masks."""
    idle = (commands[..., 0] == 0.0) & (commands[..., 2] == 0.0)
    return idle, ~idle


def amp_transition_mask(locomotion_before, epoch_before, epoch_after, dones):
    """Select valid policy transitions for AMP replay.

    A locomotion transition is valid when its command epoch remained stable.
    Terminal transitions are retained because their post-step observation is
    captured before environment reset.
    """
    return _bool(locomotion_before) & ((epoch_before == epoch_after) | _bool(dones))
