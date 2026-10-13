"""Pure terrain contracts and curriculum math for the formal Go2 AMP task."""


TERRAIN_NAMES = (
    "flat",
    "rough_flat",
    "smooth_slope",
    "rough_slope",
    "stairs",
    "discrete_obstacles",
    "hurdle",
    "gap",
    "pit",
    "narrow_passage",
)

TERRAIN_CLASS_IDS = {name: index for index, name in enumerate(TERRAIN_NAMES)}

TERRAIN_PROPORTIONS = (
    0.10,
    0.10,
    0.10,
    0.10,
    0.15,
    0.10,
    0.10,
    0.10,
    0.10,
    0.05,
)

# Twenty physical columns make the percentages exact rather than approximate.
TERRAIN_COLUMNS = (
    "flat", "flat",
    "rough_flat", "rough_flat",
    "smooth_slope", "smooth_slope",
    "rough_slope", "rough_slope",
    "stairs", "stairs", "stairs",
    "discrete_obstacles", "discrete_obstacles",
    "hurdle", "hurdle",
    "gap", "gap",
    "pit", "pit",
    "narrow_passage",
)

# Temporary compatibility name for configuration imports created before the
# formal single-stage terrain contract was frozen.
GO2_AMP_TERRAIN_COLUMNS = TERRAIN_COLUMNS


def difficulty_for_level(level, num_rows=20):
    """Map an explicit curriculum row to the closed difficulty interval [0, 1]."""
    if num_rows < 2 or level < 0 or level >= num_rows:
        raise ValueError("terrain level must be inside a curriculum with at least two rows")
    return float(level) / float(num_rows - 1)


def next_curriculum_levels(levels, forward_commands, progress,
                           success_distance, early_failure,
                           random_lower_levels, max_level=19):
    """Return curriculum levels after forward-only success/failure updates.

    Inputs may be NumPy arrays or PyTorch tensors.  A supplied random lower
    level is used only when a successful episode would advance beyond the top
    row, matching MGDP's top-level recycling behavior without hiding RNG here.
    """
    updated = levels.clone() if hasattr(levels, "clone") else levels.copy()
    forward = (forward_commands.bool() if hasattr(forward_commands, "bool")
               else forward_commands.astype(bool))
    failed = (early_failure.bool() if hasattr(early_failure, "bool")
              else early_failure.astype(bool))
    move_up = forward & (progress >= success_distance)
    move_down = forward & failed & ~move_up
    updated[move_up] += 1
    updated[move_down] -= 1
    updated[updated < 0] = 0
    overflow = updated > max_level
    updated[overflow] = random_lower_levels[overflow]
    return updated
