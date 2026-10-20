"""Terrain contracts, curriculum math, and frozen evaluation case helpers."""

from collections import Counter


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
TERRAIN_OVERVIEW_LEVELS = (0, 9, 19)


def terrain_overview_layout(levels=TERRAIN_OVERVIEW_LEVELS):
    """Return the compact terrain-inspection grid from frozen training data."""
    levels = tuple(int(level) for level in levels)
    for level in levels:
        difficulty_for_level(level)
    counts = Counter(TERRAIN_COLUMNS)
    column_count = len(TERRAIN_COLUMNS)
    for name, proportion in zip(TERRAIN_NAMES, TERRAIN_PROPORTIONS):
        expected = int(round(proportion * column_count))
        if counts[name] != expected:
            raise RuntimeError(
                "terrain proportion does not match physical columns: {}"
                .format(name))
    return tuple(
        {
            "terrain": name,
            "level": level,
            "proportion": proportion,
            "display_env": terrain_index * len(levels) + level_index,
        }
        for terrain_index, (name, proportion) in enumerate(
            zip(TERRAIN_NAMES, TERRAIN_PROPORTIONS))
        for level_index, level in enumerate(levels)
    )


def configure_evaluation_terrain_grid(cfg, columns, levels=None,
                                      formal_num_rows=20):
    """Configure the formal named generator for evaluation or inspection."""
    columns = tuple(columns)
    if not columns or any(name not in TERRAIN_NAMES for name in columns):
        raise ValueError("evaluation terrain columns must use formal names")
    if int(formal_num_rows) < 2:
        raise ValueError("formal_num_rows must be at least two")
    cfg.curriculum = True
    cfg.num_cols = len(columns)
    cfg.explicit_terrain_columns = columns
    cfg.explicit_terrain_num_rows = int(formal_num_rows)
    if levels is None:
        cfg.num_rows = int(formal_num_rows)
        cfg.explicit_terrain_levels = None
        return
    levels = tuple(int(level) for level in levels)
    if not levels:
        raise ValueError("evaluation terrain levels must not be empty")
    for level in levels:
        difficulty_for_level(level, int(formal_num_rows))
    cfg.num_rows = len(levels)
    cfg.explicit_terrain_levels = levels


def _display_row_for_level(cfg, level):
    level = int(level)
    explicit_levels = getattr(cfg, "explicit_terrain_levels", None)
    if explicit_levels is None:
        if level < 0 or level >= int(cfg.num_rows):
            raise ValueError("evaluation terrain level is outside the grid")
        return level
    try:
        return tuple(explicit_levels).index(level)
    except ValueError as error:
        raise ValueError("evaluation terrain level is not displayed") from error


def _assign_values(target, values):
    if hasattr(target, "new_tensor"):
        target[:] = target.new_tensor(values)
    else:
        target[:] = values


def _fill(target, value):
    if hasattr(target, "fill_"):
        target.fill_(value)
    else:
        target.fill(value)


def _all_true(value):
    result = value.all() if hasattr(value, "all") else all(value)
    return bool(result.item() if hasattr(result, "item") else result)


def _copy_commands(commands):
    return commands.clone() if hasattr(commands, "clone") else commands.copy()


def _set_fixed_commands(env, vx, yaw):
    fixed = _copy_commands(env.commands)
    fixed[:] = 0.0
    fixed[:, 0] = float(vx)
    fixed[:, 2] = float(yaw)
    env._fixed_evaluation_commands = fixed
    env.commands[:] = fixed


def freeze_evaluation_case(env, terrain, level, vx, yaw):
    """Place every environment on one exact case and freeze reset behavior."""
    if terrain not in TERRAIN_NAMES:
        raise ValueError("unknown Go2 AMP terrain: {}".format(terrain))
    if terrain not in ("flat", "rough_flat") and float(yaw) != 0.0:
        raise ValueError("structured terrain evaluation requires yaw=0")
    columns = tuple(env.cfg.terrain.explicit_terrain_columns)
    if terrain not in columns:
        raise ValueError("evaluation terrain is absent from the generated grid")
    column = columns.index(terrain)
    row = _display_row_for_level(env.cfg.terrain, level)
    env._suppress_curriculum = True
    _fill(env.terrain_types, column)
    _fill(env.terrain_levels, row)
    env.env_origins[:] = env.terrain_origins[
        env.terrain_levels, env.terrain_types]
    env.env_class[:] = env.terrain_class[
        env.terrain_levels, env.terrain_types]
    _set_fixed_commands(env, vx, yaw)


def freeze_terrain_overview(env, layout):
    """Place one fixed-base reference robot on each compact overview tile."""
    layout = tuple(layout)
    if len(layout) != int(env.num_envs):
        raise ValueError("terrain overview requires one environment per tile")
    columns = tuple(env.cfg.terrain.explicit_terrain_columns)
    terrain_types = [columns.index(item["terrain"]) for item in layout]
    terrain_levels = [
        _display_row_for_level(env.cfg.terrain, item["level"])
        for item in layout]
    env._suppress_curriculum = True
    _assign_values(env.terrain_types, terrain_types)
    _assign_values(env.terrain_levels, terrain_levels)
    env.env_origins[:] = env.terrain_origins[
        env.terrain_levels, env.terrain_types]
    env.env_class[:] = env.terrain_class[
        env.terrain_levels, env.terrain_types]
    _set_fixed_commands(env, 0.0, 0.0)


def assert_evaluation_case(env, terrain, level, vx, yaw, env_ids=None):
    """Fail loudly if a reset changes an explicitly requested evaluation case."""
    selected = slice(None) if env_ids is None else env_ids
    columns = tuple(env.cfg.terrain.explicit_terrain_columns)
    expected_column = columns.index(terrain)
    expected_row = _display_row_for_level(env.cfg.terrain, level)
    expected_class = TERRAIN_CLASS_IDS[terrain]
    checks = (
        ("terrain column", env.terrain_types[selected] == expected_column),
        ("terrain level", env.terrain_levels[selected] == expected_row),
        ("terrain class", env.env_class[selected] == expected_class),
        ("command vx", env.commands[selected, 0] == float(vx)),
        ("command vy", env.commands[selected, 1] == 0.0),
        ("command yaw", env.commands[selected, 2] == float(yaw)),
        ("command extra fields", env.commands[selected, 3:] == 0.0),
    )
    for label, matches in checks:
        if not _all_true(matches):
            raise RuntimeError("evaluation {} changed after reset".format(label))


def assert_terrain_overview(env, layout):
    """Verify the compact overview still maps each robot to its named tile."""
    columns = tuple(env.cfg.terrain.explicit_terrain_columns)
    expected_types = [columns.index(item["terrain"]) for item in layout]
    expected_levels = [
        _display_row_for_level(env.cfg.terrain, item["level"])
        for item in layout]
    expected_classes = [TERRAIN_CLASS_IDS[item["terrain"]] for item in layout]
    checks = (
        ("terrain columns", env.terrain_types
         == _like(env.terrain_types, expected_types)),
        ("terrain levels", env.terrain_levels
         == _like(env.terrain_levels, expected_levels)),
        ("terrain classes", env.env_class
         == _like(env.env_class, expected_classes)),
        ("zero commands", env.commands == 0.0),
    )
    for label, matches in checks:
        if not _all_true(matches):
            raise RuntimeError("terrain overview {} changed".format(label))


def _like(target, values):
    if hasattr(target, "new_tensor"):
        return target.new_tensor(values)
    return values


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
