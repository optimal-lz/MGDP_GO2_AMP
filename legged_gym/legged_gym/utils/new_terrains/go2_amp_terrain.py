"""Deterministic directional height fields for the formal Go2 AMP task."""

import numpy as np


TERRAIN_NAMES = (
    "flat",
    "rough_ground",
    "slope_course",
    "stairs_course",
    "low_obstacle_course",
    "gap_course",
    "narrow_passage",
)
TERRAIN_CLASS_IDS = {
    name: index for index, name in enumerate(TERRAIN_NAMES)
}


def _cells(distance, scale):
    return int(round(float(distance) / float(scale)))


def _height_units(height, scale):
    return int(round(float(height) / float(scale)))


def _write_profile(terrain, profile):
    terrain.height_field_raw[:, :] = profile[:, None]


def _smooth(field, passes=4):
    result = field
    for _ in range(passes):
        padded = np.pad(result, 1, mode="edge")
        result = sum(
            padded[dx:dx + result.shape[0], dy:dy + result.shape[1]]
            for dx in range(3) for dy in range(3)
        ) / 9.0
    return result


def _rough_ground(terrain, difficulty, rng):
    start = _cells(1.3, terrain.horizontal_scale)
    stop = terrain.height_field_raw.shape[0] - start
    amplitude = 0.005 + 0.045 * difficulty
    rough = rng.uniform(-1.0, 1.0, (stop - start,
                                   terrain.height_field_raw.shape[1]))
    rough = _smooth(rough)
    rough -= np.mean(rough)
    peak = np.max(np.abs(rough))
    if peak > 0.0:
        rough *= amplitude / peak
    transition = min(_cells(0.4, terrain.horizontal_scale), len(rough) // 2)
    taper = np.ones(len(rough), dtype=np.float64)
    if transition:
        edge = np.sin(np.linspace(0.0, np.pi / 2.0, transition)) ** 2
        taper[:transition] = edge
        taper[-transition:] = edge[::-1]
    rough *= taper[:, None]
    terrain.height_field_raw[start:stop] = np.rint(
        rough / terrain.vertical_scale).astype(np.int16)


def _slope_course(terrain, difficulty):
    start = _cells(1.3, terrain.horizontal_scale)
    ascent = _cells(2.1, terrain.horizontal_scale)
    platform = _cells(1.2, terrain.horizontal_scale)
    slope_ratio = 0.05 + 0.25 * difficulty
    top = _height_units(slope_ratio * 2.1, terrain.vertical_scale)
    profile = np.zeros(terrain.height_field_raw.shape[0], dtype=np.int16)
    profile[start:start + ascent] = np.rint(
        top * np.arange(1, ascent + 1) / ascent).astype(np.int16)
    top_start = start + ascent
    profile[top_start:top_start + platform] = top
    down_start = top_start + platform
    profile[down_start:down_start + ascent] = np.rint(
        top * (1.0 - np.arange(1, ascent + 1) / ascent)).astype(np.int16)
    _write_profile(terrain, profile)


def _stairs_course(terrain, difficulty):
    start = _cells(1.3, terrain.horizontal_scale)
    tread = _cells(0.30, terrain.horizontal_scale)
    platform = _cells(1.2, terrain.horizontal_scale)
    step = _height_units(0.05 + 0.09 * difficulty,
                         terrain.vertical_scale)
    profile = np.zeros(terrain.height_field_raw.shape[0], dtype=np.int16)
    for index in range(7):
        begin = start + index * tread
        profile[begin:begin + tread] = (index + 1) * step
    top_start = start + 7 * tread
    profile[top_start:top_start + platform] = 7 * step
    down_start = top_start + platform
    for index in range(7):
        begin = down_start + index * tread
        profile[begin:begin + tread] = (6 - index) * step
    _write_profile(terrain, profile)


def _low_obstacle_course(terrain, difficulty):
    width = _cells(0.30, terrain.horizontal_scale)
    height = _height_units(0.04 + 0.12 * difficulty,
                           terrain.vertical_scale)
    for position in (2.0, 4.0, 6.0):
        center = _cells(position, terrain.horizontal_scale)
        start = center - width // 2
        terrain.height_field_raw[start:start + width, :] = height


def _gap_course(terrain, difficulty):
    width = max(1, int(np.ceil(
        (0.10 + 0.25 * difficulty) / terrain.horizontal_scale)))
    depth = -_height_units(1.0, terrain.vertical_scale)
    for position in (2.6, 5.4):
        center = _cells(position, terrain.horizontal_scale)
        start = center - width // 2
        terrain.height_field_raw[start:start + width, :] = depth


def _narrow_passage(terrain, difficulty):
    rows, columns = terrain.height_field_raw.shape
    center = columns // 2
    target_width = 1.50 - 0.50 * difficulty
    target_cells = max(1, _cells(target_width, terrain.horizontal_scale))
    wall = _height_units(0.35, terrain.vertical_scale)
    open_end = _cells(1.3, terrain.horizontal_scale)
    funnel = _cells(0.7, terrain.horizontal_scale)
    corridor_end = open_end + funnel + _cells(
        4.0, terrain.horizontal_scale)

    def walls_at(row, open_width):
        start = center - open_width // 2
        stop = start + open_width
        terrain.height_field_raw[row, :start] = wall
        terrain.height_field_raw[row, stop:] = wall

    for offset, row in enumerate(range(open_end, open_end + funnel)):
        fraction = float(offset + 1) / float(funnel)
        width = int(round(columns * (1.0 - fraction)
                          + target_cells * fraction))
        walls_at(row, width)
    for row in range(open_end + funnel, corridor_end):
        walls_at(row, target_cells)
    for offset, row in enumerate(range(corridor_end,
                                       corridor_end + funnel)):
        fraction = float(offset + 1) / float(funnel)
        width = int(round(target_cells * (1.0 - fraction)
                          + columns * fraction))
        walls_at(row, width)


def build_go2_amp_terrain(terrain, terrain_name, difficulty, seed):
    """Fill one 8 m directional tile and attach its local spawn origin."""
    if terrain_name not in TERRAIN_CLASS_IDS:
        raise ValueError("unknown Go2 AMP terrain: {}".format(terrain_name))
    difficulty = min(1.0, max(0.0, float(difficulty)))
    terrain.height_field_raw.fill(0)
    rng = np.random.RandomState(int(seed))

    if terrain_name == "rough_ground":
        _rough_ground(terrain, difficulty, rng)
    elif terrain_name == "slope_course":
        _slope_course(terrain, difficulty)
    elif terrain_name == "stairs_course":
        _stairs_course(terrain, difficulty)
    elif terrain_name == "low_obstacle_course":
        _low_obstacle_course(terrain, difficulty)
    elif terrain_name == "gap_course":
        _gap_course(terrain, difficulty)
    elif terrain_name == "narrow_passage":
        _narrow_passage(terrain, difficulty)

    terrain.idx = TERRAIN_CLASS_IDS[terrain_name]
    spawn_x = 4.0 if terrain_name == "flat" else 0.65
    terrain.spawn_origin = (spawn_x, 4.0, 0.0)
