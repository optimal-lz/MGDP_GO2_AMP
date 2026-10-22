import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = (
    ROOT / "legged_gym/legged_gym/utils/new_terrains/go2_amp_terrain.py")


def load_generator():
    if not GENERATOR.is_file():
        raise AssertionError("focused Go2 AMP terrain generator is missing")
    spec = importlib.util.spec_from_file_location(
        "go2_amp_heightfield_generator", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tile:
    horizontal_scale = 0.1
    vertical_scale = 0.005

    def __init__(self):
        self.height_field_raw = np.zeros((80, 80), dtype=np.int16)


class TestGo2AmpTerrainGeometry(unittest.TestCase):
    def build(self, name, difficulty, seed=11):
        self.assertTrue(
            GENERATOR.is_file(),
            "focused Go2 AMP terrain generator is missing")
        generator = load_generator()
        tile = Tile()
        generator.build_go2_amp_terrain(
            tile, name, difficulty, seed)
        self.assertEqual(tile.height_field_raw.shape, (80, 80))
        return tile

    def metres(self, tile):
        return tile.height_field_raw.astype(np.float64) * tile.vertical_scale

    def test_slope_has_fixed_sections_and_height_continuity(self):
        for difficulty, expected_top in ((0.0, 0.105), (1.0, 0.630)):
            height = self.metres(self.build("slope_course", difficulty))[:, 40]
            self.assertTrue(np.all(height[:13] == 0.0))
            self.assertAlmostEqual(float(height[33]), expected_top, delta=0.005)
            np.testing.assert_allclose(height[34:46], expected_top, atol=0.005)
            self.assertEqual(float(height[66]), 0.0)
            self.assertTrue(np.all(height[67:] == 0.0))
            self.assertLessEqual(float(np.max(np.abs(np.diff(height)))), 0.035)

    def test_stairs_have_seven_fixed_treads_and_inherited_top(self):
        for difficulty, step_height in ((0.0, 0.05), (1.0, 0.14)):
            height = self.metres(self.build("stairs_course", difficulty))[:, 40]
            ascent = height[13:34]
            for step in range(7):
                np.testing.assert_allclose(
                    ascent[step * 3:(step + 1) * 3],
                    (step + 1) * step_height, atol=0.005)
            np.testing.assert_allclose(
                height[34:46], 7 * step_height, atol=0.005)
            self.assertTrue(np.all(height[67:] == 0.0))

    def test_low_obstacles_are_three_transverse_nonoverlapping_bars(self):
        for difficulty, expected_height in ((0.0, 0.04), (1.0, 0.16)):
            height = self.metres(
                self.build("low_obstacle_course", difficulty))
            active_rows = np.flatnonzero(np.any(height > 0.0, axis=1))
            np.testing.assert_array_equal(
                active_rows, [19, 20, 21, 39, 40, 41, 59, 60, 61])
            np.testing.assert_allclose(
                height[active_rows], expected_height, atol=0.005)

    def test_gap_course_has_two_bounded_transverse_gaps(self):
        for difficulty, expected_cells in ((0.0, 1), (1.0, 4)):
            height = self.metres(self.build("gap_course", difficulty))
            gap_rows = np.flatnonzero(np.any(height < 0.0, axis=1))
            groups = np.split(gap_rows, np.where(np.diff(gap_rows) > 1)[0] + 1)
            self.assertEqual([len(group) for group in groups],
                             [expected_cells, expected_cells])
            self.assertAlmostEqual(float(height[gap_rows].min()), -1.0)
            self.assertGreater(groups[1][0] - groups[0][-1], 20)

    def test_narrow_passage_is_open_then_funnel_corridor_and_exit(self):
        for difficulty, expected_width in ((0.0, 1.5), (1.0, 1.0)):
            height = self.metres(self.build("narrow_passage", difficulty))
            self.assertTrue(np.all(height[:13] == 0.0))
            self.assertTrue(np.all(height[67:] == 0.0))
            corridor = height[20:60]
            self.assertAlmostEqual(float(corridor.max()), 0.35)
            open_cells = np.sum(corridor[0] == 0.0)
            self.assertAlmostEqual(open_cells * 0.1, expected_width, delta=0.1)
            wall_counts = np.sum(height[13:20] > 0.0, axis=1)
            self.assertTrue(np.all(np.diff(wall_counts) >= 0))
            exit_counts = np.sum(height[60:67] > 0.0, axis=1)
            self.assertTrue(np.all(np.diff(exit_counts) <= 0))

    def test_rough_ground_is_seeded_bounded_and_tapered(self):
        first = self.build("rough_ground", 1.0, seed=17)
        same = self.build("rough_ground", 1.0, seed=17)
        other = self.build("rough_ground", 1.0, seed=18)
        np.testing.assert_array_equal(
            first.height_field_raw, same.height_field_raw)
        self.assertFalse(np.array_equal(
            first.height_field_raw, other.height_field_raw))
        height = self.metres(first)
        self.assertTrue(np.all(height[:13] == 0.0))
        self.assertTrue(np.all(height[67:] == 0.0))
        self.assertLessEqual(float(np.max(np.abs(height))), 0.055)
        self.assertLess(abs(float(np.mean(height[13:67]))), 0.005)
        self.assertLessEqual(float(np.max(np.abs(np.diff(height, axis=0)))),
                             0.025)

    def test_names_ids_spawn_origins_and_unknown_name(self):
        expected = (
            "flat", "rough_ground", "slope_course", "stairs_course",
            "low_obstacle_course", "gap_course", "narrow_passage")
        for index, name in enumerate(expected):
            tile = self.build(name, 0.5)
            self.assertEqual(tile.idx, index)
            expected_x = 4.0 if name == "flat" else 0.65
            self.assertAlmostEqual(tile.spawn_origin[0], expected_x)
            self.assertAlmostEqual(tile.spawn_origin[1], 4.0)
            self.assertAlmostEqual(tile.spawn_origin[2], 0.0)
        with self.assertRaisesRegex(ValueError, "unknown Go2 AMP terrain"):
            self.build("pit", 0.5)


if __name__ == "__main__":
    unittest.main()
