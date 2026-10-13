import ast
import importlib.util
from collections import Counter
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "legged_gym/legged_gym/envs/go2_amp/terrain.py"
SPEC = importlib.util.spec_from_file_location("go2_amp_terrain_helper", HELPER_PATH)
TERRAIN_HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TERRAIN_HELPER)


class TestAmpTerrainContract(unittest.TestCase):
    def test_exact_names_ids_proportions_and_columns(self):
        expected_names = (
            "flat", "rough_flat", "smooth_slope", "rough_slope", "stairs",
            "discrete_obstacles", "hurdle", "gap", "pit", "narrow_passage",
        )
        expected_counts = [2, 2, 2, 2, 3, 2, 2, 2, 2, 1]

        self.assertEqual(TERRAIN_HELPER.TERRAIN_NAMES, expected_names)
        self.assertEqual(
            TERRAIN_HELPER.TERRAIN_CLASS_IDS,
            {name: index for index, name in enumerate(expected_names)},
        )
        self.assertEqual(tuple(TERRAIN_HELPER.TERRAIN_PROPORTIONS),
                         (0.10, 0.10, 0.10, 0.10, 0.15,
                          0.10, 0.10, 0.10, 0.10, 0.05))
        self.assertAlmostEqual(sum(TERRAIN_HELPER.TERRAIN_PROPORTIONS), 1.0)
        counts = Counter(TERRAIN_HELPER.TERRAIN_COLUMNS)
        self.assertEqual(len(TERRAIN_HELPER.TERRAIN_COLUMNS), 20)
        self.assertEqual([counts[name] for name in expected_names], expected_counts)

    def test_difficulty_spans_closed_unit_interval(self):
        self.assertEqual(TERRAIN_HELPER.difficulty_for_level(0), 0.0)
        self.assertEqual(TERRAIN_HELPER.difficulty_for_level(19), 1.0)
        self.assertAlmostEqual(TERRAIN_HELPER.difficulty_for_level(9), 9 / 19)
        with self.assertRaises(ValueError):
            TERRAIN_HELPER.difficulty_for_level(-1)
        with self.assertRaises(ValueError):
            TERRAIN_HELPER.difficulty_for_level(20)


class TestAmpTerrainCurriculum(unittest.TestCase):
    def test_only_forward_commands_change_level(self):
        levels = np.asarray([3, 3, 3, 3, 19])
        forward = np.asarray([True, True, False, False, True])
        progress = np.asarray([5.0, 0.1, 5.0, 0.1, 5.0])
        early_failure = np.asarray([False, True, False, True, False])
        random_lower = np.asarray([0, 0, 0, 0, 7])

        updated = TERRAIN_HELPER.next_curriculum_levels(
            levels, forward, progress, success_distance=4.0,
            early_failure=early_failure, random_lower_levels=random_lower,
        )

        np.testing.assert_array_equal(updated, [4, 2, 3, 3, 7])
        np.testing.assert_array_equal(levels, [3, 3, 3, 3, 19])

    def test_failed_level_zero_is_clamped(self):
        updated = TERRAIN_HELPER.next_curriculum_levels(
            np.asarray([0]), np.asarray([True]), np.asarray([0.0]), 4.0,
            np.asarray([True]), np.asarray([0]),
        )
        np.testing.assert_array_equal(updated, [0])


class _FakeTerrainUtils:
    def __init__(self):
        self.calls = []

    def pyramid_sloped_terrain(self, terrain, **kwargs):
        self.calls.append(("slope", kwargs))

    def random_uniform_terrain(self, terrain, **kwargs):
        self.calls.append(("roughness", kwargs))

    def pyramid_stairs_terrain(self, terrain, **kwargs):
        self.calls.append(("stairs", kwargs))

    def discrete_obstacles_terrain(self, terrain, *args, **kwargs):
        self.calls.append(("obstacles", {"args": args, **kwargs}))


def _load_named_builder():
    path = ROOT / "legged_gym/legged_gym/utils/new_terrains/add_mix_terrain.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    wanted = {"GO2_AMP_TERRAIN_CLASS_IDS", "trimesh_terrain_by_name"}
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in wanted
                for target in node.targets):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name == "trimesh_terrain_by_name":
            nodes.append(node)
    fake = _FakeTerrainUtils()
    helper_calls = []

    def helper(name):
        def record(_terrain, *args, **kwargs):
            helper_calls.append((name, args, kwargs))
        return record

    namespace = {
        "terrain_utils": fake,
        "parkour_hurdle_terrain": helper("hurdle"),
        "gap_terrain": helper("gap"),
        "pit_terrain": helper("pit"),
        "narrow_passage_terrain": helper("narrow"),
        "clear_spawn_platform": helper("spawn"),
    }
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["trimesh_terrain_by_name"], fake, helper_calls


class TestAmpTerrainConstruction(unittest.TestCase):
    class Terrain:
        horizontal_scale = 0.1
        vertical_scale = 0.01

    def test_every_named_builder_sets_stable_id_and_safe_spawn(self):
        builder, _fake, helper_calls = _load_named_builder()
        for expected_id, name in enumerate(TERRAIN_HELPER.TERRAIN_NAMES):
            helper_calls.clear()
            terrain = self.Terrain()
            builder(terrain, name, 0.5, lambda _terrain: None, 20)
            self.assertEqual(terrain.idx, expected_id)
            self.assertEqual(helper_calls[-1][0], "spawn")
        with self.assertRaisesRegex(ValueError, "unknown Go2 AMP terrain"):
            builder(self.Terrain(), "crawl_tunnel", 0.5, lambda _terrain: None, 20)

    def test_geometry_gets_harder_monotonically(self):
        builder, fake, helper_calls = _load_named_builder()

        def calls_at(name, difficulty):
            fake.calls.clear()
            helper_calls.clear()
            builder(self.Terrain(), name, difficulty, lambda _terrain: None, 20)
            return list(fake.calls), list(helper_calls)

        low, _ = calls_at("smooth_slope", 0.0)
        high, _ = calls_at("smooth_slope", 1.0)
        self.assertLess(low[0][1]["slope"], high[0][1]["slope"])

        low, _ = calls_at("stairs", 0.0)
        high, _ = calls_at("stairs", 1.0)
        self.assertLess(low[0][1]["step_height"], high[0][1]["step_height"])

        for name, helper_name, key, direction in (
            ("hurdle", "hurdle", "hurdle_height", 1),
            ("gap", "gap", "gap_size", 1),
            ("pit", "pit", "depth", 1),
            ("narrow_passage", "narrow", "passage_width", -1),
        ):
            _, low_helpers = calls_at(name, 0.0)
            _, high_helpers = calls_at(name, 1.0)
            low_value = next(item[2][key] for item in low_helpers if item[0] == helper_name)
            high_value = next(item[2][key] for item in high_helpers if item[0] == helper_name)
            self.assertGreater(direction * high_value, direction * low_value)

    def test_explicit_curriculum_uses_closed_interval_without_changing_legacy(self):
        source = (ROOT / "legged_gym/legged_gym/utils/terrain.py").read_text(encoding="utf-8")
        self.assertIn("difficulty = i / (self.cfg.num_rows - 1)", source)
        self.assertIn("difficulty = i / self.cfg.num_rows", source)
        self.assertIn("self.make_terrain_by_name(explicit_columns[j], difficulty)", source)
        self.assertIn("self.make_terrain(choice, difficulty)", source)


if __name__ == "__main__":
    unittest.main()
