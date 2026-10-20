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
    def test_overview_layout_is_ten_terrains_by_three_formal_levels(self):
        layout = TERRAIN_HELPER.terrain_overview_layout()

        self.assertEqual(len(layout), 30)
        self.assertEqual(
            [(item["terrain"], item["level"]) for item in layout[:3]],
            [("flat", 0), ("flat", 9), ("flat", 19)],
        )
        self.assertEqual(
            [(item["terrain"], item["level"]) for item in layout[-3:]],
            [("narrow_passage", 0),
             ("narrow_passage", 9),
             ("narrow_passage", 19)],
        )
        self.assertEqual(layout[12]["proportion"], 0.15)
        self.assertEqual(layout[-1]["proportion"], 0.05)

    def test_evaluation_grid_uses_explicit_curriculum_and_exact_preview_levels(self):
        class Config:
            curriculum = False
            num_rows = 20
            num_cols = 20
            explicit_terrain_columns = TERRAIN_HELPER.TERRAIN_COLUMNS

        cfg = Config()
        TERRAIN_HELPER.configure_evaluation_terrain_grid(
            cfg, TERRAIN_HELPER.TERRAIN_NAMES, levels=(0, 9, 19))

        self.assertTrue(cfg.curriculum)
        self.assertEqual(cfg.num_rows, 3)
        self.assertEqual(cfg.num_cols, 10)
        self.assertEqual(cfg.explicit_terrain_columns,
                         TERRAIN_HELPER.TERRAIN_NAMES)
        self.assertEqual(cfg.explicit_terrain_levels, (0, 9, 19))
        self.assertEqual(cfg.explicit_terrain_num_rows, 20)

    def test_freeze_case_sets_actual_terrain_level_origin_and_command(self):
        class TerrainConfig:
            num_rows = 20
            num_cols = len(TERRAIN_HELPER.TERRAIN_COLUMNS)
            explicit_terrain_columns = TERRAIN_HELPER.TERRAIN_COLUMNS

        class Config:
            terrain = TerrainConfig()

        class Environment:
            cfg = Config()
            num_envs = 2
            commands = np.zeros((2, 4), dtype=np.float32)
            terrain_types = np.zeros(2, dtype=np.int64)
            terrain_levels = np.zeros(2, dtype=np.int64)
            terrain_origins = np.zeros((20, 20, 3), dtype=np.float32)
            terrain_class = np.broadcast_to(np.asarray([
                TERRAIN_HELPER.TERRAIN_CLASS_IDS[name]
                for name in TERRAIN_HELPER.TERRAIN_COLUMNS
            ], dtype=np.float32), (20, 20)).copy()
            env_origins = np.zeros((2, 3), dtype=np.float32)
            env_class = np.zeros(2, dtype=np.float32)

        env = Environment()
        env.terrain_origins[9, 8] = [72.0, 4.0, 0.2]

        TERRAIN_HELPER.freeze_evaluation_case(
            env, "stairs", level=9, vx=0.8, yaw=0.0)

        np.testing.assert_array_equal(env.terrain_types, [8, 8])
        np.testing.assert_array_equal(env.terrain_levels, [9, 9])
        np.testing.assert_allclose(env.env_origins, [[72.0, 4.0, 0.2]] * 2)
        np.testing.assert_array_equal(env.env_class, [4, 4])
        np.testing.assert_allclose(env.commands, [[0.8, 0.0, 0.0, 0.0]] * 2)
        np.testing.assert_allclose(
            env._fixed_evaluation_commands,
            [[0.8, 0.0, 0.0, 0.0]] * 2)
        self.assertTrue(env._suppress_curriculum)
        TERRAIN_HELPER.assert_evaluation_case(
            env, "stairs", level=9, vx=0.8, yaw=0.0)

        env.commands[1, 0] = 0.4
        with self.assertRaisesRegex(RuntimeError, "command"):
            TERRAIN_HELPER.assert_evaluation_case(
                env, "stairs", level=9, vx=0.8, yaw=0.0)

        env.commands[1, 0] = 0.8
        env.commands[1, 3] = 1.0
        with self.assertRaisesRegex(RuntimeError, "extra"):
            TERRAIN_HELPER.assert_evaluation_case(
                env, "stairs", level=9, vx=0.8, yaw=0.0)

    def test_overview_freezes_every_named_tile_and_checks_generated_class(self):
        class TerrainConfig:
            num_rows = 3
            num_cols = 10
            explicit_terrain_columns = TERRAIN_HELPER.TERRAIN_NAMES
            explicit_terrain_levels = (0, 9, 19)

        class Config:
            terrain = TerrainConfig()

        class Environment:
            cfg = Config()
            num_envs = 30
            commands = np.ones((30, 4), dtype=np.float32)
            terrain_types = np.zeros(30, dtype=np.int64)
            terrain_levels = np.zeros(30, dtype=np.int64)
            terrain_origins = np.zeros((3, 10, 3), dtype=np.float32)
            terrain_class = np.broadcast_to(
                np.arange(10, dtype=np.float32), (3, 10)).copy()
            env_origins = np.zeros((30, 3), dtype=np.float32)
            env_class = np.zeros(30, dtype=np.float32)

        env = Environment()
        layout = TERRAIN_HELPER.terrain_overview_layout()

        TERRAIN_HELPER.freeze_terrain_overview(env, layout)

        np.testing.assert_array_equal(env.terrain_types[:6], [0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(env.terrain_levels[:6], [0, 1, 2, 0, 1, 2])
        np.testing.assert_array_equal(env.env_class[:6], [0, 0, 0, 1, 1, 1])
        np.testing.assert_allclose(env.commands, 0.0)
        TERRAIN_HELPER.assert_terrain_overview(env, layout)

        env.env_class[4] = 9
        with self.assertRaisesRegex(RuntimeError, "classes"):
            TERRAIN_HELPER.assert_terrain_overview(env, layout)

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
        self.assertIn("difficulty = level / (formal_num_rows - 1)", source)
        self.assertIn("difficulty = i / self.cfg.num_rows", source)
        self.assertIn("self.make_terrain_by_name(explicit_columns[j], difficulty)", source)
        self.assertIn("self.make_terrain(choice, difficulty)", source)

    def test_legacy_single_row_curriculum_remains_supported(self):
        path = ROOT / "legged_gym/legged_gym/utils/terrain.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        terrain_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Terrain")
        method = next(
            node for node in terrain_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "curiculum")
        minimal_class = ast.ClassDef(
            name="LegacyTerrain", bases=[], keywords=[],
            body=[method], decorator_list=[])
        module = ast.Module(body=[minimal_class], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {}
        exec(compile(module, str(path), "exec"), namespace)

        class Config:
            num_rows = 1
            num_cols = 1

        legacy = namespace["LegacyTerrain"]()
        legacy.cfg = Config()
        legacy.calls = []
        legacy.make_terrain = (
            lambda choice, difficulty: legacy.calls.append(
                (choice, difficulty)) or object())
        legacy.make_terrain_by_name = lambda name, difficulty: object()
        legacy.add_terrain_to_map = lambda terrain, row, col: None

        legacy.curiculum()

        self.assertEqual(legacy.calls, [(0.001, 0.0)])

    def test_preview_rows_use_formal_levels_instead_of_three_equal_steps(self):
        path = ROOT / "legged_gym/legged_gym/utils/terrain.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        terrain_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Terrain")
        method = next(
            node for node in terrain_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "curiculum")
        minimal_class = ast.ClassDef(
            name="PreviewTerrain", bases=[], keywords=[],
            body=[method], decorator_list=[])
        module = ast.Module(body=[minimal_class], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {}
        exec(compile(module, str(path), "exec"), namespace)

        class Config:
            num_rows = 3
            num_cols = 1
            explicit_terrain_columns = ("stairs",)
            explicit_terrain_levels = (0, 9, 19)
            explicit_terrain_num_rows = 20

        preview = namespace["PreviewTerrain"]()
        preview.cfg = Config()
        preview.calls = []
        preview.make_terrain_by_name = (
            lambda name, difficulty: preview.calls.append(
                (name, difficulty)) or object())
        preview.add_terrain_to_map = lambda terrain, row, col: None
        preview.make_terrain = lambda choice, difficulty: object()

        preview.curiculum()

        self.assertEqual([name for name, _ in preview.calls], ["stairs"] * 3)
        np.testing.assert_allclose(
            [difficulty for _, difficulty in preview.calls],
            [0.0, 9.0 / 19.0, 1.0],
        )


if __name__ == "__main__":
    unittest.main()
