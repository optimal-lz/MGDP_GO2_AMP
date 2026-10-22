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
    def test_overview_layout_is_seven_terrains_by_four_formal_levels(self):
        layout = TERRAIN_HELPER.terrain_overview_layout()

        self.assertEqual(len(layout), 28)
        self.assertEqual(
            [(item["terrain"], item["level"]) for item in layout[:4]],
            [("flat", 0), ("flat", 3), ("flat", 6), ("flat", 9)],
        )
        self.assertEqual(
            [(item["terrain"], item["level"]) for item in layout[-4:]],
            [("narrow_passage", 0),
             ("narrow_passage", 3),
             ("narrow_passage", 6),
             ("narrow_passage", 9)],
        )
        self.assertEqual(layout[12]["proportion"], 0.20)
        self.assertEqual(layout[-1]["proportion"], 0.05)

    def test_evaluation_grid_uses_explicit_curriculum_and_exact_preview_levels(self):
        class Config:
            curriculum = False
            num_rows = 10
            num_cols = 20
            explicit_terrain_columns = TERRAIN_HELPER.TERRAIN_COLUMNS

        cfg = Config()
        TERRAIN_HELPER.configure_evaluation_terrain_grid(
            cfg, TERRAIN_HELPER.TERRAIN_NAMES, levels=(0, 3, 6, 9))

        self.assertTrue(cfg.curriculum)
        self.assertEqual(cfg.num_rows, 4)
        self.assertEqual(cfg.num_cols, 7)
        self.assertEqual(cfg.explicit_terrain_columns,
                         TERRAIN_HELPER.TERRAIN_NAMES)
        self.assertEqual(cfg.explicit_terrain_levels, (0, 3, 6, 9))
        self.assertEqual(cfg.explicit_terrain_num_rows, 10)

    def test_freeze_case_sets_actual_terrain_level_origin_and_command(self):
        class TerrainConfig:
            num_rows = 10
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
            terrain_origins = np.zeros((10, 20, 3), dtype=np.float32)
            terrain_class = np.broadcast_to(np.asarray([
                TERRAIN_HELPER.TERRAIN_CLASS_IDS[name]
                for name in TERRAIN_HELPER.TERRAIN_COLUMNS
            ], dtype=np.float32), (10, 20)).copy()
            env_origins = np.zeros((2, 3), dtype=np.float32)
            env_class = np.zeros(2, dtype=np.float32)

        env = Environment()
        stairs_column = TERRAIN_HELPER.TERRAIN_COLUMNS.index("stairs_course")
        env.terrain_origins[9, stairs_column] = [72.0, 4.0, 0.2]

        TERRAIN_HELPER.freeze_evaluation_case(
            env, "stairs_course", level=9, vx=0.8, yaw=0.0)

        np.testing.assert_array_equal(
            env.terrain_types, [stairs_column, stairs_column])
        np.testing.assert_array_equal(env.terrain_levels, [9, 9])
        np.testing.assert_allclose(env.env_origins, [[72.0, 4.0, 0.2]] * 2)
        np.testing.assert_array_equal(
            env.env_class,
            [TERRAIN_HELPER.TERRAIN_CLASS_IDS["stairs_course"]] * 2)
        np.testing.assert_allclose(env.commands, [[0.8, 0.0, 0.0, 0.0]] * 2)
        np.testing.assert_allclose(
            env._fixed_evaluation_commands,
            [[0.8, 0.0, 0.0, 0.0]] * 2)
        self.assertTrue(env._suppress_curriculum)
        TERRAIN_HELPER.assert_evaluation_case(
            env, "stairs_course", level=9, vx=0.8, yaw=0.0)

        env.commands[1, 0] = 0.4
        with self.assertRaisesRegex(RuntimeError, "command"):
            TERRAIN_HELPER.assert_evaluation_case(
                env, "stairs_course", level=9, vx=0.8, yaw=0.0)

        env.commands[1, 0] = 0.8
        env.commands[1, 3] = 1.0
        with self.assertRaisesRegex(RuntimeError, "extra"):
            TERRAIN_HELPER.assert_evaluation_case(
                env, "stairs_course", level=9, vx=0.8, yaw=0.0)

    def test_overview_freezes_every_named_tile_and_checks_generated_class(self):
        class TerrainConfig:
            num_rows = 4
            num_cols = 7
            explicit_terrain_columns = TERRAIN_HELPER.TERRAIN_NAMES
            explicit_terrain_levels = (0, 3, 6, 9)

        class Config:
            terrain = TerrainConfig()

        class Environment:
            cfg = Config()
            num_envs = 28
            commands = np.ones((28, 4), dtype=np.float32)
            terrain_types = np.zeros(28, dtype=np.int64)
            terrain_levels = np.zeros(28, dtype=np.int64)
            terrain_origins = np.zeros((4, 7, 3), dtype=np.float32)
            terrain_class = np.broadcast_to(
                np.arange(7, dtype=np.float32), (4, 7)).copy()
            env_origins = np.zeros((28, 3), dtype=np.float32)
            env_class = np.zeros(28, dtype=np.float32)

        env = Environment()
        layout = TERRAIN_HELPER.terrain_overview_layout()

        TERRAIN_HELPER.freeze_terrain_overview(env, layout)

        np.testing.assert_array_equal(
            env.terrain_types[:8], [0, 0, 0, 0, 1, 1, 1, 1])
        np.testing.assert_array_equal(
            env.terrain_levels[:8], [0, 1, 2, 3, 0, 1, 2, 3])
        np.testing.assert_array_equal(
            env.env_class[:8], [0, 0, 0, 0, 1, 1, 1, 1])
        np.testing.assert_allclose(env.commands, 0.0)
        TERRAIN_HELPER.assert_terrain_overview(env, layout)

        env.env_class[4] = 9
        with self.assertRaisesRegex(RuntimeError, "classes"):
            TERRAIN_HELPER.assert_terrain_overview(env, layout)

    def test_exact_names_ids_proportions_and_columns(self):
        expected_names = (
            "flat", "rough_ground", "slope_course", "stairs_course",
            "low_obstacle_course", "gap_course", "narrow_passage",
        )
        expected_counts = [3, 3, 4, 4, 3, 2, 1]

        self.assertEqual(TERRAIN_HELPER.TERRAIN_NAMES, expected_names)
        self.assertEqual(
            TERRAIN_HELPER.TERRAIN_CLASS_IDS,
            {name: index for index, name in enumerate(expected_names)},
        )
        self.assertEqual(tuple(TERRAIN_HELPER.TERRAIN_PROPORTIONS),
                         (0.15, 0.15, 0.20, 0.20, 0.15, 0.10, 0.05))
        self.assertAlmostEqual(sum(TERRAIN_HELPER.TERRAIN_PROPORTIONS), 1.0)
        counts = Counter(TERRAIN_HELPER.TERRAIN_COLUMNS)
        self.assertEqual(len(TERRAIN_HELPER.TERRAIN_COLUMNS), 20)
        self.assertEqual([counts[name] for name in expected_names], expected_counts)

    def test_difficulty_spans_closed_unit_interval(self):
        self.assertEqual(TERRAIN_HELPER.difficulty_for_level(0), 0.0)
        self.assertEqual(TERRAIN_HELPER.difficulty_for_level(9), 1.0)
        self.assertAlmostEqual(TERRAIN_HELPER.difficulty_for_level(3), 1 / 3)
        with self.assertRaises(ValueError):
            TERRAIN_HELPER.difficulty_for_level(-1)
        with self.assertRaises(ValueError):
            TERRAIN_HELPER.difficulty_for_level(10)

    def test_column_assignment_is_bounded_for_nondivisible_env_counts(self):
        for num_envs in (1, 7, 19, 21, 63, 4095, 4096):
            columns = TERRAIN_HELPER.terrain_column_indices(num_envs, 20)
            self.assertEqual(len(columns), num_envs)
            self.assertGreaterEqual(min(columns), 0)
            self.assertLessEqual(max(columns), 19)
        with self.assertRaises(ValueError):
            TERRAIN_HELPER.terrain_column_indices(0, 20)


class TestAmpTerrainCurriculum(unittest.TestCase):
    def test_only_forward_commands_change_level(self):
        levels = np.asarray([3, 3, 3, 3, 9])
        forward = np.asarray([True, True, False, False, True])
        progress = np.asarray([5.0, 0.1, 5.0, 0.1, 5.0])
        early_failure = np.asarray([False, True, False, True, False])
        random_lower = np.asarray([0, 0, 0, 0, 7])

        updated = TERRAIN_HELPER.next_curriculum_levels(
            levels, forward, progress, success_distance=4.0,
            early_failure=early_failure, random_lower_levels=random_lower,
            max_level=9,
        )

        np.testing.assert_array_equal(updated, [4, 2, 3, 3, 7])
        np.testing.assert_array_equal(levels, [3, 3, 3, 3, 9])

    def test_failed_level_zero_is_clamped(self):
        updated = TERRAIN_HELPER.next_curriculum_levels(
            np.asarray([0]), np.asarray([True]), np.asarray([0.0]), 4.0,
            np.asarray([True]), np.asarray([0]),
        )
        np.testing.assert_array_equal(updated, [0])


class TestAmpTerrainConstruction(unittest.TestCase):
    @staticmethod
    def load_add_terrain_to_map():
        path = ROOT / "legged_gym/legged_gym/utils/terrain.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        terrain_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Terrain")
        method = next(
            node for node in terrain_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "add_terrain_to_map")
        minimal_class = ast.ClassDef(
            name="TerrainOriginHarness", bases=[], keywords=[],
            body=[method], decorator_list=[])
        module = ast.Module(body=[minimal_class], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {"np": np}
        exec(compile(module, str(path), "exec"), namespace)
        return namespace["TerrainOriginHarness"]

    def test_named_spawn_metadata_translates_without_changing_legacy_mix(self):
        harness_type = self.load_add_terrain_to_map()
        harness = harness_type()
        harness.type = "mix"
        harness.border = 0
        harness.length_per_env_pixels = 80
        harness.width_per_env_pixels = 80
        harness.env_length = 8.0
        harness.env_width = 8.0
        harness.height_field_raw = np.zeros((160, 160), dtype=np.int16)
        harness.env_origins = np.zeros((2, 2, 3), dtype=np.float64)
        harness.terrain_type = np.zeros((2, 2), dtype=np.float64)

        class Tile:
            horizontal_scale = 0.1
            vertical_scale = 0.005

            def __init__(self, spawn=None, idx=4):
                self.height_field_raw = np.zeros((80, 80), dtype=np.int16)
                self.idx = idx
                if spawn is not None:
                    self.spawn_origin = spawn

        named = Tile((0.65, 4.0, 0.0), idx=6)
        harness.add_terrain_to_map(named, 1, 1)
        np.testing.assert_allclose(harness.env_origins[1, 1],
                                   [8.65, 12.0, 0.0])
        self.assertEqual(harness.terrain_type[1, 1], 6)

        legacy = Tile(idx=3)
        legacy.height_field_raw[30:50, 30:50] = 20
        harness.add_terrain_to_map(legacy, 0, 0)
        np.testing.assert_allclose(harness.env_origins[0, 0],
                                   [4.0, 4.0, 0.1])
        self.assertEqual(harness.terrain_type[0, 0], 3)

    def test_explicit_curriculum_uses_closed_interval_without_changing_legacy(self):
        source = (ROOT / "legged_gym/legged_gym/utils/terrain.py").read_text(encoding="utf-8")
        self.assertIn("difficulty = level / (formal_num_rows - 1)", source)
        self.assertIn("difficulty = i / self.cfg.num_rows", source)
        self.assertIn("self.make_terrain_by_name(", source)
        self.assertIn("seed=seed", source)
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
        legacy.make_terrain_by_name = lambda name, difficulty, seed=0: object()
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
            num_rows = 4
            num_cols = 1
            explicit_terrain_columns = ("stairs_course",)
            explicit_terrain_levels = (0, 3, 6, 9)
            explicit_terrain_num_rows = 10

        preview = namespace["PreviewTerrain"]()
        preview.cfg = Config()
        preview.calls = []
        preview.make_terrain_by_name = (
            lambda name, difficulty, seed=0: preview.calls.append(
                (name, difficulty)) or object())
        preview.add_terrain_to_map = lambda terrain, row, col: None
        preview.make_terrain = lambda choice, difficulty: object()

        preview.curiculum()

        self.assertEqual(
            [name for name, _ in preview.calls], ["stairs_course"] * 4)
        np.testing.assert_allclose(
            [difficulty for _, difficulty in preview.calls],
            [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0],
        )


if __name__ == "__main__":
    unittest.main()
