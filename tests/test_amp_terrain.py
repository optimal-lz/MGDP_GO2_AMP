import ast
from collections import Counter
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legged_gym"))

HELPER_PATH = ROOT / "legged_gym/legged_gym/envs/go2_amp/terrain.py"
if not HELPER_PATH.is_file():
    raise ImportError(f"missing pure Go2 AMP terrain helper: {HELPER_PATH}")
SPEC = importlib.util.spec_from_file_location("go2_amp_terrain_helper", HELPER_PATH)
TERRAIN_HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TERRAIN_HELPER)
GO2_AMP_COURSE_COLUMN_CYCLE = TERRAIN_HELPER.GO2_AMP_COURSE_COLUMN_CYCLE
GO2_AMP_TERRAIN_COLUMNS = TERRAIN_HELPER.GO2_AMP_TERRAIN_COLUMNS
assign_amp_columns = TERRAIN_HELPER.assign_amp_columns
compute_anchor_count = TERRAIN_HELPER.compute_anchor_count


class TestAmpTerrainAllocation(unittest.TestCase):
    def test_anchor_counts_and_smallest_supported_split(self):
        self.assertEqual(compute_anchor_count(2), 1)
        self.assertEqual(compute_anchor_count(64), 10)
        self.assertEqual(compute_anchor_count(4096), 614)
        for invalid in (-1, 0, 1):
            with self.assertRaisesRegex(ValueError, "at least 2"):
                compute_anchor_count(invalid)

    def test_column_contract_and_prefix_allocation(self):
        self.assertEqual(GO2_AMP_TERRAIN_COLUMNS, (
            "slope down",
            "slope up",
            "rough pyramid",
            "stairs down",
            "stairs up",
            "discrete obstacles",
        ))
        self.assertEqual(GO2_AMP_COURSE_COLUMN_CYCLE, (0, 1, 2, 2, 3, 3, 4, 4, 5, 5))

        columns, anchors = assign_amp_columns(12)
        self.assertEqual(columns.tolist(), [0, 0, 0, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        self.assertEqual(
            anchors.tolist(),
            [True, True, False, False, False, False, False, False, False, False, False, False],
        )

        columns, anchors = assign_amp_columns(8)
        self.assertEqual(columns.tolist(), [0, 0, 1, 2, 2, 3, 3, 4])
        self.assertEqual(anchors.tolist(), [True, False, False, False, False, False, False, False])
        self.assertEqual(columns[0], columns[1])
        self.assertNotEqual(bool(anchors[0]), bool(anchors[1]))

        forbidden = {"flat", "hurdle", "gap", "ramp", "bream", "new stairs down", "pit"}
        self.assertTrue(forbidden.isdisjoint(GO2_AMP_TERRAIN_COLUMNS))

    def test_every_course_block_and_partial_block_follow_the_fixed_cycle(self):
        for num_envs in range(2, 80):
            columns, anchors = assign_amp_columns(num_envs)
            self.assertTrue(np.all(columns[anchors] == 0))
            course = columns[~anchors]
            for start in range(0, len(course), len(GO2_AMP_COURSE_COLUMN_CYCLE)):
                block = course[start:start + len(GO2_AMP_COURSE_COLUMN_CYCLE)]
                self.assertEqual(
                    block.tolist(),
                    list(GO2_AMP_COURSE_COLUMN_CYCLE[:len(block)]),
                )

        counts = Counter(GO2_AMP_COURSE_COLUMN_CYCLE)
        self.assertEqual([counts[index] for index in range(6)], [1, 1, 2, 2, 2, 2])

    def test_4096_split_counts_anchor_and_course_zero_slopes(self):
        columns, anchors = assign_amp_columns(4096)
        self.assertEqual(int(anchors.sum()), 614)
        self.assertEqual(int((~anchors).sum()), 3482)
        course_zero_slopes = int(np.isin(columns[~anchors], [0, 1]).sum())
        self.assertEqual(course_zero_slopes, 698)
        self.assertEqual(int(anchors.sum()) + course_zero_slopes, 1312)
        self.assertEqual(1312 / 4096, 0.3203125)

        no_anchor = np.resize(np.asarray(GO2_AMP_COURSE_COLUMN_CYCLE), 4096)
        self.assertEqual(int(np.isin(no_anchor, [0, 1]).sum()), 820)


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
    mapping = next(
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "GO2_AMP_TERRAIN_CLASS_IDS"
                for target in node.targets)
    )
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "trimesh_terrain_by_name"
    )
    fake = _FakeTerrainUtils()
    namespace = {"terrain_utils": fake}
    module = ast.Module(body=[mapping, function], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["trimesh_terrain_by_name"], namespace["GO2_AMP_TERRAIN_CLASS_IDS"], fake


class TestAmpTerrainConstruction(unittest.TestCase):
    def test_named_builder_uses_six_existing_primitives_and_stable_class_ids(self):
        builder, class_ids, fake = _load_named_builder()
        self.assertEqual(class_ids, {
            "slope down": 0,
            "slope up": 0,
            "rough pyramid": 1,
            "stairs down": 2,
            "stairs up": 3,
            "discrete obstacles": 4,
        })
        self.assertNotIn("flat", class_ids)

        class Terrain:
            horizontal_scale = 0.1

        for name, expected_id in class_ids.items():
            terrain = Terrain()
            builder(terrain, name, 0.5, lambda _terrain: None, 20)
            self.assertEqual(terrain.idx, expected_id)

        fake.calls.clear()
        builder(Terrain(), "slope down", 0.5, lambda _terrain: None, 20)
        self.assertEqual(fake.calls[0], ("slope", {"slope": -0.2, "platform_size": 3.0}))
        fake.calls.clear()
        builder(Terrain(), "slope up", 0.5, lambda _terrain: None, 20)
        self.assertEqual(fake.calls[0], ("slope", {"slope": 0.2, "platform_size": 3.0}))

        for name in ("slope down", "slope up"):
            fake.calls.clear()
            builder(Terrain(), name, 0.0, lambda _terrain: None, 20)
            self.assertEqual(fake.calls[0][1]["slope"], 0.0)

        with self.assertRaisesRegex(ValueError, "unknown Go2 AMP terrain"):
            builder(Terrain(), "flat", 0.0, lambda _terrain: None, 20)

    def test_opt_in_builder_and_environment_keep_legacy_curriculum_boundary(self):
        terrain_source = (
            ROOT / "legged_gym/legged_gym/utils/terrain.py"
        ).read_text(encoding="utf-8")
        self.assertIn("explicit_terrain_columns", terrain_source)
        self.assertIn("trimesh_terrain_by_name", terrain_source)
        self.assertIn("choice = j / self.cfg.num_cols + 0.001", terrain_source)
        self.assertIn("self.make_terrain(choice, difficulty)", terrain_source)

        config_source = (
            ROOT / "legged_gym/legged_gym/envs/go2_amp/config.py"
        ).read_text(encoding="utf-8")
        self.assertIn("num_cols = len(GO2_AMP_TERRAIN_COLUMNS)", config_source)
        self.assertIn("explicit_terrain_columns = GO2_AMP_TERRAIN_COLUMNS", config_source)

        env_path = ROOT / "legged_gym/legged_gym/envs/go2_amp/env.py"
        tree = ast.parse(env_path.read_text(encoding="utf-8"))
        class_node = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Go2AmpRandomDog"
        )
        methods = {
            node.name: ast.unparse(node)
            for node in class_node.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("set_amp_training_iteration", methods)
        origins = methods["_get_env_origins"]
        curriculum = methods["_update_terrain_curriculum"]
        self.assertIn("self.is_amp_anchor", origins)
        self.assertIn("assign_amp_columns", origins)
        self.assertIn("self.terrain_levels[anchor_ids] = 0", curriculum)
        self.assertIn("self.terrain_types[anchor_ids] = 0", curriculum)
        self.assertIn("super()._update_terrain_curriculum(course_ids)", curriculum)
        self.assertNotIn("terrain_types == 0", curriculum)
        self.assertNotIn("env_class ==", curriculum)


if __name__ == "__main__":
    unittest.main()
