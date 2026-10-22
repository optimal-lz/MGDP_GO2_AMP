import ast
import importlib.util
import io
from pathlib import Path
import unittest
from contextlib import redirect_stdout


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "legged_gym/scripts/view_go2_amp_terrains.py"
TERRAIN_HELPER_PATH = (
    ROOT / "legged_gym/legged_gym/envs/go2_amp/terrain.py")


class TestGo2AmpTerrainOverview(unittest.TestCase):
    def test_overview_is_checkpoint_free_single_viewer_with_fixed_layout(self):
        self.assertTrue(SCRIPT.is_file())
        source = SCRIPT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }

        self.assertIn("main", functions)
        self.assertIn("terrain_overview_layout", source)
        self.assertIn("configure_evaluation_terrain_grid", source)
        self.assertIn("freeze_terrain_overview", source)
        self.assertIn("TERRAIN_NAMES", source)
        self.assertIn("TERRAIN_PROPORTIONS", source)
        self.assertNotIn("load_pair", source)
        self.assertNotIn("--run_dir", source)
        self.assertNotIn("--manifest", source)

    def test_overview_layout_and_console_have_28_named_tiles(self):
        spec = importlib.util.spec_from_file_location(
            "overview_terrain_helper", TERRAIN_HELPER_PATH)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        self.assertEqual(len(helper.terrain_overview_layout()), 28)
        self.assertEqual(helper.TERRAIN_OVERVIEW_LEVELS, (0, 3, 6, 9))

        source = SCRIPT.read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef)
                        and node.name == "_print_layout")
        module = ast.Module(body=[function], type_ignores=[])
        ast.fix_missing_locations(module)
        namespace = {
            "TERRAIN_NAMES": helper.TERRAIN_NAMES,
            "TERRAIN_PROPORTIONS": helper.TERRAIN_PROPORTIONS,
            "TERRAIN_OVERVIEW_LEVELS": helper.TERRAIN_OVERVIEW_LEVELS,
        }
        exec(compile(module, str(SCRIPT), "exec"), namespace)
        output = io.StringIO()
        with redirect_stdout(output):
            namespace["_print_layout"]()
        rows = [line for line in output.getvalue().splitlines()
                if line.strip().startswith("row")]
        self.assertEqual(len(rows), 7)
        self.assertTrue(any("narrow_passage" in row for row in rows))

    def test_camera_is_initialized_once_outside_the_simulation_loop(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "main")
        all_calls = [node for node in ast.walk(main)
                     if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute)
                     and node.func.attr == "set_camera"]
        loop = next(node for node in ast.walk(main)
                    if isinstance(node, ast.While))
        loop_calls = [node for node in ast.walk(loop)
                      if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute)
                      and node.func.attr == "set_camera"]
        self.assertEqual(len(all_calls), 1)
        self.assertEqual(loop_calls, [])


if __name__ == "__main__":
    unittest.main()
