import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "legged_gym/scripts/view_go2_amp_terrains.py"


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


if __name__ == "__main__":
    unittest.main()
