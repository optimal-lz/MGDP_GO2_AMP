import unittest
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


class ReadmeGo2AmpWorkflowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.readme = (ROOT / "README.md").read_text(encoding="utf-8")

    def test_documents_the_single_formal_entrypoints(self):
        for fragment in (
                "train_go2_amp.py",
                "--mode amp",
                "--mode no_amp",
                "--target_iteration 1000",
                "--target_iteration 2000",
                "--target_iteration 5000",
                "--resume_from",
                "vis_go2_amp.py",
                "evaluate_go2_amp_comparison.py"):
            self.assertIn(fragment, self.readme)

    def test_documents_artifacts_and_optional_visual_evidence(self):
        for fragment in (
                "last.json",
                "checkpoint_1000.json",
                "depth_latest.png",
                "height_latest.png",
                "cases.csv",
                "summary.csv",
                "completion.png",
                "--record",
                "Video_260929142335.mp4"):
            self.assertIn(fragment, self.readme)

    def test_does_not_advertise_retired_formal_workflow(self):
        for fragment in (
                "train_go2_amp_stage1.py",
                "--amp_policy_only",
                "--checkpoint_model last.pt",
                "is_amp_anchor"):
            self.assertNotIn(fragment, self.readme)

    def test_documents_revised_terrain_contract(self):
        for terrain in (
                "flat", "rough_ground", "slope_course", "stairs_course",
                "low_obstacle_course", "gap_course", "narrow_passage"):
            self.assertIn("`{}`".format(terrain), self.readme)
        self.assertIn("level 0..9", self.readme)
        self.assertIn("level 0/3/6/9", self.readme)

    def test_active_commands_use_only_revised_output_root(self):
        code_blocks = re.findall(r"```(?:bash|text)?\n(.*?)```", self.readme,
                                 flags=re.DOTALL)
        active_commands = "\n".join(code_blocks)
        self.assertIn("legged_gym/outputs/go2_amp_v1/", active_commands)
        self.assertNotIn("legged_gym/outputs/go2_amp/", active_commands)

    def test_documents_vertical_metric_and_pre_revision_boundary(self):
        self.assertIn("base_lin_vel_z_sq", self.readme)
        self.assertIn("`legged_gym/outputs/go2_amp/`", self.readme)
        self.assertIn("不得续训，也不得作为修正后 V1 的性能结论", self.readme)


if __name__ == "__main__":
    unittest.main()
