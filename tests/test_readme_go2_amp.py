import unittest
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
