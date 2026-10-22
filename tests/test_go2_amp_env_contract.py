import ast
from pathlib import Path
import unittest

from tests.ast_source import node_source


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / "legged_gym/legged_gym/envs/go2_amp/env.py"


class TestGo2AmpEnvironmentContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = ENV_PATH.read_text(encoding="utf-8")
        tree = ast.parse(cls.source)
        cls.imports = {
            alias.name: node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        class_node = next(node for node in tree.body
                          if isinstance(node, ast.ClassDef)
                          and node.name == "Go2AmpRandomDog")
        cls.methods = {node.name: node_source(cls.source, node) for node in class_node.body
                       if isinstance(node, ast.FunctionDef)}

    def test_wrap_to_pi_uses_shared_compatibility_helper(self):
        self.assertEqual(
            self.imports["wrap_to_pi"],
            "legged_gym.utils.math",
        )

    def test_commands_increment_epoch_apply_deadband_and_disable_structured_yaw(self):
        source = self.methods["_resample_commands"]
        self.assertIn("apply_fixed_evaluation_commands", source)
        self.assertIn("_fixed_evaluation_commands", source)
        self.assertIn("apply_command_deadband", source)
        self.assertIn("apply_terrain_command_policy", source)
        self.assertIn("self.command_epoch[env_ids] += 1", source)
        self.assertIn("turning_terrain_ids", source)
        self.assertIn("TERRAIN_CLASS_IDS['flat']", source)
        self.assertIn("TERRAIN_CLASS_IDS['rough_ground']", source)
        self.assertIn("self.cfg.commands.vx_deadband + 1e-06", source)

    def test_idle_and_locomotion_are_exact_complements(self):
        self.assertIn("classify_commands(self.commands)",
                      self.methods["get_idle_mask"])
        self.assertIn("classify_commands(self.commands)",
                      self.methods["get_locomotion_mask"])

    def test_only_base_contact_and_timeout_terminate(self):
        source = self.methods["check_termination"]
        self.assertIn("self.termination_contact_indices", source)
        self.assertIn("> 1.0", source)
        self.assertIn("self.time_out_buf", source)
        for forbidden in ("root_states[:, 2]", "foot_height", "roll", "pitch", "base_lin_vel"):
            self.assertNotIn(forbidden, source)

    def test_reset_uses_natural_pose_zero_velocity_and_safe_spawn_rules(self):
        dofs = self.methods["_reset_dofs"]
        root = self.methods["_reset_root_states"]
        reset = self.methods["reset_idx"]
        self.assertIn("torch_rand_float(0.9, 1.1", dofs)
        self.assertIn("self.dof_vel[env_ids] = 0.0", dofs)
        self.assertIn("self.env_origins[env_ids, 2] + 0.45", root)
        self.assertIn("torch_rand_float(-0.1, 0.1", root)
        self.assertIn("rough_ground", root)
        self.assertIn("spawn_jitter[flat | rough_ground, 1]", root)
        self.assertIn("spawn_jitter[flat, 0] +=", root)
        self.assertIn("spawn_jitter = torch.zeros", root)
        self.assertIn("self.root_states[env_ids, 7:13] = 0.0", root)
        self.assertIn("_reset_amp_histories", reset)
        self.assertNotIn("reference", reset.lower())
        self.assertNotIn("rsi", reset.lower())

    def test_reset_refreshes_derived_state_and_clears_visual_cache(self):
        self.assertIn("refresh_reset_state", self.methods)
        refresh = self.methods["refresh_reset_state"]
        for required in (
                "self.base_quat[env_ids]", "self.base_lin_vel[env_ids]",
                "self.base_ang_vel[env_ids]",
                "self.projected_gravity[env_ids]", "self._get_heights(env_ids)",
                "self.obs_dict['image_buf'][env_ids] = 0.0",
                "self.image_update_mask[env_ids] = False"):
            self.assertIn(required, refresh)
        self.assertIn("refresh_reset_state", self.methods["reset_idx"])

    def test_checkpoint_restart_suppresses_curriculum_and_forces_visual_refresh(self):
        self.assertIn("restart_after_checkpoint_load", self.methods)
        restart = self.methods["restart_after_checkpoint_load"]
        self.assertIn("_suppress_curriculum", restart)
        self.assertIn("self.reset()", restart)
        self.assertIn("refresh_visual_observation", restart)
        self.assertIn("_suppress_curriculum", self.methods["_update_terrain_curriculum"])

    def test_curriculum_has_no_anchor_stage_or_unlock(self):
        source = self.methods["_update_terrain_curriculum"]
        self.assertIn("next_curriculum_levels", source)
        self.assertIn("self.commands[env_ids, 0] > self.cfg.commands.vx_deadband", source)
        for forbidden in ("anchor", "unlock", "training_iteration", "stage"):
            self.assertNotIn(forbidden, source.lower())

    def test_stuck_is_reward_only(self):
        termination = self.methods["check_termination"]
        self.assertNotIn("stuck", termination.lower())
        self.assertIn("_reward_stuck", self.methods)

    def test_gap_edge_and_heading_penalties_use_revised_classes(self):
        edge = self.methods["_reward_feet_edge"]
        self.assertIn("TERRAIN_CLASS_IDS['gap_course']", edge)
        self.assertIn("self.terrain_levels >= 2", edge)
        self.assertNotIn("pit", edge)
        heading = self.methods["_reward_cheat"]
        self.assertIn("TERRAIN_CLASS_IDS['rough_ground']", heading)

    def test_reward_statistics_split_task_from_regularization_without_reordering(self):
        compute = self.methods["compute_reward"]
        self.assertIn("TASK_REWARD_NAMES", self.source)
        self.assertIn("self._task_reward", compute)
        self.assertIn("self._regularization_reward", compute)
        self.assertIn("torch.clamp(self._base_reward, min=0.0)", compute)
        self.assertIn("episode_lengths", self.methods["reset_idx"])


if __name__ == "__main__":
    unittest.main()
