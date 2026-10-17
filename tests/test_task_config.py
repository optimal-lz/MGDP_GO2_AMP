import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1] / "legged_gym"

def find_class(node, name):
    for child in node.body:
        if isinstance(child, ast.ClassDef) and child.name == name:
            return child
    raise AssertionError("class %s not found" % name)

def assignments(node):
    values = {}
    for child in node.body:
        if isinstance(child, ast.Assign) and len(child.targets) == 1:
            target = child.targets[0]
            if isinstance(target, ast.Name):
                try:
                    values[target.id] = ast.literal_eval(child.value)
                except (ValueError, TypeError):
                    pass
    return values

class TestGo2AmpTaskConfig(unittest.TestCase):
    def test_single_task_registration_preserves_mgdp_references(self):
        source = (ROOT / "legged_gym/envs/__init__.py").read_text(encoding="utf-8")
        self.assertEqual(source.count('task_registry.register("go2_amp"'), 1)
        self.assertNotIn('task_registry.register("go2_amp_stage', source)
        self.assertIn('task_registry.register("random_dog_stage1"', source)
        self.assertIn('task_registry.register("random_dog_stage2"', source)

    def test_active_expert_manifest_is_forward_and_turn_only(self):
        tree = ast.parse((ROOT / "legged_gym/envs/go2_amp/config.py").read_text(encoding="utf-8"))
        top = assignments(tree)
        self.assertEqual(top["AMP_EXPERT_GROUPS"], {
            "forward": ["go2_forward.txt", "go2_forward_fast.txt"],
            "turn": [
                "go2_turn_left.txt", "go2_turn_left_fast.txt", "go2_turn_left_faster.txt",
                "go2_turn_right.txt", "go2_turn_right_fast.txt", "go2_turn_right_faster.txt",
            ],
        })
        self.assertEqual(top["AMP_EXPERT_GROUP_WEIGHTS"], {"forward": 0.8, "turn": 0.2})

    def test_environment_and_training_baseline_matches_frozen_design(self):
        tree = ast.parse((ROOT / "legged_gym/envs/go2_amp/config.py").read_text(encoding="utf-8"))
        env_cfg = find_class(tree, "Go2AmpCfg")
        train_cfg = find_class(tree, "Go2AmpTrainCfg")
        env_values = assignments(find_class(env_cfg, "env"))
        self.assertEqual(env_values["num_envs"], 4096)
        self.assertEqual(env_values["num_privileged_obs"], 194)
        self.assertEqual(assignments(find_class(env_cfg, "control")), {
            "control_type": "P", "stiffness": {"joint": 30.0},
            "damping": {"joint": 0.8}, "action_scale": 0.5, "decimation": 4,
        })
        camera = assignments(find_class(env_cfg, "camera"))
        for key, value in {"resized": [16, 16], "update_interval": 5,
                           "wm_sensor_steps": 1, "wm_rollout_steps": 4,
                           "noise_gaussian": 0.03, "noise_dropout": 0.1}.items():
            self.assertEqual(camera[key], value)
        self.assertIs(camera["use_memory"], False)
        domain = assignments(find_class(env_cfg, "domain_rand"))
        self.assertEqual(domain["friction_range"], [0.5, 1.25])
        self.assertEqual(domain["latency_range"], [0.0, 0.02])
        self.assertEqual(domain["push_interval_s"], 15)
        self.assertEqual(domain["max_push_vel_xy"], 1.0)
        self.assertEqual(
            {name: domain[name] for name in (
                "randomize_friction", "randomize_action_latency",
                "push_robots",
            )},
            {
                "randomize_friction": True,
                "randomize_action_latency": True,
                "push_robots": True,
            },
        )
        for name in (
                "randomize_base_mass", "randomize_link_mass",
                "randomize_restitution", "randomize_com",
                "randomize_gains", "randomize_motor_strength",
                "randomize_motor_offset", "randomize_lag_timesteps"):
            self.assertIs(domain[name], False, name)
        noise = assignments(find_class(env_cfg, "noise"))
        self.assertIs(noise["add_height_noise"], False)
        self.assertIs(noise["add_privileged_noise"], False)
        rewards = assignments(find_class(env_cfg, "rewards"))
        self.assertIs(rewards.get("terrain_adaptive_reward", True), False)
        scales = assignments(find_class(find_class(env_cfg, "rewards"), "scales"))
        expected = {"tracking_lin_vel": 1.5, "tracking_ang_vel": 0.5,
                    "feet_air_time": 0.5, "lin_vel_z": -1.0,
                    "ang_vel_xy": -0.05, "orientation": -0.1,
                    "torques": -1e-4, "dof_acc": -2.5e-7,
                    "action_rate": -0.03, "collision": -1.0,
                    "feet_stumble": -0.1, "dof_pos_limits": -10.0,
                    "torque_limits": -0.01, "dof_error": -0.04,
                    "feet_edge": -1.0, "cheat": -1.0, "stuck": -1.0,
                    "termination": 0.0}
        self.assertEqual({key: scales[key] for key in expected}, expected)
        runner = assignments(find_class(train_cfg, "runner"))
        self.assertEqual(runner["runner_class_name"], "Go2AmpPolicyRunner")
        self.assertEqual(runner["num_steps_per_env"], 24)
        self.assertEqual(runner["amp_updates_per_iter"], 20)
        self.assertEqual(runner["amp_replay_rollouts"], 2)
        self.assertEqual(runner["save_interval"], 1000)
        self.assertEqual(assignments(find_class(train_cfg, "amp"))["reward_coef"], 0.01)
        policy = assignments(find_class(train_cfg, "policy"))
        self.assertIs(policy["use_visual_memory"], True)

if __name__ == "__main__":
    unittest.main()
