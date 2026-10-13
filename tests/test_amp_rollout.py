"""Local source contract for AMP placement in the original MGDP runner."""

import ast
import math
import statistics
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "legged_gym/rl/MGDP/runners/policy_runner.py"


def _load_runner_helpers(*names):
    tree = ast.parse(RUNNER_PATH.read_text(encoding="utf-8"))
    functions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != set(names):
        raise AssertionError(f"runner helpers missing: {sorted(set(names) - {n.name for n in functions})}")
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"math": math}
    exec(compile(module, str(RUNNER_PATH), "exec"), namespace)
    return tuple(namespace[name] for name in names)


def _distribution_sums(prefix, values):
    return {
        f"{prefix}_count": len(values),
        f"{prefix}_sum": sum(values),
        f"{prefix}_sq_sum": sum(value * value for value in values),
        f"{prefix}_abs_sum": sum(abs(value) for value in values),
        f"{prefix}_zero_count": sum(value == 0 for value in values),
    }


class TestAmpRolloutIntegration(unittest.TestCase):
    def test_rollout_sufficient_stats_do_not_sync_each_step(self):
        path = ROOT / "legged_gym/rl/MGDP/amp/session.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        session = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "AMPSession"
        )
        reward = next(
            node for node in session.body
            if isinstance(node, ast.FunctionDef) and node.name == "reward"
        )
        reward_source = ast.unparse(reward)
        self.assertNotIn(".item()", reward_source)
        self.assertIn("self.last_reward_metrics", reward_source)
        record = next(
            node for node in session.body
            if isinstance(node, ast.FunctionDef) and node.name == "record"
        )
        record_source = ast.unparse(record)
        self.assertIn("self.replay.insert(state[mask], next_state[mask])", record_source)
        self.assertNotIn("anchor", record_source)
        self.assertNotIn("course", record_source)

    def test_rollout_metrics_aggregate_every_step_and_keep_groups_distinct(self):
        new_accumulator, accumulate, finalize = _load_runner_helpers(
            "new_amp_rollout_accumulator",
            "accumulate_amp_rollout",
            "finalize_amp_rollout_metrics",
        )
        task_steps = [[-1.0, 1.0], [2.0, -2.0], [3.0, -3.0]]
        raw_steps = [[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]
        contribution_steps = [[0.0, 0.1], [0.2, 0.3], [0.4, 0.5]]
        logit_steps = [[-3.0, 3.0], [-2.0, 2.0], [-1.0, 1.0]]

        accumulator = new_accumulator()
        for task, raw, contribution, logits in zip(
                task_steps, raw_steps, contribution_steps, logit_steps):
            total = [a + b for a, b in zip(task, contribution)]
            step = {}
            for prefix, values in (
                ("task_reward", task),
                ("total_reward", total),
                ("r_amp_raw", raw),
                ("amp_reward_contribution", contribution),
                ("policy_logit_rollout", logits),
                ("anchor_r_amp_raw", raw[:1]),
                ("anchor_amp_reward_contribution", contribution[:1]),
                ("anchor_policy_logit_rollout", logits[:1]),
                ("course_r_amp_raw", raw[1:]),
                ("course_amp_reward_contribution", contribution[1:]),
                ("course_policy_logit_rollout", logits[1:]),
            ):
                step.update(_distribution_sums(prefix, values))
            accumulate(accumulator, step)

        metrics = finalize(accumulator)
        task_values = sum(task_steps, [])
        total_values = [
            task + contribution
            for tasks, contributions in zip(task_steps, contribution_steps)
            for task, contribution in zip(tasks, contributions)
        ]
        raw_values = sum(raw_steps, [])
        contribution_values = sum(contribution_steps, [])
        logit_values = sum(logit_steps, [])
        self.assertAlmostEqual(metrics["task_reward_mean"], statistics.mean(task_values))
        self.assertAlmostEqual(metrics["total_reward_mean"], statistics.mean(total_values))
        self.assertAlmostEqual(metrics["r_amp_raw_mean"], statistics.mean(raw_values))
        self.assertAlmostEqual(metrics["r_amp_raw_std"], statistics.pstdev(raw_values))
        self.assertAlmostEqual(metrics["r_amp_raw_zero_fraction"], 2 / 6)
        self.assertAlmostEqual(
            metrics["amp_reward_contribution_mean"], statistics.mean(contribution_values)
        )
        self.assertAlmostEqual(
            metrics["amp_reward_contribution_std"], statistics.pstdev(contribution_values)
        )
        self.assertAlmostEqual(metrics["amp_contribution_task_ratio"], 0.125)
        self.assertAlmostEqual(
            metrics["policy_logit_rollout_mean"], statistics.mean(logit_values)
        )
        self.assertAlmostEqual(
            metrics["policy_logit_rollout_std"], statistics.pstdev(logit_values)
        )
        self.assertAlmostEqual(metrics["anchor_policy_logit_rollout_mean"], -2.0)
        self.assertAlmostEqual(metrics["course_policy_logit_rollout_mean"], 2.0)
        self.assertIn("anchor_r_amp_raw_std", metrics)
        self.assertIn("course_amp_reward_contribution_std", metrics)
        self.assertNotIn("policy_logit", metrics)

    def test_terrain_and_update_metrics_are_forwarded_without_decisions(self):
        summarize_terrain, merge_metrics = _load_runner_helpers(
            "summarize_amp_terrain",
            "merge_amp_metrics",
        )
        cycle = [0, 1, 2, 2, 3, 3, 4, 4, 5, 5]
        course_columns = [cycle[index % len(cycle)] for index in range(3482)]
        anchors = [True] * 614 + [False] * 3482
        columns = [0] * 614 + course_columns
        levels = [0] * 4096
        terrain = summarize_terrain(anchors, columns, levels)
        self.assertEqual(terrain["anchor_count"], 614)
        self.assertEqual(terrain["course_level0_zero_slope_count"], 698)
        self.assertEqual(terrain["complete_flat_fraction"], 1312 / 4096)
        self.assertEqual(terrain["terrain_level_mean"], 0.0)

        update = {
            "discriminator_loss": 1.2,
            "gradient_penalty": 3.4,
            "expert_logit_mean": 0.5,
            "expert_logit_std": 0.6,
            "policy_logit_update_mean": -0.7,
            "policy_logit_update_std": 0.8,
            "amp_updates": 1,
            "anchor_replay_size": 100,
            "course_replay_size": 200,
            "anchor_skipped_updates": 0,
            "course_skipped_updates": 0,
        }
        merged = merge_metrics({"task_reward_mean": 1.0}, terrain, update)
        for key, value in update.items():
            self.assertEqual(merged[key], value)
        self.assertIn("policy_logit_rollout_mean", {
            "policy_logit_rollout_mean": 0.0,
            "policy_logit_update_mean": merged["policy_logit_update_mean"],
        })

        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("self.writer.add_scalar('Episode/' + key", source)
        config = (
            ROOT / "legged_gym/legged_gym/envs/go2_amp/config.py"
        ).read_text(encoding="utf-8")
        self.assertIn("feet_air_time = 0.5", config)

    def test_reward_is_applied_before_ppo_transition(self):
        source = RUNNER_PATH.read_text(encoding="utf-8")
        ast.parse(source)
        for fragment in (
            "self.env.set_amp_training_iteration(it)",
            "iteration=it",
            "is_anchor=anchor_mask",
            "self.amp.record(amp_before, amp_after, anchor_mask)",
            "self.amp.update(iteration=it)",
        ):
            self.assertIn(fragment, source)
        set_iteration = source.index("self.env.set_amp_training_iteration(it)")
        before = source.index("amp_before = self.env.get_amp_observations()")
        step = source.index("obs_dict, rewards, dones, infos = self.env.step(actions)")
        reward = source.index("rewards, amp_rollout_metrics = self.amp.reward(")
        ppo = source.index("self.alg.process_env_step(rewards, dones")
        self.assertLess(before, step)
        self.assertLess(step, reward)
        self.assertLess(reward, ppo)
        self.assertLess(set_iteration, before)
        self.assertIn("amp_after = select_next_amp_state(", source)
        ppo_update = source.index("self.alg.update()")
        amp_update = source.index("self.amp.update(iteration=it)")
        self.assertLess(ppo_update, amp_update)

    def test_runner_has_no_metric_driven_experiment_control(self):
        source = RUNNER_PATH.read_text(encoding="utf-8")
        forbidden = (
            "if amp_rollout_metrics",
            "if amp_update_metrics",
            "amp_reward_coef =",
            "launch_next",
            "auto_resume",
            "admission_passed",
        )
        for fragment in forbidden:
            self.assertNotIn(fragment, source)

    def test_final_handoff_saves_matching_policy_and_world_model(self):
        root = Path(__file__).resolve().parents[1] / "legged_gym"
        runner = (root / "rl/MGDP/runners/policy_runner.py").read_text(encoding="utf-8")
        helper = (root / "legged_gym/utils/helpers.py").read_text(encoding="utf-8")
        environment = (root / "legged_gym/envs/random_dog/random_dog.py").read_text(encoding="utf-8")
        self.assertIn("self.save(os.path.join(self.nn_dir, 'last.pt'))", runner)
        self.assertIn("self.save_world_model(os.path.join(self.nn_dir, 'wm_last.pt'))", runner)
        self.assertIn("world_model_checkpoint = 'wm_last.pt'", helper)
        self.assertIn("wm_file = getattr(self.cfg.camera, 'world_model_checkpoint'", environment)
        self.assertIn("loaded_dict['iter'] + (1 if self.amp is not None else 0)", runner)


if __name__ == "__main__":
    unittest.main()
