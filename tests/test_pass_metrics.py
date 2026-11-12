"""pass^k 指标口径测试。"""

import unittest

from memos_eval import run


def row(task_id, reward, seed=300):
    return {
        "task_id": task_id, "reward": reward, "seed": seed, "arm": "B-intra",
        "attempts": [{"memory": {}}],
    }


class PassMetricsTests(unittest.TestCase):
    def test_single_trial_counts_unscored_task_as_failure(self):
        metrics = run.pass_metrics([
            row("task_001", 1.0), row("task_002", 0.0), row("task_003", None),
        ])
        self.assertEqual(metrics["num_trials_per_task"], 1)
        self.assertEqual(metrics["passed"], 1)
        self.assertAlmostEqual(metrics["pass_hat_ks"][1], 1 / 3)
        self.assertAlmostEqual(metrics["pass_at_1_all_tasks"], 1 / 3)
        self.assertAlmostEqual(metrics["pass_at_1_valid_tasks"], 1 / 2)

    def test_pass_hat_k_matches_tau2_definition(self):
        rows = [row("A", 1.0), row("A", 1.0), row("A", 0.0), row("A", 0.0)]
        rows += [row("B", 0.0) for _ in range(4)]
        metrics = run.pass_metrics(rows)
        self.assertEqual(metrics["num_trials_per_task"], 4)
        self.assertAlmostEqual(metrics["pass_hat_ks"][1], 0.25)
        self.assertAlmostEqual(metrics["pass_hat_ks"][2], 1 / 12)
        self.assertEqual(metrics["pass_hat_ks"][3], 0.0)

    def test_uneven_trials_cap_k_at_minimum(self):
        metrics = run.pass_metrics([row("A", 1.0), row("A", 0.0), row("B", 1.0)])
        self.assertEqual(metrics["num_trials_per_task"], 1)
        # 按题取均值：A=1/2，B=1/1
        self.assertAlmostEqual(metrics["pass_hat_ks"][1], 0.75)
        self.assertAlmostEqual(metrics["pass_at_1_all_tasks"], 2 / 3)

    def test_summarize_run_exposes_pass_metrics(self):
        manifest = {
            "status": "completed", "arms": ["B-intra"],
            "tasks": [row("task_001", 1.0), row("task_002", 0.0)],
            "failed_attempts": [],
        }
        arm = run.summarize_run(manifest)["arms"]["B-intra"]
        self.assertEqual(arm["passed"], 1)
        self.assertAlmostEqual(arm["average_reward"], 0.5)
        self.assertAlmostEqual(arm["pass_hat_ks"][1], 0.5)
        self.assertAlmostEqual(arm["pass_at_1_valid_tasks"], 0.5)


if __name__ == "__main__":
    unittest.main()
