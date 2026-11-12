"""离线检查丢弃未评分结果、备份和复用原 cube 的续跑准备。"""

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from memos_eval.recover import plan_recovery, prepare_recovery
from memos_eval.run import restore_progress


class Simulation:
    @classmethod
    def model_validate(cls, value):
        return SimpleNamespace(**value)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "experiment"
        self.output.mkdir()
        self.arm = self.output / "B-cross_seed300"
        self.arm.mkdir()
        self.tasks = [f"task_{number:03}" for number in range(1, 6)]
        self.rows = [self.row(task, reward) for task, reward in zip(self.tasks, [1.0, 0.0, None])]
        self.simulations = [self.simulation(row) for row in self.rows]
        self.manifest = {
            "run_id": "original-run", "status": "failed", "arms": ["B-cross"],
            "seeds": [300], "task_order": self.tasks, "tasks": self.rows,
            "configuration": {"max_concurrency": 8, "agent_effort": "max"},
            "cubes": [{"arm": "B-cross", "seed": 300, "task_id": None,
                       "cube_id": "original-memory-cube"}],
            "active_tasks": [{"arm": "B-cross", "seed": 300, "task_id": "task_004"}],
            "failed_attempts": [], "error": {"type": "KeyboardInterrupt"},
            "finished_at": "2026-11-05T15:52:09+08:00", "invocations": [{"num_tasks": None}],
        }
        self.results = {"timestamp": "original-timestamp", "info": {"model": "original-model"},
                        "tasks": [{"id": task, "version": "original"} for task in self.tasks],
                        "simulations": self.simulations, "simulation_index": None}
        self.write(self.output / "manifest.json", self.manifest)
        self.write(self.output / "summary.json", {"status": "failed"})
        self.write(self.arm / "results.json", self.results)
        for row, simulation in zip(self.rows, self.simulations):
            directory = self.arm / row["task_id"]
            self.write(directory / "completed.json", {"task": row, "simulation": simulation})
            self.write(directory / "attempt_0/error.json", {"original": row["task_id"]})
        self.write(self.arm / "task_004/attempt_0/error.json", {"interrupted": True})
        (self.output / "run.log").write_text("original log")

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False))

    def row(self, task, reward):
        return {"arm": "B-cross", "seed": 300, "task_id": task,
                "position": self.tasks.index(task), "reward": reward,
                "attempts": [{"memory": {}}]}

    def simulation(self, row):
        return {"id": f"sim-{row['task_id']}", "task_id": row["task_id"],
                "reward_info": {"reward": row["reward"]} if row["reward"] is not None else None}

    def snapshot(self):
        return {str(path.relative_to(self.output)): path.read_bytes()
                for path in self.output.rglob("*") if path.is_file()}

    def test_preview_does_not_change_files_or_existing_memory_references(self):
        before = self.snapshot()
        plan = plan_recovery(self.output)
        self.assertEqual(plan.report["kept_scored"], 2)
        self.assertEqual(plan.report["excluded_unscored"], [["B-cross", 300, "task_003"]])
        self.assertEqual([key[2] for key in plan.report["requeued"]], self.tasks[2:])
        self.assertEqual(plan.report["archived_task_directories"], [
            "B-cross_seed300/task_003", "B-cross_seed300/task_004"])
        self.assertEqual(plan.manifest["cubes"], self.manifest["cubes"])
        self.assertTrue(plan.report["reuse_existing_memory"])
        self.assertFalse(plan.report["memory_mutations"])
        self.assertEqual(self.snapshot(), before)

    def test_prepare_requires_confirmed_quiescence_before_any_mutation(self):
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "后台写入已结束"):
            prepare_recovery(self.output)
        self.assertEqual(self.snapshot(), before)

    def test_prepare_archives_none_and_interrupted_tasks_and_native_resume_accepts_it(self):
        before = self.snapshot()
        report = prepare_recovery(self.output, confirmed_stopped=True)
        backup = Path(report["backup"])
        for relative in ("manifest.json", "summary.json", "B-cross_seed300/results.json",
                         "B-cross_seed300/task_003/completed.json",
                         "B-cross_seed300/task_003/attempt_0/error.json",
                         "B-cross_seed300/task_004/attempt_0/error.json"):
            self.assertEqual((backup / relative).read_bytes(), before[relative])
        self.assertFalse((self.arm / "task_003").exists())
        self.assertFalse((self.arm / "task_004").exists())
        manifest = json.loads((self.output / "manifest.json").read_text())
        results = json.loads((self.arm / "results.json").read_text())
        self.assertEqual(manifest["tasks"], self.rows[:2])
        self.assertEqual(results["simulations"], self.simulations[:2])
        for name in ("run_id", "cubes", "configuration", "task_order", "invocations"):
            self.assertEqual(manifest[name], self.manifest[name])
        self.assertEqual(results["tasks"], self.results["tasks"])
        self.assertEqual(results["info"], self.results["info"])
        self.assertEqual(manifest["active_tasks"], [])
        self.assertEqual(manifest["failed_attempts"], [])
        self.assertEqual(manifest["status"], "partial")
        self.assertNotIn("error", manifest)
        self.assertNotIn("finished_at", manifest)
        self.assertEqual((self.output / "run.log").read_text(), "original log")
        restored = restore_progress(manifest, self.output, Simulation)
        self.assertEqual(set(restored[("B-cross", 300)]), set(self.tasks[:2]))
        summary = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(summary["arms"]["B-cross"]["scored"], 2)
        self.assertEqual(summary["arms"]["B-cross"]["passed"], 1)

    def test_checkpoint_overrides_stale_result_and_recovers_manifest_commit_window(self):
        recovered = self.row("task_003", 1.0)
        self.write(self.arm / "task_003/completed.json", {
            "task": recovered, "simulation": self.simulation(recovered)})
        new_row = self.row("task_004", 0.0)
        self.write(self.arm / "task_004/completed.json", {
            "task": new_row, "simulation": self.simulation(new_row)})
        plan = plan_recovery(self.output)
        self.assertEqual(plan.report["kept_scored"], 4)
        self.assertEqual(plan.report["checkpoint_records_recovered"], [["B-cross", 300, "task_004"]])
        self.assertEqual(plan.report["excluded_unscored"], [])
        self.assertEqual(plan.report["requeued"], [["B-cross", 300, "task_005"]])
        prepare_recovery(self.output, confirmed_stopped=True)
        manifest = json.loads((self.output / "manifest.json").read_text())
        restored = restore_progress(manifest, self.output, Simulation)
        self.assertEqual(set(restored[("B-cross", 300)]), set(self.tasks[:4]))

    def test_scored_reward_mismatch_is_rejected_before_backup_or_mutation(self):
        broken = deepcopy(self.rows[0])
        broken_simulation = self.simulation(broken)
        broken_simulation["reward_info"]["reward"] = 0.0
        self.write(self.arm / "task_001/completed.json", {"task": broken, "simulation": broken_simulation})
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "reward 不一致"):
            prepare_recovery(self.output, confirmed_stopped=True)
        self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
