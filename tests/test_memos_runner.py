"""并发任务隔离、异步回滚及记忆查看测试。"""

import contextlib
import io
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from memos_eval import inspect as memory_inspect
from memos_eval import run


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(os.chdir, Path.cwd())
        self.root = Path(self.temp.name)
        (self.root / "tau2-bench").mkdir()
        os.chdir(self.root)
        self.main_thread = threading.get_ident()
        self.state = SimpleNamespace(
            store={}, clients=[], agents=[], cleanups=[], snapshots=[], calls=[],
            saves=[], manifests=[], lock=threading.Lock(), saved_events={},
        )

    def execute(self, *, arms=("B-cross",), task_count=2, concurrency=1,
                simulate_hook=None, settle_hook=None, native_logs=False,
                output="experiment", num_tasks=None, resume=False, extra_args=()):
        os.chdir(self.root)
        state = self.state
        main_thread = self.main_thread
        tasks = [SimpleNamespace(id=f"task_{i:03}") for i in range(1, task_count + 1)]
        registered = {}

        class Client:
            def __init__(self, *args):
                self.request_count = 0
                self.closed = False
                with state.lock:
                    state.clients.append(self)

            def ensure_user(self):
                return {"async_add": True, "preference_memory": True,
                        "skill_generation": False, "extraction": "ordinary_fine"}

            def create_cube(self, cube):
                self.request_count += 1
                with state.lock:
                    assert cube not in state.store, "不能重新创建已有 cube"
                    state.store[cube] = []
                return cube

            def delete_session(self, cube, task_id):
                self.request_count += 1
                with state.lock:
                    state.store[cube] = [r for r in state.store[cube] if r["task_id"] != task_id]
                    state.cleanups.append((cube, task_id, list(state.store[cube])))

            def close(self):
                self.closed = True

        class Agent:
            def __init__(self, **kwargs):
                self.task_id = kwargs["task_id"]
                self.cube_id = kwargs["cube_id"]
                self.client = kwargs["client"]
                self.audit_path = kwargs["audit_path"]
                self.arm = kwargs["arm"]
                self.settled = False
                with state.lock:
                    state.agents.append(self)

            def finish(self, reason, tail=None):
                pass

            def settle_writes(self, timeout):
                self.settled = True
                return settle_hook(self) if settle_hook else True

            def summary(self):
                return {"written_turns": 1, "memory_calls": 1,
                        "write_status_counts": {"submitted": 1},
                        "source_tasks": {self.task_id: 1}}

        class Simulation(SimpleNamespace):
            def model_dump(self, mode=None):
                return {"task_id": self.task_id, "termination_reason": self.termination_reason,
                        "reward_info": {"reward": self.reward_info.reward} if self.reward_info else None}

            @classmethod
            def model_validate(cls, data):
                return cls(task_id=data["task_id"], termination_reason=data["termination_reason"],
                           reward_info=SimpleNamespace(**data["reward_info"]) if data["reward_info"] else None)

        class Results:
            def __init__(self, info, tasks, simulations):
                self.simulations = simulations
                self.tasks = tasks

            def save(self, path):
                assert threading.get_ident() == main_thread
                ids = [s.task_id for s in self.simulations]
                state.saves.append(ids)
                Path(path).write_text(json.dumps({"tasks": [t.id for t in self.tasks],
                    "simulations": [s.model_dump(mode="json") for s in self.simulations]}))
                arm = Path(path).parent.name.split("_seed")[0]
                for tid in ids:
                    event = state.saved_events.get((arm, tid))
                    if event:
                        event.set()

        def build(config, task, **kwargs):
            from tau2.utils.llm_utils import llm_log_mode
            assert llm_log_mode.get() == "all"
            agent = registered["memos_agent"](
                [], "policy", config.llm_agent, config.llm_args_agent, task)
            assert agent.task_id == task.id
            assert task.id in str(agent.audit_path)
            return SimpleNamespace(
                agent=agent, task=task, message=None, to_role="user",
                agent_state=SimpleNamespace(messages=[]), user_state=SimpleNamespace(messages=[]),
            )

        def simulate(orch, **kwargs):
            agent = orch.agent
            with state.lock:
                attempt = sum(arm == agent.arm and tid == agent.task_id for arm, tid in state.calls)
                state.calls.append((agent.arm, agent.task_id))
                state.snapshots.append((agent.task_id, attempt, list(state.store[agent.cube_id])))
                state.store[agent.cube_id].append({"task_id": agent.task_id, "attempt": attempt})
            agent.client.request_count += 11
            if simulate_hook:
                simulate_hook(orch, attempt)
            return Simulation(
                task_id=orch.task.id, termination_reason="user_stop",
                reward_info=SimpleNamespace(reward=0.0 if orch.task.id == "task_001" else 1.0),
            )

        original_write = run.write_json

        def write_json(path, value):
            if path.name == "manifest.json":
                assert threading.get_ident() == main_thread
                state.manifests.append(json.loads(json.dumps(value)))
            original_write(path, value)

        argv = ["memos-eval", "--output", output]
        if resume:
            argv.append("--resume")
        else:
            argv.extend(["--arms", *arms, "--max-concurrency", str(concurrency), "--max-retries", "1"])
        if num_tasks is not None:
            argv.extend(["--num-tasks", str(num_tasks)])
        argv.extend(extra_args)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(run, "ROOT", self.root))
            stack.enter_context(patch.object(run, "load_dotenv"))
            stack.enter_context(patch.object(run, "write_json", write_json))
            stack.enter_context(patch("sys.argv", argv))
            stack.enter_context(patch.dict(os.environ, {
                "USER_API_BASE": "", "USER_API_KEY": "", "MEMOS_API_KEY": "",
            }))
            stack.enter_context(patch("memos_eval.client.MemOSClient", Client))
            stack.enter_context(patch("memos_eval.agent.MemOSAgent", Agent))
            stack.enter_context(patch("tau2.registry.registry.register_agent_factory",
                                      side_effect=lambda factory, name: registered.update({name: factory})))
            stack.enter_context(patch("tau2.runner.helpers.get_info", side_effect=lambda config:
                SimpleNamespace(user_info=SimpleNamespace(llm_args={})) ))
            stack.enter_context(patch("tau2.runner.helpers.get_tasks", return_value=tasks))
            if not native_logs:
                stack.enter_context(patch("tau2.runner.batch._TaskLogContext",
                                          side_effect=lambda *a: contextlib.nullcontext()))
            stack.enter_context(patch("tau2.runner.build.build_orchestrator", build))
            stack.enter_context(patch("tau2.runner.build._build_env_kwargs", return_value={}))
            stack.enter_context(patch("tau2.runner.simulation.run_simulation", simulate))
            stack.enter_context(patch("tau2.data_model.simulation.Results", Results))
            stack.enter_context(patch("tau2.data_model.simulation.SimulationRun", Simulation))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            run.main()
        return json.loads((self.root / output / "manifest.json").read_text())

    def test_eight_workers_overlap_with_separate_agents_and_correct_scopes(self):
        barriers = {arm: threading.Barrier(8) for arm in ("B-cross", "B-intra")}
        active = {arm: 0 for arm in barriers}
        peak = {arm: 0 for arm in barriers}
        seen_threads = {arm: set() for arm in barriers}

        def simulate(orch, attempt):
            arm = orch.agent.arm
            with self.state.lock:
                active[arm] += 1
                peak[arm] = max(peak[arm], active[arm])
                seen_threads[arm].add(threading.get_ident())
            barriers[arm].wait(timeout=5)
            with self.state.lock:
                active[arm] -= 1

        manifest = self.execute(arms=("B-cross", "B-intra"), task_count=8,
                                concurrency=8, simulate_hook=simulate)
        self.assertEqual(peak, {"B-cross": 8, "B-intra": 8})
        self.assertEqual([len(t) for t in seen_threads.values()], [8, 8])
        self.assertEqual(manifest["configuration"]["max_concurrency"], 8)
        self.assertEqual(manifest["configuration"]["memory_add"]["async_mode"], "async")
        self.assertEqual(manifest["memos_features"]["extraction"], "ordinary_fine")
        self.assertEqual(len(manifest["tasks"]), 16)
        cross = [a for a in self.state.agents if a.arm == "B-cross"]
        intra = [a for a in self.state.agents if a.arm == "B-intra"]
        self.assertEqual(len({a.cube_id for a in cross}), 1)
        self.assertEqual(len({a.cube_id for a in intra}), 8)
        self.assertTrue({a.cube_id for a in cross}.isdisjoint(a.cube_id for a in intra))
        self.assertEqual(len({id(a.client) for a in self.state.agents}), 16)
        self.assertTrue(all(c.closed for c in self.state.clients))
        self.assertFalse(any(a.settled for a in self.state.agents))
        self.assertTrue(all(r["attempts"][0]["memos_http_requests"] == 11 for r in manifest["tasks"]))
        self.assertEqual(manifest["active_tasks"], [])

    def test_results_stay_in_task_order_when_workers_finish_in_reverse(self):
        arm = "B-cross"
        self.state.saved_events = {(arm, f"task_{i:03}"): threading.Event() for i in range(1, 4)}

        def simulate(orch, attempt):
            tid = orch.task.id
            if tid != "task_003":
                following = f"task_{int(tid[-3:]) + 1:03}"
                self.assertTrue(self.state.saved_events[(arm, following)].wait(timeout=5))

        manifest = self.execute(task_count=3, concurrency=3, simulate_hook=simulate)
        self.assertEqual(self.state.saves, [
            ["task_003"], ["task_002", "task_003"], ["task_001", "task_002", "task_003"]])
        self.assertEqual([r["task_id"] for r in manifest["tasks"]], ["task_001", "task_002", "task_003"])
        self.assertEqual([r["completion_order"] for r in manifest["tasks"]], [3, 2, 1])
        self.assertTrue(all(r["started_at"] <= r["finished_at"] for r in manifest["tasks"]))

    def test_failed_attempt_settles_before_deletion_and_keeps_other_task_memory(self):
        def simulate(orch, attempt):
            if orch.task.id == "task_002" and attempt == 0:
                raise ValueError("用户模拟器空回复")

        def settle(agent):
            self.state.store[agent.cube_id].append({"task_id": agent.task_id, "attempt": "late"})
            return True

        manifest = self.execute(simulate_hook=simulate, settle_hook=settle)
        self.assertEqual(self.state.calls, [
            ("B-cross", "task_001"), ("B-cross", "task_002"), ("B-cross", "task_002")])
        self.assertEqual(len(self.state.cleanups), 1)
        self.assertEqual(self.state.cleanups[0][1:], (
            "task_002", [{"task_id": "task_001", "attempt": 0}]))
        self.assertEqual(self.state.snapshots[-1][2], [{"task_id": "task_001", "attempt": 0}])
        self.assertTrue(manifest["tasks"][1]["attempts"][0]["memory_rolled_back"])
        self.assertEqual(len({id(a.client) for a in self.state.agents}), 3)

    def test_unconfirmed_async_write_stops_without_deleting_or_retrying(self):
        def simulate(orch, attempt):
            if orch.task.id == "task_002":
                raise ValueError("用户模拟器空回复")

        with self.assertRaisesRegex(RuntimeError, "异步写入状态未确认"):
            self.execute(simulate_hook=simulate, settle_hook=lambda agent: False)
        self.assertEqual(self.state.cleanups, [])
        self.assertEqual(self.state.calls, [("B-cross", "task_001"), ("B-cross", "task_002")])
        manifest = json.loads((self.root / "experiment/manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["failed_attempts"][0]["status"], "cleanup_unconfirmed")
        self.assertFalse(manifest["failed_attempts"][0]["memory_rolled_back"])
        summary = json.loads((self.root / "experiment/summary.json").read_text())
        self.assertFalse(summary["arms"]["B-cross"]["memory_pipeline_passed"])
        self.assertTrue(all(c.closed for c in self.state.clients))

    def test_user_request_retries_preserve_current_task_memory(self):
        from tau2.data_model.message import AssistantMessage
        from tau2.user.user_simulator import UserSimulator
        from tau2.user.user_simulator_base import UserState

        def simulate(orch, attempt):
            orch.user_state = UserState(system_messages=[], messages=[])
            UserSimulator("openai/gpt-5.2").generate_next_message(
                AssistantMessage(role="assistant", content="请确认。"), orch.user_state)

        usage = {"prompt_tokens": 10, "completion_tokens": 4}
        replies = [AssistantMessage(role="assistant", content=text, cost=0.1,
                                   usage=usage, raw_data={"usage": usage})
                   for text in ("", "", "确认。")]
        with patch("tau2.user.user_simulator.generate", side_effect=replies), \
             patch("tau2.user.user_simulator.time.sleep"):
            manifest = self.execute(task_count=1, simulate_hook=simulate)
        self.assertEqual(self.state.calls, [("B-cross", "task_001")])
        self.assertEqual(self.state.cleanups, [])
        attempts = manifest["tasks"][0]["attempts"]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["user_usage"]["calls"], 3)
        self.assertAlmostEqual(attempts[0]["user_usage"]["reported_cost_usd"], 0.3)

    def test_exhausted_user_request_retries_record_error_before_task_retry(self):
        from tau2.data_model.message import AssistantMessage
        from tau2.user.user_simulator import UserSimulator
        from tau2.user.user_simulator_base import UserState

        def simulate(orch, attempt):
            orch.user_state = UserState(system_messages=[], messages=[])
            UserSimulator("openai/gpt-5.2").generate_next_message(
                AssistantMessage(role="assistant", content="请确认。"), orch.user_state)

        usage = {"prompt_tokens": 10, "completion_tokens": 4}
        replies = [AssistantMessage(role="assistant", content=text, cost=0.1,
                                   usage=usage, raw_data={"usage": usage})
                   for text in ("", "", "", "确认。")]
        with patch("tau2.user.user_simulator.generate", side_effect=replies), \
             patch("tau2.user.user_simulator.time.sleep"):
            manifest = self.execute(task_count=1, simulate_hook=simulate)
        self.assertEqual(self.state.calls, [("B-cross", "task_001")] * 2)
        self.assertEqual(len(self.state.cleanups), 1)
        failed, succeeded = manifest["tasks"][0]["attempts"]
        self.assertEqual(failed["status"], "infrastructure_error")
        self.assertEqual(failed["error_type"], "EmptyUserResponseError")
        self.assertEqual(failed["user_usage"]["calls"], 3)
        self.assertEqual(succeeded["user_usage"]["calls"], 1)
        error = json.loads((self.root / "experiment/B-cross_seed300/task_001/attempt_0/error.json").read_text())
        self.assertEqual(len(error["pending_empty_responses"]), 3)

    def test_relative_output_resolves_before_benchmark_chdir(self):
        self.execute(task_count=1, output="runs/check_relative")
        self.assertTrue((self.root / "runs/check_relative/manifest.json").is_file())
        self.assertFalse((self.root / "tau2-bench/runs").exists())

    def test_cross_runs_18_then_resumes_all_97_in_the_same_cube(self):
        first = self.execute(task_count=97, concurrency=8, num_tasks=18,
                             extra_args=("--seeds", "301"))
        self.assertEqual(first["status"], "partial")
        self.assertEqual(len(first["task_order"]), 97)
        self.assertEqual(len(first["tasks"]), 18)
        original_cube = first["cubes"][0]["cube_id"]
        original_rows = first["tasks"]
        first_results = json.loads((self.root / "experiment/B-cross_seed301/results.json").read_text())
        self.assertEqual(len(first_results["tasks"]), 97)
        second = self.execute(task_count=97, resume=True)
        self.assertEqual(second["status"], "completed")
        self.assertEqual(second["arms"], ["B-cross"])
        self.assertEqual(second["seeds"], [301])
        self.assertEqual(second["configuration"]["max_concurrency"], 8)
        self.assertEqual(second["run_id"], first["run_id"])
        self.assertEqual(second["cubes"], first["cubes"])
        self.assertEqual(second["tasks"][:18], original_rows)
        self.assertEqual(len(second["tasks"]), 97)
        self.assertEqual(len(self.state.calls), 97)
        self.assertEqual({a.cube_id for a in self.state.agents}, {original_cube})
        self.assertEqual(second["tasks"][0]["reward"], 0.0)
        self.assertEqual(self.state.calls.count(("B-cross", "task_001")), 1)
        self.assertEqual([inv["num_tasks"] for inv in second["invocations"]], [18, None])
        self.assertTrue(all(inv["started_at"] <= inv["finished_at"] for inv in second["invocations"]))

    def test_intra_resume_keeps_completed_cubes_and_builds_only_remaining_tasks(self):
        first = self.execute(arms=("B-intra",), task_count=97, concurrency=8, num_tasks=18)
        second = self.execute(task_count=97, resume=True)
        self.assertEqual(second["arms"], ["B-intra"])
        self.assertEqual(second["status"], "completed")
        self.assertEqual(second["cubes"][:18], first["cubes"])
        self.assertEqual(len(second["cubes"]), 97)
        self.assertEqual(len(self.state.calls), 97)
        self.assertEqual(len({a.cube_id for a in self.state.agents}), 97)
        self.assertEqual(self.state.cleanups, [])

    def test_checkpoint_repairs_last_result_and_manifest_save_window(self):
        first = self.execute(task_count=2, num_tasks=1)
        first["active_tasks"] = [{"arm": "B-cross", "seed": 300, "task_id": "task_001"}]
        first["tasks"] = []
        first["status"] = "running"
        (self.root / "experiment/manifest.json").write_text(json.dumps(first))
        (self.root / "experiment/B-cross_seed300/results.json").write_text("{")
        resumed = self.execute(task_count=2, resume=True)
        self.assertEqual(resumed["status"], "completed")
        self.assertEqual([r["task_id"] for r in resumed["tasks"]], ["task_001", "task_002"])
        self.assertEqual(self.state.calls, [("B-cross", "task_001"), ("B-cross", "task_002")])

    def test_resume_refuses_unknown_active_attempt_without_touching_memories(self):
        first = self.execute(task_count=2, num_tasks=1)
        first["active_tasks"] = [{"arm": "B-cross", "seed": 300, "task_id": "task_002"}]
        path = self.root / "experiment/manifest.json"
        path.write_text(json.dumps(first))
        clients_before = len(self.state.clients)
        with self.assertRaisesRegex(ValueError, "未确认"):
            self.execute(task_count=2, resume=True)
        self.assertEqual(len(self.state.clients), clients_before)
        self.assertEqual(self.state.cleanups, [])
        self.assertEqual(json.loads(path.read_text()), first)

    def test_resume_rejects_conflicting_parameters_and_old_sync_policy(self):
        first = self.execute(task_count=2, num_tasks=1)
        path = self.root / "experiment/manifest.json"
        with self.assertRaisesRegex(ValueError, "top_k.*不一致"):
            self.execute(task_count=2, resume=True, extra_args=("--top-k", "3"))
        self.assertEqual(json.loads(path.read_text()), first)
        first["configuration"]["memory_add"] = {"async_mode": "sync", "mode": "fast"}
        path.write_text(json.dumps(first))
        with self.assertRaisesRegex(ValueError, "不能混用旧版"):
            self.execute(task_count=2, resume=True)

    def test_fresh_runs_with_the_same_directory_name_have_different_cube_ids(self):
        one = self.execute(task_count=1, output="one/shared_name")
        two = self.execute(task_count=1, output="two/shared_name")
        self.assertNotEqual(one["run_id"], two["run_id"])
        self.assertNotEqual(one["cubes"][0]["cube_id"], two["cubes"][0]["cube_id"])

    def test_concurrent_native_logs_and_all_llm_calls_remain_task_local(self):
        barrier = threading.Barrier(2)

        def simulate(orch, attempt):
            from loguru import logger
            from tau2.utils.llm_utils import _write_llm_log
            barrier.wait(timeout=5)
            for index in range(2):
                logger.debug("业务日志 " + orch.task.id)
                _write_llm_log({"task_id": orch.task.id}, {"index": index}, "agent_response")

        self.execute(task_count=2, concurrency=2, simulate_hook=simulate, native_logs=True)
        root = self.root / "experiment/B-cross_seed300"
        for tid in ("task_001", "task_002"):
            log = next((root / tid).rglob("task.log"))
            text = log.read_text()
            self.assertIn("业务日志 " + tid, text)
            other = "task_002" if tid == "task_001" else "task_001"
            self.assertNotIn("业务日志 " + other, text)
            calls = list(log.parent.glob("llm_debug/*.json"))
            self.assertEqual(len(calls), 2)
            self.assertTrue(all(json.loads(p.read_text())["request"]["task_id"] == tid for p in calls))
        text = (self.root / "experiment/run.log").read_text()
        self.assertIn("业务日志 task_001", text)
        self.assertIn("业务日志 task_002", text)


class MemoryInspectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "manifest.json").write_text(json.dumps({
            "memos_url": "http://offline.invalid", "memos_user": "test-user",
            "cubes": [
                {"arm": "B-cross", "task_id": None, "cube_id": "domain-cube"},
                {"arm": "B-intra", "task_id": "task_001", "cube_id": "task1-cube"},
                {"arm": "B-intra", "task_id": "task_002", "cube_id": "task2-cube"},
            ],
        }))

    def view(self, arm, task_id):
        calls = []

        class Client:
            def __init__(self, *args):
                pass

            def list_memories(self, cube, session, limit, offset):
                calls.append((cube, session, limit, offset))
                return [SimpleNamespace(
                    memory_id=f"{cube}-memory", content="visible memory", task_id=session,
                    created_at="2026-11-04", metadata={"cube_id": cube},
                )]

        output = self.root / f"{arm}.json"
        argv = [
            "inspect-memories", "--run-dir", str(self.root), "--arm", arm,
            "--task-id", task_id, "--output", str(output),
        ]
        with patch("sys.argv", argv), patch.object(memory_inspect, "load_dotenv"), \
             patch.object(memory_inspect, "MemOSClient", Client), \
             contextlib.redirect_stdout(io.StringIO()):
            memory_inspect.main()
        return calls, json.loads(output.read_text())

    def test_intra_view_chooses_only_requested_task_cube(self):
        calls, rows = self.view("B-intra", "task_002")
        self.assertEqual(calls, [("task2-cube", "task_002", 100, 0)])
        self.assertEqual([row["cube_id"] for row in rows], ["task2-cube"])

    def test_cross_view_keeps_domain_cube_and_filters_task_session(self):
        calls, rows = self.view("B-cross", "task_002")
        self.assertEqual(calls, [("domain-cube", "task_002", 100, 0)])
        self.assertEqual(rows[0]["task_id"], "task_002")


if __name__ == "__main__":
    unittest.main()
