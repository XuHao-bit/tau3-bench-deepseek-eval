"""C 组工具路由、业务评分投影和共享库续跑边界。"""

import json
import contextlib
import copy
import io
import os
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import unittest

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.environment.tool import Tool

from memos_eval.run_skills import business_messages, configure, install_memory_tools, restore_checkpoints


class SkillRunnerTests(unittest.TestCase):
    def test_configuration_accepts_eight_workers_but_rejects_whole_task_retry(self):
        from memos_eval.run import parser
        args = parser().parse_args(["--arms", "C", "--max-concurrency", "8", "--max-retries", "0"])
        configure(args, {"max_concurrency", "max_retries"})
        self.assertEqual(args.max_concurrency, 8)
        args.max_retries = 1
        with self.assertRaisesRegex(ValueError, "max-retries"):
            configure(args, {"max_concurrency", "max_retries"})

    def test_business_projection_keeps_business_call_pairs_and_user_messages(self):
        question = UserMessage(role="user", content="请办理业务")
        business = ToolCall(id="bank", name="bank_tool", arguments={})
        memory = ToolCall(id="memory", name="memos_search", arguments={"query": "流程"})
        mixed = AssistantMessage(role="assistant", tool_calls=[memory, business])
        history = [question, mixed,
                   ToolMessage(role="tool", id="memory", content="历史技能"),
                   ToolMessage(role="tool", id="bank", content="业务执行结果"),
                   AssistantMessage(role="assistant", content="业务已完成")]
        projected = business_messages(history)
        assert len(projected) == 4
        assert projected[0] is question
        assert projected[1].tool_calls == [business]
        assert projected[2].id == "bank"
        assert mixed.tool_calls == [memory, business]

    def test_fact_write_is_removed_from_scoring_but_kept_in_original(self):
        fact = ToolCall(id="fact", name="memos_write_fact", arguments={"content": "已核验"})
        messages = [AssistantMessage(role="assistant", tool_calls=[fact]),
                    ToolMessage(role="tool", id="fact", content="保存成功"),
                    AssistantMessage(role="assistant", content="业务回答")]
        projected = business_messages(messages)
        self.assertEqual(len(projected), 1)
        self.assertEqual(len(messages), 3)


    def test_memory_dispatch_does_not_execute_user_calls_or_replay_business(self):
        calls = []

        def memos_search(query: str) -> dict:
            """查找历史技能。"""
            calls.append(query)
            return {"skills": []}

        env = SimpleNamespace(get_response=lambda message: ("business", message.name))
        install_memory_tools(env, [Tool(memos_search)])
        result = env.get_response(ToolCall(id="m1", name="memos_search", arguments={"query": "verify"}))
        assert json.loads(result.content) == {"skills": []}
        assert not result.error
        rejected = env.get_response(ToolCall(id="u1", requestor="user", name="memos_search",
                                           arguments={"query": "用户内部调用"}))
        assert rejected.error
        assert calls == ["verify"]
        assert env.get_response(ToolCall(name="bank_tool", arguments={})) == ("business", "bank_tool")


    def test_resume_recovers_checkpoint_window_but_rejects_unconfirmed_learning(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = {"seeds": [300], "domains": ["banking_knowledge"], "tasks": [],
                        "active_tasks": [{"seed": 300, "domain": "banking_knowledge", "task_id": "task_001"}]}
            task_dir = output / "C_seed300" / "task_001"
            task_dir.mkdir(parents=True)
            row = {"seed": 300, "domain": "banking_knowledge", "task_id": "task_001", "position": 0,
                   "phase_complete": True, "memory": {"unsafe_to_continue": False}}
            (task_dir / "completed.json").write_text(json.dumps({"task": row, "simulation": {"task_id": "task_001"}}))
            restored = restore_checkpoints(manifest, output)
            assert restored[(300, "banking_knowledge", "task_001")][0] == row
            assert manifest["active_tasks"] == []
            row["memory"]["unsafe_to_continue"] = True
            (task_dir / "completed.json").write_text(json.dumps({"task": row, "simulation": {}}))
            with self.assertRaisesRegex(ValueError, "学习状态"):
                restore_checkpoints(manifest, output)

    def test_parallel_run_isolates_clients_paths_and_checkpoints(self):
        """三个并发任务不能串上下文，主线程独占汇总文件。"""
        from memos_eval import run_skills
        from memos_eval.run import parser
        from tau2.registry import registry
        from tau2.utils.llm_utils import llm_log_mode

        main_thread = threading.get_ident()
        barrier = threading.Barrier(3)
        registered = {}
        agents, clients, saves = [], [], []
        state_lock = threading.Lock()
        tasks = [SimpleNamespace(id=f"task_{i:03d}") for i in range(1, 4)]

        class Client:
            def __init__(self, *args, **kwargs):
                self.request_count = 0
                self.learning_lock = threading.RLock()
                self.learning_aborted = threading.Event()
                self.snapshot_lock = threading.RLock()
                self.closed = False
                with state_lock:
                    clients.append(self)

            def ensure_skill_service(self):
                return {"extraction": "native_skill_only"}

            def create_cube(self, cube_id):
                return cube_id

            def close(self):
                self.closed = True

        class Agent:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                self.memory_tools = []
                with state_lock:
                    agents.append(self)

            def finish(self, *args):
                self.client.request_count += int(self.task_id[-1])

            def summary(self):
                return {"unsafe_to_continue": False, "learning_status": "completed"}

        class Simulation(SimpleNamespace):
            def model_copy(self, update):
                result = copy.copy(self)
                result.__dict__.update(update)
                return result

            def model_dump(self, mode=None):
                return {"task_id": self.task_id, "messages": [], "reward": self.reward_info.reward}

        class Results:
            def __init__(self, **kwargs):
                self.simulations = kwargs["simulations"]

            def save(self, path):
                self_task_ids = [s.task_id for s in self.simulations]
                assert threading.get_ident() == main_thread
                saves.append(self_task_ids)
                path.write_text(json.dumps({"task_ids": self_task_ids}))

        def build(config, task, **kwargs):
            assert llm_log_mode.get() == "all"
            agent = registered["memos_skill_agent"]([], "policy", config.llm_agent, config.llm_args_agent, task)
            barrier.wait(timeout=5)
            assert agent.task_id == task.id
            assert task.id in str(agent.audit_path)
            agent.audit_path.parent.mkdir(parents=True, exist_ok=True)
            agent.audit_path.write_text(task.id)
            orch = SimpleNamespace(environment=SimpleNamespace(get_response=lambda message: None,
                                                                get_policy=lambda: "policy"),
                                   message=None, to_role="user", agent_state=None, user_state=None)
            orch.run = lambda: Simulation(task_id=task.id, messages=[], reward_info=None,
                                         termination_reason="agent_stop")
            return orch

        previous_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "tau2-bench").mkdir()
                output = root / "run"
                args = parser().parse_args(["--arms", "C", "--max-concurrency", "8", "--max-retries", "0",
                                           "--output", str(output)])
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(run_skills, "ROOT", root))
                    stack.enter_context(patch("sys.argv", ["run", "--max-concurrency", "8", "--max-retries", "0"]))
                    stack.enter_context(patch("memos_eval.skill_client.SkillClient", Client))
                    stack.enter_context(patch("memos_eval.skill_agent.SkillAgent", Agent))
                    stack.enter_context(patch.object(registry, "register_agent_factory",
                                                    lambda factory, name: registered.update({name: factory})))
                    stack.enter_context(patch("tau2.runner.helpers.get_tasks", return_value=tasks))
                    stack.enter_context(patch("tau2.runner.helpers.get_info",
                                             return_value=SimpleNamespace(user_info=SimpleNamespace(llm_args={}))))
                    stack.enter_context(patch("tau2.runner.build.build_orchestrator", build))
                    stack.enter_context(patch("tau2.runner.build._build_env_kwargs", return_value={}))
                    stack.enter_context(patch("tau2.runner.batch._TaskLogContext", lambda *args: contextlib.nullcontext()))
                    stack.enter_context(patch("tau2.evaluator.evaluator.evaluate_simulation",
                                             return_value=SimpleNamespace(reward=1.0)))
                    stack.enter_context(patch("tau2.data_model.simulation.Results", Results))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    run_skills.run(args)
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertEqual(manifest["status"], "completed")
                self.assertEqual(manifest["active_tasks"], [])
                self.assertEqual(len({id(agent.client) for agent in agents}), 3)
                self.assertEqual(len({id(agent.client.learning_lock) for agent in agents}), 1)
                self.assertEqual(len({id(agent.client.learning_aborted) for agent in agents}), 1)
                self.assertEqual(len({id(agent.client.snapshot_lock) for agent in agents}), 1)
                self.assertTrue(all(client.closed for client in clients))
                self.assertEqual(saves[-1], [task.id for task in tasks])
                for row in manifest["tasks"]:
                    self.assertEqual(row["memos_http_requests"], int(row["task_id"][-1]))
                    task_dir = output / "C_seed300" / row["task_id"]
                    self.assertTrue((task_dir / "completed.json").exists())
                    self.assertEqual((task_dir / "attempt_0" / "memory_audit.jsonl").read_text(), row["task_id"])
        finally:
            os.chdir(previous_cwd)
