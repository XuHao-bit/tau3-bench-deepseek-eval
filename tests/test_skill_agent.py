"""C 组的轨迹边界、自主工具调用和同步学习审计。"""

import copy
import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage

from memos_eval.skill_agent import SkillAgent


def skill(procedure="Check the live state before acting."):
    return {
        "id": "skill-old", "memory": "Verify a live business precondition.",
        "metadata": {
            "memory_type": "SkillMemory", "status": "activated", "name": "live-check",
            "procedure": procedure, "experience": ["Do not reuse a historical account ID."],
            "sources": [{"role": "user", "content": "PRIVATE SOURCE CONTENT", "message_id": "m1"}],
            "info": {"task_id": "previous-task", "source_domain": "banking_knowledge"},
        },
    }


def reply(content="Done", calls=None, **kwargs):
    return AssistantMessage(role="assistant", content=content, tool_calls=calls, **kwargs)


def business_history(count=5, identical=False):
    history = [UserMessage(role="user", content="Please handle my request", raw_data={"hidden": "SECRET"})]
    for i in range(count):
        history.append(reply(None, [ToolCall(id=f"b{i}", name="lookup", arguments={"step": i})]))
        history.append(ToolMessage(role="tool", id=f"b{i}", content="same" if identical else f"observed {i}"))
    return history + [reply()]


class FakeClient:
    def __init__(self):
        self.learning_lock = threading.RLock()
        self.snapshot_lock = threading.RLock()
        self.learning_aborted = threading.Event()
        self.skills = [skill()]
        self.facts = []
        self.searches = []
        self.reads = []
        self.learns = []
        self.learn_status = "completed"
        self.writes = []

    def list_skills(self, cube):
        return copy.deepcopy(self.skills)

    def list_all_facts(self, cube):
        return copy.deepcopy(self.facts)

    def search_memories(self, cube, session, query, memory_type="all", top_k=6):
        self.searches.append((cube, session, query, memory_type, top_k))
        facts = [row for row in self.facts if row["metadata"]["info"]["fact_scope"] == "shared"
                 or row["metadata"]["session_id"] == session]
        nodes = (self.skills if memory_type == "skill" else facts if memory_type == "fact" else facts + self.skills)
        return copy.deepcopy(nodes[:top_k])

    def write_fact(self, cube, content, task_id, domain, scope="task"):
        session = f"{domain}:{task_id}"
        info = {"task_id": task_id, "source_domain": domain}
        self.writes.append((cube, session, content, scope, info))
        node = {"id": f"fact-{len(self.writes)}", "memory": content,
                "metadata": {"memory_type": "LongTermMemory", "session_id": session,
                             "info": {**info, "fact_scope": scope}}}
        self.facts.append(node)
        return {"status": "completed", "memory_ids": [node["id"]], "memories": [node]}

    def get_skills(self, cube, ids):
        self.reads.append((cube, ids))
        return copy.deepcopy([row for row in self.skills if row["id"] in ids])

    def learn(self, cube, session, messages, info):
        self.learns.append((cube, session, copy.deepcopy(messages), copy.deepcopy(info)))
        if self.learn_status == "completed":
            self.skills = [skill("Confirm both identity and the live state.")]
        return {"status": self.learn_status, "skill_ids": ["skill-old"],
                "skills": copy.deepcopy(self.skills), "raw_add": []}


class SkillAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.client = FakeClient()

    def agent(self, history=None, **kwargs):
        agent = SkillAgent(
            [], "Keep the original domain policy.", "openai/deepseek-test",
            {"extra_body": {"reasoning_effort": "max"}}, client=self.client,
            cube_id="shared-cube", task_id="current-task", domain="banking_knowledge",
            audit_path=self.root / "audit.jsonl", skills_dir=self.root, **kwargs,
        )
        return agent, agent.get_init_state(history or [])

    def events(self):
        return [json.loads(row) for row in (self.root / "audit.jsonl").read_text().splitlines()]

    def test_business_generation_does_not_automatically_search_or_inject_skill(self):
        agent, state = self.agent()
        user = UserMessage(role="user", content="My current request")
        with patch("memos_eval.skill_agent.generate", return_value=reply()) as generate:
            agent.generate_next_message(user, state)
        self.assertEqual(self.client.searches, [])
        self.assertEqual(state.messages[0].content, "My current request")
        self.assertNotIn("Verify a live business precondition", str(generate.call_args.kwargs["messages"]))
        self.assertEqual({tool.name for tool in agent.memory_tools}, {"memos_search", "memos_write_fact", "memos_learn"})

    def test_search_and_read_preserve_procedure_but_exclude_raw_sources(self):
        agent, _ = self.agent()
        result = agent.memos_search("live preconditions", memory_type="skill", top_k=6)
        self.assertEqual(self.client.searches[0][:2], ("shared-cube", "banking_knowledge:current-task"))
        self.assertEqual(result["memories"][0]["procedure"], self.client.skills[0]["metadata"]["procedure"])
        self.assertNotIn("PRIVATE SOURCE CONTENT", json.dumps(result))
        self.assertEqual(result["memories"][0]["experience"], self.client.skills[0]["metadata"]["experience"])
        self.assertEqual(agent.summary()["cross_task_skill_ids"], ["skill-old"])

    def test_learning_uses_only_real_business_history_and_updates_same_id(self):
        history = business_history()
        history += [reply(None, [ToolCall(id="mem1", name="memos_search", arguments={"query": "x"})]),
                    ToolMessage(role="tool", id="mem1", content="MEMORY TOOL CONTENT")]
        agent, state = self.agent(history)
        # 用户模拟器内部过程不属于可见业务轨迹。
        state.messages.append(ToolMessage(role="tool", id="private", content="SIMULATOR SECRET", requestor="user"))
        result = agent.memos_learn(["all"])
        self.assertEqual(result["status"], "completed")
        payload = json.dumps(self.client.learns[0][2])
        for private in ("SECRET", "MEMORY TOOL CONTENT", "SIMULATOR SECRET", "memos_search", "raw_data"):
            self.assertNotIn(private, payload)
        self.assertEqual(sum(row["role"] == "tool" for row in self.client.learns[0][2]), 5)
        self.assertIn("lookup", payload)
        changes = [json.loads(line) for line in (self.root / "skill_changes.jsonl").read_text().splitlines()]
        self.assertEqual(changes[-1]["updated_ids"], ["skill-old"])
        self.assertEqual(changes[-1]["added_ids"], [])
        latest = json.loads((self.root / "skills_latest.json").read_text())
        self.assertEqual(latest["skills"][0]["metadata"]["procedure"], self.client.skills[0]["metadata"]["procedure"])
        again = agent.memos_learn(["all"])
        self.assertEqual(again["status"], "skipped")
        self.assertEqual(len(self.client.learns), 1)

    def test_five_duplicate_tool_outputs_do_not_meet_native_gate(self):
        agent, _ = self.agent(business_history(5, identical=True))
        result = agent.memos_learn(["all"])
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["unique_tool_messages"], 1)
        self.assertEqual(self.client.learns, [])

    def test_incomplete_business_call_is_not_uploaded(self):
        history = business_history() + [reply(None, [ToolCall(id="pending", name="transfer", arguments={})])]
        agent, _ = self.agent(history)
        self.assertEqual(agent.memos_learn(["all"])["status"], "skipped")
        self.assertEqual(self.client.learns, [])

    def test_unknown_learn_is_not_repeated_and_stops_further_learning(self):
        agent, _ = self.agent(business_history())
        self.client.learn_status = "unknown"
        self.assertEqual(agent.memos_learn(["all"])["status"], "unknown")
        self.assertTrue(agent.unsafe_to_continue)
        agent.memos_learn(["all"])
        with patch("memos_eval.skill_agent.generate") as generate:
            agent.finish("user_stop")
        generate.assert_not_called()
        self.assertEqual(len(self.client.learns), 1)

    def test_failed_after_snapshot_does_not_invent_deleted_skills(self):
        agent, _ = self.agent(business_history())
        with patch.object(self.client, "list_skills", side_effect=[copy.deepcopy(self.client.skills), RuntimeError("read failed")]):
            result = agent.memos_learn(["all"])
        self.assertEqual(result["status"], "unknown")
        self.assertTrue(agent.unsafe_to_continue)
        change = json.loads((self.root / "skill_changes.jsonl").read_text().splitlines()[-1])
        self.assertIsNone(change["after_skills"])
        self.assertEqual(change["removed_ids"], [])
        self.assertIsNone(agent.summary()["final_skill_count"])

    def test_post_phase_can_learn_but_cannot_execute_business_tools(self):
        agent, state = self.agent(business_history())
        before = list(state.messages)
        responses = [
            reply(None, [ToolCall(id="bad", name="bank_transfer", arguments={})]),
            reply(None, [ToolCall(id="learn", name="memos_learn", arguments={"message_ids": ["all"]})]),
            reply("Finished learning"),
        ]
        with patch("memos_eval.skill_agent.generate", side_effect=responses) as generate:
            agent.finish("user_stop")
        self.assertEqual(len(self.client.learns), 1)
        self.assertEqual(state.messages, before)
        self.assertTrue(all({tool.name for tool in call.kwargs["tools"]} == {"memos_search", "memos_write_fact", "memos_learn"}
                            for call in generate.call_args_list))
        self.assertEqual(agent.summary()["post_learning_usage"]["calls"], 3)
        self.assertTrue(any(row["event"] == "post_learning_tool_rejected" for row in self.events()))

    def test_infrastructure_failure_does_not_trigger_post_phase(self):
        agent, _ = self.agent(business_history())
        with patch("memos_eval.skill_agent.generate") as generate:
            agent.finish("infrastructure_error")
        generate.assert_not_called()
        self.assertEqual(self.client.learns, [])

    def test_fact_write_defaults_to_task_and_search_returns_full_content(self):
        agent, _ = self.agent()
        receipt = agent.memos_write_fact("The customer supplied a phone number.")
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(self.client.writes[0][3], "task")
        result = agent.memos_search("customer phone", memory_type="fact")
        self.assertEqual(result["memories"][0]["content"], "The customer supplied a phone number.")
        self.assertEqual(result["memories"][0]["memory_type"], "fact")
        self.assertEqual(agent.summary()["business_read_fact_ids"], ["fact-1"])
        latest = json.loads((self.root / "facts_latest.json").read_text())
        self.assertEqual(latest["facts"][0]["memory"], "The customer supplied a phone number.")

    def test_cross_task_reads_use_returned_source_not_initial_snapshot(self):
        agent, _ = self.agent()
        late = skill()
        late["id"] = "late-arriving-skill"
        self.client.skills = [late]
        agent.memos_search("live state", memory_type="skill")
        self.assertEqual(agent.summary()["business_cross_task_skill_ids"], ["late-arriving-skill"])
        event = agent.summary()["read_events"][-1]
        self.assertEqual(event["returned_versions"][0]["source_task_id"], "previous-task")

    def test_parallel_learning_keeps_before_after_and_snapshot_serialized(self):
        first, _ = self.agent(business_history())
        second = SkillAgent([], "Policy", "openai/deepseek-test", client=self.client,
                            cube_id="shared-cube", task_id="second-task", domain="banking_knowledge",
                            audit_path=self.root / "second.jsonl", skills_dir=self.root)
        second.get_init_state(business_history())
        native_learn = self.client.learn
        active = 0
        max_active = 0
        count_lock = threading.Lock()

        def slow_learn(*args, **kwargs):
            nonlocal active, max_active
            with count_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.03)
            result = native_learn(*args, **kwargs)
            with count_lock:
                active -= 1
            return result

        with patch.object(self.client, "learn", side_effect=slow_learn), ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda agent: agent.memos_learn(["all"]), [first, second]))
        self.assertEqual([result["status"] for result in results], ["completed", "completed"])
        self.assertEqual(max_active, 1)
        self.assertEqual(len((self.root / "skill_changes.jsonl").read_text().splitlines()), 2)

    def test_unknown_learning_blocks_other_tasks_before_they_write(self):
        first, _ = self.agent(business_history())
        second = SkillAgent([], "Policy", "openai/deepseek-test", client=self.client,
                            cube_id="shared-cube", task_id="second-task", domain="banking_knowledge",
                            audit_path=self.root / "second.jsonl", skills_dir=self.root)
        second.get_init_state(business_history())
        self.client.learn_status = "unknown"
        first.memos_learn(["all"])
        blocked = second.memos_learn(["all"])
        self.assertEqual(blocked["reason"], "blocked_by_unconfirmed_learning")
        self.assertTrue(second.unsafe_to_continue)
        self.assertEqual(len(self.client.learns), 1)
        self.assertEqual(second.memos_write_fact("A public policy", scope="shared")["reason"], "blocked_by_unconfirmed_learning")
        self.assertEqual(self.client.writes, [])

    def test_confirmed_extraction_failure_without_write_allows_next_task_learning(self):
        first, _ = self.agent(business_history())
        receipt = {"status": "failed", "write_state": "not_started", "skill_ids": [], "skills": [],
                   "raw_add": {"data": []}, "error": "原生提炼响应截断"}
        with patch.object(self.client, "learn", return_value=receipt) as learn:
            result = first.memos_learn(["all"])
            first.memos_learn(["all"])
        self.assertEqual(learn.call_count, 1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"], "原生提炼响应截断")
        self.assertFalse(first.summary()["unsafe_to_continue"])
        self.assertTrue(first.summary()["learning_failure_without_write"])
        self.assertFalse(self.client.learning_aborted.is_set())
        second = SkillAgent([], "Policy", "openai/deepseek-test", client=self.client,
                            cube_id="shared-cube", task_id="next-task", domain="banking_knowledge",
                            audit_path=self.root / "next.jsonl", skills_dir=self.root)
        second.get_init_state(business_history())
        self.assertEqual(second.memos_learn(["all"])["status"], "completed")

    def test_failed_learning_after_write_started_still_aborts_shared_chain(self):
        agent, _ = self.agent(business_history())
        receipt = {"status": "failed", "write_state": "started", "skill_ids": [], "skills": [],
                   "raw_add": {"data": []}, "error": "写入时失败"}
        with patch.object(self.client, "learn", return_value=receipt):
            self.assertEqual(agent.memos_learn(["all"])["status"], "failed")
        self.assertTrue(agent.summary()["unsafe_to_continue"])
        self.assertTrue(self.client.learning_aborted.is_set())

    def test_failed_learning_with_changed_snapshot_cannot_claim_no_write(self):
        agent, _ = self.agent(business_history())
        receipt = {"status": "failed", "write_state": "not_started", "skill_ids": [], "skills": [],
                   "raw_add": {"data": []}, "error": "提炼失败"}
        def changed_snapshot(*args):
            self.client.skills = [skill("Unexpectedly changed procedure.")]
            return receipt
        with patch.object(self.client, "learn", side_effect=changed_snapshot):
            agent.memos_learn(["all"])
        self.assertTrue(agent.summary()["unsafe_to_continue"])
        self.assertTrue(self.client.learning_aborted.is_set())

    def test_skipped_learning_does_not_block_other_tasks(self):
        agent, _ = self.agent(business_history(2))
        agent.memos_learn(["all"])
        self.assertFalse(self.client.learning_aborted.is_set())

    def test_fast_fact_write_does_not_wait_for_another_task_learning(self):
        agent, _ = self.agent()
        held = threading.Event()
        release = threading.Event()

        def hold_learning():
            with self.client.learning_lock:
                held.set()
                release.wait(2)

        with ThreadPoolExecutor(max_workers=2) as pool:
            holder = pool.submit(hold_learning)
            self.assertTrue(held.wait(1))
            writing = pool.submit(agent.memos_write_fact, "Current verification state")
            try:
                self.assertEqual(writing.result(timeout=0.5)["status"], "completed")
            finally:
                release.set()
                holder.result(timeout=1)

    def test_finish_with_confirmed_learning_does_not_wait_for_other_tasks(self):
        agent, _ = self.agent(business_history())
        self.assertEqual(agent.memos_learn(["all"])["status"], "completed")
        held = threading.Event()
        release = threading.Event()

        def hold_learning():
            with self.client.learning_lock:
                held.set()
                release.wait(2)

        with patch("memos_eval.skill_agent.generate", return_value=reply("Learning is complete")), \
                ThreadPoolExecutor(max_workers=2) as pool:
            holder = pool.submit(hold_learning)
            self.assertTrue(held.wait(1))
            finishing = pool.submit(agent.finish, "user_stop")
            try:
                finishing.result(timeout=0.5)
            finally:
                release.set()
                holder.result(timeout=1)
        self.assertFalse(agent.summary()["unsafe_to_continue"])
        self.assertEqual(agent.summary()["final_skill_count"], 1)
        latest = json.loads((self.root / "skills_latest.json").read_text())
        self.assertEqual(latest["status"], "completed")


if __name__ == "__main__":
    unittest.main()
