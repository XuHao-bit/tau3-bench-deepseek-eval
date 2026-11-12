"""自动记忆的 QA 边界、最终一致性和调用上限测试。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from memos_eval.agent import MemOSAgent
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)


def hit(memory_id="m1", content="Verify the current account before changing it.", **metadata):
    return SimpleNamespace(
        memory_id=memory_id, content=content, task_id="previous_task",
        created_at="2026-11-04", metadata=metadata,
    )


class FakeClient:
    def __init__(self):
        self.hits = [hit()]
        self.searches = []
        self.writes = []
        self.status_queries = []
        self.search_error = False
        self.write_error = False
        self.status = "submitted"
        self.receipt_status = "submitted"

    def search(self, cube_id, session_id, query, top_k):
        self.searches.append((cube_id, session_id, query, top_k))
        if self.search_error:
            raise RuntimeError("search unavailable")
        return self.hits

    def add(self, cube_id, session_id, messages, info):
        self.writes.append((cube_id, session_id, messages, info))
        if self.write_error:
            raise RuntimeError("write response unavailable")
        return {"status": self.receipt_status, "task_id": f"add-{len(self.writes)}", "memory_ids": []}

    def add_status(self, cube_id, session_id, task_id):
        self.status_queries.append((cube_id, session_id, task_id))
        return {
            "status": self.status, "task_id": task_id,
            "memory_ids": ["new-memory"] if self.status == "completed" else [],
        }


def reply(content, finish_reason=None):
    raw = {"choices": [{"finish_reason": finish_reason}]} if finish_reason else None
    return AssistantMessage(role="assistant", content=content, raw_data=raw)


def tool_call(name="lookup", call_id="call1"):
    return AssistantMessage(
        role="assistant", content=None,
        tool_calls=[ToolCall(id=call_id, name=name, arguments={})],
    )


class MemOSAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.client = FakeClient()
        # 本地替身只计文本长度，不下载 tokenizer 词表。
        encoding = SimpleNamespace(encode=lambda text, **kwargs: list(text))
        tokenizer = patch("memos_eval.agent.tiktoken.get_encoding", return_value=encoding)
        tokenizer.start()
        self.addCleanup(tokenizer.stop)

    def agent(self, **kwargs):
        agent = MemOSAgent(
            [], "Original bank policy.", "openai/deepseek-test",
            {"extra_body": {"reasoning_effort": "max"}},
            client=self.client, cube_id="cube", task_id="current_task", arm="B-cross",
            audit_path=Path(self.temp.name) / "audit.jsonl", **kwargs,
        )
        return agent, agent.get_init_state([reply("Hello")])

    def events(self):
        return [json.loads(line) for line in (Path(self.temp.name) / "audit.jsonl").read_text().splitlines()]

    def test_current_user_query_and_copied_injection_leave_history_unmodified(self):
        agent, state = self.agent()
        user = UserMessage(role="user", content="Change my address", raw_data={"hidden": "never upload"})
        with patch("memos_eval.agent.generate", return_value=reply("Please verify your account", "stop")) as generate:
            agent.generate_next_message(user, state)
        outgoing = generate.call_args.kwargs["messages"]
        self.assertEqual(self.client.searches[0], ("cube", "current_task", "Change my address", 6))
        self.assertIn("<memory_context", outgoing[2].content)
        self.assertIn("## 相关记忆\n", outgoing[2].content)
        self.assertIn("<reference_data>", outgoing[2].content)
        self.assertEqual(user.content, "Change my address")
        self.assertEqual(state.messages[1].content, "Change my address")
        self.assertIn("Original bank policy.", outgoing[0].content)
        qa = self.client.writes[0][2]
        self.assertEqual(qa, [
            {"role": "user", "content": "Change my address"},
            {"role": "assistant", "content": "Please verify your account"},
        ])
        self.assertNotIn("hidden", json.dumps(qa))
        self.assertNotIn("Hello", json.dumps(qa))
        self.assertEqual(self.client.status_queries, [])
        self.assertEqual(agent.summary()["write_status_counts"], {"submitted": 1})

    def test_tool_chain_keeps_native_history_but_only_saves_final_qa(self):
        agent, state = self.agent()
        with patch("memos_eval.agent.generate", side_effect=[tool_call(), reply("Done")]) as generate:
            agent.generate_next_message(UserMessage(role="user", content="Look up my account"), state)
            self.assertEqual(self.client.writes, [])
            result = ToolMessage(role="tool", id="call1", content="Current balance: 12")
            agent.generate_next_message(MultiToolMessage(role="tool", tool_messages=[result]), state)
        self.assertEqual(len(self.client.searches), 1)
        first = generate.call_args_list[0].kwargs["messages"][2].content
        second = generate.call_args_list[1].kwargs["messages"][2].content
        self.assertEqual(first, second)
        self.assertEqual(self.client.writes[0][2], [
            {"role": "user", "content": "Look up my account"},
            {"role": "assistant", "content": "Done"},
        ])
        outgoing = generate.call_args_list[1].kwargs["messages"]
        self.assertEqual(outgoing[3].tool_calls[0].name, "lookup")
        self.assertEqual(outgoing[4].content, "Current balance: 12")
        self.assertEqual(agent.summary()["injections"], 2)
        self.assertEqual(agent.summary()["injected_tokens"], 2 * agent._block_tokens)

    def test_memory_body_is_complete_and_duplicates_keep_server_order(self):
        long_content = "long reference " * 3000
        self.client.hits = [hit("duplicate", long_content, memory_key="Full record") for _ in range(7)]
        agent, state = self.agent()
        with patch("memos_eval.agent.generate", return_value=reply("Done")) as generate:
            agent.generate_next_message(UserMessage(role="user", content="Question"), state)
        outgoing = generate.call_args.kwargs["messages"][2].content
        self.assertEqual(outgoing.count(long_content), 6)
        self.assertIn("[6] Full record", outgoing)
        self.assertNotIn("[7]", outgoing)
        self.assertEqual(agent.summary()["read_tokens"], 6 * len(long_content))
        self.assertGreater(agent.summary()["write_tokens"], 0)

    def test_new_turn_writes_only_current_qa_and_never_reuploads_memory(self):
        agent, state = self.agent()
        with patch("memos_eval.agent.generate", side_effect=[reply("First answer"), reply("Second answer")]):
            agent.generate_next_message(UserMessage(role="user", content="First"), state)
            first_read = agent.summary()["read_tokens"]
            agent.generate_next_message(UserMessage(role="user", content="Second"), state)
        self.assertEqual(self.client.searches[1][2], "Second")
        self.assertEqual(self.client.writes[1][2], [
            {"role": "user", "content": "Second"},
            {"role": "assistant", "content": "Second answer"},
        ])
        self.assertEqual(agent.summary()["read_tokens"], 2 * first_read)
        self.assertEqual(len(self.client.status_queries), 1)
        self.assertEqual(agent.summary()["write_status_counts"], {"submitted": 2})

    def test_normal_finish_never_writes_a_tool_only_tail(self):
        agent, state = self.agent()
        with patch("memos_eval.agent.generate", return_value=tool_call("transfer", "transfer1")):
            agent.generate_next_message(UserMessage(role="user", content="Please transfer me"), state)
        result = ToolMessage(role="tool", id="transfer1", content="Transferred")
        agent.finish("user_stop", MultiToolMessage(role="tool", tool_messages=[result]))
        agent.finish("user_stop", result)
        self.assertEqual(self.client.writes, [])
        self.assertEqual(sum(row["event"] == "pending_write_skipped" for row in self.events()), 1)

    def test_abnormal_finish_discards_incomplete_round(self):
        agent, state = self.agent()
        with patch("memos_eval.agent.generate", return_value=tool_call()):
            agent.generate_next_message(UserMessage(role="user", content="Check"), state)
        agent.finish("max_steps", ToolMessage(role="tool", id="call1", content="result"))
        self.assertEqual(self.client.writes, [])
        self.assertEqual(self.events()[-1]["reason"], "max_steps")

    def test_truncated_answer_is_not_submitted(self):
        agent, state = self.agent()
        with patch("memos_eval.agent.generate", return_value=reply("Partial answer", "length")):
            agent.generate_next_message(UserMessage(role="user", content="Question"), state)
        agent.finish("user_stop")
        self.assertEqual(self.client.writes, [])
        self.assertTrue(any(row["event"] == "write_skipped" and row["reason"] == "incomplete_reply" for row in self.events()))

    def test_failed_search_stops_generation_as_infrastructure_error(self):
        self.client.search_error = True
        agent, state = self.agent()
        with patch("memos_eval.agent.generate") as generate:
            with self.assertRaisesRegex(RuntimeError, "search unavailable"):
                agent.generate_next_message(UserMessage(role="user", content="Question"), state)
        generate.assert_not_called()
        self.assertEqual(self.client.writes, [])
        self.assertEqual(agent.summary()["search_errors"], 1)

    def test_uncertain_submission_is_not_repeated_and_prevents_unsafe_cleanup(self):
        self.client.write_error = True
        agent, state = self.agent()
        with patch("memos_eval.agent.generate", return_value=reply("Answer")):
            agent.generate_next_message(UserMessage(role="user", content="Question"), state)
        agent.finish("user_stop")
        self.assertEqual(len(self.client.writes), 1)
        self.assertEqual(agent.summary()["write_status_counts"], {"unknown": 1})
        self.assertFalse(agent.settle_writes(timeout=0))
        self.assertEqual(self.client.status_queries, [])

    def test_unknown_receipt_keeps_ticket_and_can_later_complete(self):
        self.client.receipt_status = "unknown"
        agent, state = self.agent()
        with patch("memos_eval.agent.generate", return_value=reply("Answer")):
            agent.generate_next_message(UserMessage(role="user", content="Question"), state)
        error = next(row for row in self.events() if row["event"] == "memory_write_error")
        self.assertEqual(error["status"], "unknown")
        self.assertEqual(error["add_task_id"], "add-1")
        self.assertEqual(agent.summary()["submitted_turns"], 0)
        self.client.status = "completed"
        self.assertTrue(agent.settle_writes(timeout=0))
        self.assertEqual(agent.summary()["submitted_turns"], 1)
        self.assertEqual(agent.summary()["completed_turns"], 1)
        self.assertEqual(len(self.client.writes), 1)

    def test_completed_submission_and_first_retrieval_are_distinct_observations(self):
        agent, state = self.agent()
        with patch("memos_eval.agent.generate", side_effect=[reply("First answer"), reply("Second answer")]):
            agent.generate_next_message(UserMessage(role="user", content="First"), state)
            self.client.status = "completed"
            self.client.hits = [hit("new-memory", "First answer")]
            agent.generate_next_message(UserMessage(role="user", content="Second"), state)
        first = agent.summary()["write_submissions"][0]
        self.assertEqual(first["status"], "completed")
        self.assertEqual(first["submitted_turn"], 1)
        self.assertEqual(first["first_retrievable_turn"], 2)
        status_event = next(row for row in self.events() if row["event"] == "memory_write_status")
        self.assertEqual(status_event["query_turn"], 2)
        self.assertEqual(status_event["submitted_turn"], 1)
        self.assertGreaterEqual(first["first_retrievable_delay_seconds"], 0)

    def test_cleanup_waits_for_terminal_status_and_ignores_policy_call_limit(self):
        agent, state = self.agent(max_memory_calls=2)
        with patch("memos_eval.agent.generate", return_value=reply("Answer")):
            agent.generate_next_message(UserMessage(role="user", content="Question"), state)
        self.assertFalse(agent.settle_writes(timeout=0))
        self.client.status = "failed"
        self.assertTrue(agent.settle_writes(timeout=0))
        self.assertEqual(agent.summary()["write_status_counts"], {"failed": 1})
        self.assertEqual(agent.summary()["memory_calls"], 2)
        self.assertEqual(agent.summary()["cleanup_status_calls"], 2)

    def test_call_limit_stops_memory_operations_but_not_business(self):
        agent, state = self.agent(max_memory_calls=1)
        with patch("memos_eval.agent.generate", side_effect=[reply("One"), reply("Two")]):
            agent.generate_next_message(UserMessage(role="user", content="Question one"), state)
            agent.generate_next_message(UserMessage(role="user", content="Question two"), state)
        self.assertEqual(len(self.client.searches), 1)
        self.assertEqual(self.client.writes, [])
        self.assertEqual(state.messages[-1].content, "Two")
        self.assertEqual(agent.summary()["memory_calls"], 1)

    def test_init_state_resets_task_local_state(self):
        agent, state = self.agent()
        with patch("memos_eval.agent.generate", return_value=reply("Answer")):
            agent.generate_next_message(UserMessage(role="user", content="Question"), state)
        self.assertGreater(agent.summary()["read_tokens"], 0)
        fresh = agent.get_init_state()
        self.assertEqual(agent.summary()["source_tasks"], {})
        self.assertEqual(agent.summary()["write_submissions"], [])
        self.assertEqual(agent.turn, 0)
        self.assertEqual(fresh.messages, [])


if __name__ == "__main__":
    unittest.main()
