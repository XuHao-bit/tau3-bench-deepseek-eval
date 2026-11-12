"""离线检查真实生成链路的逐次日志及基础设施异常分类。"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from litellm import ModelResponse

from tau2.data_model.message import AssistantMessage, SystemMessage
from tau2.environment.tool import Tool
from tau2.runner.batch import _TaskLogContext
from tau2.user.user_simulator import EmptyUserResponseError, UserSimulator
from tau2.user.user_simulator_base import UserState
from tau2.utils.llm_utils import llm_log_dir, llm_log_mode


def response(content="", tool_calls=None):
    return ModelResponse(
        model="gpt-5.2",
        choices=[{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": content, "tool_calls": tool_calls,
        }}],
        usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    )


class EmptyRetryLoggingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        directory_token = llm_log_dir.set(self.root)
        mode_token = llm_log_mode.set("all")
        self.addCleanup(llm_log_dir.reset, directory_token)
        self.addCleanup(llm_log_mode.reset, mode_token)
        self.state = UserState(
            system_messages=[SystemMessage(role="system", content="模拟用户")], messages=[],
        )
        self.incoming = AssistantMessage(role="assistant", content="请确认操作。")
        self.sleep = self.enterContext(patch("tau2.user.user_simulator.time.sleep"))
        self.enterContext(patch("tau2.utils.llm_utils.get_response_cost", return_value=0.1))

    def test_logs_every_attempt_with_same_request_without_executing_tool(self):
        operation = Mock()

        def confirm() -> str:
            """确认操作。"""
            operation()
            return "已确认"

        user = UserSimulator(llm="openai/gpt-5.2", tools=[Tool(confirm)], llm_args={
            "seed": 300, "extra_body": {"reasoning_effort": "low"},
            "empty_response_retries": 2,
        })
        tools = [{"id": "confirm_1", "type": "function", "function": {
            "name": "confirm", "arguments": "{}",
        }}]
        with patch("tau2.utils.llm_utils.completion", side_effect=[
            response(), response(None), response(None, tools),
        ]) as completion:
            message, state = user.generate_next_message(self.incoming, self.state)
        self.assertEqual(completion.call_count, 3)
        for request in completion.call_args_list:
            self.assertEqual(request, completion.call_args_list[0])
            self.assertNotIn("empty_response_retries", request.kwargs)
        operation.assert_not_called()
        self.assertEqual(message.tool_calls[0].requestor, "user")
        self.assertEqual(state.messages, [self.incoming, message])
        self.assertAlmostEqual(message.cost, 0.3)
        self.assertEqual(message.usage, {"prompt_tokens": 30, "completion_tokens": 12})
        self.assertEqual(message.raw_data["usage"]["prompt_tokens"], 10)
        logs = [json.loads(p.read_text()) for p in self.root.glob("*user_simulator_response*.json")]
        self.assertEqual(len(logs), 3)
        self.assertEqual(sum(log["response"]["usage"]["completion_tokens"] for log in logs), 12)
        self.assertEqual(sum(bool(log["response"]["tool_calls"]) for log in logs), 1)

    def test_exhaustion_remains_infrastructure_error_in_native_task_logs(self):
        user = UserSimulator(llm="openai/gpt-5.2")
        with patch("tau2.utils.llm_utils.completion", side_effect=[response() for _ in range(3)]):
            with self.assertRaises(EmptyUserResponseError):
                with _TaskLogContext("empty-test", self.root, SimpleNamespace(id="task_020"), True):
                    user.generate_next_message(self.incoming, self.state)
        task_dir = self.root / "artifacts/task_task_020/sim_empty-test"
        status = json.loads((task_dir / "sim_status.json").read_text())
        self.assertEqual(status["reason"], "infrastructure_error")
        self.assertEqual(status["error_type"], "EmptyUserResponseError")
        logs = list((task_dir / "llm_debug").glob("*user_simulator_response*.json"))
        self.assertEqual(len(logs), 3)
        self.assertEqual(self.state.messages, [self.incoming])
        self.assertEqual(len(self.state.pending_empty_responses), 3)


if __name__ == "__main__":
    unittest.main()
