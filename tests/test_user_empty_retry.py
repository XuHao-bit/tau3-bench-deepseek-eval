"""用户模拟器空响应的请求重试、历史与开销测试。"""

import unittest
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from threading import Barrier, Lock
from unittest.mock import patch

from memos_eval.run import reported_usage
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.user.user_simulator import EmptyUserResponseError, UserSimulator
from tau2.utils.llm_utils import get_cost, get_token_usage


def response(content=None, *, tool_calls=None, number=1, cost=0.1, usage=True):
    tokens = {"prompt_tokens": number * 10, "completion_tokens": number}
    raw_usage = {**tokens, "completion_tokens_details": {"reasoning_tokens": number}}
    return AssistantMessage(
        role="assistant", content=content, tool_calls=tool_calls, cost=cost,
        usage=tokens if usage else None,
        raw_data={"id": f"response-{number}", "usage": raw_usage if usage else None,
                  "choices": [{"finish_reason": "stop"}]},
    )


class UserEmptyRetryTests(unittest.TestCase):
    def setUp(self):
        self.sleep = patch("tau2.user.user_simulator.time.sleep").start()
        self.addCleanup(patch.stopall)

    def simulator(self, **llm_args):
        user = UserSimulator(
            "openai/gpt-5.2", instructions="询问账户状态。", llm_args=llm_args,
        )
        user.set_seed(300)
        return user, user.get_init_state()

    def test_default_retries_keep_identical_request_and_account_for_every_call(self):
        user, state = self.simulator(extra_body={"reasoning_effort": "low"})
        incoming = AssistantMessage(role="assistant", content="账户已确认。")
        snapshots = []
        replies = iter([
            response(None, number=1, cost=0.1),
            response(" \n ", tool_calls=[], number=2, cost=0.2),
            response("请继续。", number=3, cost=0.3),
        ])

        def generate(**kwargs):
            snapshots.append(deepcopy(kwargs))
            return next(replies)

        with patch("tau2.user.user_simulator.generate", side_effect=generate):
            result, result_state = user.generate_next_message(incoming, state)

        self.assertIs(result_state, state)
        self.assertEqual(len(snapshots), 3)
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual(snapshots[1], snapshots[2])
        self.assertEqual(snapshots[0]["seed"], 300)
        self.assertEqual(snapshots[0]["extra_body"], {"reasoning_effort": "low"})
        self.assertNotIn("empty_response_retries", snapshots[0])
        self.assertEqual([m.content for m in state.messages], ["账户已确认。", "请继续。"])
        self.assertEqual(state.pending_empty_responses, [])
        retries = result.raw_data["empty_response_retries"]
        self.assertEqual([r["raw_data"]["id"] for r in retries], ["response-1", "response-2"])
        self.assertEqual(result.raw_data["usage"]["prompt_tokens"], 30)
        self.assertEqual(result.usage, {"prompt_tokens": 60, "completion_tokens": 6})
        self.assertAlmostEqual(result.cost, 0.6)
        self.assertAlmostEqual(get_cost([result])[1], 0.6)
        self.assertEqual(get_token_usage([result]), {"prompt_tokens": 60, "completion_tokens": 6})
        self.assertEqual(reported_usage(state, "user"), {
            "calls": 3, "input_tokens": 60, "output_tokens": 6,
            "reasoning_tokens": 6, "reported_cost_usd": result.cost,
        })
        self.assertEqual(self.sleep.call_count, 2)

    def test_configured_retry_count_is_local_and_does_not_mutate_original_args(self):
        for retries in (0, 1, 2):
            with self.subTest(retries=retries):
                original = {"empty_response_retries": retries, "seed": 301}
                user = UserSimulator("openai/gpt-5.2", instructions="询问。", llm_args=original)
                state = user.get_init_state()
                with patch("tau2.user.user_simulator.generate", side_effect=lambda **kw: response()) as generate:
                    with self.assertRaises(EmptyUserResponseError):
                        user.generate_next_message(AssistantMessage(role="assistant", content="你好。"), state)
                self.assertEqual(generate.call_count, retries + 1)
                self.assertTrue(all("empty_response_retries" not in c.kwargs for c in generate.call_args_list))
                self.assertEqual(original, {"empty_response_retries": retries, "seed": 301})
                self.assertEqual(len(state.pending_empty_responses), retries + 1)

    def test_retry_after_tool_results_appends_each_result_once(self):
        for multiple in (False, True):
            with self.subTest(multiple=multiple):
                user, state = self.simulator()
                state.messages.append(UserMessage(role="user", tool_calls=[
                    ToolCall(id="check-1", name="check_account", arguments={}, requestor="user"),
                ]))
                result = ToolMessage(role="tool", id="check-1", content="账户有效。", requestor="user")
                incoming = MultiToolMessage(role="tool", tool_messages=[result]) if multiple else result
                with patch("tau2.user.user_simulator.generate", side_effect=[response(), response("谢谢。")]) as generate:
                    user.generate_next_message(incoming, state)
                self.assertEqual(generate.call_count, 2)
                self.assertEqual(len([m for m in state.messages if isinstance(m, ToolMessage)]), 1)
                for call in generate.call_args_list:
                    self.assertEqual(len([m for m in call.kwargs["messages"] if isinstance(m, ToolMessage)]), 1)

    def test_tool_only_response_is_returned_without_retry_or_execution(self):
        user, state = self.simulator()
        call = ToolCall(id="check-1", name="check_account", arguments={"account": "123"})
        with patch("tau2.user.user_simulator.generate", return_value=response(tool_calls=[call])) as generate:
            result, _ = user.generate_next_message(AssistantMessage(role="assistant", content="请核对。"), state)
        self.assertEqual(generate.call_count, 1)
        self.sleep.assert_not_called()
        self.assertEqual(result.tool_calls[0].requestor, "user")
        self.assertEqual(result.tool_calls[0].arguments, {"account": "123"})
        self.assertEqual(state.pending_empty_responses, [])

    def test_exhaustion_is_infrastructure_error_and_keeps_billable_empty_calls(self):
        user, state = self.simulator()
        incoming = AssistantMessage(role="assistant", content="操作完成。")
        with patch("tau2.user.user_simulator.generate", side_effect=[
            response(None, number=1), response("", number=2), response(" \t", tool_calls=[], number=3),
        ]) as generate:
            with self.assertRaises(EmptyUserResponseError) as raised:
                user.generate_next_message(incoming, state)
        self.assertIsInstance(raised.exception, RuntimeError)
        self.assertEqual(generate.call_count, 3)
        self.assertEqual(state.messages, [incoming])
        self.assertEqual(len(state.pending_empty_responses), 3)
        totals = reported_usage(state, "user")
        self.assertEqual(totals["calls"], 3)
        self.assertEqual(totals["input_tokens"], 60)
        self.assertEqual(totals["output_tokens"], 6)
        self.assertEqual(totals["reasoning_tokens"], 6)
        self.assertAlmostEqual(totals["reported_cost_usd"], 0.3)
        self.assertEqual(reported_usage(state, "assistant")["calls"], 0)

    def test_nonempty_response_is_not_retried(self):
        user, state = self.simulator()
        with patch("tau2.user.user_simulator.generate", return_value=response("继续。")) as generate:
            result, _ = user.generate_next_message(AssistantMessage(role="assistant", content="你好。"), state)
        self.assertEqual(result.content, "继续。")
        self.assertEqual(generate.call_count, 1)
        self.sleep.assert_not_called()
        self.assertEqual(state.pending_empty_responses, [])

    def test_model_exception_is_not_retried(self):
        user, state = self.simulator()
        failure = TimeoutError("连接超时")
        with patch("tau2.user.user_simulator.generate", side_effect=failure) as generate:
            with self.assertRaises(TimeoutError) as raised:
                user.generate_next_message(AssistantMessage(role="assistant", content="你好。"), state)
        self.assertIs(raised.exception, failure)
        self.assertEqual(generate.call_count, 1)
        self.sleep.assert_not_called()
        self.assertEqual(state.pending_empty_responses, [])

    def test_model_exception_after_empty_preserves_earlier_usage(self):
        user, state = self.simulator()
        with patch("tau2.user.user_simulator.generate", side_effect=[response(), TimeoutError("连接超时")]) as generate:
            with self.assertRaises(TimeoutError):
                user.generate_next_message(AssistantMessage(role="assistant", content="你好。"), state)
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(len(state.pending_empty_responses), 1)
        self.assertEqual(reported_usage(state, "user")["calls"], 1)
        self.assertAlmostEqual(reported_usage(state, "user")["reported_cost_usd"], 0.1)

    def test_unknown_cost_or_usage_stays_unknown_and_raw_attempts_survive(self):
        user, state = self.simulator()
        with patch("tau2.user.user_simulator.generate", side_effect=[
            response(cost=None, usage=False), response("继续。", number=2),
        ]):
            result, _ = user.generate_next_message(AssistantMessage(role="assistant", content="你好。"), state)
        self.assertIsNone(result.cost)
        self.assertIsNone(result.usage)
        self.assertEqual(result.raw_data["usage"]["prompt_tokens"], 20)
        self.assertIsNone(result.raw_data["empty_response_retries"][0]["cost"])
        self.assertEqual(state.pending_empty_responses, [])

    def test_concurrent_users_keep_retry_records_separate(self):
        users = []
        for seed in (300, 301):
            user, state = self.simulator()
            user.set_seed(seed)
            users.append((user, state))
        counts = {}
        lock = Lock()
        barrier = Barrier(2)

        def generate(**kwargs):
            seed = kwargs["seed"]
            with lock:
                count = counts[seed] = counts.get(seed, 0) + 1
            if count == 1:
                barrier.wait(timeout=10)
            return response(None if count == 1 else f"用户 {seed}", number=seed)

        def run(pair):
            user, state = pair
            return user.generate_next_message(AssistantMessage(role="assistant", content="你好。"), state)

        with patch("tau2.user.user_simulator.generate", side_effect=generate):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(run, users))
        self.assertEqual(counts, {300: 2, 301: 2})
        for seed, (result, state) in zip((300, 301), results):
            self.assertEqual(result.content, f"用户 {seed}")
            records = result.raw_data["empty_response_retries"]
            self.assertEqual([r["raw_data"]["id"] for r in records], [f"response-{seed}"])
            self.assertEqual(state.pending_empty_responses, [])
            self.assertEqual(reported_usage(state, "user")["calls"], 2)


if __name__ == "__main__":
    unittest.main()
