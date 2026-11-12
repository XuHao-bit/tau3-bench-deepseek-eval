"""原生 Skill 同步提炼的状态、跨线程日志和本地文件诊断。"""

import logging
import importlib.util
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from local_memos.native_skills import NativeSkillDiagnostics, configure_skill_only_reader


def request():
    return SimpleNamespace(writable_cube_ids=["cube-c"], session_id="banking:one", task_id="learn-one")


def skill(memory_id="skill-one", url="http://localhost:8001/download/example.zip"):
    return SimpleNamespace(id=memory_id, metadata=SimpleNamespace(url=url))


class NativeSkillServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.diag = NativeSkillDiagnostics(
            self.directory / "learning.jsonl", self.directory, lambda: "trace-one", secrets=["private-key-value"])
        self.logger = logging.getLogger("native_skill_test")
        self.logger.addHandler(self.diag)
        self.logger.setLevel(logging.WARNING)
        self.addCleanup(self.logger.removeHandler, self.diag)

    def run_extraction(self, extract, messages=None):
        restored = messages if messages is not None else [{"role": "tool", "content": str(i)} for i in range(5)]
        wrapped = self.diag.observe_extraction(extract, lambda items: restored)

        def process(_request):
            with ThreadPoolExecutor(max_workers=1) as executor:
                items = executor.submit(wrapped, []).result()
            return {"code": 200, "data": [{"memory_id": item.id, "memory_type": "SkillMemory"} for item in items]}

        return self.diag.run_add(request(), process)["eval_diagnostics"]

    def test_completed_skill_includes_cross_thread_observation_and_local_file(self):
        (self.directory / "example.zip").write_bytes(b"skill")
        diagnostics = self.run_extraction(lambda items: [skill()])
        self.assertEqual(diagnostics["status"], "completed")
        self.assertEqual(diagnostics["restored_tool_messages"], 5)
        self.assertEqual(diagnostics["generated_skill_ids"], ["skill-one"])
        self.assertTrue(diagnostics["file_status"][0]["exists"])

    def test_empty_extraction_is_distinct_from_tool_gate_skip(self):
        diagnostics = self.run_extraction(lambda items: [])
        self.assertEqual(diagnostics["status"], "processed_no_skill")
        diagnostics = self.run_extraction(lambda items: [], [{"role": "tool", "content": "one"}])
        self.assertEqual(diagnostics["status"], "skipped")
        self.assertEqual(diagnostics["reason"], "tool_messages_below_five")

    def test_swallowed_error_is_not_reported_as_successful_empty_learning(self):
        def extract(items):
            self.logger.warning("LLM skill memory extraction failed: private-key-value")
            return []

        diagnostics = self.run_extraction(extract)
        self.assertEqual(diagnostics["status"], "failed")
        self.assertNotIn("private-key-value", str(diagnostics))
        self.assertIn("[redacted]", diagnostics["warnings"][0]["message"])

    def test_missing_zip_marks_partial_even_when_memory_item_was_generated(self):
        diagnostics = self.run_extraction(lambda items: [skill()])
        self.assertEqual(diagnostics["status"], "partial")
        self.assertFalse(diagnostics["file_status"][0]["exists"])

    def test_add_exception_is_journaled_and_propagates_without_repeating(self):
        calls = []

        def process(_request):
            calls.append("called")
            raise RuntimeError("database unavailable")

        with self.assertRaisesRegex(RuntimeError, "database unavailable"):
            self.diag.run_add(request(), process)
        self.assertEqual(calls, ["called"])
        self.assertIn('"status": "failed"', (self.directory / "learning.jsonl").read_text())

    def test_add_without_skill_entry_reports_unknown(self):
        diagnostics = self.diag.run_add(request(), lambda _: {"code": 200, "data": []})["eval_diagnostics"]
        self.assertEqual(diagnostics["status"], "unknown")
        self.assertEqual(diagnostics["write_state"], "unknown")

    def test_empty_native_return_before_persistence_is_explicitly_not_started(self):
        def extract(items):
            self.logger.warning("JSON decode failed after native attempts")
            return []

        diagnostics = self.run_extraction(extract)
        self.assertEqual(diagnostics["status"], "failed")
        self.assertFalse(diagnostics["skill_persistence_started"])
        self.assertTrue(diagnostics["skill_extraction_returned"])
        self.assertEqual(diagnostics["write_state"], "not_started")

    def test_persistence_started_but_native_empty_return_remains_unknown(self):
        def write(*args, **kwargs):
            raise OSError("local write failed")

        writer = self.diag.observe_persistence(write)

        def extract(items):
            try:
                writer()
            except OSError:
                self.logger.warning("Error writing skills to file")
            return []

        diagnostics = self.run_extraction(extract)
        self.assertEqual(diagnostics["status"], "failed")
        self.assertTrue(diagnostics["skill_persistence_started"])
        self.assertEqual(diagnostics["write_state"], "unknown")

    def test_fast_fact_write_does_not_wait_for_skill_learning_lock(self):
        self.diag._serial.acquire()
        process = Mock(return_value={"code": 200, "data": [{"memory_id": "fact-one", "memory_type": "LongTermMemory"}]})
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                payload = executor.submit(self.diag.run_fast_add, request(), process).result(timeout=2)
            finally:
                self.diag._serial.release()
        self.assertEqual(payload["eval_diagnostics"]["status"], "completed")
        self.assertFalse(payload["eval_diagnostics"]["skill_extraction_entered"])
        self.assertEqual(payload["eval_diagnostics"]["fact_memory_ids"], ["fact-one"])
        process.assert_called_once()
        self.assertFalse((self.directory / "learning.jsonl").exists())


@unittest.skipUnless(importlib.util.find_spec("memos"), "原生函数接线测试使用 local_memos/.venv")
class NativeSkillOnlyReaderTests(unittest.TestCase):
    def setUp(self):
        from memos.mem_reader.read_multi_modal import MultiModalParser
        from memos.mem_reader.multi_modal_struct import MultiModalStructMemReader
        from memos.mem_reader.simple_struct import count_tokens_text
        from memos.mem_reader.read_skill_memory.process_skill_memory import process_skill_memory_fine

        self.reader = object.__new__(MultiModalStructMemReader)
        self.reader.embedder = SimpleNamespace(embed=lambda texts: [[0.0] for _ in texts])
        self.reader.llm = SimpleNamespace(generate=Mock(return_value="{}"))
        self.reader.general_llm = self.reader.llm
        self.reader.multi_modal_parser = MultiModalParser(self.reader.embedder, self.reader.llm)
        self.reader.chat_window_max_tokens = 1024
        self.reader._count_tokens = count_tokens_text
        self.reader.searcher = SimpleNamespace(search=Mock(return_value=[]))
        self.reader.graph_db = object()
        self.reader.oss_config = None
        self.reader.skills_dir_config = {"skills_local_tmp_dir": "/unused/tmp", "skills_local_dir": "/unused/files", "skills_oss_dir": "skill_memory/"}
        self.reader._process_string_fine = Mock(side_effect=AssertionError("不应提炼伴生普通记忆"))
        self.reader._process_tool_trajectory_fine = Mock(side_effect=AssertionError("不应提炼伴生工具记忆"))
        self.reader.multi_modal_parser.process_transfer = Mock(side_effect=AssertionError("不应逐来源提炼"))
        self.native = Mock(wraps=process_skill_memory_fine)
        configure_skill_only_reader(self.reader, self.native)

    def test_fast_fact_uses_native_parser_without_invoking_skill_or_llm(self):
        items = self.reader._process_multi_modal_data(
            [{"role": "assistant", "content": "A confirmed fact.", "message_id": "fact-message"}],
            {"user_id": "owner", "session_id": "one"}, mode="fast")
        self.assertTrue(items)
        self.assertIn("A confirmed fact.", items[0].memory)
        self.assertEqual(items[0].metadata.memory_type, "LongTermMemory")
        self.native.assert_not_called()
        self.reader.llm.generate.assert_not_called()

    def test_fine_calls_real_native_skill_gate_with_unmodified_sources(self):
        messages = [{"role": "tool", "content": f"Confirmed result {i}", "message_id": f"m{i}", "tool_call_id": f"c{i}"} for i in range(4)]
        info = {"user_id": "owner", "session_id": "one", "custom_tags": None}
        context = SimpleNamespace(manager_user_id=None, project_id=None)
        with unittest.mock.patch.dict("os.environ", {"SKILLS_REPO_BACKEND": "LOCAL"}):
            result = self.reader._process_multi_modal_data(messages, info, mode="fine", user_name="cube-c", user_context=context)
        self.assertEqual(result, [])
        self.native.assert_called_once()
        call = self.native.call_args.kwargs
        sources = [s for item in call["fast_memory_items"] for s in item.metadata.sources]
        self.assertEqual([(s.message_id, s.tool_call_id, s.content) for s in sources], [(m["message_id"], m["tool_call_id"], m["content"]) for m in messages])
        self.assertIs(call["llm"], self.reader.general_llm)
        self.assertIs(call["graph_db"], self.reader.graph_db)
        self.assertIs(call["embedder"], self.reader.embedder)
        self.assertIs(call["skills_dir_config"], self.reader.skills_dir_config)
        self.assertIsNone(call["oss_config"])
        self.assertIs(call["user_context"], context)
        self.assertEqual(call["user_name"], "cube-c")
        self.reader.llm.generate.assert_not_called()
        self.reader._process_string_fine.assert_not_called()
        self.reader._process_tool_trajectory_fine.assert_not_called()
        self.reader.multi_modal_parser.process_transfer.assert_not_called()

    def test_native_skill_recall_is_scoped_without_changing_business_fact_search(self):
        from memos.mem_reader.read_skill_memory.process_skill_memory import _recall_related_skill_memories

        with unittest.mock.patch.dict("os.environ", {"SKILLS_REPO_BACKEND": "LOCAL"}):
            self.reader._process_multi_modal_data(
                [{"role": "tool", "content": "One result", "message_id": "m1"}],
                {"user_id": "owner"}, mode="fine")
        skill_searcher = self.native.call_args.kwargs["searcher"]
        result = _recall_related_skill_memories(
            "task", [{"role": "user", "content": "workflow"}], skill_searcher,
            self.reader.llm, False, {"user_id": "owner"}, "cube-c")
        self.assertEqual(result, [])
        skill_call = self.reader.searcher.search.call_args
        self.assertEqual(skill_call.kwargs["memory_type"], "SkillMemory")
        self.assertTrue(skill_call.kwargs["include_skill_memory"])
        self.assertEqual(skill_call.kwargs["user_name"], "cube-c")
        self.reader.searcher.search("fact query", memory_type="All", include_skill_memory=False)
        self.assertEqual(self.reader.searcher.search.call_args.kwargs["memory_type"], "All")
        self.assertFalse(self.reader.searcher.search.call_args.kwargs["include_skill_memory"])

    def test_fine_keeps_native_task_split_without_fabricating_skill(self):
        messages = [{"role": "tool", "content": f"Result {i}", "message_id": f"m{i}"} for i in range(5)]
        with unittest.mock.patch.dict("os.environ", {"SKILLS_REPO_BACKEND": "LOCAL"}):
            result = self.reader._process_multi_modal_data(messages, {"user_id": "owner"}, mode="fine")
        self.assertEqual(result, [])
        self.reader.llm.generate.assert_called_once()
        self.reader._process_string_fine.assert_not_called()
        self.reader._process_tool_trajectory_fine.assert_not_called()

    def test_real_native_json_failures_return_empty_before_any_file_write(self):
        import json
        import memos.mem_reader.read_skill_memory.process_skill_memory as native

        with tempfile.TemporaryDirectory() as directory:
            recorder = NativeSkillDiagnostics(Path(directory) / "learning.jsonl", Path(directory), lambda: "native-failure")
            native.logger.addHandler(recorder)
            self.addCleanup(native.logger.removeHandler, recorder)
            self.reader.general_llm.generate.side_effect = [
                json.dumps([{"task_name": "case", "message_indices": [[0, 4]]}]),
                "lookup query", '{"name": "truncated', '{"name": "truncated', "",
            ]
            self.native.side_effect = recorder.observe_extraction(
                native.process_skill_memory_fine, native._reconstruct_messages_from_memory_items)
            writer = Mock(wraps=recorder.observe_persistence(native._write_skills_to_file))
            messages = [{"role": "tool", "content": f"Result {i}", "message_id": f"m{i}"} for i in range(5)]

            def process(_request):
                items = self.reader._process_multi_modal_data(messages, {"user_id": "owner"}, mode="fine")
                self.assertEqual(items, [])
                return {"code": 200, "data": []}

            with unittest.mock.patch.dict("os.environ", {"SKILLS_REPO_BACKEND": "LOCAL"}), unittest.mock.patch.object(native, "_write_skills_to_file", writer):
                diagnostics = recorder.run_add(request(), process)["eval_diagnostics"]
            self.assertEqual(diagnostics["status"], "failed")
            self.assertEqual(diagnostics["write_state"], "not_started")
            self.assertFalse(diagnostics["skill_persistence_started"])
            self.assertEqual(self.reader.general_llm.generate.call_count, 5)
            self.assertEqual(diagnostics["generated_skill_ids"], [])
            writer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
