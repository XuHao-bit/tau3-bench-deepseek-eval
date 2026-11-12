"""同次原生学习更新同一 Skill 时，合并证据后仅写一次本地技能。"""

import copy
import importlib.util
import json
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from local_memos.native_skills import NativeSkillDiagnostics, coalesce_native_skill_updates


class SkillUpdateCoalesceBoundaryTests(unittest.TestCase):
    def candidate(self, target, task, index):
        return ({"name": task, "update": True, "old_memory_id": target}, task,
                [{"idx": index, "role": "tool", "content": f"evidence {index}"}])

    def test_overlapping_recall_does_not_merge_distinct_update_targets(self):
        results = [self.candidate("old-a", "first", 0), self.candidate("old-b", "second", 1)]
        shared_recall = [SimpleNamespace(id=target) for target in ("old-a", "old-b")]
        reextract = Mock(side_effect=AssertionError("不同实际目标不应合并"))
        wrapped = coalesce_native_skill_updates(lambda *args: results, reextract)
        chunks = {task: messages for _, task, messages in results}
        observed = wrapped(chunks, {task: shared_recall for task in chunks}, object(), [])
        self.assertEqual(observed, results)
        reextract.assert_not_called()

    def test_failed_merge_remains_failed_when_upstream_catches_the_exception(self):
        with tempfile.TemporaryDirectory() as temp:
            recorder = NativeSkillDiagnostics(Path(temp) / "learning.jsonl", Path(temp), lambda: "merge-trace")
            results = [self.candidate("old-a", "first", 0), self.candidate("old-a", "second", 1)]
            old = SimpleNamespace(id="old-a", metadata=SimpleNamespace(memory_type="SkillMemory"))
            wrapped = coalesce_native_skill_updates(
                lambda *args: results, lambda **kwargs: None,
                recorder.record_coalesced_update, recorder.fail_coalesced_update)

            def process(request):
                with self.assertRaisesRegex(RuntimeError, "未生成有效"):
                    wrapped({task: messages for _, task, messages in results}, {"first": [old], "second": [old]}, object(), [])
                return {"data": []}

            request = SimpleNamespace(writable_cube_ids=["cube"], session_id="task", task_id="learn")
            diagnostics = recorder.run_add(request, process)["eval_diagnostics"]
            self.assertEqual(diagnostics["status"], "failed")
            self.assertEqual(diagnostics["coalesced_updates"][0]["status"], "failed")
            self.assertEqual(len(diagnostics["coalesced_updates"][0]["candidates"]), 2)


@unittest.skipUnless(importlib.util.find_spec("memos"), "使用 local_memos/.venv 测试原生 Skill 文件流程")
class NativeSkillUpdateCoalesceTests(unittest.TestCase):
    def setUp(self):
        import memos.mem_reader.read_skill_memory.process_skill_memory as native
        from memos.memories.textual.item import SourceMessage, TextualMemoryItem, TreeNodeTextualMemoryMetadata

        self.native = native
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files = self.root / "files"
        self.files.mkdir()
        self.old_id = "e8783a4c-7460-4bff-b4e1-67c7ea9ab4b2"
        self.old_name = "account_contact_detail_update_with_identity_verification"
        self.old_zip = self.files / f"{self.old_name}.zip"
        with zipfile.ZipFile(self.old_zip, "w") as archive:
            archive.writestr(f"{self.old_name}/SKILL.md", "old procedure")
        self.old = TextualMemoryItem(id=self.old_id, memory="Update contact details after identity verification", metadata=TreeNodeTextualMemoryMetadata(
            user_id="test-user", session_id="prior-task", memory_type="SkillMemory",
            name=self.old_name, description="Update contact details after identity verification",
            procedure="Verify identity before changing contact details.", experience=[], preference=[], examples=[],
            url=f"http://localhost:8001/download/{self.old_name}.zip"))
        sources = [SourceMessage(role="tool", type="chat", message_id=f"m{i}", content=f"{'LEFT_EVIDENCE' if i < 3 else 'RIGHT_EVIDENCE'} confirmed step {i}") for i in range(6)]
        self.raw = [TextualMemoryItem(memory="visible trajectory", metadata=TreeNodeTextualMemoryMetadata(
            user_id="test-user", session_id="current-task", memory_type="LongTermMemory", sources=sources))]
        self.deleted_ids = []
        self.searcher = SimpleNamespace(search=lambda *args, **kwargs: [self.old])
        self.graph = SimpleNamespace(delete_node_by_prams=lambda **kwargs: self.deleted_ids.extend(kwargs["memory_ids"]))
        self.embedder = SimpleNamespace(embed=lambda texts: [[0.1, 0.2] for _ in texts])
        self.config = {"skills_local_tmp_dir": str(self.root / "tmp"), "skills_local_dir": str(self.files), "skills_oss_dir": "skills/"}

    def schema(self, name, evidence):
        return {"name": name, "description": "Identity verification and contact updates", "procedure": evidence,
                "experience": [evidence], "preference": [], "examples": [], "scripts": {}, "tool": {}, "others": {},
                "update": True, "old_memory_id": self.old_id}

    def run_native(self, coalesce):
        test = self

        class LLM:
            def __init__(self):
                self.first = True
                self.lock = threading.Lock()
                self.extraction_prompts = []

            def generate(self, messages, **kwargs):
                prompt = messages[0]["content"]
                with self.lock:
                    if self.first:
                        self.first = False
                        return json.dumps([
                            {"task_name": "identity_check", "message_indices": [[0, 2]]},
                            {"task_name": "contact_change", "message_indices": [[3, 5]]},
                        ])
                    self.extraction_prompts.append(prompt)
                left, right = "LEFT_EVIDENCE" in prompt, "RIGHT_EVIDENCE" in prompt
                if left and right:
                    return json.dumps(test.schema("identity_and_contact_update", "LEFT_EVIDENCE then RIGHT_EVIDENCE"))
                if left:
                    return json.dumps(test.schema("identity_verification_for_account_access_and_updates", "LEFT_EVIDENCE"))
                if right:
                    return json.dumps(test.schema(test.old_name, "RIGHT_EVIDENCE"))
                raise AssertionError("缺少真实片段证据")

        llm = LLM()
        original_completed = self.native.as_completed

        def files_old_name_first(futures):
            if isinstance(futures, dict) and futures and all(isinstance(value, dict) and "name" in value for value in futures.values()):
                # 固定文件完成顺序，重现第二个更新再次移除旧名 ZIP。
                return iter(sorted(futures, key=lambda future: futures[future]["name"] != self.old_name))
            return original_completed(futures)

        events = []
        batch = self.native._batch_extract_skills
        if coalesce:
            batch = coalesce_native_skill_updates(batch, self.native._extract_skill_memory_by_llm_md, events.append)
        with patch.object(self.native, "_batch_extract_skills", batch), patch.object(self.native, "as_completed", files_old_name_first), patch.dict("os.environ", {"SKILLS_REPO_BACKEND": "LOCAL"}), patch("sys.argv", ["test", "--port", "8001"]):
            items = self.native.process_skill_memory_fine(
                copy.deepcopy(self.raw), {"user_id": "test-user", "session_id": "current-task"},
                searcher=self.searcher, graph_db=self.graph, llm=llm, embedder=self.embedder,
                skills_dir_config=self.config, rewrite_query=False, user_name="cube-c")
        return items, llm, events

    def test_native_unmerged_double_update_reproduces_missing_zip(self):
        items, _, _ = self.run_native(False)
        self.assertEqual([item.id for item in items], [self.old_id, self.old_id])
        self.assertFalse(self.old_zip.exists())
        self.assertEqual(self.deleted_ids, [self.old_id, self.old_id])

    def test_coalesced_native_update_preserves_both_sources_and_one_real_zip(self):
        items, llm, events = self.run_native(True)
        self.assertEqual([item.id for item in items], [self.old_id])
        self.assertEqual(self.deleted_ids, [self.old_id])
        self.assertEqual(len(llm.extraction_prompts), 3)
        self.assertTrue(all(marker in llm.extraction_prompts[-1] for marker in ("LEFT_EVIDENCE", "RIGHT_EVIDENCE")))
        self.assertEqual({source.message_id for source in items[0].metadata.sources}, {f"m{i}" for i in range(6)})
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["target_memory_id"], self.old_id)
        self.assertEqual(len(events[0]["candidates"]), 2)
        self.assertEqual(events[0]["message_indices"], list(range(6)))
        filename = items[0].metadata.url.rsplit("/", 1)[-1]
        with zipfile.ZipFile(self.files / filename) as archive:
            markdown = archive.read("identity_and_contact_update/SKILL.md").decode()
        self.assertIn("LEFT_EVIDENCE", markdown)
        self.assertIn("RIGHT_EVIDENCE", markdown)


if __name__ == "__main__":
    unittest.main()
