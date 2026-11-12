"""原生 Skill 的同步学习、隔离读取和运行中查看。"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from memos_eval.client import MemOSClientError
from memos_eval.skill_client import SkillClient


def skill(memory_id="s1", *, status="activated"):
    return {
        "id": memory_id,
        "memory": "处理前检查授权与当前业务状态。",
        "metadata": {
            "memory_type": "SkillMemory",
            "name": "先检查当前状态",
            "procedure": "先检查状态，再执行授权动作，最后验证结果。",
            "experience": ["不能用历史状态代替实时检查。"],
            "sources": [{"role": "tool", "content": "可追溯的原始结果"}],
            "status": status,
            "info": {"task_id": "task_001"},
        },
    }


def response(data=None, **extra):
    return httpx.Response(200, json={"code": 200, "data": data, **extra})


def grouped(nodes, cube="cube-a"):
    return {"skill_mem": [{"cube_id": cube, "memories": nodes}]}


def client(handler):
    return SkillClient("http://memos.invalid", transport=httpx.MockTransport(handler))


class SkillClientTests(unittest.TestCase):
    def test_skill_health_rejects_ordinary_memory_profile(self):
        with client(lambda req: httpx.Response(200, json={
            "status": "healthy", "eval_features": {"skill_generation": False,
            "extraction": "ordinary_fine"}})) as c:
            with self.assertRaisesRegex(MemOSClientError, "原生 Skill"):
                c.ensure_skill_service()

    def test_skill_health_reports_native_sync_configuration(self):
        features = {"skill_generation": True, "extraction": "native_skill_only",
                    "sync_add": True, "sync_fact_write": True, "skill_storage": "LOCAL",
                    "same_skill_update": "coalesce_native_evidence"}
        with client(lambda req: httpx.Response(200, json={
                "status": "healthy", "eval_features": features})) as c:
            self.assertEqual(c.ensure_skill_service(), features)

    def test_search_uses_skill_quota_and_preserves_complete_skill(self):
        saved = skill()

        def handler(req):
            self.assertEqual(req.url.path, "/product/search")
            body = json.loads(req.content)
            self.assertEqual(body["readable_cube_ids"], ["cube-a"])
            self.assertEqual(body["session_id"], "banking_knowledge:task_005")
            self.assertEqual(body["search_memory_type"], "SkillMemory")
            self.assertEqual(body["mode"], "fast")
            self.assertEqual(body["skill_mem_top_k"], 4)
            self.assertTrue(body["include_skill_memory"])
            for name in ("include_preference", "search_tool_memory", "internet_search"):
                self.assertFalse(body[name])
            return response({
                "skill_mem": [
                    {"cube_id": "cube-b", "memories": [skill("outside")]},
                    {"cube_id": "cube-a", "memories": [saved, skill("deleted", status="deleted")]},
                ],
                "text_mem": [{"cube_id": "cube-a", "memories": [skill("ordinary")]}],
            })

        with client(handler) as c:
            hits = c.search_skills("cube-a", "banking_knowledge:task_005", "授权", 4)
        self.assertEqual(hits, [saved])

    def test_read_uses_cube_and_id_filter_and_rejects_extra_nodes(self):
        saved = skill()

        def handler(req):
            self.assertEqual(req.url.path, "/product/get_memory")
            body = json.loads(req.content)
            self.assertEqual(body["mem_cube_id"], "cube-a")
            self.assertEqual(body["filter"], {"or": [{"id": "s1"}, {"id": "s2"}]})
            self.assertTrue(body["include_skill_memory"])
            self.assertFalse(body["include_tool_memory"])
            self.assertFalse(body["include_preference"])
            return response(grouped([saved, skill("unrequested")]))

        with client(handler) as c:
            self.assertEqual(c.get_skills("cube-a", ["s1", "s2", "s1"]), [saved])
            self.assertEqual(c.get_skills("cube-a", []), [])
            self.assertEqual(c.request_count, 1)

    def test_list_reads_all_skill_pages_without_other_memory_groups(self):
        def handler(req):
            body = json.loads(req.content)
            self.assertNotIn("filter", body)
            self.assertNotIn("page_size", body)
            return response({**grouped([skill(), skill("s2")]),
                "pref_mem": [{"cube_id": "cube-a", "memories": [skill("preference")]}]})

        with client(handler) as c:
            self.assertEqual([row["id"] for row in c.list_skills("cube-a")], ["s1", "s2"])

    def test_sync_learning_reads_all_skill_ids_and_keeps_same_id_update(self):
        messages = [{"role": "user", "content": "本题可见请求"}]
        paths = []

        def handler(req):
            paths.append(req.url.path)
            body = json.loads(req.content)
            if req.url.path == "/product/add":
                self.assertEqual(body["messages"], messages)
                self.assertEqual(body["writable_cube_ids"], ["cube-a"])
                self.assertEqual(body["session_id"], "banking:task_005")
                self.assertEqual(body["async_mode"], "sync")
                self.assertEqual(body["mode"], "fine")
                self.assertFalse(body["is_upload_skill"])
                self.assertNotIn("chat_history", body)
                self.assertEqual(req.extensions["timeout"]["read"], 1800)
                return response([
                    {"cube_id": "cube-a", "memory_id": "s1", "memory_type": "SkillMemory"},
                    {"cube_id": "cube-a", "memory_id": "s2", "memory_type": "SkillMemory"},
                    {"cube_id": "cube-a", "memory_id": "fact", "memory_type": "LongTermMemory"},
                ], eval_diagnostics={"status": "completed", "skill_count": 2})
            return response(grouped([skill(), skill("s2")]))

        with client(handler) as c:
            receipt = c.learn("cube-a", "banking:task_005", messages, {"task_id": "task_005"})
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["skill_ids"], ["s1", "s2"])
        self.assertEqual(receipt["skills"][0]["metadata"]["procedure"], skill()["metadata"]["procedure"])
        self.assertEqual(paths, ["/product/add", "/product/get_memory"])
        self.assertEqual(receipt["diagnostics"]["skill_count"], 2)

    def test_learning_zero_skill_is_not_ordinary_fact_success(self):
        def handler(req):
            self.assertEqual(req.url.path, "/product/add")
            return response([{"cube_id": "cube-a", "memory_id": "fact", "memory_type": "LongTermMemory"}],
                            eval_diagnostics={"status": "processed_no_skill", "skill_count": 0})

        with client(handler) as c:
            receipt = c.learn("cube-a", "task", [{"role": "user", "content": "q"}], {})
            self.assertEqual(c.request_count, 1)
        self.assertEqual(receipt["status"], "processed_no_skill")
        self.assertEqual(receipt["skills"], [])

    def test_learning_retains_extraction_failure_even_http_200(self):
        with client(lambda req: response([], eval_diagnostics={
                "status": "failed", "skill_count": 0, "error": "技能提炼失败"})) as c:
            receipt = c.learn("cube-a", "task", [{"role": "user", "content": "q"}], {})
        self.assertEqual(receipt["status"], "failed")

    def test_learning_failure_preserves_explicit_no_write_evidence(self):
        with client(lambda req: response([], eval_diagnostics={
                "status": "failed", "skill_count": 0, "write_state": "not_started",
                "error": "原生提炼响应不完整"})) as c:
            receipt = c.learn("cube-a", "task", [{"role": "user", "content": "q"}], {})
            self.assertEqual(c.request_count, 1)
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["write_state"], "not_started")
        self.assertTrue(receipt["empty_add_result"])
        self.assertEqual(receipt["skill_ids"], [])
        self.assertEqual(receipt["error"], "原生提炼响应不完整")

    def test_missing_add_data_is_not_explicit_empty_no_write_result(self):
        with client(lambda req: response(None, eval_diagnostics={
                "status": "failed", "write_state": "not_started", "error": "解析失败"})) as c:
            receipt = c.learn("cube-a", "task", [{"role": "user", "content": "q"}], {})
        self.assertEqual(receipt["status"], "unknown")
        self.assertFalse(receipt.get("empty_add_result", False))

    def test_no_write_diagnostic_cannot_hide_a_nonempty_add_result(self):
        with client(lambda req: response([
                {"cube_id": "cube-a", "memory_type": "LongTermMemory", "memory_id": "unexpected"}],
                eval_diagnostics={"status": "failed", "write_state": "not_started", "error": "提炼失败"})) as c:
            receipt = c.learn("cube-a", "task", [{"role": "user", "content": "q"}], {})
        self.assertEqual(receipt["status"], "partial")
        self.assertFalse(receipt["empty_add_result"])

    def test_zero_skill_without_diagnostics_is_unknown(self):
        with client(lambda req: response([])) as c:
            receipt = c.learn("cube-a", "task", [{"role": "user", "content": "q"}], {})
        self.assertEqual(receipt["status"], "unknown")

    def test_learning_timeout_is_not_repeated(self):
        calls = []

        def handler(req):
            calls.append(req.url.path)
            raise httpx.ReadTimeout("未确认写入", request=req)

        with client(handler) as c:
            receipt = c.learn("cube-a", "task", [{"role": "user", "content": "q"}], {})
        self.assertEqual(receipt["status"], "unknown")
        self.assertEqual(calls, ["/product/add"])

    def test_missing_readback_is_partial_without_resubmission(self):
        paths = []

        def handler(req):
            paths.append(req.url.path)
            if req.url.path == "/product/add":
                return response([
                    {"cube_id": "cube-a", "memory_id": "s1", "memory_type": "SkillMemory"},
                    {"cube_id": "cube-a", "memory_id": "s2", "memory_type": "SkillMemory"},
                ], eval_diagnostics={"status": "completed"})
            return response(grouped([skill()]))

        with client(handler) as c:
            receipt = c.learn("cube-a", "task", [{"role": "user", "content": "q"}], {})
        self.assertEqual(receipt["status"], "partial")
        self.assertEqual(receipt["missing_skill_ids"], ["s2"])
        self.assertEqual(paths, ["/product/add", "/product/get_memory"])

    def test_client_exposes_a_shared_learning_lock_and_abort_event(self):
        with client(lambda req: response({})) as c:
            with c.learning_lock:
                with c.learning_lock:
                    self.assertFalse(c.learning_aborted.is_set())
            with c.snapshot_lock:
                with c.snapshot_lock:
                    self.assertFalse(c.learning_aborted.is_set())
            c.learning_aborted.set()
            self.assertTrue(c.learning_aborted.is_set())

    def test_write_fact_uses_fast_and_unreserved_scope_metadata(self):
        saved = fact("f1", scope="task", task_id="task_005")
        saved["metadata"]["source_domain"] = "banking_knowledge"
        paths = []

        def handler(req):
            paths.append(req.url.path)
            body = json.loads(req.content)
            if req.url.path == "/product/add":
                self.assertEqual(body["async_mode"], "sync")
                self.assertEqual(body["mode"], "fast")
                self.assertEqual(body["session_id"], "banking_knowledge:task_005")
                self.assertEqual(body["messages"], [{"role": "assistant", "content": "已查询当前状态"}])
                self.assertEqual(body["writable_cube_ids"], ["cube-a"])
                self.assertEqual(body["info"]["kind"], "fact")
                self.assertEqual(body["info"]["fact_scope"], "task")
                self.assertNotIn("visibility", body["info"])
                self.assertEqual(body["info"]["task_id"], "task_005")
                self.assertEqual(body["info"]["source_domain"], "banking_knowledge")
                return response([{"memory_id": "f1", "memory_type": "LongTermMemory", "cube_id": "cube-a"}],
                                eval_diagnostics={"status": "completed", "operation": "write_fact"})
            return response({"text_mem": [{"cube_id": "cube-a", "memories": [saved]}]})

        with client(handler) as c:
            receipt = c.write_fact("cube-a", "已查询当前状态", "task_005", "banking_knowledge")
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["memory_ids"], ["f1"])
        self.assertEqual(receipt["memories"], [saved])
        self.assertEqual(paths, ["/product/add", "/product/get_memory"])

    def test_fact_search_filters_visibility_before_top_k(self):
        public = fact("public", scope="shared", task_id="task_001", score=0.8)
        current = fact("current", scope="task", task_id="task_005", score=0.7)
        hidden = fact("private-other", scope="task", task_id="task_001", score=1.0)
        source_qa = fact("original-qa", scope="shared", task_id="task_001", score=0.99)
        source_qa["metadata"].pop("kind")
        data = [hidden, source_qa, public, current]

        def handler(req):
            body = json.loads(req.content)
            self.assertEqual(body["search_memory_type"], "LongTermMemory")
            self.assertFalse(body["include_skill_memory"])
            self.assertEqual(body["filter"], {"or": [
                {"kind": "fact", "fact_scope": "shared"},
                {"kind": "fact", "fact_scope": "task", "session_id": "banking_knowledge:task_005"},
            ]})
            visible = [row for row in data if any(
                all(row["metadata"].get(key) == value for key, value in condition.items())
                for condition in body["filter"]["or"])]
            return response({"text_mem": [{"cube_id": "cube-a", "memories": visible[:body["top_k"]]}]})

        with client(handler) as c:
            memories = c.search_memories("cube-a", "banking_knowledge:task_005", "状态", "fact", 2)
        self.assertEqual([row["id"] for row in memories], ["public", "current"])

    def test_all_search_combines_skill_and_scoped_fact_by_native_score(self):
        saved_skill = skill()
        saved_skill["metadata"]["relativity"] = 0.8

        def handler(req):
            body = json.loads(req.content)
            self.assertEqual(body["search_memory_type"], "All")
            self.assertEqual(body["skill_mem_top_k"], 2)
            self.assertEqual(body["filter"]["or"][0], {"memory_type": "SkillMemory"})
            return response({**grouped([saved_skill]), "text_mem": [
                {"cube_id": "cube-b", "memories": [fact("outside", score=1)]},
                {"cube_id": "cube-a", "memories": [
                    fact("public", scope="shared", score=0.9),
                    fact("private-other", scope="task", task_id="task_001", score=1),
                    fact("current", scope="task", task_id="task_005", score=0.7),
                ]},
            ]})

        with client(handler) as c:
            memories = c.search_memories("cube-a", "banking_knowledge:task_005", "流程", top_k=2)
        self.assertEqual([row["id"] for row in memories], ["public", "s1"])

    def test_human_fact_snapshot_includes_private_but_no_qa_or_fine_sources(self):
        ordinary = fact("fine-source")
        ordinary["metadata"].pop("kind")

        def handler(req):
            body = json.loads(req.content)
            self.assertEqual(body["filter"], {"kind": "fact"})
            self.assertFalse(body["include_skill_memory"])
            return response({"text_mem": [{"cube_id": "cube-a", "memories": [
                fact("public", scope="shared"), fact("private", scope="task"), ordinary,
            ]}]})

        with client(handler) as c:
            self.assertEqual([row["id"] for row in c.list_all_facts("cube-a")], ["public", "private"])

    def test_scoped_fact_snapshot_excludes_other_task_private_items(self):
        def handler(req):
            body = json.loads(req.content)
            self.assertEqual(body["filter"]["or"][1]["session_id"], "banking_knowledge:task_005")
            return response({"text_mem": [{"cube_id": "cube-a", "memories": [
                fact("public", scope="shared", task_id="task_001"),
                fact("private-other", scope="task", task_id="task_001"),
                fact("private-current", scope="task", task_id="task_005"),
            ]}]})

        with client(handler) as c:
            result = c.list_facts("cube-a", "banking_knowledge:task_005")
        self.assertEqual([row["id"] for row in result], ["public", "private-current"])

    def test_write_fact_timeout_does_not_retry(self):
        calls = []

        def handler(req):
            calls.append(req.url.path)
            raise httpx.ReadTimeout("未确认写入", request=req)

        with client(handler) as c:
            result = c.write_fact("cube-a", "当前状态", "task_005", "banking_knowledge")
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(calls, ["/product/add"])

    def test_write_fact_rejects_readback_with_wrong_scope(self):
        def handler(req):
            if req.url.path == "/product/add":
                return response([{"memory_id": "f1", "memory_type": "LongTermMemory", "cube_id": "cube-a"}],
                                eval_diagnostics={"status": "completed", "operation": "write_fact"})
            return response({"text_mem": [{"cube_id": "cube-a", "memories": [fact("f1", scope="shared")]}]})

        with client(handler) as c:
            result = c.write_fact("cube-a", "当前状态", "task_005", "banking_knowledge")
        self.assertEqual(result["status"], "partial")

    def test_fact_input_rejects_invalid_scope_without_http(self):
        with client(lambda req: self.fail("不应发请求")) as c:
            with self.assertRaises(ValueError):
                c.write_fact("cube-a", "状态", "task_005", "banking_knowledge", "public")
            with self.assertRaises(ValueError):
                c.search_memories("cube-a", "task", "query", "all", 0)

    def test_skill_viewer_resolves_c_cubes_and_seed_without_b_cubes(self):
        from memos_eval import inspect_skills

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "manifest.json").write_text(json.dumps({
                "memos_url": "http://memos.invalid", "memos_user": "tau3-eval",
                "cubes": [
                    {"arm": "B-cross", "seed": 300, "cube_id": "ordinary"},
                    {"arm": "C", "seed": 300, "cube_id": "cube-a"},
                    {"arm": "C", "seed": 301, "cube_id": "cube-b"},
                ],
            }))
            fake_client = unittest.mock.MagicMock()
            fake_client.list_skills.return_value = [skill()]
            output = root / "actual_skills.json"
            with patch.object(inspect_skills, "SkillClient", return_value=fake_client), \
                    patch("sys.argv", ["inspect_skills", "--run-dir", str(root),
                                       "--seed", "300", "--output", str(output)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                inspect_skills.main()
            fake_client.list_skills.assert_called_once_with("cube-a")
            self.assertEqual(json.loads(output.read_text())[0]["skills"][0]["id"], "s1")

    def test_viewer_can_export_explicit_facts_without_changing_skill_default(self):
        from memos_eval import inspect_skills

        fake_client = unittest.mock.MagicMock()
        fake_client.list_skills.return_value = [skill()]
        fake_client.list_facts.return_value = [fact()]
        stream = io.StringIO()
        with patch.object(inspect_skills, "SkillClient", return_value=fake_client), \
                patch("sys.argv", ["inspect_skills", "--cube-id", "cube-a", "--memory-type", "all",
                                   "--session-id", "banking_knowledge:task_005"]), \
                contextlib.redirect_stdout(stream):
            inspect_skills.main()
        fake_client.list_skills.assert_called_once_with("cube-a")
        fake_client.list_facts.assert_called_once_with("cube-a", "banking_knowledge:task_005")
        row = json.loads(stream.getvalue())[0]
        self.assertEqual(row["skill_count"], 1)
        self.assertEqual(row["fact_count"], 1)
        self.assertEqual(row["facts"][0]["metadata"]["fact_scope"], "task")


def fact(memory_id="f1", *, scope="task", task_id="task_005", score=0.5):
    return {"id": memory_id, "memory": "assistant: 已核对当前状态", "metadata": {
        "memory_type": "LongTermMemory", "status": "activated", "kind": "fact",
        "fact_scope": scope, "task_id": task_id, "source_domain": "banking_knowledge",
        "session_id": f"banking_knowledge:{task_id}", "relativity": score,
    }}


if __name__ == "__main__":
    unittest.main()
