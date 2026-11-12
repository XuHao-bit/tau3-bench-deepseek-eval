import json
import unittest

import httpx

from memos_eval.client import MemOSClient, MemOSClientError


def node(memory_id="m1", cube="cube-a", session="task_001"):
    return {
        "id": memory_id,
        "memory": "需要确认当前业务条件。",
        "metadata": {
            "user_name": cube,
            "session_id": session,
            "created_at": "2026-11-04T10:00:00",
            "info": {"task_id": session},
        },
    }


def response(data, **extra):
    return httpx.Response(200, json={"code": 200, "data": data, **extra})


def client(handler):
    return MemOSClient("http://memos.invalid", transport=httpx.MockTransport(handler))


class TestMemOSClient(unittest.TestCase):
    def test_async_add_submits_qa_once_without_history_or_mode(self):
        messages = [
            {"role": "user", "content": "请检查业务条件"},
            {"role": "assistant", "content": "条件满足"},
        ]
        paths = []

        def handler(request):
            paths.append(request.url.path)
            body = json.loads(request.content)
            if request.url.path == "/product/add":
                assert body["messages"] == messages
                assert body["async_mode"] == "async"
                assert "mode" not in body
                assert body["writable_cube_ids"] == ["cube-a"]
                assert "chat_history" not in body
                return response([{"status": "submitted", "task_id": body["task_id"],
                                  "memory_ids": [], "cube_id": "cube-a"}])
            raise AssertionError("add 不应隐式读回，以免验证失败触发重复写入")

        receipt = client(handler).add("cube-a", "task_001", messages, {"task_id": "task_001"})
        assert receipt["status"] == "submitted"
        assert receipt["memory_ids"] == []
        assert paths == ["/product/add"]


    def test_get_missing_id_returns_no_memory(self):
        def handler(request):
            assert request.url.path == "/product/get_memory"
            return response({"text_mem": [{"cube_id": "cube-a", "memories": []}]})

        assert client(handler).get("cube-a", ["missing"]) == []


    def test_search_keeps_order_duplicates_and_preference_but_no_skill_or_tools(self):
        def handler(request):
            body = json.loads(request.content)
            assert body["readable_cube_ids"] == ["cube-a"]
            assert body["include_skill_memory"] is False
            assert body["include_preference"] is True
            assert body["search_tool_memory"] is False
            assert body["top_k"] == 6
            assert body["dedup"] == "no"
            assert body["relativity"] == 0
            assert "mode" not in body
            assert body["internet_search"] is False
            assert body["session_id"] == "task_002"
            return response({"text_mem": [
                {"cube_id": "cube-b", "memories": [node("other", "cube-b")]},
                {"cube_id": "cube-a", "memories": [node(), node()]},
            ], "pref_mem": [{"cube_id": "cube-a", "memories": [node("pref")]}],
               "skill_mem": [{"cube_id": "cube-a", "memories": [node("skill")]}]})

        hits = client(handler).search("cube-a", "task_002", "业务条件")
        assert [hit.memory_id for hit in hits] == ["m1", "m1", "pref"]
        assert hits[0].task_id == "task_001"
        assert hits[-1].metadata["category"] == "preference"

    def test_add_timeout_keeps_receipt_ticket_without_resubmission(self):
        calls = []

        def handler(request):
            calls.append(json.loads(request.content))
            raise httpx.ReadTimeout("超时", request=request)

        receipt = client(handler).add("cube-a", "task_001", [{"role": "user", "content": "q"}], {})
        assert receipt["status"] == "unknown"
        assert receipt["task_id"] == calls[0]["task_id"]
        assert len(calls) == 1

    def test_preference_injects_preference_text_instead_of_context_summary(self):
        pref = node("pref")
        pref["memory"] = "用户正在咨询客服渠道。"
        pref["metadata"]["preference"] = "偏好邮件沟通。"
        def handler(request):
            return response({"pref_mem": [{"cube_id": "cube-a", "memories": [pref]}]})
        hit = client(handler).search("cube-a", "task_001", "沟通")[0]
        assert hit.content == "偏好邮件沟通。"
        assert hit.metadata["context_summary"] == "用户正在咨询客服渠道。"

    def test_status_is_a_short_nonblocking_query(self):
        def handler(request):
            assert request.url.path == "/product/add/status"
            assert request.extensions["timeout"]["read"] == 3
            assert json.loads(request.content) == {
                "mem_cube_id": "cube-a", "session_id": "task_001",
                "user_id": "tau3-eval", "task_id": "ticket-1"}
            return response({"status": "submitted", "task_id": "ticket-1", "memory_ids": []})

        assert client(handler).add_status("cube-a", "task_001", "ticket-1")["status"] == "submitted"

    def test_health_rejects_server_without_async_extraction(self):
        def handler(request):
            return httpx.Response(200, json={"status": "healthy"})

        with self.assertRaisesRegex(MemOSClientError, "异步普通记忆"):
            client(handler).ensure_user()

    def test_health_reports_actual_extraction_profile(self):
        features = {"async_add": True, "preference_memory": True,
                    "skill_generation": False, "extraction": "ordinary_fine"}
        def handler(request):
            return httpx.Response(200, json={"status": "healthy", "eval_features": features})

        assert client(handler).ensure_user() == features


    def test_get_by_ids_filters_using_cube_scoped_endpoint(self):
        def handler(request):
            assert request.url.path == "/product/get_memory"
            body = json.loads(request.content)
            assert body["mem_cube_id"] == "cube-a"
            assert body["filter"] == {"or": [{"id": "m1"}, {"id": "other"}]}
            return response({"text_mem": [
                {"cube_id": "cube-a", "memories": [node()]},
                {"cube_id": "cube-b", "memories": [node("other", "cube-b")]},
            ]})

        assert [m.memory_id for m in client(handler).get("cube-a", ["m1", "other"])] == ["m1"]


    def test_delete_session_verifies_empty_and_preserves_cube_scope(self):
        calls = []

        def handler(request):
            calls.append(request.url.path)
            body = json.loads(request.content)
            if request.url.path == "/product/delete_memory":
                assert body == {"writable_cube_ids": ["cube-a"], "session_id": "task_001"}
                return response({"status": "success"})
            assert body["filter"] == {"and": [{"session_id": "task_001"}]}
            assert body["mem_cube_id"] == "cube-a"
            return response({"text_mem": [{"cube_id": "cube-a", "memories": []}]})

        client(handler).delete_session("cube-a", "task_001")
        assert calls == ["/product/delete_memory", "/product/get_memory"]


    def test_delete_session_rejects_reported_success_when_memory_remains(self):
        def handler(request):
            if request.url.path == "/product/delete_memory":
                return response({"status": "success"})
            return response({"text_mem": [{"cube_id": "cube-a", "memories": [node()]}]})

        with self.assertRaisesRegex(MemOSClientError, "清理后仍有记忆"):
            client(handler).delete_session("cube-a", "task_001")


    def test_list_memories_uses_real_server_pagination_and_session_filter(self):
        def handler(request):
            body = json.loads(request.content)
            assert body["page"] == 1
            assert body["page_size"] == 3
            assert body["include_skill_memory"] is False
            assert body["filter"] == {"and": [{"session_id": "task_001"}]}
            return response({"text_mem": [{"cube_id": "cube-a", "memories": [
                node("m1"), node("m2"), node("m3"),
            ]}]})

        assert [m.memory_id for m in client(handler).list_memories("cube-a", "task_001", 2, 1)] == ["m2", "m3"]


    def test_http_200_with_failure_status_is_not_success(self):
        def handler(request):
            return response({"status": "failure"}, message="storage failed")

        with self.assertRaisesRegex(MemOSClientError, "storage failed"):
            client(handler).delete_session("cube-a", "task_001")


    def test_create_cube_sends_owner_and_custom_id(self):
        def handler(request):
            assert json.loads(request.content) == {
                "cube_name": "cube-a", "cube_id": "cube-a", "owner_id": "tau3-eval"
            }
            return response({"cube_id": "cube-a"})

        assert client(handler).create_cube("cube-a") == "cube-a"
