"""本地 MemOS 主服务的普通记忆接口。"""

from dataclasses import dataclass, field
from typing import Any
import uuid

import httpx


class MemOSClientError(RuntimeError):
    """MemOS 请求失败或返回的业务状态不成功。"""


@dataclass(frozen=True)
class MemoryHit:
    memory_id: str
    content: str
    task_id: str | None = None
    created_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MemOSClient:
    """只使用本地主服务接口，不连接 MemOS Cloud。"""

    def __init__(
        self,
        base_url: str,
        user_id: str = "tau3-eval",
        api_key: str | None = None,
        timeout: float = 300,
        *,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.user_id = user_id
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._http = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            headers=headers,
            timeout=timeout,
            transport=transport,
        )
        self.request_count = 0

    def close(self) -> None:
        """关闭客户端连接。"""
        self._http.close()

    def _request(self, method: str, path: str, body: Any = None, *, timeout=None) -> dict:
        self.request_count += 1
        try:
            options = {"timeout": timeout} if timeout is not None else {}
            response = self._http.request(method, path, json=body, **options)
        except httpx.RequestError as exc:
            raise MemOSClientError(f"MemOS {path} 连接失败：{type(exc).__name__}") from exc
        if response.is_error:
            raise MemOSClientError(f"MemOS {path} HTTP {response.status_code}")
        try:
            result = response.json()
        except ValueError as exc:
            raise MemOSClientError(f"MemOS {path} 没有返回 JSON") from exc
        if not isinstance(result, dict):
            raise MemOSClientError(f"MemOS {path} 返回结构不符合主服务协议")
        data = result.get("data")
        if result.get("code", 200) != 200 or (
            isinstance(data, dict) and data.get("status") == "failure"
        ):
            raise MemOSClientError(f"MemOS {path} 失败：{result.get('message', '未知错误')}")
        return result

    def ensure_user(self) -> dict:
        """检查服务可用；cube owner 须由部署端预先创建。

        当前主服务没有用户注册 HTTP 路由，create_cube 会验证 owner。
        """
        result = self._request("GET", "health")
        if result.get("status") != "healthy":
            raise MemOSClientError("MemOS 健康检查未通过")
        features = result.get("eval_features") or {}
        if not (features.get("async_add") and features.get("preference_memory")
                and features.get("skill_generation") is False
                and features.get("extraction") == "ordinary_fine"):
            raise MemOSClientError("MemOS 未启用本实验的异步普通记忆服务，请更新并重启 local_memos")
        return features

    def create_cube(self, cube_id: str) -> str:
        """为已存在的 owner 创建独立 cube，返回真实 cube ID。"""
        result = self._request(
            "POST",
            "product/create_cube",
            {"cube_name": cube_id, "cube_id": cube_id, "owner_id": self.user_id},
        )
        actual_id = (result.get("data") or {}).get("cube_id")
        if not actual_id:
            raise MemOSClientError("MemOS 创建 cube 后未返回 cube_id")
        return str(actual_id)

    @staticmethod
    def _hit(node: dict, cube_id: str, category: str = "text_mem") -> MemoryHit:
        metadata = dict(node.get("metadata") or {})
        metadata["cube_id"] = cube_id
        metadata.setdefault("memory_key", node.get("memory_key") or metadata.get("key"))
        content = node.get("memory") or node.get("content") or ""
        if category == "pref_mem" and metadata.get("preference"):
            metadata.setdefault("context_summary", content)
            content = metadata["preference"]
        info = metadata.get("info") or {}
        return MemoryHit(
            memory_id=str(node.get("id") or node.get("memory_id") or ""),
            content=content,
            task_id=(
                info.get("task_id")
                or metadata.get("task_id")
                or metadata.get("session_id")
            ),
            created_at=metadata.get("created_at") or node.get("created_at"),
            metadata=metadata,
        )

    @classmethod
    def _grouped_hits(cls, data: dict, cube_id: str) -> list[MemoryHit]:
        hits = []
        for category in ("text_mem", "pref_mem"):
            for group in data.get(category) or []:
                if group.get("cube_id") != cube_id:
                    continue
                for node in group.get("memories") or []:
                    hit = cls._hit(node, cube_id, category)
                    if hit.memory_id and hit.metadata.get("status") != "deleted":
                        hit.metadata["category"] = "preference" if category == "pref_mem" else "detail_factual"
                        hits.append(hit)
        return hits

    def search(
        self, cube_id: str, session_id: str, query: str, top_k: int = 6
    ) -> list[MemoryHit]:
        """按 cube 硬隔离检索，session 仅作为软相关度信号。"""
        result = self._request(
            "POST",
            "product/search",
            {
                "user_id": self.user_id,
                "readable_cube_ids": [cube_id],
                "session_id": session_id,
                "query": query,
                "top_k": top_k,
                "dedup": "no",
                "relativity": 0,
                "include_skill_memory": False,
                "include_preference": True,
                "search_tool_memory": False,
                "internet_search": False,
            },
        )
        hits = self._grouped_hits(result.get("data") or {}, cube_id)
        return hits[:top_k]

    def add(
        self, cube_id: str, session_id: str, messages: list[dict], info: dict
    ) -> dict:
        """提交一次异步 QA，票据用于查询，网络异常时不重发。"""
        if not messages:
            raise MemOSClientError("没有可提交的 QA")
        task_id = str(uuid.uuid4())
        try:
            result = self._request(
                "POST",
                "product/add",
                {
                    "user_id": self.user_id,
                    "writable_cube_ids": [cube_id],
                    "session_id": session_id,
                    "messages": messages,
                    "async_mode": "async",
                    "task_id": task_id,
                    "info": info,
                },
            )
        except MemOSClientError as exc:
            return {"status": "unknown", "task_id": task_id, "memory_ids": [], "error": str(exc)}
        entries = result.get("data") or []
        if len(entries) != 1 or entries[0].get("cube_id") != cube_id:
            return {"status": "unknown", "task_id": task_id, "memory_ids": [],
                    "error": "MemOS 未返回本次异步提交的回执"}
        receipt = dict(entries[0])
        if receipt.get("task_id") != task_id:
            return {"status": "unknown", "task_id": task_id, "memory_ids": [],
                    "error": "MemOS 异步写入票据不匹配"}
        return receipt

    def add_status(self, cube_id: str, session_id: str, task_id: str) -> dict:
        """即时查询后台票据，不等待记忆处理完成。"""
        result = self._request("POST", "product/add/status", {
            "mem_cube_id": cube_id, "session_id": session_id,
            "user_id": self.user_id, "task_id": task_id,
        }, timeout=3)
        return result.get("data") or {"status": "unknown", "task_id": task_id, "memory_ids": []}

    def _scoped_memories(
        self,
        cube_id: str,
        *,
        metadata_filter: dict | None = None,
        page_size: int | None = None,
    ) -> list[MemoryHit]:
        body: dict[str, Any] = {
            "mem_cube_id": cube_id,
            "include_preference": True,
            "include_tool_memory": False,
            "include_skill_memory": False,
        }
        if metadata_filter is not None:
            body["filter"] = metadata_filter
        if page_size is not None:
            body.update(page=1, page_size=page_size)
        result = self._request("POST", "product/get_memory", body)
        return self._grouped_hits(result.get("data") or {}, cube_id)

    def get(self, cube_id: str, memory_ids: list[str]) -> list[MemoryHit]:
        """在指定 cube 内按 ID 读取。

        裸 get_memory_by_ids 不接收 cube，且可能回退到服务默认 namespace；
        使用支持 cube 与 ID 条件的 get_memory 端点完成同一读取。
        """
        if not memory_ids:
            return []
        ids = list(dict.fromkeys(memory_ids))
        scoped = self._scoped_memories(
            cube_id, metadata_filter={"or": [{"id": mid} for mid in ids]}
        )
        return [hit for hit in scoped if hit.memory_id in ids]

    def list_memories(
        self,
        cube_id: str,
        session_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryHit]:
        """查看 cube 中已落库的普通记忆，支持按任务 session 筛选。

        服务端对 text/tool 分别分页；取足前缀后统一应用 offset。
        """
        if limit < 1 or offset < 0:
            raise ValueError("limit 必须为正数，offset 不得为负数")
        metadata_filter = {"and": [{"session_id": session_id}]} if session_id else None
        hits = self._scoped_memories(
            cube_id, metadata_filter=metadata_filter, page_size=offset + limit
        )
        return hits[offset : offset + limit]

    def delete_session(self, cube_id: str, session_id: str) -> None:
        """清理失败任务的记忆并验证，保留同 cube 中其它任务的数据。"""
        self._request(
            "POST",
            "product/delete_memory",
            {"writable_cube_ids": [cube_id], "session_id": session_id},
        )
        if self.list_memories(cube_id, session_id=session_id, limit=1):
            raise MemOSClientError(
                f"MemOS 清理后仍有记忆：cube={cube_id}, session={session_id}"
            )
