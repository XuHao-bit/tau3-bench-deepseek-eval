"""C 组显式事实记忆与原生 Skill 的同步写入、学习和隔离读取。"""

import threading
import uuid

from memos_eval.client import MemOSClient, MemOSClientError


class SkillClient(MemOSClient):
    """事实只来自显式写入，Skill 通过原生提炼生成。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_lock = threading.RLock()
        self.snapshot_lock = threading.RLock()
        self.learning_aborted = threading.Event()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def ensure_skill_service(self) -> dict:
        """确认连接的是启用同步原生 Skill 提炼的服务。"""
        result = self._request("GET", "health")
        features = result.get("eval_features") or {}
        if not (result.get("status") == "healthy"
                and features.get("skill_generation") is True
                and features.get("sync_add") is True
                and features.get("sync_fact_write") is True
                and features.get("same_skill_update") == "coalesce_native_evidence"
                and features.get("extraction") == "native_skill_only"):
            raise MemOSClientError("MemOS 未启用 C 组原生 Skill 同步提炼服务")
        return features

    @staticmethod
    def _skills(data: dict, cube_id: str) -> list[dict]:
        """只解析当前 cube 的 Skill 分组，保留实际结构字段。"""
        rows = []
        for group in data.get("skill_mem") or []:
            if group.get("cube_id") != cube_id:
                continue
            for node in group.get("memories") or []:
                metadata = node.get("metadata") or {}
                if (node.get("id") and metadata.get("memory_type") == "SkillMemory"
                        and metadata.get("status") != "deleted"):
                    rows.append(node)
        return rows

    def search_skills(
        self, cube_id: str, session_id: str, query: str, top_k: int = 6,
    ) -> list[dict]:
        """用 Skill 专用额度检索，不将普通记忆交给 C 的 agent。"""
        if not 1 <= top_k <= 6:
            raise ValueError("Skill top_k 必须在 1～6 之间")
        result = self._request("POST", "product/search", {
            "user_id": self.user_id,
            "readable_cube_ids": [cube_id],
            "session_id": session_id,
            "query": query,
            "mode": "fast",
            "search_memory_type": "SkillMemory",
            "include_skill_memory": True,
            "skill_mem_top_k": top_k,
            "top_k": top_k,
            "include_preference": False,
            "search_tool_memory": False,
            "internet_search": False,
            "dedup": "no",
            "relativity": 0,
        })
        return self._skills(result.get("data") or {}, cube_id)[:top_k]

    def _read_skills(self, cube_id: str, memory_ids: list[str] | None = None) -> list[dict]:
        body = {
            "mem_cube_id": cube_id,
            "include_skill_memory": True,
            "include_preference": False,
            "include_tool_memory": False,
        }
        if memory_ids is not None:
            body["filter"] = {"or": [{"id": mid} for mid in memory_ids]}
        result = self._request("POST", "product/get_memory", body)
        return self._skills(result.get("data") or {}, cube_id)

    def get_skills(self, cube_id: str, memory_ids: list[str]) -> list[dict]:
        """按 ID 在指定 cube 读取完整 Skill，不使用无 scope 的旧端点。"""
        ids = list(dict.fromkeys(memory_ids))
        if not ids:
            return []
        return [node for node in self._read_skills(cube_id, ids) if node["id"] in ids]

    def list_skills(self, cube_id: str) -> list[dict]:
        """列出当前 cube 已保存的全部 Skill，供快照和查看使用。"""
        return self._read_skills(cube_id)

    @staticmethod
    def _fact_filter(session_id: str | None = None) -> dict:
        if session_id is None:
            return {"kind": "fact"}
        return {"or": [
            {"kind": "fact", "fact_scope": "shared"},
            {"kind": "fact", "fact_scope": "task", "session_id": session_id},
        ]}

    @staticmethod
    def _metadata_value(metadata: dict, key: str):
        return metadata.get(key, (metadata.get("info") or {}).get(key))

    @classmethod
    def _facts(cls, data: dict, cube_id: str, session_id: str | None = None) -> list[dict]:
        """只返回显式写入、且在当前任务可见的事实。"""
        rows = []
        for group in data.get("text_mem") or []:
            if group.get("cube_id") != cube_id:
                continue
            for node in group.get("memories") or []:
                metadata = node.get("metadata") or {}
                scope = cls._metadata_value(metadata, "fact_scope")
                if (not node.get("id") or metadata.get("memory_type") != "LongTermMemory"
                        or metadata.get("status") == "deleted"
                        or cls._metadata_value(metadata, "kind") != "fact"
                        or scope not in {"task", "shared"}):
                    continue
                if session_id is None or scope == "shared" or metadata.get("session_id") == session_id:
                    rows.append(node)
        return rows

    def search_memories(
        self, cube_id: str, session_id: str, query: str,
        memory_type: str = "all", top_k: int = 6,
    ) -> list[dict]:
        """按类型检索；可见性在服务端 top-k 前过滤，读取时再确认 scope。"""
        if memory_type not in {"fact", "skill", "all"}:
            raise ValueError("memory_type 必须是 fact、skill 或 all")
        if not 1 <= top_k <= 6:
            raise ValueError("top_k 必须在 1～6 之间")
        if memory_type == "skill":
            return self.search_skills(cube_id, session_id, query, top_k)
        visible_filter = self._fact_filter(session_id)
        if memory_type == "all":
            visible_filter = {"or": [{"memory_type": "SkillMemory"}, *visible_filter["or"]]}
        result = self._request("POST", "product/search", {
            "user_id": self.user_id,
            "readable_cube_ids": [cube_id],
            "session_id": session_id,
            "query": query,
            "mode": "fast",
            "search_memory_type": "All" if memory_type == "all" else "LongTermMemory",
            "filter": visible_filter,
            "include_skill_memory": memory_type == "all",
            "skill_mem_top_k": top_k,
            "top_k": top_k,
            "include_preference": False,
            "search_tool_memory": False,
            "internet_search": False,
            "dedup": "no",
            "relativity": 0,
        })
        data = result.get("data") or {}
        nodes = self._facts(data, cube_id, session_id)
        if memory_type == "all":
            nodes += self._skills(data, cube_id)
        nodes.sort(key=lambda node: float((node.get("metadata") or {}).get("relativity") or 0), reverse=True)
        return nodes[:top_k]

    def _read_facts(self, cube_id: str, metadata_filter: dict) -> list[dict]:
        result = self._request("POST", "product/get_memory", {
            "mem_cube_id": cube_id,
            "filter": metadata_filter,
            "include_skill_memory": False,
            "include_preference": False,
            "include_tool_memory": False,
        })
        return self._facts(result.get("data") or {}, cube_id)

    def list_facts(self, cube_id: str, session_id: str | None = None) -> list[dict]:
        """观察者可列全部事实；传 session 时仅列本题事实和共享事实。"""
        nodes = self._read_facts(cube_id, self._fact_filter(session_id))
        return [node for node in nodes if session_id is None
                or self._metadata_value(node["metadata"], "fact_scope") == "shared"
                or node["metadata"].get("session_id") == session_id]

    def list_all_facts(self, cube_id: str) -> list[dict]:
        """供人查看共享 cube 中全部显式事实，不作为 agent 的跨题读取。"""
        return self.list_facts(cube_id)

    def write_fact(
        self, cube_id: str, content: str, task_id: str, domain: str, scope: str = "task",
    ) -> dict:
        """同步 fast 保存 DS 主动写入的事实，不进行 LLM 二次提炼。"""
        if scope not in {"task", "shared"}:
            raise ValueError("fact scope 必须是 task 或 shared")
        if not content.strip():
            raise ValueError("事实正文不能为空")
        session_id = f"{domain}:{task_id}"
        receipt = {"status": "unknown", "memory_ids": [], "memories": [],
                   "raw_add": None, "diagnostics": {}}
        try:
            result = self._request("POST", "product/add", {
                "user_id": self.user_id,
                "writable_cube_ids": [cube_id],
                "session_id": session_id,
                "async_mode": "sync", "mode": "fast", "is_upload_skill": False,
                "messages": [{"role": "assistant", "content": content}],
                "info": {"kind": "fact", "fact_scope": scope, "arm": "C",
                         "source_type": "explicit_fact", "task_id": task_id,
                         "source_domain": domain},
            })
        except MemOSClientError as exc:
            receipt["error"] = str(exc)
            return receipt
        receipt["raw_add"] = result
        receipt["diagnostics"] = result.get("eval_diagnostics") or {}
        entries = result.get("data") or []
        if not isinstance(entries, list):
            receipt["error"] = "MemOS fast 写入返回格式不符合主服务协议"
            return receipt
        ids = list(dict.fromkeys(str(row["memory_id"]) for row in entries
                                if row.get("cube_id") == cube_id
                                and row.get("memory_type") == "LongTermMemory" and row.get("memory_id")))
        receipt["memory_ids"] = ids
        if not ids:
            receipt["error"] = "MemOS fast 写入没有返回事实 ID"
            return receipt
        try:
            nodes = self._read_facts(cube_id, {"or": [{"id": mid, "kind": "fact"} for mid in ids]})
        except MemOSClientError as exc:
            receipt.update(status="partial", error=f"事实写入后无法读回：{exc}")
            return receipt
        nodes = [node for node in nodes if node["id"] in ids
                 and node["metadata"].get("session_id") == session_id
                 and self._metadata_value(node["metadata"], "fact_scope") == scope
                 and self._metadata_value(node["metadata"], "task_id") == task_id
                 and self._metadata_value(node["metadata"], "source_domain") == domain]
        receipt["memories"] = nodes
        missing = [mid for mid in ids if mid not in {node["id"] for node in nodes}]
        if missing:
            receipt.update(status="partial", missing_memory_ids=missing,
                           error="部分事实写入后未按原 scope／来源读回")
        else:
            receipt["status"] = "completed"
        native_status = receipt["diagnostics"].get("status")
        if native_status in {"failed", "partial", "unknown"}:
            receipt["status"] = native_status
        return receipt

    def learn(
        self, cube_id: str, session_id: str, messages: list[dict], info: dict,
    ) -> dict:
        """同步提交一次真实轨迹；写入结果不明时不自动重发。"""
        if not messages:
            raise ValueError("没有可提交的轨迹")
        task_id = str(uuid.uuid4())
        receipt = {
            "status": "unknown", "task_id": task_id,
            "skill_ids": [], "skills": [], "raw_add": None,
            "diagnostics": {},
        }
        try:
            result = self._request("POST", "product/add", {
                "user_id": self.user_id,
                "writable_cube_ids": [cube_id],
                "session_id": session_id,
                "task_id": task_id,
                "async_mode": "sync",
                "mode": "fine",
                "is_upload_skill": False,
                "messages": messages,
                "info": info,
            }, timeout=1800)
        except MemOSClientError as exc:
            receipt["error"] = str(exc)
            return receipt

        receipt["raw_add"] = result
        diagnostics = result.get("eval_diagnostics") or {}
        receipt["diagnostics"] = diagnostics
        receipt["write_state"] = diagnostics.get("write_state", "unknown")
        native_status = diagnostics.get("status", "unknown")
        if native_status in {"completed", "processed_no_skill", "skipped", "failed", "partial", "unknown"}:
            receipt["status"] = native_status
        entries = result.get("data")
        if not isinstance(entries, list):
            receipt.update(status="unknown", error="MemOS 同步学习返回格式不符合主服务协议")
            return receipt
        receipt["empty_add_result"] = entries == []
        ids = list(dict.fromkeys(
            str(row["memory_id"]) for row in entries
            if row.get("cube_id") == cube_id
            and row.get("memory_type") == "SkillMemory" and row.get("memory_id")
        ))
        receipt["skill_ids"] = ids
        if diagnostics.get("error"):
            receipt["error"] = diagnostics["error"]
        if receipt["write_state"] == "not_started" and entries:
            receipt.update(status="partial", error="未开始写入的诊断与非空 add 结果不一致："
                           + (diagnostics.get("error") or "未提供提炼错误"))
        if not ids:
            if receipt["status"] == "completed":
                receipt.update(status="unknown", error="提炼状态为 completed，但 add 没有返回 Skill ID")
            return receipt

        try:
            skills = self.get_skills(cube_id, ids)
        except MemOSClientError as exc:
            receipt.update(status="partial", error=f"Skill 写入后无法读回：{exc}")
            return receipt
        receipt["skills"] = skills
        missing = [mid for mid in ids if mid not in {node["id"] for node in skills}]
        if missing:
            receipt.update(status="partial", missing_skill_ids=missing,
                           error="部分 Skill ID 写入后无法在当前 cube 读回")
        elif receipt["status"] in {"failed", "processed_no_skill", "skipped"}:
            receipt.update(status="partial", error="原生提炼状态与实际 Skill 产物不一致")
        return receipt
