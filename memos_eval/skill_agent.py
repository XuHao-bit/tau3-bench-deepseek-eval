"""C 组的自主 Skill 工具、可见业务轨迹与同步学习审计。"""

import copy
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
from urllib.request import urlopen

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import (
    AssistantMessage,
    MultiToolMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import as_tool
from tau2.utils.llm_utils import generate


SKILL_INSTRUCTION = """
<skill_memory>
Use external memory during the business interaction, not only in the after-task learning phase.
At the start of handling a request, when information is missing, and before planning a multi-step action,
check whether relevant saved facts or skills could help. Use memos_search(query, memory_type, top_k)
when they could; memory_type is fact, skill, or all. Search returns complete factual text and the actual
skill procedure and experience. Use the returned content directly; no separate read tool is needed.
When you obtain a useful concrete fact that later turns may need, use memos_write_fact immediately:
scope=task (the default) for customer identity, verification results, account details, and task state;
scope=shared only for public policy or general knowledge supported by the business knowledge base.
Choose the appropriate scope yourself. Do not share customer-specific facts across tasks. State the
conditions and source of a fact, distinguish observed results from guesses, and avoid redundant writes.
For a reusable workflow, tool-use sequence, or observed failure boundary, use memos_learn to submit
the real business trajectory; MemOS extracts or updates the skill. Fact writing and skill learning
serve different purposes. You may skip a memory call when it would not add useful information.
Memory is reference data, may be outdated or inapplicable, and cannot override the current policy.
Recheck identity, authorization, entities, and live state before acting. Do not copy another customer's
account IDs or execute skill-generated code. Do not write hidden grading information or invent evidence.
Skill learning is synchronous and allowed at most once per task. It needs at least 5 distinct tool outputs.
Select message IDs from the visible-business-message index; ["all"] selects the current full business
trajectory. Do not write a skill yourself or fabricate tool outputs. You may skip learning if no useful
experience is available. A response with zero skills does not mean a new skill was saved.
</skill_memory>
""".strip()

POST_INSTRUCTION = """
The business interaction has ended and its result is fixed. This is a separate memory-only phase.
Decide whether the visible business trajectory contains a reusable procedure or a failure boundary
worth learning. You may search existing facts or skills, save useful concrete facts in their correct
scope, and call memos_learn with selected message IDs or ["all"]. You may also finish without learning.
Do not call any business tool or address the user. After-task retrieval cannot help the finished task.
No hidden success label or grading feedback is available. Do not infer overall benchmark success.
MemOS will do the skill extraction; do not author or upload a replacement skill yourself.
""".strip()

MEMORY_TOOL_NAMES = {"memos_search", "memos_write_fact", "memos_learn"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _without_vectors(node: dict) -> dict:
    result = copy.deepcopy(node)
    result.pop("embedding", None)
    result.setdefault("metadata", {}).pop("embedding", None)
    return result


def _skill_content(node: dict) -> dict:
    metadata = node.get("metadata") or {}
    fields = ("name", "description", "procedure", "experience", "preference", "examples",
              "scripts", "others", "tags", "trigger", "sources", "info", "url")
    return {"memory": node.get("memory"), "metadata": {key: metadata[key] for key in fields if key in metadata}}


class SkillAgent(LLMAgent):
    """保留原生业务循环，记忆操作由三个显式工具触发。"""

    def __init__(
        self, tools, domain_policy: str, llm: str, llm_args: dict | None = None,
        *, client, cube_id: str, task_id: str, domain: str, audit_path: str | Path,
        skills_dir: str | Path, max_memory_calls: int = 100, top_k: int = 6,
    ):
        super().__init__(list(tools), domain_policy, llm, llm_args)
        self.client = client
        self.cube_id = cube_id
        self.task_id = task_id
        self.domain = domain
        self.session_id = f"{domain}:{task_id}"
        self.audit_path = Path(audit_path)
        self.skills_dir = Path(skills_dir)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.max_memory_calls = max_memory_calls
        self.top_k = top_k
        self.memory_tools = [as_tool(self.memos_search), as_tool(self.memos_write_fact), as_tool(self.memos_learn)]
        self.tools = list(tools) + self.memory_tools
        self._reset()

    def _reset(self) -> None:
        self._state = None
        self._finished = False
        self._learn_attempted = False
        self._phase = "business"
        self.unsafe_to_continue = False
        self.stats = Counter()
        self.read_events = []
        self.learning_status = "not_requested"
        self.learning_result = None
        self.learning_failure_without_write = False
        self.initial_skills = []
        self.final_skills = []
        self.initial_facts = []
        self.final_facts = []
        self.fact_writes = []
        self._final_snapshot_confirmed = True
        self.post_learning_usage = {
            "calls": 0, "input_tokens": 0, "output_tokens": 0,
            "reasoning_tokens": 0, "reported_cost_usd": 0.0,
        }

    @property
    def system_prompt(self) -> str:
        return super().system_prompt + "\n\n" + SKILL_INSTRUCTION

    def get_init_state(self, message_history=None):
        self._reset()
        self._state = super().get_init_state(list(message_history or []))
        with self.client.learning_lock:
            self.initial_skills = self._snapshot("ready")
            self.final_skills = self.initial_skills
            self.initial_facts = copy.deepcopy(self.final_facts)
        self._audit("skill_task_start", skill_count=len(self.initial_skills),
                    available_skill_ids=[row["id"] for row in self.initial_skills],
                    fact_count=len(self.initial_facts))
        return self._state

    def _audit(self, event: str, **fields) -> None:
        record = {
            "time": _now(), "event": event, "arm": "C", "task_id": self.task_id,
            "domain": self.domain, "cube_id": self.cube_id,
            "session_id": self.session_id, "phase": self._phase, **fields,
        }
        with self.audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _snapshot(self, status: str) -> list[dict]:
        with self.client.learning_lock, self.client.snapshot_lock:
            skills = [_without_vectors(node) for node in self.client.list_skills(self.cube_id)]
            facts = [_without_vectors(node) for node in self.client.list_all_facts(self.cube_id)]
            header = {"cube_id": self.cube_id, "snapshot_at": _now(), "status": status,
                      "task_id": self.task_id, "source_domain": self.domain}
            _write_json(self.skills_dir / "skills_latest.json", {**header, "skill_count": len(skills), "skills": skills})
            _write_json(self.skills_dir / "facts_latest.json", {**header, "fact_count": len(facts), "facts": facts})
            self.final_facts = facts
            return skills

    def _fact_snapshot(self, status: str) -> None:
        """事实快照只锁观察文件，不等待其他任务的 Skill 提炼。"""
        with self.client.snapshot_lock:
            facts = [_without_vectors(node) for node in self.client.list_all_facts(self.cube_id)]
            _write_json(self.skills_dir / "facts_latest.json", {
                "cube_id": self.cube_id, "snapshot_at": _now(), "status": status,
                "task_id": self.task_id, "source_domain": self.domain,
                "fact_count": len(facts), "facts": facts,
            })
            self.final_facts = facts

    def visible_trajectory(self) -> list[dict]:
        """仅投影 agent 已接收的业务消息，编号与原生历史索引对应。"""
        rows = []
        if self._state is None:
            return rows
        business_call_ids = {}
        for index, message in enumerate(self._state.messages, 1):
            row = {"message_id": f"m{index:04d}"}
            if isinstance(message, UserMessage):
                if message.is_tool_call():
                    continue
                row.update(role="user", content=message.content or "")
            elif isinstance(message, AssistantMessage):
                calls = [call for call in message.tool_calls or []
                         if call.requestor == "assistant" and call.name not in MEMORY_TOOL_NAMES]
                # 记忆工具调用的解释文字也不回灌给提炼器。
                if message.tool_calls and not calls:
                    continue
                row.update(role="assistant", content=message.content or "")
                if calls:
                    row["tool_calls"] = []
                    for call in calls:
                        business_call_ids[call.id] = call.name
                        row["tool_calls"].append({"id": call.id, "type": "function", "function": {
                            "name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)}})
            elif isinstance(message, ToolMessage):
                if message.requestor != "assistant" or message.id not in business_call_ids:
                    continue
                row.update(role="tool", tool_call_id=message.id, content=message.content or "")
            else:
                continue
            rows.append(row)
        return rows

    def _message_index(self) -> str:
        rows = self.visible_trajectory()
        index = [{"id": row["message_id"], "role": row["role"]} for row in rows]
        tools = len({row["content"] for row in rows if row["role"] == "tool" and row["content"]})
        return ("Visible business message IDs for memos_learn (or use [\"all\"]):\n"
                + json.dumps(index, ensure_ascii=False)
                + f"\nDistinct nonempty business tool outputs currently available: {tools}.")

    def generate_next_message(self, message, state):
        self._state = state
        if isinstance(message, MultiToolMessage):
            state.messages.extend(row for row in message.tool_messages if row.requestor == "assistant")
        elif message is not None:
            if isinstance(message, UserMessage) and message.is_tool_call():
                raise ValueError("用户模拟器内部工具调用不能进入 agent。")
            state.messages.append(message)
        response = generate(
            model=self.llm, tools=self.tools,
            messages=state.system_messages + [SystemMessage(role="system", content=self._message_index())]
                     + state.messages,
            call_name="agent_response", **self.llm_args,
        )
        response.validate()
        state.messages.append(response)
        self._audit("agent_response", response=self._response_record(response),
                    visible_message_id=f"m{len(state.messages):04d}")
        return response, state

    @staticmethod
    def _response_record(response) -> dict:
        return {"role": "assistant", "content": response.content,
                "tool_calls": [call.model_dump() for call in response.tool_calls or []]}

    @staticmethod
    def _public_skill(node: dict) -> dict:
        metadata = node.get("metadata") or {}
        info = metadata.get("info") or {}
        result = {"memory_id": str(node.get("id") or node.get("memory_id") or ""),
                  "memory_type": "skill", "content": node.get("memory") or "",
                  "description": metadata.get("description") or node.get("memory") or "",
                  "source_task_id": info.get("task_id") or metadata.get("task_id") or metadata.get("session_id"),
                  "source_domain": info.get("source_domain") or metadata.get("source_domain")}
        for key in ("name", "procedure", "experience", "preference", "examples", "scripts", "others",
                    "tags", "trigger", "url", "created_at", "updated_at"):
            if key in metadata:
                result[key] = metadata[key]
        result["source_refs"] = [
            {key: source[key] for key in ("role", "message_id", "tool_call_id", "chat_time") if key in source}
            for source in metadata.get("sources") or [] if isinstance(source, dict)
        ]
        return result

    @staticmethod
    def _public_fact(node: dict) -> dict:
        metadata = node.get("metadata") or {}
        info = metadata.get("info") or {}
        return {
            "memory_id": str(node.get("id") or node.get("memory_id") or ""),
            "memory_type": "fact", "content": node.get("memory") or "",
            "scope": info.get("fact_scope") or metadata.get("fact_scope"),
            "source_task_id": info.get("task_id") or metadata.get("task_id"),
            "source_domain": info.get("source_domain") or metadata.get("source_domain"),
            "session_id": metadata.get("session_id"),
            "created_at": metadata.get("created_at"), "updated_at": metadata.get("updated_at"),
        }

    @classmethod
    def _public_memory(cls, node: dict) -> dict:
        if (node.get("metadata") or {}).get("memory_type") == "SkillMemory":
            result = cls._public_skill(node)
        else:
            result = cls._public_fact(node)
        metadata = node.get("metadata") or {}
        session = metadata.get("session_id") or ""
        if ":" in session:
            domain, task_id = session.split(":", 1)
            result["source_domain"] = result.get("source_domain") or domain
            result["source_task_id"] = result.get("source_task_id") or task_id
        return result

    def _reserve(self, operation: str) -> bool:
        if self.stats["memory_calls"] >= self.max_memory_calls:
            self._audit("memory_budget_exhausted", operation=operation)
            return False
        self.stats["memory_calls"] += 1
        self.stats[f"{operation}_calls"] += 1
        return True

    def _remember_read(self, operation: str, memories: list[dict]) -> None:
        def cross_task(row):
            task = row.get("source_task_id")
            domain = row.get("source_domain")
            return bool(task) and (task != self.task_id or bool(domain and domain != self.domain))
        skills = [row for row in memories if row["memory_type"] == "skill"]
        facts = [row for row in memories if row["memory_type"] == "fact"]
        self.read_events.append({
            "phase": self._phase, "operation": operation,
            "after_message_index": len(self._state.messages) if self._state else 0,
            "skill_ids": [row["memory_id"] for row in skills],
            "fact_ids": [row["memory_id"] for row in facts],
            "cross_task_skill_ids": [row["memory_id"] for row in skills if cross_task(row)],
            "cross_task_fact_ids": [row["memory_id"] for row in facts if cross_task(row)],
            "source_tasks": [row.get("source_task_id") for row in memories],
            "returned_versions": copy.deepcopy(memories),
        })

    def memos_search(
        self, query: str, memory_type: Literal["fact", "skill", "all"] = "all", top_k: int = 6,
    ) -> dict:
        """检索相关事实或原生技能，直接返回完整正文和可用方法。

        Args:
            query: 当前业务问题、方法或陷阱的语义描述，不使用题目 ID。
            memory_type: fact 读取当前题事实及公共事实；skill 读取共享技能；all 同时检索。
            top_k: 返回记忆数量，范围 1 到 6。
        """
        if not query.strip() or not 1 <= top_k <= self.top_k or memory_type not in {"fact", "skill", "all"}:
            return {"status": "skipped", "reason": "invalid_search_arguments", "memories": []}
        if not self._reserve("search"):
            return {"status": "skipped", "reason": "memory_call_budget", "memories": []}
        started = time.perf_counter()
        try:
            nodes = self.client.search_memories(self.cube_id, self.session_id, query,
                                                memory_type=memory_type, top_k=top_k)
        except Exception as error:
            self.stats["search_errors"] += 1
            self.unsafe_to_continue = True
            self._audit("memory_search_error", query=query, memory_type=memory_type, error=str(error))
            return {"status": "failed", "error": str(error), "memories": []}
        memories = [self._public_memory(node) for node in nodes]
        self._remember_read("search", memories)
        for kind in ("fact", "skill"):
            self.stats[f"{self._phase}_{kind}_hits"] += sum(row["memory_type"] == kind for row in memories)
        self._audit("memory_search", query=query, memory_type=memory_type, top_k=top_k, memories=memories,
                    elapsed_seconds=time.perf_counter() - started)
        return {"status": "completed", "memories": memories}

    def _blocked_write(self) -> dict | None:
        if self.client.learning_aborted.is_set():
            self.unsafe_to_continue = True
            self._audit("memory_write_blocked", reason="blocked_by_unconfirmed_learning")
            return {"status": "skipped", "reason": "blocked_by_unconfirmed_learning"}
        return None

    def memos_write_fact(self, content: str, scope: Literal["task", "shared"] = "task") -> dict:
        """同步保存一条具体事实；客户与核验信息默认仅当前任务可见。

        Args:
            content: 有当前对话或业务工具依据的完整事实文本，注明条件和来源。
            scope: task 适用于客户、核验、账户和当前任务状态；shared 仅用于公共政策或通用知识。
        """
        if not content.strip() or scope not in {"task", "shared"}:
            return {"status": "skipped", "reason": "invalid_fact_arguments"}
        if not self._reserve("write_fact"):
            return {"status": "skipped", "reason": "memory_call_budget"}
        blocked = self._blocked_write()
        if blocked:
            return blocked
        started = time.perf_counter()
        self._audit("fact_write_started", content=content, scope=scope)
        try:
            receipt = self.client.write_fact(self.cube_id, content, self.task_id, self.domain, scope=scope)
        except Exception as error:
            receipt = {"status": "unknown", "memory_ids": [], "memories": [], "error": str(error)}
        if receipt["status"] in {"unknown", "partial", "failed"}:
            self.unsafe_to_continue = True
            self.client.learning_aborted.set()
        record = {"time": _now(), "task_id": self.task_id, "domain": self.domain,
                  "cube_id": self.cube_id, "phase": self._phase, "content": content,
                  "scope": scope, "receipt": receipt, "elapsed_seconds": time.perf_counter() - started}
        self.fact_writes.append(record)
        with self.client.snapshot_lock:
            with (self.skills_dir / "fact_changes.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            try:
                self._fact_snapshot(receipt["status"])
            except Exception as error:
                receipt = {**receipt, "snapshot_status": "failed", "snapshot_error": str(error)}
                self._audit("fact_snapshot_error", error=str(error), storage_status=receipt["status"])
        self._audit("fact_write_finished", content=content, scope=scope, receipt=receipt)
        return {"status": receipt["status"], "memory_ids": receipt.get("memory_ids", []),
                "memories": [self._public_memory(row) for row in receipt.get("memories", [])],
                **({"error": receipt["error"]} if receipt.get("error") else {})}

    def _selected_trajectory(self, message_ids: list[str]) -> tuple[list[dict], str | None]:
        rows = self.visible_trajectory()
        indices = {row["message_id"]: index for index, row in enumerate(rows)}
        if message_ids == ["all"]:
            selected = rows
        elif not message_ids or any(mid not in indices for mid in message_ids):
            return [], "unknown_or_empty_message_ids"
        else:
            start, end = min(indices[mid] for mid in message_ids), max(indices[mid] for mid in message_ids)
            calls = {call["id"]: i for i, row in enumerate(rows) for call in row.get("tool_calls", [])}
            returns = {row["tool_call_id"]: i for i, row in enumerate(rows) if row["role"] == "tool"}
            while True:
                previous = (start, end)
                for row in rows[start:end + 1]:
                    ids = [call["id"] for call in row.get("tool_calls", [])]
                    if row["role"] == "tool":
                        ids.append(row["tool_call_id"])
                    for call_id in ids:
                        if call_id not in calls or call_id not in returns:
                            return [], "incomplete_business_call"
                        start, end = min(start, calls[call_id]), max(end, returns[call_id])
                if previous == (start, end):
                    break
            users = [i for i in range(start + 1) if rows[i]["role"] == "user"]
            if users:
                start = users[-1]
            selected = rows[start:end + 1]
        requested = {call["id"] for row in selected for call in row.get("tool_calls", [])}
        returned = {row["tool_call_id"] for row in selected if row["role"] == "tool"}
        if requested != returned:
            return [], "incomplete_business_call"
        return copy.deepcopy(selected), None

    def memos_learn(self, message_ids: list[str]) -> dict:
        """提交真实可见业务轨迹，让 MemOS 同步提炼或更新原生 Skill。

        Args:
            message_ids: 可见业务消息编号，或 ["all"] 表示当前完整业务过程。
                需要至少五条内容不同的业务工具返回；每题最多一次实际提炼请求。
        """
        if self._learn_attempted or self.unsafe_to_continue:
            return {"status": "skipped", "reason": "learn_already_attempted_or_unconfirmed"}
        if not self._reserve("learn"):
            return {"status": "skipped", "reason": "memory_call_budget"}
        messages, error = self._selected_trajectory(message_ids)
        unique_tools = {row["content"] for row in messages if row["role"] == "tool" and row["content"]}
        if error or len(unique_tools) < 5:
            result = {"status": "skipped", "reason": error or "fewer_than_five_distinct_tool_messages",
                      "unique_tool_messages": len(unique_tools)}
            self._audit("skill_learning_skipped", requested_message_ids=message_ids, **result)
            return result
        queued_at = time.perf_counter()
        self._audit("skill_learning_queued", unique_tool_messages=len(unique_tools))
        with self.client.learning_lock:
            wait_seconds = time.perf_counter() - queued_at
            self.stats["learning_queue_wait_seconds"] += wait_seconds
            self._audit("skill_learning_dequeued", queue_wait_seconds=wait_seconds)
            blocked = self._blocked_write()
            if blocked:
                return blocked
            result = self._learn_locked(messages, unique_tools)
            if self.unsafe_to_continue:
                self.client.learning_aborted.set()
            return result

    def _archive_versions(self, skills: list[dict], stage: str) -> list[dict]:
        """在学习锁内保存本地技能文件，便于查看覆盖前后的实际版本。"""
        artifacts = []
        for node in skills:
            url = (node.get("metadata") or {}).get("url")
            if not url:
                continue
            parsed = urlparse(url)
            if parsed.hostname not in {"localhost", "127.0.0.1"}:
                continue
            directory = self.skills_dir / "skill_versions" / self.domain / self.task_id / stage
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{node['id']}.zip"
            try:
                with urlopen(url, timeout=10) as response:
                    path.write_bytes(response.read())
                artifacts.append({"memory_id": node["id"], "url": url, "path": str(path), "status": "saved"})
            except Exception as error:
                artifacts.append({"memory_id": node["id"], "url": url, "status": "failed", "error": str(error)})
        return artifacts

    def _learn_locked(self, messages: list[dict], unique_tools: set[str]) -> dict:
        self._learn_attempted = True
        try:
            before = self._snapshot("learning")
        except Exception as failure:
            self.learning_status = "failed"
            self.unsafe_to_continue = True
            self._audit("skill_learning_error", error=str(failure), request_sent=False)
            return {"status": "failed", "error": str(failure), "request_sent": False}
        before_artifacts = self._archive_versions(before, "before")
        info = {"arm": "C", "task_id": self.task_id, "source_domain": self.domain,
                "source_type": "agent_visible_trajectory",
                "submitted_message_ids": [row["message_id"] for row in messages]}
        self._audit("skill_learning_started", messages=messages, info=info,
                    unique_tool_messages=len(unique_tools), before_skills=before)
        started = time.perf_counter()
        try:
            receipt = self.client.learn(self.cube_id, self.session_id, messages, info)
        except Exception as failure:
            receipt = {"status": "unknown", "skill_ids": [], "skills": [], "error": str(failure)}
        self.learning_status = receipt.get("status", "unknown")
        self.learning_result = receipt
        raw_add = receipt.get("raw_add")
        failed_before_write = (
            self.learning_status == "failed"
            and receipt.get("write_state") == "not_started"
            and isinstance(raw_add, dict) and raw_add.get("data") == []
            and receipt.get("skill_ids") == []
        )
        native_unconfirmed = self.learning_status in {"failed", "partial", "unknown"}
        self.unsafe_to_continue = native_unconfirmed or self.client.learning_aborted.is_set()
        if native_unconfirmed and not failed_before_write:
            self.client.learning_aborted.set()
        try:
            after = self._snapshot(self.learning_status)
            self.final_skills = after
        except Exception as failure:
            after = None
            self._final_snapshot_confirmed = False
            self.unsafe_to_continue = True
            self.learning_status = "unknown"
            receipt = {**receipt, "status": "unknown", "snapshot_error": str(failure)}
            self.learning_result = receipt
            latest_path = self.skills_dir / "skills_latest.json"
            latest = json.loads(latest_path.read_text())
            latest.update(status="unknown", status_updated_at=_now(), error=str(failure))
            _write_json(latest_path, latest)
        after_artifacts = self._archive_versions(after, "after") if after is not None else []
        before_map = {row["id"]: row for row in before}
        after_map = {row["id"]: row for row in after} if after is not None else {}
        self.learning_failure_without_write = failed_before_write and after is not None and before_map == after_map
        if self.learning_failure_without_write:
            self.unsafe_to_continue = self.client.learning_aborted.is_set()
            self.stats["confirmed_learning_failures_without_write"] += 1
            self._audit("skill_learning_failed_without_write", error=receipt.get("error"),
                        write_state=receipt["write_state"], skill_count=0)
        change = {
            "time": _now(), "cube_id": self.cube_id, "task_id": self.task_id,
            "domain": self.domain, "phase": self._phase, "status": self.learning_status,
            "added_ids": [mid for mid in after_map if mid not in before_map],
            "updated_ids": [mid for mid in after_map if mid in before_map
                            and _skill_content(after_map[mid]) != _skill_content(before_map[mid])],
            "removed_ids": [mid for mid in before_map if mid not in after_map] if after is not None else [],
            "after_snapshot_confirmed": self._final_snapshot_confirmed,
            "learning_failure_without_write": self.learning_failure_without_write,
            "before_skills": before, "after_skills": after,
            "before_artifacts": before_artifacts, "after_artifacts": after_artifacts,
            "receipt": receipt, "elapsed_seconds": time.perf_counter() - started,
        }
        with (self.skills_dir / "skill_changes.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(change, ensure_ascii=False) + "\n")
        self._audit("skill_learning_finished", **{key: value for key, value in change.items()
                    if key not in {"time", "cube_id", "task_id", "domain", "phase"}})
        return {"status": self.learning_status, "skill_ids": receipt.get("skill_ids", []),
                "memories": [self._public_memory(row) for row in receipt.get("skills", [])],
                "learning_failure_without_write": self.learning_failure_without_write,
                "added_ids": change["added_ids"], "updated_ids": change["updated_ids"],
                **({"error": receipt["error"]} if receipt.get("error") else {})}

    def finish(self, termination_reason, tail=None) -> None:
        """业务结果固定后执行有限的记忆决策，不改变原生历史或环境。"""
        if self._finished:
            return
        self._finished = True
        reason = getattr(termination_reason, "value", str(termination_reason))
        if self.client.learning_aborted.is_set():
            self.unsafe_to_continue = True
            self._audit("post_learning_skipped", reason="blocked_by_unconfirmed_learning")
            return
        if self._state is None or reason in {"infrastructure_error", "agent_error", "user_error"} or self.unsafe_to_continue:
            self._audit("post_learning_skipped", reason=reason, unsafe_to_continue=self.unsafe_to_continue)
            return
        self._phase = "post_learning"
        messages = [SystemMessage(role="system", content=self.system_prompt + "\n\n" + POST_INSTRUCTION),
                    UserMessage(role="user", content=json.dumps({"visible_business_trajectory": self.visible_trajectory(),
                                                               "learn_already_attempted": self._learn_attempted},
                                                              ensure_ascii=False))]
        self._audit("post_learning_started", visible_business_trajectory=self.visible_trajectory(), termination_reason=reason)
        tools = {tool.name: tool for tool in self.memory_tools}
        try:
            for _ in range(6):
                response = generate(model=self.llm, tools=self.memory_tools, messages=list(messages),
                                    call_name="skill_learning_decision", **self.llm_args)
                response.validate()
                self.post_learning_usage["calls"] += 1
                usage = (response.raw_data or {}).get("usage") or response.usage or {}
                self.post_learning_usage["input_tokens"] += usage.get("prompt_tokens", 0) or 0
                self.post_learning_usage["output_tokens"] += usage.get("completion_tokens", 0) or 0
                self.post_learning_usage["reasoning_tokens"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
                self.post_learning_usage["reported_cost_usd"] += response.cost or 0
                messages.append(response)
                self._audit("post_learning_response", response=self._response_record(response), usage=usage, cost=response.cost)
                if not response.is_tool_call():
                    break
                for call in response.tool_calls:
                    if call.name not in tools:
                        result = {"status": "skipped", "reason": "business_tools_disabled_after_interaction"}
                        self._audit("post_learning_tool_rejected", tool_name=call.name)
                    else:
                        result = tools[call.name](**call.arguments)
                    messages.append(ToolMessage(role="tool", id=call.id, content=json.dumps(result, ensure_ascii=False)))
                if self.unsafe_to_continue:
                    break
        except Exception as error:
            self.unsafe_to_continue = True
            self._audit("post_learning_error", error_type=type(error).__name__, error=str(error))
        finally:
            # 已确认的本题学习快照足以结算，不等待其他任务的后续学习。
            needs_snapshot = not (self._learn_attempted and self._final_snapshot_confirmed)
            if not self.unsafe_to_continue and needs_snapshot:
                try:
                    self.final_skills = self._snapshot("task_completed")
                except Exception as error:
                    self._final_snapshot_confirmed = False
                    self.unsafe_to_continue = True
                    self._audit("memory_snapshot_error", error=str(error))
            self._audit("post_learning_finished", learning_status=self.learning_status,
                        usage=self.post_learning_usage, unsafe_to_continue=self.unsafe_to_continue)
            self._phase = "business"

    def summary(self) -> dict:
        def ids(field, phase=None):
            return sorted({mid for event in self.read_events if phase is None or event["phase"] == phase
                           for mid in event[field]})
        return {
            **dict(self.stats), "initial_skill_count": len(self.initial_skills),
            "initial_fact_count": len(self.initial_facts),
            "final_fact_count": len(self.final_facts) if self._final_snapshot_confirmed else None,
            "final_skill_count": len(self.final_skills) if self._final_snapshot_confirmed else None,
            "last_observed_skill_count": len(self.final_skills), "learning_status": self.learning_status,
            "learning_failure_without_write": self.learning_failure_without_write,
            "learn_attempted": self._learn_attempted,
            "unsafe_to_continue": self.unsafe_to_continue or self.client.learning_aborted.is_set(),
            "read_skill_ids": ids("skill_ids"), "cross_task_skill_ids": ids("cross_task_skill_ids"),
            "business_read_skill_ids": ids("skill_ids", "business"),
            "business_cross_task_skill_ids": ids("cross_task_skill_ids", "business"),
            "post_read_skill_ids": ids("skill_ids", "post_learning"), "read_events": self.read_events,
            "read_fact_ids": ids("fact_ids"), "cross_task_fact_ids": ids("cross_task_fact_ids"),
            "business_read_fact_ids": ids("fact_ids", "business"),
            "business_cross_task_fact_ids": ids("cross_task_fact_ids", "business"),
            "post_read_fact_ids": ids("fact_ids", "post_learning"),
            "fact_writes": self.fact_writes,
            "post_learning_usage": self.post_learning_usage,
        }
