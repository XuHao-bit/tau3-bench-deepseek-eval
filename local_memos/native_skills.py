"""保留原生 Skill 算法，为同步学习提供提炼与文件状态。"""

import copy
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse


EVAL_FEATURES = {
    "async_add": False,
    "sync_add": True,
    "preference_memory": False,
    "skill_generation": True,
    "extraction": "native_skill_only",
    "sync_fact_write": True,
    "skill_storage": "LOCAL",
    "companion_fine_extraction": False,
    "tool_trajectory_extraction": False,
    "skill_learning_recall": "SkillMemory",
    "same_skill_update": "coalesce_native_evidence",
}


def _now():
    return datetime.now(timezone.utc).isoformat()


def coalesce_native_skill_updates(original_batch, reextract, observe=None, failed=None):
    """同次原生提取中，合并更新同一旧 Skill 的证据再提炼一次。"""
    def extract(task_chunks, related_memories_map, llm, chat_history):
        results = original_batch(task_chunks, related_memories_map, llm, chat_history)
        targets = {}
        for index, (candidate, _, _) in enumerate(results):
            if candidate.get("update") and candidate.get("old_memory_id"):
                targets.setdefault(str(candidate["old_memory_id"]), []).append(index)
        replacements, consumed = {}, set()
        for target, indices in targets.items():
            if len(indices) < 2:
                continue
            entries = [results[index] for index in indices]
            task_types = [entry[1] for entry in entries]
            candidates = [copy.deepcopy(entry[0]) for entry in entries]
            messages, seen = [], set()
            for _, _, chunk in entries:
                for message in chunk:
                    key = (message["idx"], message.get("role"), message.get("content"))
                    if key not in seen:
                        seen.add(key)
                        messages.append(message)
            messages.sort(key=lambda message: message["idx"])
            event = {"target_memory_id": target, "task_types": task_types,
                     "message_indices": [message["idx"] for message in messages],
                     "candidates": candidates}
            try:
                old = next(memory for task in task_types for memory in related_memories_map.get(task, [])
                           if str(memory.id) == target and memory.metadata.memory_type == "SkillMemory")
                merged = reextract(messages=messages, old_memories=[old], llm=llm, chat_history=chat_history)
                if not isinstance(merged, dict) or not merged.get("name"):
                    raise RuntimeError("同一 Skill 的合并证据未生成有效原生提炼结果")
                if merged.get("update") and str(merged.get("old_memory_id")) != target:
                    raise RuntimeError("同一 Skill 的合并提炼返回了其他更新目标")
            except Exception as exc:
                event.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
                if observe is not None:
                    observe(event)
                if failed is not None:
                    failed(exc)
                raise
            canonical = task_types[0]
            merged["_task_type"] = canonical
            # 原生后续附件和 sources 同样读取这两个字典，保持完整证据引用。
            task_chunks[canonical] = messages
            related_memories_map[canonical] = [old]
            replacements[indices[0]] = (merged, canonical, messages)
            consumed.update(indices[1:])
            event.update(status="coalesced", result=copy.deepcopy(merged))
            if observe is not None:
                observe(event)
        return [replacements.get(index, item) for index, item in enumerate(results) if index not in consumed]

    return extract


class _SkillOnlyRecall:
    """仅供原生 Skill 学习内部召回旧技能，不读取事实的原始来源。"""

    def __init__(self, searcher):
        self._searcher = searcher

    def search(self, query, *args, **kwargs):
        kwargs["memory_type"] = "SkillMemory"
        return self._searcher.search(query, *args, **kwargs)


def configure_skill_only_reader(reader, skill_process):
    """复用原生 fast 解析，仅将 fine 输入交给原生 Skill 提炼。"""
    original_process = reader._process_multi_modal_data
    skill_searcher = _SkillOnlyRecall(reader.searcher)

    def process(scene_data_info, info, mode="fine", **kwargs):
        fast_items = original_process(scene_data_info, info, mode="fast", **kwargs)
        if mode == "fast":
            return fast_items
        if mode != "fine":
            raise ValueError(f"C 服务不支持写入模式：{mode}")
        inputs = [item for item in fast_items if not reader._is_file_url_only_item(item)]
        return skill_process(
            fast_memory_items=inputs,
            info=info,
            searcher=skill_searcher,
            graph_db=reader.graph_db,
            llm=reader.general_llm,
            embedder=reader.embedder,
            oss_config=reader.oss_config,
            skills_dir_config=reader.skills_dir_config,
            **kwargs,
        )

    reader._process_multi_modal_data = process


class NativeSkillDiagnostics(logging.Handler):
    """按原生 trace 关联线程池中的提炼结果与警告，单次学习串行执行。"""

    def __init__(self, journal, files_dir, get_trace, secrets=()):
        super().__init__(logging.WARNING)
        self.journal = Path(journal)
        self.files_dir = Path(files_dir)
        self.get_trace = get_trace
        self.secrets = tuple(value for value in secrets if value)
        self._state_lock = threading.RLock()
        self._serial = threading.Lock()
        self._active = None

    def _record(self):
        with self._state_lock:
            if self._active is not None and self._active["trace_id"] == self.get_trace():
                return self._active
        return None

    def _clean(self, value):
        message = str(value)
        for secret in self.secrets:
            message = message.replace(secret, "[redacted]")
        return message

    def emit(self, record):
        active = self._record()
        if active is not None:
            with self._state_lock:
                active["warnings"].append({
                    "level": record.levelname,
                    "logger": record.name,
                    "message": self._clean(record.getMessage()),
                })

    def record_coalesced_update(self, event):
        active = self._record()
        if active is not None:
            with self._state_lock:
                active["coalesced_updates"].append(event)

    def fail_coalesced_update(self, error):
        active = self._record()
        if active is not None:
            with self._state_lock:
                active["error"] = {"type": type(error).__name__, "message": self._clean(error)}

    def observe_extraction(self, original, reconstruct):
        """只观察原生函数的输入与输出，不改变其提炼和更新判断。"""
        def extract(fast_memory_items, *args, **kwargs):
            active = self._record()
            if active is not None:
                restored = reconstruct(fast_memory_items)
                with self._state_lock:
                    active["skill_extraction_entered"] = True
                    active["restored_tool_messages"] = sum(message.get("role") == "tool" for message in restored)
            items = original(fast_memory_items, *args, **kwargs)
            if active is not None:
                with self._state_lock:
                    active["skill_extraction_returned"] = True
                    active["generated_skill_ids"] = [str(item.id) for item in items]
                    active["skill_count"] = len(items)
                    active["file_status"] = []
                    for item in items:
                        url = getattr(item.metadata, "url", "") or ""
                        name = Path(unquote(urlparse(url).path)).name if url else ""
                        active["file_status"].append({
                            "skill_id": str(item.id), "url": url,
                            "exists": bool(name and (self.files_dir / name).is_file()),
                        })
            return items

        return extract

    def observe_persistence(self, original):
        """原生写技能文件前标记；后续节点删除和落库均在该阶段之后。"""
        def write(*args, **kwargs):
            active = self._record()
            if active is not None:
                with self._state_lock:
                    active["skill_persistence_started"] = True
            return original(*args, **kwargs)

        return write

    def _finish_status(self, record):
        if record.get("error"):
            return "failed"
        if not record["skill_extraction_entered"]:
            return "unknown"
        if record["restored_tool_messages"] < 5:
            record["reason"] = "tool_messages_below_five"
            return "skipped"
        if record["skill_count"]:
            if any(not item["exists"] for item in record["file_status"]):
                return "partial"
            return "completed"
        if record["warnings"]:
            record["reason"] = "native_extraction_warning"
            return "failed"
        return "processed_no_skill"

    def _finish_write_state(self, record):
        if (record["skill_extraction_returned"] and not record["skill_persistence_started"]
                and record["skill_count"] == 0 and not record["generated_skill_ids"]
                and record.get("add_returned_empty") is True):
            return "not_started"
        if record["status"] in {"completed", "partial"}:
            return record["status"]
        return "unknown"

    def run_add(self, request, process):
        """执行一次原生 add，保存诊断；异常透传，不重新写入。"""
        with self._serial:
            record = {
                "trace_id": self.get_trace(), "cube_ids": request.writable_cube_ids,
                "session_id": request.session_id, "task_id": request.task_id,
                "started_at": _now(), "skill_extraction_entered": False,
                "skill_extraction_returned": False, "skill_persistence_started": False,
                "restored_tool_messages": None, "generated_skill_ids": [],
                "skill_count": 0, "file_status": [], "warnings": [],
                "operation": "learn", "companion_fine_extraction": False,
                "coalesced_updates": [],
            }
            with self._state_lock:
                self._active = record
            try:
                response = process(request)
                payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
                record["add_returned_empty"] = payload.get("code", 200) == 200 and payload.get("data") == []
            except Exception as exc:
                record["error"] = {"type": type(exc).__name__, "message": self._clean(exc)}
                raise
            finally:
                with self._state_lock:
                    record["status"] = self._finish_status(record)
                    record["write_state"] = self._finish_write_state(record)
                    record["finished_at"] = _now()
                    self.journal.parent.mkdir(parents=True, exist_ok=True)
                    with self.journal.open("a", encoding="utf-8") as output:
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    self._active = None
            payload["eval_diagnostics"] = copy.deepcopy(record)
            return payload

    def run_fast_add(self, request, process):
        """原生 fast 保存事实，不进入 Skill 学习队列或共享诊断状态。"""
        record = {
            "trace_id": self.get_trace(), "cube_ids": request.writable_cube_ids,
            "session_id": request.session_id, "task_id": request.task_id,
            "started_at": _now(), "operation": "write_fact",
            "skill_extraction_entered": False, "memory_ids": [], "fact_memory_ids": [],
        }
        try:
            response = process(request)
            payload = response.model_dump(mode="json") if hasattr(response, "model_dump") else dict(response)
            ids = [str(row["memory_id"]) for row in payload.get("data") or [] if row.get("memory_id")]
            record.update(memory_ids=ids, fact_memory_ids=ids, status="completed" if ids else "unknown")
        except Exception as exc:
            record.update(status="failed", error={"type": type(exc).__name__, "message": self._clean(exc)})
            raise
        finally:
            record["finished_at"] = _now()
            with self.journal.with_name("fact_writes.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        payload["eval_diagnostics"] = record
        return payload


def install_native_skills(app, components, runtime):
    """C 同步写事实，或仅执行原生 Skill 提炼与更新。"""
    import os

    from fastapi import HTTPException
    from fastapi.responses import JSONResponse
    import memos.mem_reader.read_skill_memory.process_skill_memory as skill_module
    from memos.api.routers.server_router import add_handler
    from memos.context.context import get_current_trace_id
    from memos.multi_mem_cube.single_cube import SingleCubeView
    from memos.types.general_types import SearchMode

    runtime = Path(runtime)
    recorder = NativeSkillDiagnostics(
        runtime / "skill_learning.jsonl", runtime / "files", get_current_trace_id,
        secrets=[os.getenv(name) for name in ("OPENAI_API_KEY", "MOS_EMBEDDER_API_KEY", "MEMRADER_API_KEY")],
    )
    skill_module.logger.addHandler(recorder)
    skill_module._write_skills_to_file = recorder.observe_persistence(skill_module._write_skills_to_file)
    skill_module._batch_extract_skills = coalesce_native_skill_updates(
        skill_module._batch_extract_skills, skill_module._extract_skill_memory_by_llm_md,
        recorder.record_coalesced_update, recorder.fail_coalesced_update)
    skill_process = recorder.observe_extraction(
        skill_module.process_skill_memory_fine, skill_module._reconstruct_messages_from_memory_items)
    configure_skill_only_reader(components["mem_reader"], skill_process)
    # 同步 fine 已完成本次提炼，不投递到关闭的 scheduler。
    SingleCubeView._schedule_memory_tasks = lambda *args, **kwargs: None

    def search_text(self, search_req, user_context, search_mode):
        dispatch = {SearchMode.FAST: self._fast_search, SearchMode.FINE: self._fine_search,
                    SearchMode.MIXTURE: self._mix_search}
        if search_mode not in dispatch:
            raise ValueError(f"不支持的检索模式：{search_mode}")
        return dispatch[search_mode](search_req, user_context)

    SingleCubeView._search_text = search_text
    original_add = add_handler.handle_add_memories

    def add(request):
        if request.async_mode != "sync" or request.mode not in {"fast", "fine"} or request.is_upload_skill or request.is_feedback:
            raise HTTPException(status_code=422, detail="C 服务仅接收 sync + fast 事实或 sync + fine 技能学习")
        if len(request.writable_cube_ids or []) != 1:
            raise HTTPException(status_code=422, detail="C 服务一次仅向一个明确的 cube 学习")
        if request.mode == "fast":
            return JSONResponse(recorder.run_fast_add(request, original_add))
        return JSONResponse(recorder.run_add(request, original_add))

    add_handler.handle_add_memories = add
    app.router.routes[:] = [route for route in app.router.routes if getattr(route, "path", None) != "/health"]

    @app.get("/health")
    def health():
        return {"status": "healthy", "service": "memos", "version": app.version,
                "eval_features": EVAL_FEATURES}

    return recorder
