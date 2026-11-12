"""本地实验的异步普通记忆写入：后台抽取、落库核验与票据查询。"""

import copy
import asyncio
import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager
from contextvars import copy_context


EVAL_FEATURES = {
    "async_add": True,
    "preference_memory": True,
    "skill_generation": False,
    "extraction": "ordinary_fine",
}


def _now():
    return datetime.now(timezone.utc).isoformat()


class AsyncMemoryWriter:
    """同一 cube/session 顺序执行，不同任务最多八路并行。"""

    def __init__(self, process, verify, journal: Path, workers=8):
        self.process = process
        self.verify = verify
        self.journal = journal
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="memos-qa")
        self.records = {}
        self.tails = {}
        if journal.exists():
            for line in journal.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.records[record["task_id"]] = record
            for record in list(self.records.values()):
                if record["status"] == "submitted":
                    self._update(record["task_id"], status="unknown", error="服务已重启，后台完成状态未确认")

    def _update(self, task_id, **updates):
        with self.lock:
            record = self.records[task_id]
            record.update(updates)
            with self.journal.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            return copy.deepcopy(record)

    def submit(self, request, trace_id=None):
        cube_ids = request.writable_cube_ids or [request.user_id]
        if len(cube_ids) != 1:
            raise ValueError("本地异步实验一次只写一个 cube")
        cube = cube_ids[0]
        session = request.session_id or "default_session"
        task_id = request.task_id or str(uuid.uuid4())
        with self.lock:
            if task_id in self.records:
                record = self.records[task_id]
                if (record["cube_id"], record["session_id"], record["user_id"]) != (cube, session, request.user_id):
                    raise ValueError("异步票据已属于其他任务")
                return copy.deepcopy(record)
            self.records[task_id] = {
                "task_id": task_id, "status": "submitted", "memory_ids": [],
                "cube_id": cube, "session_id": session, "user_id": request.user_id,
                "submitted_at": _now(),
                "trace_id": trace_id,
            }
            receipt = self._update(task_id)
            previous = self.tails.get((cube, session))
            self.tails[(cube, session)] = self.executor.submit(
                copy_context().run, self._run, task_id, request.model_copy(deep=True), previous)
            return receipt

    def _run(self, task_id, request, previous):
        try:
            if previous is not None:
                previous.result()
            self._update(task_id, started_at=_now())
            # HTTP 始终为 async；后台运行主服务的普通 fine factual/preference 路径。
            background = request.model_copy(update={"async_mode": "sync", "mode": "fine"})
            response = self.process(background)
            entries = response.data or []
            ids = [str(row["memory_id"]) for row in entries if row.get("memory_id")]
            record = self.records[task_id]
            visible = set(self.verify(record["cube_id"], ids))
            missing = sorted(set(ids) - visible)
            if missing:
                raise RuntimeError(f"后台写入后有 {len(missing)} 个 memory ID 无法读回")
            self._update(task_id, status="completed", memory_ids=ids, completed_at=_now())
        except Exception as exc:
            self._update(task_id, status="failed", error=str(exc), completed_at=_now())

    def status(self, cube, session, task_id, user_id):
        with self.lock:
            record = self.records.get(task_id)
            if record is None or (record["cube_id"], record["session_id"], record["user_id"]) != (cube, session, user_id):
                return {"task_id": task_id, "status": "unknown", "memory_ids": []}
            return copy.deepcopy(record)

    def close(self):
        self.executor.shutdown(wait=True)


def configure_ordinary_reader(reader):
    """保留原生事实与偏好抽取，禁用 Skill 和工具轨迹抽取入口。"""
    import memos.mem_reader.multi_modal_struct as module
    from memos.multi_mem_cube.single_cube import SingleCubeView
    from memos.types.general_types import SearchMode

    def no_skill(*args, **kwargs):
        return []

    module.process_skill_memory_fine = no_skill
    reader._process_tool_trajectory_fine = lambda *args, **kwargs: []
    # 写入任务由本地 executor 完整处理，不再投递给未启动的原生 scheduler。
    SingleCubeView._schedule_memory_tasks = lambda *args, **kwargs: None

    def search_text(self, search_req, user_context, search_mode):
        dispatch = {SearchMode.FAST: self._fast_search, SearchMode.FINE: self._fine_search,
                    SearchMode.MIXTURE: self._mix_search}
        if search_mode not in dispatch:
            raise ValueError(f"不支持的检索模式：{search_mode}")
        return dispatch[search_mode](search_req, user_context)

    # 嵌入或数据库请求失败必须传播到 HTTP，不能返回成功空记忆。
    SingleCubeView._search_text = search_text


def install_async_memory(app, components, journal):
    """替换当前服务实例的 add 分发，不修改 MemOS 源文件。"""
    from fastapi import HTTPException
    from memos.api.product_models import MemoryResponse
    from memos.api.routers.server_router import add_handler
    from memos.context.context import get_current_trace_id

    configure_ordinary_reader(components["mem_reader"])
    original_add = add_handler.handle_add_memories
    text_mem = components["naive_mem_cube"].text_mem

    def verify(cube, ids):
        if not ids:
            return []
        return [str(row["id"]) for row in text_mem.get_by_ids(ids, user_name=cube)]

    writer = AsyncMemoryWriter(original_add, verify, Path(journal))

    def add(request):
        if request.async_mode != "async":
            return original_add(request)
        return MemoryResponse(message="Memory submitted", data=[writer.submit(request, get_current_trace_id())])

    add_handler.handle_add_memories = add

    @app.post("/product/add/status")
    def add_status(body: dict):
        required = ["mem_cube_id", "session_id", "task_id", "user_id"]
        if not all(body.get(key) for key in required):
            raise HTTPException(status_code=422, detail="缺少 cube/session/task/user 标识")
        return {"code": 200, "message": "Memory submission status", "data": writer.status(
            body["mem_cube_id"], body["session_id"], body["task_id"], body["user_id"])}

    app.router.routes[:] = [route for route in app.router.routes if getattr(route, "path", None) != "/health"]

    @app.get("/health")
    def health():
        return {"status": "healthy", "service": "memos", "version": app.version,
                "eval_features": EVAL_FEATURES}

    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application):
        async with original_lifespan(application) as state:
            try:
                yield state
            finally:
                # 先等后台写入终止，再由原生命周期关闭 graph store。
                await asyncio.to_thread(writer.close)

    app.router.lifespan_context = lifespan
    return writer
