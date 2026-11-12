"""使用独立数据目录启动本地 MemOS Python 主服务。"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
SKILL_PROFILE = os.getenv("MEMOS_EVAL_PROFILE") == "skills"
PORT = int(os.getenv("MEMOS_EVAL_PORT", "8001" if SKILL_PROFILE else "8000"))
RUNTIME = ROOT / "local_memos" / ("runtime_skills" if SKILL_PROFILE else "runtime")
load_dotenv(ROOT / "tau2-bench" / ".env")
# 后续 MemOS 模块不能再次从其它目录覆盖当前凭据。
os.environ["PYTHON_DOTENV_DISABLED"] = "1"
base = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
key = os.getenv("OPENAI_API_KEY")
if not base or not key:
    raise SystemExit("缺少 benchmark 已使用的 OpenAI 兼容 API 配置")

embedding_model = os.getenv("MOS_EMBEDDER_MODEL", "text-embedding-3-large")
embedding_dimension = int(os.getenv("EMBEDDING_DIMENSION", "3072"))
embedding_base = os.getenv("MOS_EMBEDDER_API_BASE") or base
embedding_key = os.getenv("MOS_EMBEDDER_API_KEY")
if embedding_base.rstrip("/") != base.rstrip("/") and not embedding_key:
    raise SystemExit("独立 embedding 渠道需要配置 MOS_EMBEDDER_API_KEY")

files = RUNTIME / "files"
files.mkdir(parents=True, exist_ok=True)
os.environ.update(
    {
        "MEMOS_BASE_PATH": str(RUNTIME),
        "FILE_LOCAL_PATH": str(files),
        "GRAPH_DB_BACKEND": "neo4j",
        "MOS_NEO4J_SHARED_DB": "true",
        "NEO4J_URI": "bolt://127.0.0.1:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "local-eval",
        "NEO4J_DB_NAME": "neo4j",
        "EMBEDDING_DIMENSION": str(embedding_dimension),
        "MOS_EMBEDDER_BACKEND": "universal_api",
        "MOS_EMBEDDER_MODEL": embedding_model,
        "MOS_EMBEDDER_API_BASE": embedding_base,
        "MOS_EMBEDDER_API_KEY": embedding_key or key,
        "MOS_EMBEDDER_TIMEOUT": os.getenv("MOS_EMBEDDER_TIMEOUT", "30"),
        "OPENAI_API_BASE": base,
        "MEMRADER_API_BASE": base,
        "MEMRADER_API_KEY": key,
        "MEMRADER_MODEL": "deepseek-v4.1-flash-1017",
        "MOS_CHAT_MODEL": "deepseek-v4.1-flash-1017",
        "MEMREADER_GENERAL_MODEL": "deepseek-v4.1-flash-1017",
        "FEEDBACK_MODEL": "deepseek-v4.1-flash-1017",
        "SUGGESTION_MODEL": "deepseek-v4.1-flash-1017",
        "MEMSCHEDULER_MODEL": "deepseek-v4.1-flash-1017",
        "PREFERENCE_EXTRACTOR_MODEL": "deepseek-v4.1-flash-1017",
        "MEM_READER_BACKEND": "multimodal_struct",
        "MOS_ENABLE_REORGANIZE": "false",
        "MEMSCHEDULER_USE_REDIS_QUEUE": "false",
        "API_SCHEDULER_ON": "false",
        "ENABLE_CHAT_API": "false",
        "ENABLE_INTERNET": "false",
        # PreferenceMemory 由 fine reader 写入同一 text_mem。
        "ENABLE_PREFERENCE_MEMORY": "true",
        "ENABLE_ACTIVATION_MEMORY": "false",
        "MEM_READER_MEM_VERSION_SWITCH": "off",
        "MOS_RERANKER_BACKEND": "cosine_local",
    }
)

if SKILL_PROFILE:
    os.environ.update({
        "SKILLS_REPO_BACKEND": "LOCAL",
        "SKILLS_LOCAL_TMP_DIR": str(RUNTIME / "skill_tmp"),
        "SKILLS_LOCAL_DIR": str(files),
        "SKILLS_OSS_DIR": "skill_memory/",
        "ENABLE_PREFERENCE_MEMORY": "false",
    })

# 记录主服务内部的模型用量，不保存请求正文、向量或凭据。
_original_send = httpx.Client.send


def _usage_send(self, request, *args, **kwargs):
    operation = request.url.path.rsplit("/", 1)[-1]
    tracked = operation in {"embeddings", "completions"}
    body = json.loads(request.content) if tracked else {}
    started = time.perf_counter()
    response = _original_send(self, request, *args, **kwargs)
    if tracked and not body.get("stream"):
        response.read()
        payload = response.json()
        from memos.context.context import get_current_trace_id

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": get_current_trace_id(),
            "operation": operation,
            "model": body.get("model"),
            "status_code": response.status_code,
            "duration_seconds": time.perf_counter() - started,
            "usage": payload.get("usage"),
        }
        with (RUNTIME / "model_usage.jsonl").open("a") as output:
            output.write(json.dumps(record) + "\n")
    return response


httpx.Client.send = _usage_send

from memos.mem_user.user_manager import UserManager  # noqa: E402


users = UserManager()
if not users.validate_user("tau3-eval"):
    users.create_user("tau3-eval", user_id="tau3-eval")

if __name__ == "__main__":
    import uvicorn
    import memos
    from memos.api.server_api import app
    from memos.api.routers.server_router import components
    from internal_info_compat import install_internal_info_compat

    # 老删除接口读取此属性；新偏好记忆已存放在 text_mem 中。
    components["naive_mem_cube"].pref_mem = None
    install_internal_info_compat(components["graph_db"])
    if SKILL_PROFILE:
        from native_skills import EVAL_FEATURES, install_native_skills
        from skill_metadata_compat import install_skill_metadata_compat

        install_skill_metadata_compat(components["graph_db"])
        install_native_skills(app, components, RUNTIME)
    else:
        from async_memory import EVAL_FEATURES, install_async_memory

        install_async_memory(app, components, RUNTIME / "async_writes.jsonl")
    (RUNTIME / "runtime_info.json").write_text(
        json.dumps(
            {
                "memos_version": memos.__version__,
                "memos_source": memos.__file__,
                "neo4j_version": "5.26.6",
                "graph_backend": "neo4j",
                "shared_database": True,
                "chunker": "sentence / chonkie 1.1.0 / gpt2",
                "reader": "multimodal_struct",
                "embedding_backend": "universal_api",
                "embedding_api_base": embedding_base,
                "embedding_model": embedding_model,
                "embedding_dimension": embedding_dimension,
                "embedding_timeout_seconds": int(os.environ["MOS_EMBEDDER_TIMEOUT"]),
                "reorganize": False,
                "redis_queue": False,
                "scheduler_running": False,
                "eval_features": EVAL_FEATURES,
                "async_workers": 0 if SKILL_PROFILE else 8,
                "profile": "native_skill_only" if SKILL_PROFILE else "ordinary_async",
                "skill_storage": "LOCAL" if SKILL_PROFILE else None,
                "preference_storage": None if SKILL_PROFILE else "text_mem / PreferenceMemory",
                "tool_trajectory_extraction": False,
                "neo4j_internal_info_json_roundtrip": True,
                "host": "127.0.0.1",
                "port": PORT,
                "server_pid": os.getpid(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    uvicorn.run(app, host="127.0.0.1", port=PORT, workers=1)
