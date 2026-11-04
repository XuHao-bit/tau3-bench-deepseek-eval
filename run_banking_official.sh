#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$ROOT/.tools/bin" ]; then
  export PATH="$ROOT/.tools/bin:$PATH"
fi
cd "$ROOT/tau2-bench"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

USER_LLM="${USER_LLM:-openai/gpt-5.2}"
# Optional: separate endpoint/key for the user simulator (e.g. official OpenAI gpt-5.2)
USER_API_BASE="${USER_API_BASE:-}"
USER_API_KEY="${USER_API_KEY:-}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"
NUM_TRIALS="${NUM_TRIALS:-1}"
NUM_TASKS="${NUM_TASKS:-}"
# 新配置单独保存，避免自动续跑复用旧结果。
SAVE_TO="${SAVE_TO:-banking_full_dsv4_max_user_low}"
SEED="${SEED:-300}"

command -v srt >/dev/null 2>&1 || {
  echo "srt not found. Install @anthropic-ai/sandbox-runtime@0.0.23, or put it in .tools/bin/."
  exit 1
}
command -v rg >/dev/null 2>&1 || {
  echo "rg not found. Install ripgrep, or put it in .tools/bin/."
  exit 1
}

if [ "${SKIP_PREFLIGHT:-0}" != "1" ]; then
  USER_LLM="$USER_LLM" USER_API_BASE="$USER_API_BASE" USER_API_KEY="$USER_API_KEY" python - <<'PY'
import os
import requests
from dotenv import load_dotenv

load_dotenv(".env")
base = (
    os.getenv("USER_API_BASE")
    or os.getenv("OPENAI_BASE_URL")
    or os.getenv("OPENAI_API_BASE")
    or ""
).rstrip("/")
key = os.getenv("USER_API_KEY") or os.getenv("OPENAI_API_KEY")
model = os.environ["USER_LLM"].split("/", 1)[-1]
if not base or not key:
    raise SystemExit("USER_API_BASE/OPENAI_BASE_URL or API key is not configured")
r = requests.post(
    base + "/chat/completions",
    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    json={
        "model": model,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_completion_tokens": 128,
        "reasoning_effort": "low",
    },
    timeout=60,
)
if not r.ok:
    print(f"user model {model} is not available on this endpoint:")
    print(r.text[:500])
    raise SystemExit(1)
print(f"user model preflight ok: {model}")
PY
fi

if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
  exit 0
fi

# 通过 extra_body 保留推理档位，避免旧版 LiteLLM 丢弃参数。
USER_ARGS=$(USER_API_BASE="$USER_API_BASE" USER_API_KEY="$USER_API_KEY" python - <<'PY'
import json
import os

args = {"extra_body": {"reasoning_effort": "low"}}
if os.environ.get("USER_API_BASE"):
    args["api_base"] = os.environ["USER_API_BASE"]
if os.environ.get("USER_API_KEY"):
    args["api_key"] = os.environ["USER_API_KEY"]
print(json.dumps(args))
PY
)

RUN=(
  tau2 run
  --domain banking_knowledge
  --retrieval-config alltools
  --agent-llm openai/deepseek-v4.1-flash-1017
  --agent-llm-args '{"temperature":0.0,"extra_body":{"reasoning_effort":"max"}}'
  --user-llm "$USER_LLM"
  --user-llm-args "$USER_ARGS"
  --num-trials "$NUM_TRIALS"
  --seed "$SEED"
  --max-steps 200
  --max-concurrency "$MAX_CONCURRENCY"
  --max-retries 1
  --auto-resume
  --save-to "$SAVE_TO"
  --verbose-logs 
  --llm-log-mode all
)

if [ -n "$NUM_TASKS" ]; then
  RUN+=(--num-tasks "$NUM_TASKS")
fi

if command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu "${RUN[@]}"
else
  exec "${RUN[@]}"
fi
