#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UPSTREAM="$ROOT/tau2-bench"
VENV="$ROOT/.venv"
TAG="${TAU2_TAG:-v1.0.1}"
REPO="${TAU2_REPO:-https://github.com/sierra-research/tau2-bench.git}"

find_python() {
  local candidate
  for candidate in "${PYTHON_BIN:-}" "$(command -v python3.12 || true)" "$ROOT/../.venv-metis/bin/python" "$(command -v python3 || true)"; do
    [ -n "$candidate" ] || continue
    if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

PYTHON="$(find_python)" || {
  echo "需要 Python >= 3.12；可以设置 PYTHON_BIN=/path/to/python3.12 后重试。" >&2
  exit 1
}

if [ ! -d "$UPSTREAM/.git" ]; then
  git clone --depth 1 --branch "$TAG" "$REPO" "$UPSTREAM"
fi

for PATCH in \
  "$ROOT/patches/tau2-bench-v1.0.1-dsv4.patch" \
  "$ROOT/patches/tau2-bench-v1.0.1-empty-user-retry.patch"; do
  if git -C "$UPSTREAM" apply --check "$PATCH" 2>/dev/null; then
    git -C "$UPSTREAM" apply "$PATCH"
  elif ! git -C "$UPSTREAM" apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "补丁 $PATCH 无法应用到 $UPSTREAM；请确认 TAU2_TAG=$TAG。" >&2
    exit 1
  fi
done

if [ ! -x "$VENV/bin/python" ]; then
  "$PYTHON" -m venv "$VENV"
fi

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/.cache/pip}"
(
  cd "$UPSTREAM"
  "$VENV/bin/pip" install -c "$ROOT/constraints.txt" -e ".[knowledge]" tiktoken
)

if [ ! -f "$UPSTREAM/.env" ]; then
  cp "$ROOT/env.example" "$UPSTREAM/.env"
  chmod 600 "$UPSTREAM/.env"
fi

(
  cd "$UPSTREAM"
  "$VENV/bin/tau2" check-data
)

echo
echo "环境已就绪：$VENV"
echo "下一步：编辑 tau2-bench/.env，填入模型和 embedding API 配置。"
echo "A 组：./run_banking_official.sh"
echo "B/C 组：按 docs/reproduce_memos.md 安装并启动本地 MemOS。"
