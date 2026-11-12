#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PATH="$ROOT/.tools/bin:$PATH"
RUN=("$ROOT/.venv/bin/python" -m memos_eval.run "$@")
if command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -dimsu "${RUN[@]}"
else
  exec "${RUN[@]}"
fi
