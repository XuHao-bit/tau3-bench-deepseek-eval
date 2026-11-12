#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for service in server neo4j; do
  pid_file="$HERE/runtime/$service.pid"
  if [ -f "$pid_file" ]; then
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid"
      echo "正在停止 ${service}，PID=${pid}；等待已提交的处理完成"
      while kill -0 "$pid" 2>/dev/null; do
        sleep 0.5
      done
    fi
    rm -f "$pid_file"
    echo "已停止 $service"
  fi
done
