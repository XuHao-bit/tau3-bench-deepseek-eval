#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$HERE/.venv/bin/python" - "$HERE" <<'PY'
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

here = Path(sys.argv[1])
pid_file = here / "runtime_skills/server.pid"
if not pid_file.exists():
    print("没有 C 服务 PID 记录")
    raise SystemExit(0)
pid = int(pid_file.read_text().strip())
command = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True).stdout
if not command.strip():
    pid_file.unlink()
    print("C 服务已停止")
    raise SystemExit(0)
if str(here / "serve_skills.py") not in command:
    raise SystemExit("PID 已不属于 C 服务，保留进程，请检查 PID 记录")
os.kill(pid, signal.SIGTERM)
deadline = time.monotonic() + 45
while time.monotonic() < deadline:
    process = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True).stdout.strip()
    if not process or process.startswith("Z"):
        pid_file.unlink()
        print("C 服务已停止；共享 Neo4j 和 B 服务仍保留")
        break
    time.sleep(0.5)
else:
    raise SystemExit("C 服务仍在退出，请等待同步学习完成；未强制终止")
PY
