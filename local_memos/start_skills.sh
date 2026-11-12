#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"$HERE/.venv/bin/python" - "$HERE" <<'PY'
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

here = Path(sys.argv[1])
runtime = here / "runtime_skills"
runtime.mkdir(parents=True, exist_ok=True)
address = "http://127.0.0.1:8001/health"

def healthy():
    try:
        with urlopen(address, timeout=2) as response:
            data = json.load(response)
        features = data.get("eval_features", {})
        return (features.get("extraction") == "native_skill_only"
                and features.get("same_skill_update") == "coalesce_native_evidence")
    except (URLError, OSError):
        return False

if healthy():
    print("C 原生 Skill 服务已就绪：http://127.0.0.1:8001")
    raise SystemExit(0)
try:
    with socket.create_connection(("127.0.0.1", 7687), timeout=2):
        pass
except OSError:
    raise SystemExit("共享 Neo4j 尚未启动，请先运行 ./local_memos/start.sh")
try:
    with socket.create_connection(("127.0.0.1", 8001), timeout=1):
        raise SystemExit("8001 已被其他服务占用，请检查后再启动 C 服务")
except OSError:
    pass
with (runtime / "server.log").open("ab") as log:
    process = subprocess.Popen(
        [str(here / ".venv/bin/python"), str(here / "serve_skills.py"), "--port", "8001"],
        cwd=here.parent, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
(runtime / "server.pid").write_text(str(process.pid))
deadline = time.monotonic() + 45
while time.monotonic() < deadline:
    if healthy():
        print("C 原生 Skill 服务已就绪：http://127.0.0.1:8001")
        break
    if process.poll() is not None:
        raise SystemExit("C 服务启动失败，请查看 local_memos/runtime_skills/server.log")
    time.sleep(0.5)
else:
    raise SystemExit("C 服务尚未就绪，请查看 local_memos/runtime_skills/server.log")
PY
