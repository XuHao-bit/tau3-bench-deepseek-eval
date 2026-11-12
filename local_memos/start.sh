#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME="$HERE/runtime"
mkdir -p "$RUNTIME"
shopt -s nullglob
java_dirs=("$RUNTIME"/jdk-21*-jre)
if [ "${#java_dirs[@]}" -eq 0 ] || [ ! -f "$RUNTIME/neo4j/lib/neo4j-5.26.6.jar" ]; then
  echo "请先运行 local_memos/setup.sh"
  exit 1
fi
if [ ! -f "$RUNTIME/neo4j.pid" ] || ! kill -0 "$(cat "$RUNTIME/neo4j.pid")" 2>/dev/null; then
  nohup "${java_dirs[0]}/Contents/Home/bin/java" -Xms256m -Xmx512m \
    --add-opens=java.base/java.nio=ALL-UNNAMED \
    --add-opens=java.base/java.lang=ALL-UNNAMED \
    -cp "$RUNTIME/neo4j/lib/*" org.neo4j.server.CommunityEntryPoint \
    --home-dir="$RUNTIME/neo4j" --config-dir="$HERE" \
    > "$RUNTIME/neo4j-console.log" 2>&1 &
  echo "$!" > "$RUNTIME/neo4j.pid"
fi
"$HERE/.venv/bin/python" - <<'PY'
import socket
import time

deadline = time.monotonic() + 45
while time.monotonic() < deadline:
    try:
        with socket.create_connection(("127.0.0.1", 7687), timeout=1):
            break
    except OSError:
        time.sleep(0.5)
else:
    raise SystemExit("Neo4j 未就绪，请查看 local_memos/runtime/neo4j/logs")
PY
if [ ! -f "$RUNTIME/server.pid" ] || ! kill -0 "$(cat "$RUNTIME/server.pid")" 2>/dev/null; then
  nohup "$HERE/.venv/bin/python" "$HERE/serve.py" \
    > "$RUNTIME/server.log" 2>&1 &
  echo "$!" > "$RUNTIME/server.pid"
fi
"$HERE/.venv/bin/python" - <<'PY'
import time
from urllib.error import URLError
from urllib.request import urlopen

deadline = time.monotonic() + 45
while time.monotonic() < deadline:
    try:
        with urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
            if response.status == 200:
                print("MemOS 已就绪：http://127.0.0.1:8000")
                break
    except (URLError, OSError):
        time.sleep(0.5)
else:
    raise SystemExit("MemOS 未就绪，请查看 local_memos/runtime/server.log")
PY
