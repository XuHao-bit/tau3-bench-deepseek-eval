"""查看实验 cube 中实际保存的 MemOS 记忆，可导出 JSON。"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from memos_eval.client import MemOSClient


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / "tau2-bench/.env")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, help="包含 manifest.json 的实验目录")
    p.add_argument("--cube-id", action="append", help="也可直接指定一个或多个 cube")
    p.add_argument("--arm", choices=["B-cross", "B-intra"])
    p.add_argument("--task-id", help="只查看指定题目的 session")
    p.add_argument("--memos-url", default=os.getenv("MEMOS_URL", "http://127.0.0.1:8000"))
    p.add_argument("--memos-user", default=os.getenv("MEMOS_USER_ID", "tau3-eval"))
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--output", type=Path, help="导出实际保存的记忆 JSON")
    args = p.parse_args()
    if not args.run_dir and not args.cube_id:
        p.error("至少指定 --run-dir 或 --cube-id")
    if args.arm and not args.run_dir:
        p.error("--arm 需要同时指定 --run-dir；直接查询 cube 时请省略 --arm")
    cubes = args.cube_id or []
    if args.run_dir:
        manifest = json.loads((args.run_dir / "manifest.json").read_text())
        args.memos_url = manifest["memos_url"]
        args.memos_user = manifest["memos_user"]
        cubes += [c["cube_id"] for c in manifest["cubes"]
                  if (not args.arm or c["arm"] == args.arm)
                  and (not args.task_id or c["task_id"] in [None, args.task_id])]
    client = MemOSClient(args.memos_url, args.memos_user, os.getenv("MEMOS_API_KEY"))
    rows = []
    for cube in dict.fromkeys(cubes):
        for hit in client.list_memories(cube, args.task_id, args.limit, args.offset):
            rows.append({"cube_id": cube, "memory_id": hit.memory_id,
                         "task_id": hit.task_id, "created_at": hit.created_at,
                         "content": hit.content, "metadata": hit.metadata})
    text = json.dumps(rows, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
