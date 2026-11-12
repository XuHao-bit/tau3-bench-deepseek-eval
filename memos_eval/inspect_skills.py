"""查看 C 组共享 cube 的原生 Skill 和显式事实记忆，可按 ID 查询或导出。"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from memos_eval.skill_client import SkillClient


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / "tau2-bench/.env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="包含 manifest.json 的 C 组实验目录")
    parser.add_argument("--cube-id", action="append", help="直接指定一个或多个共享 cube")
    parser.add_argument("--seed", type=int, help="只查看指定 seed 的共享库")
    parser.add_argument("--memory-id", action="append", help="只查看指定记忆 ID，可重复")
    parser.add_argument("--memory-type", choices=["skill", "fact", "all"], default="skill",
                        help="默认查看 Skill；fact 只查看 DS 显式写入的事实")
    parser.add_argument("--session-id", help="事实只查看指定 session 的私有项和共享项")
    parser.add_argument("--memos-url", default=os.getenv("MEMOS_SKILL_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--memos-user", default=os.getenv("MEMOS_USER_ID", "tau3-eval"))
    parser.add_argument("--output", type=Path, help="导出当前记忆正文和元信息 JSON（省略向量）")
    args = parser.parse_args()
    if not args.run_dir and not args.cube_id:
        parser.error("至少指定 --run-dir 或 --cube-id")
    if args.seed is not None and not args.run_dir:
        parser.error("--seed 需要指定 --run-dir")
    cubes = list(args.cube_id or [])
    if args.run_dir:
        manifest = json.loads((args.run_dir / "manifest.json").read_text())
        args.memos_url = manifest["memos_url"]
        args.memos_user = manifest["memos_user"]
        cubes.extend(record["cube_id"] for record in manifest["cubes"]
                     if record["arm"] == "C"
                     and (args.seed is None or record["seed"] == args.seed))
    if not cubes:
        parser.error("没有匹配的 C 组共享 cube")

    client = SkillClient(args.memos_url, args.memos_user, os.getenv("MEMOS_API_KEY"))
    rows = []
    try:
        for cube in dict.fromkeys(cubes):
            skills = []
            facts = []
            if args.memory_type in {"skill", "all"}:
                skills = (client.get_skills(cube, args.memory_id) if args.memory_id
                          else client.list_skills(cube))
            if args.memory_type in {"fact", "all"}:
                facts = client.list_facts(cube, args.session_id)
                if args.memory_id:
                    facts = [node for node in facts if node["id"] in args.memory_id]
            for node in skills + facts:
                node.pop("embedding", None)
                (node.get("metadata") or {}).pop("embedding", None)
            row = {"cube_id": cube, "skill_count": len(skills), "skills": skills}
            if args.memory_type in {"fact", "all"}:
                row.update(fact_count=len(facts), facts=facts)
            rows.append(row)
    finally:
        client.close()
    text = json.dumps(rows, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
