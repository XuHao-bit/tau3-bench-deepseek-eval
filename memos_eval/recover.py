"""保留已有 memory，移出未评分记录，让原实验可以 --resume。

默认只预览。执行 --prepare 前，调用者必须确认评测进程已停止，
MemOS 已提交的后台写入也已结束；脚本不检查进程，也不调用 MemOS。
"""

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from memos_eval.run import summarize_run, write_json


@dataclass
class RecoveryPlan:
    output: Path
    manifest: dict
    results: dict[Path, dict]
    directories: list[Path]
    report: dict


def _key(row: dict) -> tuple:
    return row["arm"], row["seed"], row["task_id"]


def plan_recovery(output: Path) -> RecoveryPlan:
    """合并逐题检查点；只保留 reward 非空且有对应轨迹的记录。"""
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    rows = {_key(row): row for row in manifest["tasks"]}
    task_order = manifest["task_order"]
    expected = {(arm, seed, task) for seed in manifest["seeds"]
                for arm in manifest["arms"] for task in task_order}
    if not set(rows).issubset(expected):
        raise ValueError("任务记录包含原实验范围外的题目")
    simulations = {}
    result_files = {}
    directories = []
    recovered_checkpoints = []
    for seed in manifest["seeds"]:
        for arm in manifest["arms"]:
            arm_dir = output / f"{arm}_seed{seed}"
            result_path = arm_dir / "results.json"
            data = json.loads(result_path.read_text())
            current = {sim["task_id"]: sim for sim in data["simulations"]}
            for path in sorted(arm_dir.glob("*/completed.json")):
                checkpoint = json.loads(path.read_text())
                row = checkpoint["task"]
                key = _key(row)
                if key not in expected or key[:2] != (arm, seed) or path.parent.name != key[2]:
                    raise ValueError(f"检查点与实验任务不一致：{path}")
                if checkpoint["simulation"]["task_id"] != row["task_id"]:
                    raise ValueError(f"检查点轨迹与任务不一致：{path}")
                if key not in rows:
                    recovered_checkpoints.append(key)
                rows[key] = row
                current[row["task_id"]] = checkpoint["simulation"]
            for task_id, simulation in current.items():
                key = (arm, seed, task_id)
                if key not in rows:
                    raise ValueError(f"轨迹缺少任务记录或检查点：{arm}/{seed}/{task_id}")
                simulations[key] = simulation
            result_files[result_path] = data

    kept = {key: row for key, row in rows.items() if row.get("reward") is not None}
    for key, row in kept.items():
        if key not in simulations:
            raise ValueError(f"已评分任务缺少轨迹：{key}")
        reward = (simulations[key].get("reward_info") or {}).get("reward")
        if reward != row["reward"]:
            raise ValueError(f"任务记录与轨迹的 reward 不一致：{key}")
    for seed in manifest["seeds"]:
        for arm in manifest["arms"]:
            arm_dir = output / f"{arm}_seed{seed}"
            for task_dir in sorted(arm_dir.glob("task_*/")):
                key = (arm, seed, task_dir.name)
                if key not in expected:
                    raise ValueError(f"存在原实验范围外的任务目录：{task_dir}")
                if key not in kept:
                    directories.append(task_dir)
            data = result_files[arm_dir / "results.json"]
            data["simulations"] = [simulations[(arm, seed, task)] for task in task_order
                                   if (arm, seed, task) in kept]
            if data.get("simulation_index") is not None:
                ids = {sim["id"] for sim in data["simulations"]}
                data["simulation_index"] = [entry for entry in data["simulation_index"]
                                             if entry["id"] in ids]

    sort_key = lambda key: (manifest["seeds"].index(key[1]),
                            manifest["arms"].index(key[0]), task_order.index(key[2]))
    report = {
        "output": str(output),
        "kept_scored": len(kept),
        "excluded_unscored": [list(key) for key in sorted(set(rows) - set(kept), key=sort_key)],
        "requeued": [list(key) for key in sorted(expected - set(kept), key=sort_key)],
        "checkpoint_records_recovered": [list(key) for key in recovered_checkpoints],
        "archived_task_directories": [str(path.relative_to(output)) for path in directories],
        "previous_active_tasks": manifest.get("active_tasks", []),
        "previous_failed_attempts": manifest.get("failed_attempts", []),
        "reuse_existing_memory": True,
        "memory_mutations": False,
    }
    manifest["tasks"] = [kept[key] for key in sorted(kept, key=sort_key)]
    manifest["active_tasks"] = []
    manifest["failed_attempts"] = []
    manifest["status"] = "partial"
    manifest.pop("error", None)
    manifest.pop("finished_at", None)
    return RecoveryPlan(output, manifest, result_files, directories, report)


def prepare_recovery(output: Path, *, confirmed_stopped: bool = False) -> dict:
    """备份旧记录并准备续跑；不会删除、清空或重建任何 memory cube。"""
    if not confirmed_stopped:
        raise ValueError("须先确认评测进程已停止，且 MemOS 后台写入已结束")
    plan = plan_recovery(output)
    prepared_at = datetime.now(ZoneInfo("Asia/Shanghai"))
    backup = plan.output / f"recovery_{prepared_at:%Y%m%d_%H%M%S_%f}"
    report = {**plan.report, "prepared_at": prepared_at.isoformat(),
              "backup": str(backup), "quiescence_confirmed": True}
    plan.manifest.setdefault("recoveries", []).append({
        "prepared_at": report["prepared_at"], "backup": backup.name,
        "kept_scored": report["kept_scored"],
        "excluded_unscored": report["excluded_unscored"],
        "requeued": report["requeued"], "reuse_existing_memory": True,
    })
    summary = summarize_run(plan.manifest)
    backup.mkdir()
    for path in [plan.output / "manifest.json", plan.output / "summary.json", *plan.results]:
        if path.exists():
            destination = backup / path.relative_to(plan.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    write_json(backup / "recovery.json", report)
    for path in plan.directories:
        destination = backup / path.relative_to(plan.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), destination)
    for path, data in plan.results.items():
        write_json(path, data)
    write_json(plan.output / "manifest.json", plan.manifest)
    write_json(plan.output / "summary.json", summary)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="原实验目录")
    parser.add_argument("--prepare", action="store_true", help="备份并移出未评分记录；省略时只预览")
    parser.add_argument("--confirmed-stopped", action="store_true",
                        help="已确认评测进程停止且 MemOS 后台写入结束")
    args = parser.parse_args()
    report = (prepare_recovery(args.output, confirmed_stopped=args.confirmed_stopped)
              if args.prepare else plan_recovery(args.output).report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
