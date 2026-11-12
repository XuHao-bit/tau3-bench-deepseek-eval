"""运行 B 类自动记忆或 C 类自主 Skill 工具评测。"""

import argparse
import json
import os
import random
import sys
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import ContextVar
from datetime import datetime
from math import comb
from pathlib import Path
from queue import Empty, Queue
from zoneinfo import ZoneInfo

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
ATTEMPT_CONTEXT = ContextVar("memos_attempt")


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", nargs="+", choices=["B-cross", "B-intra", "C"],
                   default=["B-cross", "B-intra"])
    p.add_argument("--domain", default="banking_knowledge")
    p.add_argument("--domains", nargs="+", help="C 可按顺序运行多个领域，共用同一个 Skill cube")
    p.add_argument("--task-ids", nargs="+")
    p.add_argument("--seeds", "--seed", nargs="+", type=int, default=[300])
    p.add_argument("--output", type=Path)
    p.add_argument("--num-tasks", type=int, help="本次运行每组、每个 seed 的前 N 题；省略则运行全部")
    p.add_argument("--resume", action="store_true", help="保留已有结果和 cube，继续同一实验")
    p.add_argument("--memos-url", default=os.getenv("MEMOS_URL", "http://127.0.0.1:8000"))
    p.add_argument("--memos-user", default=os.getenv("MEMOS_USER_ID", "tau3-eval"))
    p.add_argument("--agent-llm", default="openai/deepseek-v4.1-flash-1017")
    p.add_argument("--user-llm", default=os.getenv("USER_LLM", "openai/gpt-5.2"))
    p.add_argument("--retrieval-config", default="alltools")
    p.add_argument("--max-concurrency", type=int, default=8)
    p.add_argument("--max-retries", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--max-memory-calls", type=int, default=100)
    return p


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def resume_configuration(args, manifest: dict) -> None:
    """续跑沿用实验配置，显式传入的不同配置直接拒绝。"""
    config = manifest["configuration"]
    if config.get("memory_add") != {
        "async_mode": "async", "messages": "current_user_and_final_assistant"
    }:
        raise ValueError("该目录不是异步 QA 版本，不能混用旧版 sync／轨迹写入结果。")
    supplied = {arg.split("=", 1)[0][2:].replace("-", "_")
                for arg in sys.argv[1:] if arg.startswith("--")}
    if "seed" in supplied:
        supplied.add("seeds")
    saved = {key: manifest[key] for key in ("arms", "seeds", "domain", "memos_url", "memos_user")}
    saved["task_ids"] = manifest["task_order"]
    saved.update({key: config[key] for key in (
        "agent_llm", "user_llm", "retrieval_config", "max_concurrency", "max_retries",
        "max_steps", "top_k", "max_memory_calls")})
    for key, value in saved.items():
        if key in supplied and getattr(args, key) != value:
            raise ValueError(f"续跑配置 {key} 与原实验不一致，请使用原配置或新输出目录。")
        setattr(args, key, value)


def restore_progress(manifest: dict, output: Path, simulation_type) -> dict:
    """合并已提交的逐题检查点，拒绝继续状态不确定的尝试。"""
    rows = {(r["arm"], r["seed"], r["task_id"]): r for r in manifest["tasks"]}
    restored = {}
    for seed in manifest["seeds"]:
        for arm in manifest["arms"]:
            arm_dir = output / f"{arm}_seed{seed}"
            checkpoints = {}
            for path in arm_dir.glob("*/completed.json"):
                checkpoint = json.loads(path.read_text())
                row = checkpoint["task"]
                if (row["arm"], row["seed"]) != (arm, seed):
                    raise ValueError(f"检查点与实验组不一致：{path}")
                key = (arm, seed, row["task_id"])
                rows[key] = row
                checkpoints[row["task_id"]] = simulation_type.model_validate(checkpoint["simulation"])
            simulations = {}
            result_path = arm_dir / "results.json"
            if result_path.exists():
                try:
                    data = json.loads(result_path.read_text())
                except json.JSONDecodeError:
                    # results 汇总可能写到一半；完整逐题检查点可重新生成它。
                    required = {key[2] for key in rows if key[:2] == (arm, seed)}
                    if not required.issubset(checkpoints):
                        raise ValueError(f"结果文件未写完整且缺少逐题检查点：{result_path}") from None
                else:
                    simulations = {s["task_id"]: simulation_type.model_validate(s)
                                   for s in data["simulations"]}
            simulations.update(checkpoints)
            expected = {key[2] for key in rows if key[:2] == (arm, seed)}
            if set(simulations) != expected:
                raise ValueError(f"结果与任务记录不一致，不能安全续跑：{arm_dir}")
            restored[(arm, seed)] = simulations
            for task_dir in arm_dir.glob("*/"):
                if task_dir.name not in expected and any(task_dir.glob("attempt_*")):
                    raise ValueError(f"存在未确认写入状态的尝试，不能自动续跑：{task_dir}")
    active = [r for r in manifest.get("active_tasks", [])
              if (r["arm"], r["seed"], r["task_id"]) not in rows]
    if active or manifest.get("failed_attempts"):
        raise ValueError("存在未确认终止的尝试或后台写入，不能自动 resume；现有结果和记忆未改动。")
    manifest["tasks"] = sorted(rows.values(), key=lambda r: (
        manifest["seeds"].index(r["seed"]), manifest["arms"].index(r["arm"]), r["position"]))
    manifest["active_tasks"] = []
    return restored


def rollback_attempt(client, cube_id: str, task_id: str, agent) -> None:
    """写入任务终止后清理失败 session，保留同域其他题的记忆。"""
    if agent is not None and not agent.settle_writes(timeout=300):
        raise RuntimeError(f"{task_id} 的异步写入状态未确认，停止运行，保留现有记忆。")
    client.delete_session(cube_id, task_id)


def reported_usage(state, role: str) -> dict:
    """汇总保留下来的调用记录，包含失败尝试中已产生的开销。"""
    totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
              "reasoning_tokens": 0, "reported_cost_usd": 0.0}

    def add_usage(usage):
        if not usage:
            return
        totals["calls"] += 1
        totals["input_tokens"] += usage.get("prompt_tokens", 0) or 0
        totals["output_tokens"] += usage.get("completion_tokens", 0) or 0
        totals["reasoning_tokens"] += (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0

    for message in getattr(state, "messages", []):
        if message.role != role:
            continue
        raw = message.raw_data or {}
        add_usage(raw.get("usage"))
        for retry in raw.get("empty_response_retries", []):
            add_usage((retry.get("raw_data") or {}).get("usage") or retry.get("usage"))
        totals["reported_cost_usd"] += message.cost or 0
    if role == "user":
        # 重试耗尽时没有有效消息，开销仍保留在状态里。
        for retry in getattr(state, "pending_empty_responses", []):
            add_usage((retry.get("raw_data") or {}).get("usage") or retry.get("usage"))
            totals["reported_cost_usd"] += retry.get("cost") or 0
    return totals


def task_passed(row: dict) -> bool:
    """与 tau2.metrics.agent_metrics.is_successful 相同的 reward 判定容差。"""
    reward = row["reward"]
    return reward is not None and abs(reward - 1.0) <= 1e-6


def pass_hat_k(trials: int, successes: int, k: int) -> float:
    """pass^k 定义与 tau2.metrics.agent_metrics.pass_hat_k 一致。"""
    if trials < k:
        raise ValueError(f"Number of trials {trials} is less than k {k}.")
    return comb(successes, k) / comb(trials, k)


def pass_metrics(rows: list[dict]) -> dict:
    """按题把各 seed／trial 聚合成 pass^k。

    k 上限取每题实际 trial 数的最小值；单 seed 时只有 pass^1。
    口径与 tau2 官方指标一致：infrastructure_error 等未评分题按失败计入 pass^1；
    pass_at_1_valid_tasks 则只统计已评分题目，用于对照 average_reward。
    """
    by_task = {}
    for row in rows:
        by_task.setdefault(row["task_id"], []).append(row)
    max_k = min((len(task_rows) for task_rows in by_task.values()), default=0)
    pass_hat_ks = {
        k: sum(pass_hat_k(len(task_rows), sum(map(task_passed, task_rows)), k)
               for task_rows in by_task.values()) / len(by_task)
        for k in range(1, max_k + 1)
    }
    scored = [row for row in rows if row["reward"] is not None]
    passed = sum(map(task_passed, scored))
    return {
        "passed": passed,
        "num_trials_per_task": max_k,
        "pass_hat_ks": pass_hat_ks,
        "pass_at_1_all_tasks": passed / len(rows) if rows else None,
        "pass_at_1_valid_tasks": passed / len(scored) if scored else None,
    }


def summarize_run(manifest: dict) -> dict:
    """区分记忆提交链路、后台完成状态与任务成功率。"""
    arms = {}
    for arm in manifest["arms"]:
        rows = [row for row in manifest["tasks"] if row["arm"] == arm]
        fatal_attempts = [row for row in manifest.get("failed_attempts", []) if row["arm"] == arm]
        scored = [row for row in rows if row["reward"] is not None]
        checks = []
        cross_hits = 0
        status_counts = {status: 0 for status in ("submitted", "completed", "unknown", "failed")}
        for row in rows:
            memory = row["attempts"][-1].get("memory", {})
            sources = memory.get("source_tasks", {})
            states = memory.get("write_status_counts", {})
            for status in status_counts:
                status_counts[status] += states.get(status, 0)
            cross_hits += sum(count for task, count in sources.items()
                              if task not in {row["task_id"], "unknown"})
            checks.append({
                "task_id": row["task_id"],
                "scored": row["reward"] is not None,
                "write_submitted": memory.get("written_turns", 0) > 0,
                "no_memory_errors": not any(memory.get(key, 0) for key in
                    ["search_errors", "failed_write_turns"])
                    and not any(states.get(key, 0) for key in ("unknown", "failed")),
                "scope_correct": arm != "B-intra" or all(
                    task == row["task_id"] for task in sources),
                "read_tokens": memory.get("read_tokens", 0),
                "write_status_counts": states,
                "source_tasks": sources,
            })
        arms[arm] = {
            "tasks": len(rows), "scored": len(scored),
            "failed_attempts": len(fatal_attempts),
            **pass_metrics(rows),
            "average_reward": sum(row["reward"] for row in scored) / len(scored) if scored else None,
            "cross_task_hits": cross_hits, "memory_checks": checks,
            "write_status_counts": status_counts,
            "memory_pipeline_passed": bool(checks) and not fatal_attempts and all(
                all(check[key] for key in ["scored", "write_submitted", "no_memory_errors", "scope_correct"])
                for check in checks),
            "all_memory_writes_completed": bool(checks) and not fatal_attempts
                and status_counts["completed"] > 0
                and not any(status_counts[key] for key in ("submitted", "unknown", "failed")),
        }
    return {"status": manifest["status"], "arms": arms}


def main() -> None:
    load_dotenv(ROOT / "tau2-bench/.env")
    args = parser().parse_args()
    if args.max_retries < 0 or args.max_concurrency < 1 or (args.num_tasks is not None and args.num_tasks < 1):
        raise ValueError("max-retries 不能为负数，max-concurrency 和 num-tasks 必须为正数")
    if "C" in args.arms or (args.resume and args.output is not None and
            json.loads((args.output / "manifest.json").read_text()).get("arms") == ["C"]):
        args.arms = ["C"] if args.resume and "--arms" not in sys.argv else args.arms
        from memos_eval.run_skills import run
        run(args)
        return
    if args.resume and args.output is None:
        raise ValueError("--resume 必须指定原实验的 --output 目录。")
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    # 相对输出路径在切换 benchmark 工作目录之前解析，与查看脚本一致。
    output = (args.output or ROOT / "runs" / f"memos_b_{stamp}").resolve()
    saved_manifest = None
    if args.resume:
        saved_manifest = json.loads((output / "manifest.json").read_text())
        resume_configuration(args, saved_manifest)
    else:
        output.mkdir(parents=True, exist_ok=False)
    os.environ["PATH"] = str(ROOT / ".tools/bin") + os.pathsep + os.environ.get("PATH", "")
    os.chdir(ROOT / "tau2-bench")

    from loguru import logger
    from tau2.data_model.simulation import Results, SimulationRun, TextRunConfig
    from tau2.orchestrator.orchestrator import Role
    from tau2.registry import registry
    from tau2.runner.batch import _TaskLogContext
    from tau2.runner.build import _build_env_kwargs, build_orchestrator
    from tau2.runner.helpers import get_info, get_tasks
    from tau2.runner.simulation import run_simulation
    from tau2.utils.llm_utils import set_llm_log_mode
    from tau2.utils.utils import get_now

    from memos_eval.agent import MemOSAgent
    from memos_eval.client import MemOSClient

    manifest = saved_manifest if saved_manifest is not None else {
        "run_id": f"{output.name}-{uuid.uuid4().hex[:8]}", "status": "starting", "domain": args.domain,
        "started_at": now(), "arms": args.arms, "seeds": args.seeds,
        "memos_url": args.memos_url, "memos_user": args.memos_user,
        "cubes": [], "tasks": [], "active_tasks": [], "failed_attempts": [],
        "configuration": {
            "agent_llm": args.agent_llm, "agent_effort": "max",
            "user_llm": args.user_llm, "user_effort": "low",
            "max_concurrency": args.max_concurrency, "num_trials_per_pass": 1,
            "max_retries": args.max_retries, "max_steps": args.max_steps,
            "retrieval_config": args.retrieval_config, "top_k": args.top_k,
            "max_memory_calls": args.max_memory_calls,
            "memory_add": {"async_mode": "async", "messages": "current_user_and_final_assistant"},
            "memory_search": {"include_preference": True, "include_skill_memory": False,
                              "search_tool_memory": False, "query": "current_user"},
            "cross_visibility": "already_indexed_at_search_time",
        },
    }
    restored = restore_progress(manifest, output, SimulationRun) if args.resume else {}
    # 完整日志写文件，终端只接收下面的简短进度，避免大量工具结果堵塞终端。
    logger.remove()
    run_handler = logger.add(output / "run.log", level="DEBUG", enqueue=True)
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    client = MemOSClient(args.memos_url, args.memos_user, os.getenv("MEMOS_API_KEY"))
    events = Queue()

    def factory(tools, domain_policy, llm, llm_args, task, **kwargs):
        context = ATTEMPT_CONTEXT.get()
        agent = MemOSAgent(
            tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args,
            client=context["client"], cube_id=context["cube_id"], task_id=task.id,
            arm=context["arm"], audit_path=context["audit_path"],
            top_k=args.top_k, max_memory_calls=args.max_memory_calls,
        )
        context["agent"] = agent
        return agent

    registry.register_agent_factory(factory, "memos_agent")
    user_args = {"extra_body": {"reasoning_effort": "low"}, "empty_response_retries": 2}
    for env, key in [("USER_API_BASE", "api_base"), ("USER_API_KEY", "api_key")]:
        if os.getenv(env):
            user_args[key] = os.environ[env]

    def create_cube(cube_client, arm, seed, task_id=None, attempt=None):
        suffix = f"-{task_id}-a{attempt}" if task_id else ""
        cube = cube_client.create_cube(f"tau3-{args.domain}-{arm}-{manifest['run_id']}-{seed}{suffix}")
        events.put(("cube", {"arm": arm, "seed": seed, "task_id": task_id,
                             "attempt": attempt, "cube_id": cube, "created_at": now()}))
        return cube

    def drain_events():
        changed = False
        while True:
            try:
                event, record = events.get_nowait()
            except Empty:
                break
            changed = True
            if event == "cube":
                manifest["cubes"].append(record)
            elif event == "attempt":
                manifest["active_tasks"] = [row for row in manifest["active_tasks"]
                    if (row["arm"], row["seed"], row["task_id"]) !=
                       (record["arm"], record["seed"], record["task_id"])]
                manifest["active_tasks"].append(record)
                print(f"{record['arm']} {record['task_id']} attempt={record['attempt']} 开始", flush=True)
            elif event == "fatal_attempt":
                manifest["failed_attempts"].append(record)
                manifest["active_tasks"] = [row for row in manifest["active_tasks"]
                    if (row["arm"], row["seed"], row["task_id"]) !=
                       (record["arm"], record["seed"], record["task_id"])]
        if changed:
            write_json(manifest_path, manifest)

    def run_task(config, arm, seed, trial_seed, position, task, arm_dir, cross_cube):
        # ContextVar 不会自动继承到线程池，逐 worker 设置完整 LLM 调用日志。
        set_llm_log_mode("all")
        started_at = now()
        attempts = []
        simulation = None
        for attempt in range(args.max_retries + 1):
            attempt_client = MemOSClient(args.memos_url, args.memos_user, os.getenv("MEMOS_API_KEY"))
            token = None
            try:
                cube = cross_cube or create_cube(attempt_client, arm, seed, task.id, attempt)
                attempt_dir = arm_dir / task.id / f"attempt_{attempt}"
                attempt_dir.mkdir(parents=True)
                context = {"client": attempt_client, "cube_id": cube, "arm": arm,
                           "audit_path": attempt_dir / "memory_audit.jsonl", "agent": None}
                token = ATTEMPT_CONTEXT.set(context)
                orch = None
                request_start = attempt_client.request_count
                attempt_started = now()
                events.put(("attempt", {"arm": arm, "seed": seed, "task_id": task.id,
                    "attempt": attempt, "cube_id": cube, "started_at": attempt_started}))
                try:
                    with _TaskLogContext(str(uuid.uuid4()), attempt_dir, task, True):
                        orch = build_orchestrator(config, task, seed=trial_seed)
                        simulation = run_simulation(
                            orch, env_kwargs=_build_env_kwargs(config, task) or None)
                        tail = orch.message if orch.to_role == Role.AGENT else None
                        context["agent"].finish(simulation.termination_reason, tail)
                    simulation.trial = 0
                    attempts.append({"attempt": attempt, "status": "scored",
                        "started_at": attempt_started, "finished_at": now(),
                        "reward": simulation.reward_info.reward, "cube_id": cube,
                        "memory": context["agent"].summary(),
                        "memos_http_requests": attempt_client.request_count - request_start,
                        "agent_usage": reported_usage(orch.agent_state, "assistant"),
                        "user_usage": reported_usage(orch.user_state, "user")})
                    break
                except Exception as exc:
                    if context["agent"] is not None:
                        context["agent"].finish("infrastructure_error")
                    write_json(attempt_dir / "error.json", {
                        "type": type(exc).__name__, "message": str(exc),
                        "traceback": traceback.format_exc(),
                        "pending_empty_responses": getattr(
                            getattr(orch, "user_state", None), "pending_empty_responses", [])})
                    # 只清理本任务；若后台写入仍可能到达，停止运行而非重试。
                    try:
                        rollback_attempt(attempt_client, cube, task.id, context["agent"])
                    except Exception as cleanup_error:
                        events.put(("fatal_attempt", {
                            "arm": arm, "seed": seed, "task_id": task.id,
                            "attempt": attempt, "cube_id": cube,
                            "started_at": attempt_started, "finished_at": now(),
                            "status": "cleanup_unconfirmed", "memory_rolled_back": False,
                            "error_type": type(exc).__name__, "error": str(exc),
                            "cleanup_error": str(cleanup_error),
                            "memory": context["agent"].summary() if context["agent"] is not None else {},
                            "memos_http_requests": attempt_client.request_count - request_start,
                            "agent_usage": reported_usage(getattr(orch, "agent_state", None), "assistant"),
                            "user_usage": reported_usage(getattr(orch, "user_state", None), "user"),
                        }))
                        raise
                    attempts.append({"attempt": attempt, "status": "infrastructure_error",
                        "started_at": attempt_started, "finished_at": now(),
                        "error_type": type(exc).__name__, "cube_id": cube,
                        "memory_rolled_back": True,
                        "memory": context["agent"].summary() if context["agent"] is not None else {},
                        "memos_http_requests": attempt_client.request_count - request_start,
                        "agent_usage": reported_usage(getattr(orch, "agent_state", None), "assistant"),
                        "user_usage": reported_usage(getattr(orch, "user_state", None), "user")})
                    logger.warning(f"{task.id} attempt={attempt} 异常，已清理本题记忆：{type(exc).__name__}")
                    simulation = None
            finally:
                if token is not None:
                    ATTEMPT_CONTEXT.reset(token)
                attempt_client.close()
        if simulation is None:
            simulation = SimulationRun(
                id=str(uuid.uuid4()), task_id=task.id, trial=0, seed=trial_seed,
                start_time=get_now(), end_time=get_now(), duration=0,
                termination_reason="infrastructure_error", messages=[], info={"attempts": attempts})
        return simulation, {
            "arm": arm, "seed": seed, "trial_seed": trial_seed,
            "position": position, "task_id": task.id, "attempts": attempts,
            "started_at": started_at, "finished_at": now(),
            "reward": simulation.reward_info.reward if simulation.reward_info else None,
            "termination_reason": str(simulation.termination_reason),
        }

    try:
        manifest["memos_features"] = client.ensure_user()
        tasks = get_tasks(args.domain, "base", args.task_ids)
        if args.task_ids:
            by_id = {task.id: task for task in tasks}
            tasks = [by_id[tid] for tid in args.task_ids]
        if args.resume and [task.id for task in tasks] != manifest["task_order"]:
            raise ValueError("benchmark 任务集合或题序与原实验不一致，不能续跑。")
        manifest["task_order"] = [task.id for task in tasks]
        manifest["last_invocation"] = {"started_at": now(), "num_tasks": args.num_tasks,
                                       "completed_before": len(manifest["tasks"])}
        manifest.setdefault("invocations", []).append(manifest["last_invocation"])
        manifest["status"] = "running"
        manifest.pop("error", None)
        manifest.pop("finished_at", None)
        write_json(manifest_path, manifest)
        for seed in args.seeds:
            trial_seed = random.Random(seed).randint(0, 1000000)
            for arm in args.arms:
                arm_dir = output / f"{arm}_seed{seed}"
                arm_existed = arm_dir.exists()
                arm_dir.mkdir(exist_ok=args.resume)
                config = TextRunConfig(
                    domain=args.domain, agent="memos_agent", llm_agent=args.agent_llm,
                    llm_args_agent={"temperature": 0.0,
                                    "extra_body": {"reasoning_effort": "max"}},
                    llm_user=args.user_llm, llm_args_user=user_args,
                    retrieval_config=args.retrieval_config, num_trials=1,
                    max_concurrency=args.max_concurrency, max_steps=args.max_steps,
                    max_errors=10, max_retries=args.max_retries, seed=seed,
                )
                info = get_info(config)
                info.user_info.llm_args = {k: v for k, v in user_args.items() if k != "api_key"}
                results = Results(info=info, tasks=tasks, simulations=[])
                previous = restored.get((arm, seed), {})
                ordered = {position: previous[task.id] for position, task in enumerate(tasks)
                           if task.id in previous}
                if args.resume and ordered:
                    results.simulations = [ordered[index] for index in sorted(ordered)]
                    results.save(arm_dir / "results.json")
                cross_cube = None
                if arm == "B-cross":
                    cubes = [cube["cube_id"] for cube in manifest["cubes"]
                             if cube["arm"] == arm and cube["seed"] == seed and cube["task_id"] is None]
                    if len(cubes) > 1 or (args.resume and arm_existed and not cubes):
                        raise ValueError("原实验的 B-cross cube 未唯一确认，不能重新创建或替换。")
                    cross_cube = cubes[0] if cubes else create_cube(client, arm, seed)
                drain_events()
                selected = list(enumerate(tasks))[:args.num_tasks] if args.num_tasks is not None else list(enumerate(tasks))
                remaining = iter((position, task) for position, task in selected if position not in ordered)
                failure = None
                with ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
                    pending = {}

                    def submit_next():
                        item = next(remaining, None)
                        if item is not None:
                            position, task = item
                            pending[pool.submit(run_task, config, arm, seed, trial_seed,
                                position, task, arm_dir, cross_cube)] = position

                    for _ in range(args.max_concurrency):
                        submit_next()
                    while pending:
                        done, _ = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                        drain_events()
                        completed = []
                        for future in done:
                            position = pending.pop(future)
                            try:
                                completed.append((position, *future.result()))
                            except Exception as exc:
                                failure = failure or exc
                        for position, simulation, row in sorted(completed, key=lambda item: item[2]["finished_at"]):
                            row["completion_order"] = len(manifest["tasks"]) + 1
                            # 逐题提交先落盘；续跑可修复随后 results／manifest 的保存窗口。
                            write_json(arm_dir / row["task_id"] / "completed.json", {
                                "task": row, "simulation": simulation.model_dump(mode="json")})
                            ordered[position] = simulation
                            results.simulations = [ordered[index] for index in sorted(ordered)]
                            results.save(arm_dir / "results.json")
                            manifest["tasks"].append(row)
                            manifest["tasks"].sort(key=lambda task: (
                                args.seeds.index(task["seed"]), args.arms.index(task["arm"]), task["position"]))
                            manifest["active_tasks"] = [active for active in manifest["active_tasks"]
                                if (active["arm"], active["seed"], active["task_id"]) !=
                                   (row["arm"], row["seed"], row["task_id"])]
                            write_json(manifest_path, manifest)
                            print(f"{arm} {len(ordered)}/{len(tasks)} {row['task_id']} reward={row['reward']}", flush=True)
                        if failure is None:
                            for _ in done:
                                submit_next()
                    drain_events()
                if failure is not None:
                    raise failure
        total = len(tasks) * len(args.arms) * len(args.seeds)
        manifest["status"] = "completed" if len(manifest["tasks"]) == total else "partial"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        client.close()
        manifest["finished_at"] = now()
        if "last_invocation" in manifest:
            manifest["last_invocation"]["finished_at"] = manifest["finished_at"]
        write_json(manifest_path, manifest)
        summary = summarize_run(manifest)
        write_json(output / "summary.json", summary)
        for arm, stats in summary["arms"].items():
            rates = " ".join(f"pass^{k}={value:.4f}"
                             for k, value in stats["pass_hat_ks"].items())
            print(f"{arm}: {stats['passed']}/{stats['scored']} 已评分题通过"
                  + (f"，{rates}" if rates else "，无可用 trial"), flush=True)
        logger.remove(run_handler)
    print(f"结果：{output}", flush=True)


if __name__ == "__main__":
    main()
