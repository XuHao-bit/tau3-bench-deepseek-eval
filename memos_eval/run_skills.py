"""C 组共享 cube 的原生 Skill 在线评测。"""

import json
import os
import random
import sys
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextvars import ContextVar
from pathlib import Path

from memos_eval.run import ROOT, now, pass_metrics, reported_usage, write_json


MEMORY_TOOLS = {"memos_search", "memos_write_fact", "memos_learn"}
C_TASK_CONTEXT = ContextVar("memos_c_task")


def business_messages(messages: list) -> list:
    """评分只重放业务；完整的记忆调用另存于原始轨迹中。"""
    memory_ids = {
        call.id for message in messages if message.role == "assistant"
        for call in (getattr(message, "tool_calls", None) or [])
        if call.name in MEMORY_TOOLS
    }
    projected = []
    for message in messages:
        if message.role == "tool" and message.requestor == "assistant" and message.id in memory_ids:
            continue
        if message.role == "assistant" and message.tool_calls:
            calls = [call for call in message.tool_calls if call.name not in MEMORY_TOOLS]
            if len(calls) != len(message.tool_calls):
                if not calls and not message.content:
                    continue
                message = message.model_copy(update={"tool_calls": calls or None})
        projected.append(message)
    return projected


def install_memory_tools(environment, tools: list) -> None:
    """将记忆工具接入原生计步，用户工具和业务执行仍走原路由。"""
    from pydantic import ValidationError
    from tau2.data_model.message import ToolMessage

    by_name = {tool.name: tool for tool in tools}
    original = environment.get_response

    def get_response(message):
        if message.name not in by_name:
            return original(message)
        error = False
        if message.requestor != "assistant":
            content = {"status": "rejected", "error": "记忆工具仅供 assistant 使用"}
            error = True
        else:
            tool = by_name[message.name]
            try:
                arguments = tool.params.model_validate(message.arguments).model_dump()
            except ValidationError as exc:
                content = {"status": "invalid_arguments", "error": str(exc)}
                error = True
            else:
                content = tool(**arguments)
                error = content.get("status") in {"failed", "unknown", "partial"} if isinstance(content, dict) else False
        return ToolMessage(role="tool", id=message.id, requestor=message.requestor,
                           content=json.dumps(content, ensure_ascii=False, default=str), error=error)

    environment.get_response = get_response


def task_directory(output: Path, seed: int, domains: list[str], domain: str, task_id: str) -> Path:
    arm = output / f"C_seed{seed}"
    return arm / task_id if len(domains) == 1 else arm / domain / task_id


def restore_checkpoints(manifest: dict, output: Path) -> dict:
    """恢复逐题提交窗口；未确认的原生更新不能通过重做当前题回滚。"""
    restored = {}
    for seed in manifest["seeds"]:
        arm_dir = output / f"C_seed{seed}"
        for checkpoint in arm_dir.glob("**/completed.json"):
            data = json.loads(checkpoint.read_text())
            row = data["task"]
            if not row.get("phase_complete") or row.get("memory", {}).get("unsafe_to_continue"):
                raise ValueError(f"学习状态未确认，不能自动续跑：{checkpoint}")
            key = (row["seed"], row["domain"], row["task_id"])
            restored[key] = (row, data["simulation"])
    for active in manifest.get("active_tasks", []):
        key = (active["seed"], active["domain"], active["task_id"])
        if key not in restored:
            raise ValueError("存在未完成尝试／学习状态未确认，不能自动续跑；现有 cube 保留。")
    for row in manifest.get("tasks", []):
        if (row["seed"], row["domain"], row["task_id"]) not in restored:
            raise ValueError("缺少完整逐题检查点，不能自动续跑。")
    manifest["tasks"] = sorted((row for row, _ in restored.values()),
                               key=lambda row: (manifest["seeds"].index(row["seed"]), row["position"]))
    manifest["active_tasks"] = []
    return restored


def configure(args, supplied: set[str]) -> None:
    """C 并发做题、同步学习，禁止整题自动重试。"""
    if args.arms != ["C"]:
        raise ValueError("C 使用专用 Skill 服务，请与 B 分开运行。")
    if "max_concurrency" not in supplied:
        args.max_concurrency = 8
    if "max_retries" not in supplied:
        args.max_retries = 0
    if "memos_url" not in supplied:
        args.memos_url = os.getenv("MEMOS_SKILL_URL", "http://127.0.0.1:8001")
    if args.max_concurrency < 1 or args.max_retries != 0:
        raise ValueError("C 要求 --max-concurrency >= 1、--max-retries 0，Skill 学习单独串行。")
    if not 1 <= args.top_k <= 6:
        raise ValueError("C 的 top-k 必须在 1～6 之间。")


def summary(manifest: dict) -> dict:
    rows = manifest["tasks"]
    scored = [row for row in rows if row["reward"] is not None]
    keyed = [dict(row, task_id=f"{row['domain']}:{row['task_id']}") for row in rows]
    return {
        "status": manifest["status"],
        "arms": {"C": {
            "tasks": len(rows), "scored": len(scored), **pass_metrics(keyed),
            "average_reward": sum(row["reward"] for row in scored) / len(scored) if scored else None,
            "unsafe_learning_tasks": [row["task_id"] for row in rows if row["memory"].get("unsafe_to_continue")],
            "memory_checks": [{"domain": row["domain"], "task_id": row["task_id"],
                               "seed": row["seed"], **row["memory"]} for row in rows],
        }},
    }


def run(args) -> None:
    from loguru import logger
    from tau2.data_model.simulation import Results, SimulationRun, TextRunConfig
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.orchestrator.orchestrator import Role
    from tau2.registry import registry
    from tau2.runner.batch import _TaskLogContext
    from tau2.runner.build import _build_env_kwargs, build_orchestrator
    from tau2.runner.helpers import get_info, get_tasks
    from tau2.utils.llm_utils import set_llm_log_mode

    from memos_eval.skill_agent import SkillAgent
    from memos_eval.skill_client import SkillClient

    supplied = {arg.split("=", 1)[0][2:].replace("-", "_") for arg in sys.argv[1:] if arg.startswith("--")}
    if "seed" in supplied:
        supplied.add("seeds")
    configure(args, supplied)
    if args.resume and args.output is None:
        raise ValueError("--resume 必须指定原输出目录。")
    output = (args.output or ROOT / "runs" / f"memos_C_{now()[:19].replace(':', '').replace('-', '')}").resolve()
    configuration = {key: getattr(args, key) for key in (
        "agent_llm", "user_llm", "max_concurrency", "max_retries", "max_steps",
        "retrieval_config", "top_k", "max_memory_calls")}
    configuration.update({"agent_effort": "max", "user_effort": "low",
                          "memory_add": {"async_mode": "sync", "mode": "fine", "messages": "agent_selected_visible_trajectory",
                                         "extraction": "native_skill_only", "same_skill_update": "coalesce_native_evidence"},
                          "fact_add": {"async_mode": "sync", "mode": "fast", "scope_default": "task"},
                          "memory_search": {"types": ["fact", "skill", "all"], "returns": "full_content"},
                          "skill_learning_concurrency": 1,
                          "num_trials_per_pass": 1, "max_learn_per_task": 1, "post_learning_generations": 6})
    domains = getattr(args, "domains", None) or [args.domain]
    if args.resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["arms"] != ["C"] or manifest["configuration"].get("memory_add") != configuration["memory_add"]:
            raise ValueError("该目录的 C 工具／学习管线与当前配置不同，请使用新输出目录。")
        saved = {key: manifest[key] for key in ("seeds", "memos_url", "memos_user")}
        saved.update({key: manifest["configuration"][key] for key in configuration if hasattr(args, key)})
        for key, value in saved.items():
            if key in supplied and getattr(args, key) != value:
                raise ValueError(f"续跑配置 {key} 与原实验不同。")
            setattr(args, key, value)
        if "domains" in supplied and domains != manifest["domains"]:
            raise ValueError("续跑 domains 与原实验不同。")
        if "domain" in supplied and [args.domain] != manifest["domains"]:
            raise ValueError("续跑 domain 与原实验不同。")
        domains = manifest["domains"]
        if "task_ids" in supplied and any(
            args.task_ids != [entry["task_id"] for entry in manifest["task_order"] if entry["domain"] == domain]
            for domain in domains
        ):
            raise ValueError("续跑 task-ids 与原实验不同。")
        configuration = manifest["configuration"]
        restored = restore_checkpoints(manifest, output)
    else:
        output.mkdir(parents=True, exist_ok=False)
        manifest = {"run_id": f"{output.name}-{uuid.uuid4().hex[:8]}", "arms": ["C"],
                    "status": "starting", "domain": domains[0], "domains": domains, "seeds": args.seeds,
                    "started_at": now(), "memos_url": args.memos_url, "memos_user": args.memos_user,
                    "configuration": configuration, "cubes": [], "tasks": [], "active_tasks": []}
        restored = {}
    os.environ["PATH"] = str(ROOT / ".tools/bin") + os.pathsep + os.environ.get("PATH", "")
    os.chdir(ROOT / "tau2-bench")
    logger.remove()
    handler = logger.add(output / "run.log", level="DEBUG", enqueue=True)
    set_llm_log_mode("all")
    client = SkillClient(args.memos_url, args.memos_user, os.getenv("MEMOS_API_KEY"), timeout=1800)
    def factory(tools, domain_policy, llm, llm_args, task, **kwargs):
        context = C_TASK_CONTEXT.get()
        agent = SkillAgent(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args,
                           client=context["client"], cube_id=context["cube_id"], task_id=task.id,
                           domain=context["domain"], audit_path=context["audit_path"],
                           skills_dir=context["skills_dir"], top_k=args.top_k,
                           max_memory_calls=args.max_memory_calls)
        context["agent"] = agent
        return agent

    registry.register_agent_factory(factory, "memos_skill_agent")
    user_args = {"extra_body": {"reasoning_effort": "low"}, "empty_response_retries": 2}
    for env, key in (("USER_API_BASE", "api_base"), ("USER_API_KEY", "api_key")):
        if os.getenv(env):
            user_args[key] = os.environ[env]
    configurations = {}
    task_lists = {}
    for domain in domains:
        ids = args.task_ids
        if args.resume:
            ids = [entry["task_id"] for entry in manifest["task_order"] if entry["domain"] == domain]
        tasks = get_tasks(domain, "base", ids)
        if ids:
            by_id = {task.id: task for task in tasks}
            tasks = [by_id[task_id] for task_id in ids]
        task_lists[domain] = tasks
        configurations[domain] = TextRunConfig(
            domain=domain, agent="memos_skill_agent", llm_agent=args.agent_llm,
            llm_args_agent={"temperature": 0.0, "extra_body": {"reasoning_effort": "max"}},
            llm_user=args.user_llm, llm_args_user=user_args,
            retrieval_config=args.retrieval_config, num_trials=1, max_concurrency=args.max_concurrency,
            max_steps=args.max_steps, max_errors=10, max_retries=0, seed=args.seeds[0])
    ordered_tasks = [(domain, task) for domain in domains for task in task_lists[domain]]
    order = [{"domain": domain, "task_id": task.id} for domain, task in ordered_tasks]
    if args.resume and manifest["task_order"] != order:
        raise ValueError("任务集合或顺序与原运行不同。")
    manifest["task_order"] = order
    manifest["last_invocation"] = {"started_at": now(), "num_tasks": args.num_tasks,
                                   "completed_before": len(manifest["tasks"])}
    manifest.setdefault("invocations", []).append(manifest["last_invocation"])

    def save_results(seed: int, simulations: dict) -> None:
        for domain in domains:
            config = configurations[domain].model_copy(update={"seed": seed})
            info = get_info(config)
            info.user_info.llm_args = {key: value for key, value in user_args.items() if key != "api_key"}
            results = Results(info=info, tasks=task_lists[domain], simulations=[
                simulations[(domain, task.id)] for task in task_lists[domain] if (domain, task.id) in simulations])
            directory = output / f"C_seed{seed}"
            if len(domains) > 1:
                directory /= domain
            directory.mkdir(parents=True, exist_ok=True)
            results.save(directory / "results.json")

    def run_task(position, domain, task, active, cube, arm_dir, seed, trial_seed):
        """每题独立客户端和上下文；只有 Skill 学习使用共享锁。"""
        set_llm_log_mode("all")
        directory = task_directory(output, seed, domains, domain, task.id)
        attempt_dir = directory / "attempt_0"
        task_client = SkillClient(args.memos_url, args.memos_user, os.getenv("MEMOS_API_KEY"), timeout=1800)
        task_client.learning_lock = client.learning_lock
        task_client.learning_aborted = client.learning_aborted
        task_client.snapshot_lock = client.snapshot_lock
        context = {"client": task_client, "cube_id": cube, "domain": domain,
                   "audit_path": attempt_dir / "memory_audit.jsonl", "skills_dir": arm_dir}
        token = C_TASK_CONTEXT.set(context)
        simulation = None
        orch = None
        phase_error = None
        active = dict(active, started_at=now())
        try:
            attempt_dir.mkdir(parents=True, exist_ok=False)
            print(f"C {position + 1}/{len(ordered_tasks)} {domain}/{task.id} 开始", flush=True)
            config = configurations[domain].model_copy(update={"seed": seed})
            with _TaskLogContext(str(uuid.uuid4()), attempt_dir, task, True):
                orch = build_orchestrator(config, task, seed=trial_seed)
                agent = context["agent"]
                install_memory_tools(orch.environment, agent.memory_tools)
                simulation = orch.run()
                simulation.policy = orch.environment.get_policy()
                scoring = simulation.model_copy(update={"messages": business_messages(simulation.messages)})
                simulation.reward_info = evaluate_simulation(
                    simulation=scoring, task=task, evaluation_type=EvaluationType.ALL,
                    solo_mode=False, domain=domain, env_kwargs=_build_env_kwargs(config, task) or None)
                write_json(attempt_dir / "business_simulation.json", simulation.model_dump(mode="json"))
                tail = orch.message if orch.to_role == Role.AGENT else None
                agent.finish(simulation.termination_reason, tail)
        except BaseException as exc:
            phase_error = exc
            attempt_dir.mkdir(parents=True, exist_ok=True)
            write_json(attempt_dir / "error.json", {"type": type(exc).__name__, "message": str(exc),
                                                   "traceback": traceback.format_exc()})
        finally:
            C_TASK_CONTEXT.reset(token)
            task_client.close()
        memory = context["agent"].summary() if "agent" in context else {"unsafe_to_continue": True}
        if phase_error is not None:
            memory["unsafe_to_continue"] = True
        row = {**active, "finished_at": now(), "trial_seed": trial_seed,
               "phase_complete": phase_error is None and not memory.get("unsafe_to_continue"),
               "reward": simulation.reward_info.reward if simulation is not None and simulation.reward_info else None,
               "termination_reason": str(simulation.termination_reason) if simulation is not None else "infrastructure_error",
               "memory": memory, "memos_http_requests": task_client.request_count,
               "agent_usage": reported_usage(getattr(orch, "agent_state", None), "assistant"),
               "user_usage": reported_usage(getattr(orch, "user_state", None), "user")}
        if phase_error is not None:
            row["error"] = {"type": type(phase_error).__name__, "message": str(phase_error)}
        return row, simulation

    try:
        manifest["memos_features"] = client.ensure_skill_service()
        manifest["status"] = "running"
        manifest.pop("error", None)
        write_json(output / "manifest.json", manifest)
        for seed in args.seeds:
            arm_dir = output / f"C_seed{seed}"
            arm_dir.mkdir(exist_ok=True)
            cubes = [entry["cube_id"] for entry in manifest["cubes"] if entry["seed"] == seed]
            if len(cubes) > 1:
                raise ValueError("C 的共享 cube 不唯一。")
            cube = cubes[0] if cubes else client.create_cube(f"tau3-C-{manifest['run_id']}-{seed}")
            if not cubes:
                manifest["cubes"].append({"arm": "C", "seed": seed, "task_id": None,
                                          "cube_id": cube, "created_at": now()})
                write_json(arm_dir / "skills_latest.json", {"cube_id": cube, "timestamp": now(),
                                                           "count": 0, "skills": []})
            write_json(output / "manifest.json", manifest)
            simulations = {(domain, task_id): SimulationRun.model_validate(simulation)
                           for (trial, domain, task_id), (_, simulation) in restored.items() if trial == seed}
            trial_seed = random.Random(seed).randint(0, 1000000)
            selected = ordered_tasks[:args.num_tasks] if args.num_tasks is not None else ordered_tasks
            remaining = iter((position, domain, task) for position, (domain, task) in enumerate(selected)
                             if (seed, domain, task.id) not in restored)
            failure = None
            with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
                pending = {}

                def submit_next():
                    entry = next(remaining, None)
                    if entry is None:
                        return False
                    position, domain, task = entry
                    active = {"arm": "C", "seed": seed, "domain": domain, "task_id": task.id,
                              "position": position, "cube_id": cube, "submitted_at": now()}
                    manifest["active_tasks"].append(active)
                    write_json(output / "manifest.json", manifest)
                    future = executor.submit(run_task, position, domain, task, active, cube, arm_dir, seed, trial_seed)
                    pending[future] = entry
                    return True

                for _ in range(args.max_concurrency):
                    if not submit_next():
                        break
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in sorted(done, key=lambda item: item.result()[0]["finished_at"]):
                        position, domain, task = pending.pop(future)
                        row, simulation = future.result()
                        row["completion_order"] = len(manifest["tasks"]) + 1
                        if simulation is not None:
                            simulation.trial = 0
                            directory = task_directory(output, seed, domains, domain, task.id)
                            write_json(directory / "completed.json", {"task": row, "simulation": simulation.model_dump(mode="json")})
                            simulations[(domain, task.id)] = simulation
                            save_results(seed, simulations)
                        manifest["tasks"].append(row)
                        if row["phase_complete"]:
                            manifest["active_tasks"] = [active for active in manifest["active_tasks"]
                                if (active["seed"], active["domain"], active["task_id"]) != (seed, domain, task.id)]
                        else:
                            failure = failure or RuntimeError(
                                f"{domain}/{task.id} 的业务或记忆状态未确认；共享 cube 和日志已保留。")
                        write_json(output / "manifest.json", manifest)
                        write_json(output / "summary.json", summary(manifest))
                        memory = row["memory"]
                        print(f"C {row['completion_order']}/{len(ordered_tasks) * len(args.seeds)} {task.id} reward={row['reward']} "
                              f"skills={memory.get('final_skill_count', '?')} learn={memory.get('learning_status', '?')}", flush=True)
                    if failure is None:
                        while len(pending) < args.max_concurrency and submit_next():
                            pass
                if failure is not None:
                    raise failure
        manifest["status"] = "completed" if len(manifest["tasks"]) == len(ordered_tasks) * len(args.seeds) else "partial"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        client.close()
        manifest["finished_at"] = now()
        manifest["last_invocation"]["finished_at"] = manifest["finished_at"]
        write_json(output / "manifest.json", manifest)
        write_json(output / "summary.json", summary(manifest))
        logger.remove(handler)
    print(f"结果：{output}", flush=True)
