#!/usr/bin/env python3
"""Summarize a tau2 results.json into compact, reproducible tables."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean


def reward_of(sim):
    info = sim.get("reward_info") or {}
    return info.get("reward")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    data = json.loads(args.results.read_text(encoding="utf-8"))
    sims = data.get("simulations") or []
    tasks = data.get("tasks") or []
    if not sims:
        raise SystemExit("results.json contains no simulations")

    rewards = [reward_of(s) for s in sims]
    reward_counter = Counter("null" if r is None else str(r) for r in rewards)
    rewarded = sum(1 for r in rewards if r == 1.0)
    valid = sum(1 for r in rewards if r is not None)
    durations = [float(s.get("duration") or 0.0) for s in sims]
    agent_cost = sum(float(s.get("agent_cost") or 0.0) for s in sims)
    user_cost = sum(float(s.get("user_cost") or 0.0) for s in sims)
    termination = Counter(str(s.get("termination_reason")) for s in sims)
    reward_basis = Counter(
        "|".join((s.get("reward_info") or {}).get("reward_basis") or [])
        for s in sims
    )

    errors = []
    for sim in sims:
        err = (sim.get("info") or {}).get("error")
        if err:
            errors.append(
                {
                    "task_id": sim.get("task_id"),
                    "termination_reason": sim.get("termination_reason"),
                    "error_type": (sim.get("info") or {}).get("error_type"),
                    "error": err,
                }
            )

    info = data.get("info") or {}
    summary = {
        "result_timestamp": data.get("timestamp"),
        "task_count": len(tasks),
        "simulation_count": len(sims),
        "num_trials": info.get("num_trials"),
        "seed": info.get("seed"),
        "trial_seed": sims[0].get("seed"),
        "max_steps": info.get("max_steps"),
        "max_errors": info.get("max_errors"),
        "retrieval_config": info.get("retrieval_config"),
        "agent": info.get("agent_info"),
        "user_simulator": {
            "llm": (info.get("user_info") or {}).get("llm"),
            "llm_args": (info.get("user_info") or {}).get("llm_args"),
        },
        "reward_counts": dict(sorted(reward_counter.items())),
        "rewarded_count": rewarded,
        "valid_count": valid,
        "pass_at_1_all_tasks": rewarded / len(sims),
        "pass_at_1_valid_tasks": rewarded / valid if valid else None,
        "mean_trial_reward": sum(r or 0.0 for r in rewards) / len(sims),
        "termination_reason_counts": dict(sorted(termination.items())),
        "reward_basis_counts": dict(sorted(reward_basis.items())),
        "duration_seconds": {
            "mean": mean(durations),
            "max": max(durations),
            "sum": sum(durations),
        },
        "cost": {
            "agent_cost_sum": agent_cost,
            "user_cost_sum": user_cost,
            "total_cost_sum": agent_cost + user_cost,
        },
        "errors": errors,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fields = [
        "task_id",
        "trial",
        "seed",
        "reward",
        "termination_reason",
        "duration_seconds",
        "agent_cost",
        "user_cost",
        "error",
    ]
    with (args.out_dir / "per_task.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for sim in sorted(sims, key=lambda x: str(x.get("task_id"))):
            writer.writerow(
                {
                    "task_id": sim.get("task_id"),
                    "trial": sim.get("trial"),
                    "seed": sim.get("seed"),
                    "reward": reward_of(sim),
                    "termination_reason": sim.get("termination_reason"),
                    "duration_seconds": sim.get("duration"),
                    "agent_cost": sim.get("agent_cost"),
                    "user_cost": sim.get("user_cost"),
                    "error": (sim.get("info") or {}).get("error"),
                }
            )


if __name__ == "__main__":
    main()
