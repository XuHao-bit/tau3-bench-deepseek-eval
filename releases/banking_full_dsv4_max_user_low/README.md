# A：DeepSeek v4.1 Flash 在 τ³-bench 银行知识库上的单次评测

## 结果

| 指标 | 值 |
| --- | --- |
| benchmark | `tau2-bench` v1.0.1，`banking_knowledge` |
| 检索配置 | `alltools` |
| agent | `openai/deepseek-v4.1-flash-1017`，temperature 0.0，`reasoning_effort=max` |
| 用户模拟器 | `openai/gpt-5.2`，temperature 0.0，`reasoning_effort=low` |
| 题目数 | 97 |
| trial 数 | 1 |
| seed | 300（实际 trial seed 626729） |
| 最大步数 | 200 |
| 成功题数 | 46 / 97 |
| pass@1（全部题） | 0.4742 |
| pass@1（排除基础设施错误） | 46 / 96 = 0.4792 |
| 终止原因 | 96 `user_stop`，1 `infrastructure_error` |
| 记录成本 | agent 8.2465，用户模拟器 3.0289，合计 11.2753 |

`task_018` 的失败是本地 orchestrator 的消息校验异常，不是模型给出的业务失败：

```text
UserMessage must have either content or tool_calls. Got UserMessage
is_final_chunk: True
```

因此同时报告 46/97 和 46/96 两个口径。主表使用 46/97，不把基础设施错误悄悄排除。

这是一次完整 97 题 pass，不是官方排行榜的 4-trial 协议；不能把它当作 4-trial 分数，也没有用重复 trial 估计方差。

## 文件

| 文件 | 说明 |
| --- | --- |
| `results.json.gz` | 原始 `results.json` 的压缩版；保留 tasks、全部 simulation 消息、判分明细和错误堆栈。本地绝对路径已替换为 `<WORKSPACE>`、`<TAU2_BENCH>`、`<TMP>`。 |
| `summary.json` | 自动汇总的成功率、终止原因、成本、耗时和错误。 |
| `per_task.csv` | 97 题逐题 reward、终止原因、耗时、成本。 |
| `run_config.json` | 模型、参数、题数、seed、并发和命令快照。 |
| `SHA256SUMS` | 文件校验和。 |

约 744 MB 的 `artifacts/`（每次 LLM 请求的 verbose debug）没有放入仓库；它可由 `--verbose-logs --llm-log-mode all` 重新生成，不影响从 `results.json.gz` 复核分数。

## 复现

要求：macOS/Linux、Python >= 3.12、可用的 OpenAI-compatible endpoint，以及：

- `srt`：`@anthropic-ai/sandbox-runtime@0.0.23`
- `rg`：ripgrep
- 能访问 `deepseek-v4.1-flash-1017` 和 `gpt-5.2` 的 API 凭据

```bash
git clone <this-repo> tau3-bench-deepseek-eval
cd tau3-bench-deepseek-eval
./setup.sh
```

`setup.sh` 会：

1. 拉取 `sierra-research/tau2-bench` v1.0.1；
2. 应用 `patches/tau2-bench-v1.0.1-dsv4.patch`；
3. 创建 `.venv` 并安装 `.[knowledge]`；
4. 从 `env.example` 生成 `tau2-bench/.env`；
5. 运行 `tau2 check-data`。

编辑 `tau2-bench/.env`，填入 endpoint 和 key。若用户模拟器使用独立 endpoint：

```bash
export USER_API_BASE=https://your-user-endpoint/v1
export USER_API_KEY=sk-...
```

启动评测：

```bash
./run_banking_official.sh
```

默认参数与本次运行一致：

```text
domain=banking_knowledge
retrieval=alltools
agent=openai/deepseek-v4.1-flash-1017
user=openai/gpt-5.2
num_trials=1
seed=300
max_steps=200
max_concurrency=8
max_retries=1
save_to=banking_full_dsv4_max_user_low
```

结果写入：

```text
tau2-bench/data/simulations/banking_full_dsv4_max_user_low/results.json
```

重新生成汇总：

```bash
gzip -dc releases/banking_full_dsv4_max_user_low/results.json.gz \
  > /tmp/results.json
python3 scripts/summarize_results.py /tmp/results.json \
  --out-dir /tmp/a_summary
```

## 可复现性边界

- 原始运行使用私有 OpenAI-compatible 网关。仓库不包含网关地址、key 或账号。没有相同模型和相同 serving 配置时，无法保证数值复现；换模型、量化版本或推理档位后结果不可直接比较。
- `deepseek-v4.1-flash-1017` 是本次实际请求的 model ID；第三方平台是否长期固定对应同一权重，需要以平台说明为准。
- `alltools` 依赖 embedding 和 sandbox 工具。`setup.sh` 只准备 Python 环境；`srt`、`rg` 需按上面的要求安装。
- 成本来自 LiteLLM 的模型价格映射，可能与实际账单不同；应以网关账单为准。
- 单次 97 题 pass 的方差没有估计。若要与官方 4-trial 结果比较，应把 `NUM_TRIALS` 设为 4 并重新运行，不能复用本目录的 `results.json.gz`。
