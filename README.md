# τ³-bench：DeepSeek v4.1 Flash 银行知识库评测

本仓库是 `banking_knowledge` 领域一次完整 A 组评测的复现包：

- agent：`openai/deepseek-v4.1-flash-1017`，`reasoning_effort=max`；
- 用户模拟器：`openai/gpt-5.2`，`reasoning_effort=low`；
- 检索：`alltools`；
- 97 题，1 trial，seed 300；
- **46/97 题 reward=1.0，pass@1 = 0.4742**；
- `task_018` 是本地消息校验的 `infrastructure_error`，排除该题时为 46/96 = 0.4792。

这是**单次完整 pass**，不是官方排行榜的 4-trial 协议。完整结果、逐题表、配置和校验和见
[`releases/banking_full_dsv4_max_user_low/`](releases/banking_full_dsv4_max_user_low/)。

## 快速开始

要求 Python >= 3.12、可用的 OpenAI-compatible endpoint，以及：

- `srt`：`@anthropic-ai/sandbox-runtime@0.0.23`
- `rg`：ripgrep

```bash
git clone https://github.com/XuHao-bit/tau3-bench-deepseek-eval.git
cd tau3-bench-deepseek-eval
./setup.sh
```

`setup.sh` 会拉取 `sierra-research/tau2-bench` v1.0.1，应用
`patches/tau2-bench-v1.0.1-dsv4.patch`，安装 `.[knowledge]`，并生成
`tau2-bench/.env`。编辑该文件，填入 endpoint 和 key：

```bash
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
OPENAI_API_BASE=https://your-openai-compatible-endpoint/v1
```

如果用户模拟器使用独立 endpoint：

```bash
export USER_API_BASE=https://your-user-endpoint/v1
export USER_API_KEY=sk-...
```

运行与本次归档一致的评测：

```bash
./run_banking_official.sh
```

默认配置为 `num_trials=1`、`seed=300`、`max_steps=200`、`max_concurrency=8`。
结果写入：

```text
tau2-bench/data/simulations/banking_full_dsv4_max_user_low/results.json
```

## 目录

```text
.
├── run_banking_official.sh          A 组评测命令
├── setup.sh                         拉取上游、打补丁、安装环境
├── env.example                      配置模板（不含密钥）
├── patches/
│   └── tau2-bench-v1.0.1-dsv4.patch 模型注册、embedding base URL、judge 配置
├── scripts/
│   └── summarize_results.py         从 results.json 生成 summary/CSV
└── releases/
    └── banking_full_dsv4_max_user_low/
        ├── README.md                结果与复现细节
        ├── results.json.gz          完整 results.json（本地路径已脱敏）
        ├── summary.json             汇总指标
        ├── per_task.csv             97 题逐题结果
        ├── run_config.json          运行配置快照
        └── SHA256SUMS
```

## 复现边界

原始运行使用私有 OpenAI-compatible 网关。仓库不包含网关地址、密钥或账号；
没有相同的 `deepseek-v4.1-flash-1017` 和 `gpt-5.2` serving 配置时，不能保证数值复现。
换模型、量化版本或推理档位后，结果不可直接比较。

`results.json.gz` 保留了 tasks、完整 simulation 消息、判分明细和错误堆栈；约 744 MB 的
verbose `artifacts/` 未归档，可用 `--verbose-logs --llm-log-mode all` 重新生成。
成本来自 LiteLLM 价格映射，实际费用以网关账单为准。

## 上游

- τ³-bench / `tau2-bench`：https://github.com/sierra-research/tau2-bench
- 固定版本：`v1.0.1`（`fc0055dc4e0a316c3f83133267fbd6faaa770992`）
