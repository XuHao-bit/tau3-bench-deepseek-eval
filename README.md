# τ³-bench：DeepSeek v4.1 Flash 银行知识库评测

本仓库提供 `banking_knowledge` 领域的 A 组基线，以及 B／C 组本地 MemOS 实验复现代码。

| 实验 | 记忆方式 |
|---|---|
| A | DeepSeek，无 MemOS |
| B-intra / B-cross | harness 自动检索、注入并异步保存事实／偏好；题内隔离或跨题共享 |
| C | DeepSeek 自主调用搜索、fast 事实写入、原生 Skill 同步提炼工具 |

**B／C 的安装、完整评测、18／5 题试跑、resume 和记忆查看：见 [复现步骤](docs/reproduce_memos.md)。**
本地 MemOS 启动器支持 macOS Apple Silicon，不依赖 Cloud 或 plugin。

已归档的 A 组配置与结果：

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
模型配置和用户空回复重试补丁，安装 `.[knowledge]`，并生成
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

## B / C 本地 MemOS 实验

```bash
source .venv/bin/activate
./local_memos/setup.sh
# 按 docs/reproduce_memos.md 配置 tau2-bench/.env
./local_memos/start.sh

# B：自动注入记忆；也可将 B-cross 换成 B-intra
./run_banking_memos.sh --arms B-cross --seeds 300 --max-concurrency 8 \
  --output runs/banking_B_cross_seed300

# C：自主操作事实及原生 Skill
./local_memos/start_skills.sh
./run_banking_memos.sh --arms C --seeds 300 --max-concurrency 8 --max-retries 0 \
  --output runs/banking_C_seed300
```

B／C 使用新 cube；并发题可读到检索当时已落库的内容，seed 不能保证相同的记忆演化顺序。
详细运行、续跑和观测命令见 [B／C 复现步骤](docs/reproduce_memos.md)。

## 目录

```text
.
├── run_banking_official.sh          A 组评测命令
├── run_banking_memos.sh             B / C 组评测入口
├── memos_eval/                      自动注入、工具、runner、恢复与观测
├── local_memos/                     固定版本 MemOS + Neo4j 本地部署适配
├── docs/reproduce_memos.md          B / C 完整复现步骤
├── view_memos.sh / view_skills.sh   查看实际保存的记忆
├── tests/                          离线验证
├── constraints.txt                 benchmark 关键依赖版本
├── setup.sh                         拉取上游、打补丁、安装环境
├── env.example                      配置模板（不含密钥）
├── patches/
│   ├── tau2-bench-v1.0.1-dsv4.patch 模型注册、embedding base URL、judge 配置
│   └── tau2-bench-v1.0.1-empty-user-retry.patch 用户空回复重试
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
- MemOS：https://github.com/MemTensor/MemOS
- 固定 commit：`a7367d07e55db61099f7b4e2c1108bc5831a24f3`（Python 主服务，MemoryOS 2.0.33）
