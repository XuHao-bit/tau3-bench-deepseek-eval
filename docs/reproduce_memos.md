# B / C 组复现

所有命令从本仓库根目录执行。当前本地服务启动器支持 **macOS Apple Silicon**，需要 Python 3.12、Git、curl、Node.js/npm、ripgrep；无需 MemOS Cloud、plugin、Docker 或 OSS 账号。

## 实验设置

| 项目 | B-intra | B-cross | C |
|---|---|---|---|
| 记忆操作 | harness 自动检索、注入、写入 | 同 B-intra | agent 自主调用 3 个工具 |
| 内容 | 事实和偏好；不提炼 Skill | 事实和偏好；不提炼 Skill | 事实及原生 Skill |
| 写入 | 每次 user＋最终 assistant 的 QA，异步 fine 提炼 | 同 B-intra | 事实同步 fast；Skill 同步 fine |
| 隔离范围 | 每题／每次尝试独立 cube | 每个 seed 共用一个 cube | 每个 seed 共用一个 cube；事实默认限当前任务，公共事实可 shared，Skill 跨题共享 |
| 并发 | 8 | 8 | 做题 8，Skill 提炼在服务端串行 |
| 整题基础设施重试 | 最多额外 1 次，重试前清理失败尝试的记忆 | 同 B-intra | 0，避免重复更新 Skill |
| MemOS 地址 | `127.0.0.1:8000` | `127.0.0.1:8000` | `127.0.0.1:8001` |

共同配置：`banking_knowledge`、`alltools`、97 题、每个 seed 每题 1 trial、seed 300、max steps 200；agent `openai/deepseek-v4.1-flash-1017`，temperature 0，`reasoning_effort=max`；用户 `openai/gpt-5.2`，`reasoning_effort=low`；最多检索 6 条记忆。

B 在收到新 user 消息时检索，并在随后的 agent 模型调用中注入检索块。工具往返期间可能反复看到同一块，但并不意味着又检索或写入了一次。只在本轮产生最终自然语言回答后提交 QA，不保存完整工具轨迹。

C 的工具：

- `memos_search(query, memory_type, top_k)`：搜索 fact、skill 或 all，直接返回正文、procedure、experience；无需额外 read。
- `memos_write_fact(content, scope)`：同步 fast 保存正文，不经 LLM 提炼；scope 默认 task，公共政策可选 shared。
- `memos_learn(message_ids)`：提交 agent 选中的真实可见业务轨迹，原生 MemOS 提炼或更新 Skill。每题最多一次，至少 5 条不同的工具结果，允许零产物；`["all"]` 选择当前完整业务轨迹。

C 结束业务对话后另有最多 6 次生成的记忆学习阶段；该阶段不能调用业务工具，也不能改变该题已固定的业务成绩。业务中和题后合计最多学习一次。提炼不接收隐藏标准答案、reward 或 judge 反馈。Skill 的生成和更新仍使用 MemOS 原生流程，本仓库适配了服务部署、事实隔离、同 ID 更新合并及写入诊断。

## 安装和配置

```bash
git clone https://github.com/XuHao-bit/tau3-bench-deepseek-eval.git
cd tau3-bench-deepseek-eval
npm install -g @anthropic-ai/sandbox-runtime@0.0.23
brew install ripgrep
./setup.sh
source .venv/bin/activate
./local_memos/setup.sh
```

`setup.sh` 拉取 tau2-bench `v1.0.1`，应用模型注册和用户空回复重试补丁；`local_memos/setup.sh` 拉取 MemOS commit `a7367d07e55db61099f7b4e2c1108bc5831a24f3` 到仓库内的 `MemOS/`，建立独立虚拟环境并下载 Java 21、Maven 3.9.9、Neo4j 5.26.6。两个环境的关键依赖分别固定在 `constraints.txt` 和 `local_memos/constraints.txt`。下载中断可重新运行安装脚本；已有完整 Java/Maven 发行包可分别放到 `local_memos/downloads/java21.tar.gz` 和 `maven.tar.gz` 复用。

编辑自动创建的 `tau2-bench/.env`（参考 `env.example`）：

```dotenv
OPENAI_API_KEY=YOUR_GATEWAY_KEY
OPENAI_API_BASE=https://your-compatible-endpoint/v1
OPENAI_BASE_URL=https://your-compatible-endpoint/v1

# MemOS embedding 独立渠道；若使用官方 embedding，填官方 API key。
MOS_EMBEDDER_API_BASE=https://api.openai.com/v1
MOS_EMBEDDER_API_KEY=YOUR_EMBEDDING_API_KEY
MOS_EMBEDDER_MODEL=text-embedding-3-large
EMBEDDING_DIMENSION=3072

# 可选：用户模拟器使用另一渠道。
# USER_API_BASE=https://your-user-endpoint/v1
# USER_API_KEY=YOUR_USER_KEY
```

网关须支持实际 model ID `deepseek-v4.1-flash-1017`、`gpt-5.2`、工具调用及上述推理档位。MemOS 内部提炼模型也使用该 DeepSeek 网关。`MOS_EMBEDDER_*` 仅配置 MemOS；benchmark 的 `KB_search_dense` 使用 `OPENAI_*` 渠道及上游配置的 embedding 模型，需要该渠道支持。

## B：运行与续跑

启动服务：

```bash
source .venv/bin/activate
./local_memos/start.sh
```

完整评测；两个命令可放到两个终端并行运行，每个终端从仓库根目录激活环境：

```bash
source .venv/bin/activate
./run_banking_memos.sh --arms B-cross --seeds 300 --max-concurrency 8 \
  --output runs/banking_B_cross_seed300
```

```bash
source .venv/bin/activate
./run_banking_memos.sh --arms B-intra --seeds 300 --max-concurrency 8 \
  --output runs/banking_B_intra_seed300
```

先试 18 题时，在相应命令中增加 `--num-tasks 18`。它保留全量任务清单，完成后用原输出目录续跑剩余题目：

```bash
./run_banking_memos.sh --resume --output runs/banking_B_cross_seed300
./run_banking_memos.sh --resume --output runs/banking_B_intra_seed300
```

指定小样本可改用 `--task-ids task_001 task_002 task_003`，但该参数会限定本实验的完整任务集合。银行共有 97 题，ID 有缺号，不一定到 task_097 为止。多个 trial 可用 `--seeds 300 301 302`；不同 seed 使用独立记忆。

查看当前记忆：

```bash
./view_memos.sh --run-dir runs/banking_B_cross_seed300 --arm B-cross --limit 3
./view_memos.sh --run-dir runs/banking_B_intra_seed300 --arm B-intra --limit 3
```

若 B 出现 reward=None，先停止评测并等待后台写入结束，修复 API 后可备份移出未评分尝试，复用原 cube 续跑：

```bash
python -m memos_eval.recover runs/banking_B_cross_seed300 --prepare --confirmed-stopped
./run_banking_memos.sh --resume --output runs/banking_B_cross_seed300
```

该操作保留有效 0/1 成绩和原记忆，失败／中断记录备份在 run 的 `recovery_*`；不适用于 C。

## C：运行与续跑

```bash
source .venv/bin/activate
./local_memos/start.sh
./local_memos/start_skills.sh
mkdir -p runs
set -o pipefail
./run_banking_memos.sh --arms C --seeds 300 --max-concurrency 8 --max-retries 0 \
  --output runs/banking_C_seed300 \
  2>&1 | tee runs/banking_C_seed300.console.log
```

先试 5 题，在上述评测命令增加 `--num-tasks 5`；完成后继续全部剩余题目：

```bash
source .venv/bin/activate
set -o pipefail
./run_banking_memos.sh --resume --output runs/banking_C_seed300 \
  2>&1 | tee -a runs/banking_C_seed300.console.log
```

新实验的输出目录必须不存在，runner 会创建新的空 cube；续跑必须保留原输出目录、Neo4j 数据、`local_memos/runtime/` 及 `local_memos/runtime_skills/` 中的状态和 Skill ZIP。新终端查看当前内容：

```bash
source .venv/bin/activate
./view_skills.sh --run-dir runs/banking_C_seed300 --memory-type all
./view_skills.sh --run-dir runs/banking_C_seed300 --memory-type skill \
  --output runs/banking_C_seed300/current_skills.json
```

明确 `write_state=not_started` 的学习失败会保留记录并继续，不自动补学；`partial/unknown` 写入会阻止自动续跑，须检查现场，不能直接删 session、重做当前题。成功的 `/product/add` 不保证有 Skill，需看 receipt 的状态、返回 ID、读回正文及文件检查。

## 日志、指标与停止

| 位置 | 用途 |
|---|---|
| `<run>/summary.json` | 完成数、已评分数、平均 reward、pass 指标 |
| `<run>/manifest.json` | 配置、task 顺序、cube ID、尝试与记忆状态 |
| `<run>/run.log` | 完整运行日志 |
| `<run>/<arm>_seed300/results.json` | 整组对话与评分结果 |
| `<run>/<arm>_seed300/<task>/attempt_0/artifacts/` | 完整轨迹和每次模型请求／响应 |
| 同级 `memory_audit.jsonl` | 实际检索、注入、写入、学习；C 区分业务和题后阶段 |
| C run 的 `skills_latest.json` / `facts_latest.json` | 当前保存内容快照 |
| C run 的 `skill_changes.jsonl` / `fact_changes.jsonl` | 逐次内容更新 |
| `local_memos/runtime*/server.log` / `model_usage.jsonl` | MemOS 服务异常、内部模型开销 |

`pass_at_1_all_tasks` 将未评分题计入分母；`pass_at_1_valid_tasks` 和 `average_reward` 只使用已评分题。单 seed 只有一个 trial，不能据此估计 pass@4；`pass_hat_ks` 是 pass^k 口径，不是至少成功一次的 pass@k。整题重试只用于基础设施异常，reward=0 不触发整题重试。用户模拟器空回复会额外原样重试 2 次，不重复执行业务工具或记忆写入。

评测结束后按顺序停止，先 C、再共享数据库：

```bash
./local_memos/stop_skills.sh   # 运行过 C 时执行
./local_memos/stop.sh
```

本地服务只监听本机，Neo4j 关闭鉴权。若原工作区已经有服务占用 8000/8001/7687/7474，先停止原工作区服务再在本仓库启动；不要同时启动两套本地数据库。修改 embedding 配置后需重启服务；已有记忆要保持相同模型及维度。

## 复现边界与离线检查

在线共享记忆意味着题号不是学习顺序。B 异步写入和 C 并发任务只能读到检索时已完成的写入；即使 seed 一致，也不保证共享记忆演化和最终分数完全相同。若需观察严格的先后学习，可用新目录并设 `--max-concurrency 1`，这与并发 8 是不同实验设置。同模型 ID 也不保证第三方 serving 版本始终一致。

代码和部署适配保存在本仓库；API key、运行数据库、模型缓存、B/C 轨迹与结果不提交到 GitHub。A 组历史归档继续保留在 `releases/`。这些步骤复现当前协议，不会重演历史运行的偶发中断与人工恢复。

已安装两个环境后，可运行不调用真实模型、不写运行数据库的测试：

```bash
source .venv/bin/activate
python -m unittest discover -s tests
local_memos/.venv/bin/python -m unittest discover -s tests -p 'test_native_skill*.py'
```
