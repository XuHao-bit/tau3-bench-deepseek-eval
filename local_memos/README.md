# 本地 MemOS 实验服务

使用本仓库内按固定 commit 拉取的 `MemOS/src/memos` 的 Python 主服务，独立虚拟环境和原生 Neo4j 5.26.6。当前启动脚本面向 macOS Apple Silicon。Java、Maven、数据库、下载缓存和日志均在本目录内，不安装系统服务，不使用 Cloud 或 plugin。

```bash
# 首次安装依赖
./local_memos/setup.sh

# 启动／停止
./local_memos/start.sh
./local_memos/stop.sh
```

服务地址为 `http://127.0.0.1:8000`，数据库仅监听本机 `7687`／`7474`。数据库关闭鉴权以用于这套本机实验，不对外提供网络服务。

API 凭据从 `tau2-bench/.env` 读取，不写入部署文件。Embedding 使用 `text-embedding-3-large`、3072 维，可由 `MOS_EMBEDDER_API_BASE`、`MOS_EMBEDDER_API_KEY` 单独配置；内部 DeepSeek 使用 `OPENAI_*` 网关。`multimodal_struct` reader、关闭重组、Redis 队列和原生 scheduler。

B 组使用本地异步适配：`/product/add` 接收 `async_mode="async"` 后返回票据，八个后台 worker 执行普通 fine 事实／偏好抽取、embedding 和落库核验；同 cube/session 按提交顺序处理。客户端不传 `mode` 或 `chat_history`，不等待处理完成。`/product/add/status` 查询 `submitted/completed/failed/unknown`；`/health` 的 `eval_features` 标明该配置，旧服务不能直接用于新版 B。

`ENABLE_PREFERENCE_MEMORY=true` 启用抽取，`PreferenceMemory` 与事实记忆一同存入 Neo4j 的 `text_mem`，不需要独立偏好向量库。部署入口禁用 Skill 和工具轨迹抽取，不投递原生 scheduler 后续任务；检索只返回事实和偏好记忆。`serve.py` 创建 owner `tau3-eval`，并为老删除接口提供 `pref_mem=None` 属性，该属性不控制当前偏好记忆的抽取与存储。

启动器为当前 graph store 的 `internal_info` 字段提供 JSON 存储和读取还原，满足 Neo4j 属性不支持嵌套 map 的限制；多 chunk 聚合的 `ingest_batch_id` 仍以原字典形式由 MemOS 读取。其它记忆字段和公共接口不变。

配置保存在 `runtime/runtime_info.json`，后台票据与状态保存在 `runtime/async_writes.jsonl`。重启后未确认结束的票据标为 `unknown`，不会自动重发；`completed` 表示处理返回且返回的 ID 已读回，不代表抽取内容正确。MemOS 内部仍可能吞掉抽取模型异常并返回原文或空结果，需结合服务日志分析。

`runtime/model_usage.jsonl` 仅记录内部模型请求的 model、usage、HTTP 状态、trace ID 和耗时，不保存请求正文、向量或凭据。后台任务保留请求的 trace context，票据包含 `trace_id` 便于关联。原始服务日志仅留在忽略提交的本机 runtime 目录。

## C：原生 Skill 服务

共享 Neo4j 启动后，C 使用独立进程、8001 端口和 `runtime_skills/`，不会更改 B 的提炼配置：

```bash
./local_memos/start_skills.sh
./local_memos/stop_skills.sh
```

服务的 `sync + fine` 仅执行原生 Skill 提炼、更新和完整附件生成，省去普通文本、工具轨迹、偏好及逐来源的伴生提炼；`sync + fast` 直接保存 DS 提交的事实原文，不调用 LLM、不生成 Skill。Skill 学习使用服务端串行锁，fast 事实写入不排该队列。LOCAL 技能 ZIP 保存到 `runtime_skills/files/`，本地文件处理需要 `alibabacloud-oss-v2` 包，`setup.sh` 会安装，不需要 OSS 账号。

Skill 学习内部只召回旧 `SkillMemory`，不把任务范围事实的原始来源加入其他任务的提炼上下文。这个类型约束只作用于传给原生提炼器的 searcher；业务中的事实检索仍由评测端按 `fact_scope` 与 session 过滤。

同次提炼的多个候选若实际更新同一个旧 Skill，先合并这些候选对应的消息证据，再调用原生提炼 helper 生成一个结果，供原生附件与更新流程使用。不同更新目标不合并；`coalesced_updates` 记录原候选、证据索引及结果，合并失败显式报错，避免同 ID 后写覆盖和旧名 ZIP 被重复删除。

更新 C 服务代码后，先 `stop_skills.sh` 再 `start_skills.sh`。C 的 Neo4j 适配将结构化 Skill／Tool 字段保存为 JSON，读回还原列表和字典，并恢复来源消息引用。

`/health` 标记 `native_skill_only`、`sync_fact_write=true` 和 `companion_fine_extraction=false`。add 保留原生 `data` 并增加 `eval_diagnostics`；学习诊断区分工具消息不足、0 产物、提炼警告／错误和文件缺失，写入 `runtime_skills/skill_learning.jsonl`。事实诊断标记 `operation=write_fact`，写入 `fact_writes.jsonl`；所有内部开销保存在 `model_usage.jsonl`。Skill／fact 按 ID 读回、可见范围和新旧版本对照由评测端完成。

学习诊断在原生技能文件写入入口记录 `skill_persistence_started`。原生提炼正常返回空、尚未进入该持久化阶段且 add 返回空 data 时，`write_state=not_started`；提炼失败仍保留 failed 及原始错误。已经进入文件／节点写入或异常状态不明时，不会标记为未写入。
