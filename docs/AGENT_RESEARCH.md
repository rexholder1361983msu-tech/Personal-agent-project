# Agent 项目调研与构建建议

本文基于当前 MiniClaw 源码、测试，以及截至 2026-09-09 可访问的官方文档整理。本文只记录分析，不包含实现改动。

## 一、当前项目判断

MiniClaw 是一个轻量级的 OpenAI-compatible ReAct Agent，核心循环为：模型请求 -> 工具调用 -> 工具结果 -> 模型请求。现有边界包括：

- `miniclaw/llm`：模型协议与 OpenAI-compatible 适配器。
- `miniclaw/tools`：工具协议、注册表和内置工具。
- `miniclaw/skills`：通过 `SKILL.md` 加载技能，并将技能包装成工具。
- `miniclaw/memory`：短期历史、长期向量记忆和向量存储抽象。
- `miniclaw/gateway`：CLI 与 FastAPI Web 接入。

最重要的现状偏差是：默认 CLI/Web 路径只组装 LLM、Agent 和内置工具，没有接入技能、记忆或真正的会话存储；Web 请求还共享 Agent 实例的历史。`examples/02_with_skills.py` 和 `examples/03_with_memory.py` 各自重新实现了一套执行流程，因此能力存在，但没有成为统一运行时的一部分。

当前高风险工具 (`shell`、`python_eval`、文件读写、网络抓取) 主要依赖应用层检查，不能视为安全隔离。生产环境还缺少取消、预算、审批、限流、认证、审计和标准 tracing。

## 二、社区项目对照

| 项目 | 最值得借鉴的设计 | 适用边界 |
|---|---|---|
| LangGraph | checkpointer 管单线程状态，store 管跨线程长期数据；显式状态、恢复和人工介入 | 复杂、可恢复的工作流；完整引入会增加复杂度 |
| PydanticAI | 类型驱动工具 schema、结构化输出、依赖注入、确定性测试模型、OTel | 最适合吸收其类型安全与测试方式 |
| OpenAI Agents SDK | Session、RunState、工具审批、`Agent.as_tool`、内置 trace/span | 与现有 Skill-as-Tool 最接近；注意部分能力与 OpenAI 生态绑定 |
| AG2 | 事件日志/WAL、多 Agent 网络、OTel GenAI 语义、trace evaluation | 适合平台化多 Agent；对当前单 Agent 项目偏重 |
| smolagents | 明确区分代码执行与安全边界，推荐 Docker/E2B/Modal 等 sandbox | 直接提醒：黑名单和本地执行不等于隔离 |

## 三、推荐目标架构

不要先做“万能多智能体平台”，建议先把 MiniClaw 进化成一个可恢复的单 Agent Runtime：

```text
Gateway / CLI / Job
        |
ConversationService  -- tenant/user/session 隔离
        |
AgentRuntime         -- 一次运行的状态、预算、取消、审批、恢复
        |
Policy + ToolExecutor -- 权限、超时、并发、审计、安全执行
        |
LLM / ToolRegistry / SkillRegistry / Memory
        |
SessionStore + EventStore + VectorStore + OTel
```

建议的最小领域对象：

- `RunState`：`run_id`、`session_id`、step、messages、pending approval、usage、错误和当前 agent/skill。
- `SessionStore`：按 `tenant_id/user_id/session_id` 保存短期对话，先 SQLite，后续可换 PostgreSQL/Redis。
- `RunEvent`：`RunStarted`、`ModelRequested`、`ToolCalled`、`ToolReturned`、`RunPaused`、`RunCompleted`、`RunFailed`。
- `ToolPolicy`：是否允许、是否需要审批、超时、输出上限、网络权限和资源预算。
- `MemoryService`：召回、上下文组装、记忆提取、写入、删除、TTL 和 namespace；不要无条件把完整对话写入向量库。

技能仍然可以保留 `SKILL.md`，但 `SkillAsTool` 应调用统一的 `AgentRuntime`，不能再复制一套 Agent loop。

## 四、分阶段实施顺序

### Phase 0：先固定行为

- 明确默认入口的能力清单和配置优先级。
- 为模型适配器、工具异常、配置占位符、技能执行、Web 会话补集成测试。
- 用确定性 Mock/Scripted Model 固定工具调用序列，禁止测试访问真实模型。

### Phase 1：统一运行时

- 抽出 `build_runtime(config)` 作为唯一组装入口。
- 将 `ReActAgent` 的循环改为 `AgentRuntime.run()`，技能、CLI、Web、定时任务全部复用。
- 增加 `RunLimits`：最大 step、token、工具调用次数、总耗时、并发数和单工具超时。

### Phase 2：会话与恢复

- 引入 SQLite `SessionStore` 和 `RunState` 持久化。
- 每个请求按身份和 session 获取独立状态，彻底消除共享 `agent.history`。
- 支持暂停/审批后用同一 `run_id` 恢复；长历史支持窗口截断和摘要。

### Phase 3：工具安全

- 普通纯函数工具可进程内执行；文件、Shell、Python 和网络工具走独立 `ToolExecutor`。
- 高风险执行优先使用 Docker sandbox：非 root、只读根文件系统、限制 CPU/内存/PIDs、默认无网络、超时和输出上限。
- 权限策略、审批和审计独立于工具本身；不要把字符串黑名单当安全边界。

### Phase 4：观测与质量

- 每次运行产生事件流和 OTel trace；默认脱敏，不把 prompt、工具参数和结果无条件写入观测后端。
- 建立离线评测集：任务成功率、工具选择准确率、参数正确率、步骤数、延迟、token 成本和安全拒绝率。
- 增加故障恢复、并发隔离、工具超时、审批拒绝、SSRF、路径穿越和大输出测试。

### Phase 5：再决定是否多 Agent/图编排

只有在单 Agent Runtime 已能稳定恢复和观测后，再按需求选择：

- 流程确定、需要人工介入和回放：吸收 LangGraph 的显式状态图和 checkpoint 思路。
- 多角色协作：先用 `Agent-as-tool`，只有需要控制权转移时才做 handoff。
- 复杂多 Agent 网络：再评估 AG2，而不是一开始引入。

## 五、第一版验收标准

第一版不以“能聊天”为验收，而以以下行为为准：

1. 两个用户并发请求不会互相看到历史、工具结果或记忆。
2. 进程重启后可以从持久化 session 继续对话。
3. 危险工具在无权限或未审批时不会执行，工具调用可审计。
4. 工具超时、模型异常和进程中断有结构化状态，能够重试或恢复。
5. 每次运行可看到模型、工具、耗时、token 和错误链路，敏感内容可关闭。
6. CLI、Web、技能和后台任务使用同一条 runtime 路径。

## 六、官方资料

- LangGraph persistence: https://docs.langchain.com/oss/python/langgraph/persistence
- PydanticAI: https://pydantic.dev/docs/ai/
- OpenAI Agents SDK sessions: https://openai.github.io/openai-agents-python/sessions/
- OpenAI Agents SDK tracing: https://openai.github.io/openai-agents-python/tracing/
- AG2 telemetry: https://docs.ag2.ai/docs/user-guide/telemetry/
- smolagents secure code execution: https://huggingface.co/docs/smolagents/en/tutorials/secure_code_execution
- OpenAI Agents SDK repository: https://github.com/openai/openai-agents-python
- LangGraph repository: https://github.com/langchain-ai/langgraph
- PydanticAI repository: https://github.com/pydantic/pydantic-ai
- AG2 repository: https://github.com/ag2ai/ag2
- smolagents repository: https://github.com/huggingface/smolagents