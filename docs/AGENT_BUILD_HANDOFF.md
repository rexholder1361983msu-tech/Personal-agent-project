# 自研 Agent 项目启动交接

本文用于在新项目中继续实现自己的 Python Agent。它记录了本次对话已经确认的背景、判断和下一步，不代表已经完成代码实现。

## 项目目标

构建一个轻量、可扩展、可恢复、可观测且具备安全工具边界的 Python Agent 项目。第一阶段不追求完整多智能体平台，优先把单 Agent Runtime 做扎实。

## 现有参考项目

参考项目是 MiniClaw，位置为当前仓库的 `miniclaw/`。它已经提供：

- OpenAI-compatible LLM 抽象。
- ReAct 模型调用和工具调用循环。
- Tool 与 ToolRegistry。
- `SKILL.md` 技能加载机制。
- 短期历史、长期向量记忆抽象。
- CLI 和 FastAPI Web Gateway。
- 基础单元测试。

## 已确认的问题

- 默认 CLI/Web 没有真正接入技能和记忆。
- Web 请求共享 Agent history，存在用户会话串线风险。
- 技能和记忆示例各自复制了 Agent 执行循环。
- Shell、Python、文件读写、网络抓取工具没有真正的安全隔离。
- 缺少 SessionStore、RunState、恢复、审批、取消、预算、审计和标准 tracing。
- 配置、依赖、Web 入口和部分文档存在实现与描述不一致。

## 目标架构

```text
Gateway / CLI / Job
        |
ConversationService       # tenant/user/session 隔离
        |
AgentRuntime              # 一次运行的状态、预算、取消、审批、恢复
        |
Policy + ToolExecutor     # 权限、超时、并发、审计、安全执行
        |
LLM / ToolRegistry / SkillRegistry / Memory
        |
SessionStore + EventStore + VectorStore + OpenTelemetry
```

## 第一阶段范围

第一阶段只实现以下内容：

1. 统一 `build_runtime(config)` 组装入口。
2. 抽出统一的 `AgentRuntime` 执行路径。
3. 定义 `RunState`，包含 `run_id`、`session_id`、step、messages、usage 和状态。
4. 实现 SQLite `SessionStore`，按 `tenant_id/user_id/session_id` 隔离会话。
5. 修复 Web 共享 history 问题。
6. 让 CLI、Web 和 Skill 使用同一条 Runtime 路径。
7. 为并发会话、持久化恢复和工具循环增加确定性测试。

第一阶段暂不实现：

- 完整多 Agent 网络。
- LangGraph 或 AG2 的整体迁移。
- 生产级分布式队列。
- 复杂向量记忆重构。
- 真实 Docker sandbox 的完整部署平台。

## 第二阶段候选范围

- `RunLimits`：最大 step、token、工具次数、总耗时、并发数和单工具超时。
- 事件模型：`RunStarted`、`ModelRequested`、`ToolCalled`、`ToolReturned`、`RunPaused`、`RunCompleted`、`RunFailed`。
- 工具权限、人工审批、审计日志和结构化错误。
- Docker sandbox：非 root、资源限制、只读文件系统、默认无网络和输出上限。
- OpenTelemetry trace/span 与敏感内容脱敏。
- 长历史压缩、长期记忆 namespace、TTL、删除和来源记录。

## 社区项目参考结论

- LangGraph：参考 checkpointer 与 store 的短期/长期状态分离。
- PydanticAI：参考类型驱动工具 schema、结构化输出和确定性测试模型。
- OpenAI Agents SDK：参考 Session、RunState、审批、Agent-as-tool 和 tracing。
- AG2：参考事件日志、WAL 和 OpenTelemetry GenAI 语义。
- smolagents：参考代码执行必须与安全隔离明确分层，黑名单不是 sandbox。

## 第一阶段验收标准

- 两个用户并发请求不会互相看到历史、工具结果或记忆。
- 进程重启后可以从 SQLite session 继续对话。
- CLI、Web、Skill 使用同一个 AgentRuntime，而不是各自复制循环。
- 工具调用、工具结果和失败状态可以通过测试稳定复现。
- 测试不访问真实 LLM，使用 Mock/Scripted Model。
- 现有测试保持通过，新增测试覆盖会话隔离和恢复。

## 新项目开始时的建议提示词

```text
请基于 docs/AGENT_BUILD_HANDOFF.md 开始实现第一阶段。
先阅读现有代码和测试，保持公开 API 兼容，优先实现统一 AgentRuntime、SQLite SessionStore、RunState、Web 会话隔离和确定性测试。
不要先引入 LangGraph/AG2，也不要重写无关模块。每次完成一个小切片后立即运行对应测试。
高风险工具暂时保留接口，但不要把黑名单当作安全隔离；真正的 sandbox 放到后续阶段。
```

## 官方资料

- LangGraph persistence: https://docs.langchain.com/oss/python/langgraph/persistence
- PydanticAI: https://pydantic.dev/docs/ai/
- OpenAI Agents SDK sessions: https://openai.github.io/openai-agents-python/sessions/
- OpenAI Agents SDK tracing: https://openai.github.io/openai-agents-python/tracing/
- AG2 telemetry: https://docs.ag2.ai/docs/user-guide/telemetry/
- smolagents secure code execution: https://huggingface.co/docs/smolagents/en/tutorials/secure_code_execution

详细调研见 [AGENT_RESEARCH.md](AGENT_RESEARCH.md)。