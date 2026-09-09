# 第一阶段实施计划（Phase 1 Plan）

> 状态：待评审。本文是第一轮输出的计划，尚未改动任何代码。
> 依据：`docs/AGENT_BUILD_HANDOFF.md`、`docs/AGENT_RESEARCH.md`。

## 0. 关键前提：工作区没有现成代码

按交接文档要求“先阅读现有代码和测试”，已对整个工作区做了扫描：

- `project/` 下只有 `docs/AGENT_BUILD_HANDOFF.md` 和 `docs/AGENT_RESEARCH.md`。
- 没有 `miniclaw/` 包、没有 `pyproject.toml`、没有任何测试、没有 git 仓库。
- 环境：Python 3.11.0；`httpx 0.28.1` 已安装；`pytest`、`fastapi` 未安装。

这与交接文档自述一致：“本文用于在新项目中继续实现自己的 Python Agent……不代表已经完成代码实现”。MiniClaw 参考源码留在原仓库，没有随文档带入本工作区。

因此本计划按“从零构建、按文档描述的 API 形状保持兼容”制定。如果你随后把真实 `miniclaw/` 源码放进 `project/`，切片 S0–S2 改为“在现有代码上重构”，S3–S6 的接口设计（AgentRuntime / RunState / SessionStore / ConversationService / build_runtime）不变，只调整落点文件。

## 1. 现状观察

1. 入口缺失：没有可运行的项目骨架，CLI/Web/Runtime 均不存在。
2. 文档记录的已知问题（将在实现中直接避免并用测试锁定）：
   - Web 请求共享 Agent 实例的 `history`，多用户会话串线；
   - CLI/Web 默认不接技能与记忆；
   - 技能与记忆示例各自复制了一套执行循环；
   - 无 SessionStore / RunState / 恢复机制。
3. 高风险工具（shell、python_eval、文件、网络）只有应用层检查，不能当安全隔离。

## 2. 根因假设与验证方式

**Web history 串线的根因假设**：Web 网关在模块级/应用级持有一个单例 `Agent`（或 `ReActAgent`），其 `history` 列表随请求被追加修改；所有请求共享同一实例，因此用户 A 的消息、工具结果会进入用户 B 的上下文。验证方式分两层：

- L1（服务层）：用 ScriptedModel 驱动两个会话交错对话，断言第二次请求时模型实际收到的 messages 只包含各自会话的历史（ScriptedModel 记录每次收到的输入即可断言）。
- L2（Web 层）：FastAPI TestClient 以不同身份（tenant/user）调用 `/chat`，相同或不同 `session_id`，断言返回与后续上下文互不串线。

由于本工作区是从零构建，“修复”体现为：运行时无会话内状态（stateless runtime + 显式 RunState），持久化按 `tenant_id/user_id/session_id` 三键隔离，并用上述测试锁定。并发写 SQLite 的正确性用 WAL 模式 + 多线程/asyncio 交错测试验证。

**持久化恢复的验证**：写入会话 → 丢弃内存对象 → 重新打开 store（模拟进程重启）→ 继续对话 → 断言模型第二次收到的上下文包含重启前的消息。

**工具循环的验证**：ScriptedModel 按脚本返回（1）tool_call（2）final answer，断言：工具恰好执行一次、tool 结果以正确 role/`tool_call_id` 回填、step/usage 计数正确、工具抛异常时 RunState 进入 FAILED 且带结构化错误、超过 max_steps 时终止。

## 3. 目标 API（锁定形状，供评审）

```python
# 领域对象
Message(role, content, tool_calls, tool_call_id)          # OpenAI 风格
ToolCall(id, name, arguments)                             # arguments 为 JSON 字符串
Usage(prompt_tokens, completion_tokens)
ModelResponse(message, usage, finish_reason)

# 模型协议（async，测试注入 ScriptedModel，不访问真实 LLM）
class ModelClient(Protocol):
    async def complete(self, messages: list[Message], tools: list[ToolSpec]) -> ModelResponse: ...

# 运行状态
class RunStatus(str, Enum): RUNNING / COMPLETED / FAILED   # PAUSED 预留给第二阶段审批
@dataclass
class RunState:
    run_id, tenant_id, user_id, session_id
    step: int
    status: RunStatus
    messages: list[Message]
    usage: Usage
    tool_calls_used: int
    error: str | None
    updated_at: float

# 运行时（无状态：状态显式传入传出，可被任意入口复用）
class AgentRuntime:
    def __init__(self, model, tools: ToolRegistry, limits: RunLimits, on_event: Callable | None = None): ...
    async def run(self, state: RunState, user_input: str) -> RunState

# 会话存储（同步 SQLite，三键隔离；ConversationService 提供 async 封装）
class SessionStore(Protocol): ...
class SQLiteSessionStore:
    def append_messages(self, tenant_id, user_id, session_id, messages) -> None: ...
    def get_messages(self, tenant_id, user_id, session_id) -> list[Message]: ...
    def list_sessions(self, tenant_id, user_id) -> list[SessionMeta]: ...

class ConversationService:
    async def chat(self, *, tenant_id, user_id, session_id, user_input) -> RunState
    # 内部：load → runtime.run → save，全程按三键过滤

# 唯一组装入口（CLI/Web/Skill 都从这里拿运行时）
@dataclass
class RuntimeBundle: runtime, service, store, skills
def build_runtime(config: AgentConfig, *, model=None, store=None) -> RuntimeBundle
# model/store 注入点专供测试：ScriptedModel + 临时 SQLite 文件
```

- `RunLimits`：本阶段仅 `max_steps`、`tool_timeout_s`、`max_tool_calls`；token 预算、取消、审批留给第二阶段。
- 事件：仅一个轻量 `on_event` 回调（RunStarted / ModelRequested / ToolCalled / ToolReturned / RunCompleted / RunFailed），用于测试断言与最小观测；不做 EventStore。

## 4. 切片计划（每个切片完成即运行对应测试）

### S0 项目骨架
- 新增：`pyproject.toml`（依赖：fastapi、uvicorn、httpx；dev：pytest、pytest-asyncio）、`miniclaw/__init__.py`、`tests/conftest.py`、`git init` + 首次提交。
- 测试：`tests/test_sanity.py`（导入与版本冒烟）。
- 安装：`pip install -e ".[dev]"`。

### S1 LLM 域模型与模型协议
- 新增：`miniclaw/llm/messages.py`、`miniclaw/llm/base.py`（协议）、`miniclaw/llm/openai_compat.py`（httpx AsyncClient 适配器）、`miniclaw/llm/scripted.py`（ScriptedModel：脚本队列 + 请求记录）。
- 测试：`tests/test_llm_adapter.py`（用 `httpx.MockTransport` 断言请求体与响应归一化，零网络）、`tests/test_scripted_model.py`。

### S2 工具协议与内置工具
- 新增：`miniclaw/tools/base.py`（Tool 协议：name/description/parameters/`async execute`）、`miniclaw/tools/registry.py`、`miniclaw/tools/builtin.py`（echo、current_time、calculator 三个纯函数安全工具；shell/python_eval/file/web 仅保留接口与显式 `enabled=False`，不做黑名单式“隔离”，真正的 sandbox 属后续阶段）。
- 测试：`tests/test_tools.py`（注册/查重/schema、未知工具报错、禁用工具不可执行）。

### S3 AgentRuntime + RunState（核心切片）
- 新增：`miniclaw/runtime/state.py`、`miniclaw/runtime/limits.py`、`miniclaw/runtime/events.py`、`miniclaw/runtime/loop.py`（`AgentRuntime.run`：用户消息 → [模型 → 工具执行 → 结果回填]* → 终答）。
- 测试：`tests/test_runtime_tool_loop.py`：tool_call→final 主路径；多工具并行调用；工具异常→FAILED+结构化 error；max_steps 超限；usage/step/event 序列断言。

### S4 SQLite SessionStore + ConversationService
- 新增：`miniclaw/session/store.py`（WAL、三键主键、消息按 seq 有序）、`miniclaw/session/service.py`。
- 测试：`tests/test_session_store.py`（三键隔离：同 session_id 不同 tenant/user 互不可见；重开文件恢复；append 顺序）；`tests/test_conversation_isolation.py`（asyncio 交错两会话，ScriptedModel 记录的输入互不串线；重启恢复后继续对话）。

### S5 build_runtime 统一组装 + CLI/Web/Skill 接入
- 新增：`miniclaw/runtime/factory.py`（`build_runtime`）、`miniclaw/gateway/web.py`（`create_app(bundle)`；`POST /v1/chat`、`GET /v1/sessions/{id}/messages`；身份取 header/body，缺省 tenant=`default`）、`miniclaw/gateway/cli.py`（REPL + 一次性 `--message` 模式）、`miniclaw/skills/`（SKILL.md front-matter 解析 + `SkillAsTool`：子任务通过**同一个** AgentRuntime 实例执行，propagate tenant/user，子会话命名空间 `session_id + "/skill/<name>"`，注册表排除自身防递归）。
- 测试：`tests/test_web_api.py`（两用户同 session_id 隔离；重启 app 同 db 后历史仍在）；`tests/test_cli.py`；`tests/test_skill_as_tool.py`（技能走 runtime：以模型调用计数/事件序列证明没有第二套循环）。

### S6 验收与文档
- 新增：`tests/test_acceptance_phase1.py`（逐条映射交接文档验收标准）、`README.md`。
- 更新：`docs/PHASE1_PLAN.md` 勾选完成项。
- 全量回归：`pytest -q`。

## 5. 兼容性与边界

- 公开 API 以文档描述为准：保留 `miniclaw` 包名与 llm/tools/skills/memory/gateway 分层，`ReActAgent` 语义由 `AgentRuntime.run` 承接（若后续放入真实 MiniClaw 源码，提供薄适配而不是改调用方）。
- 本阶段不做：LangGraph/AG2、多 Agent 网络、分布式队列、向量记忆重构、Docker sandbox、token 预算/取消/审批（接口预留 RunStatus.PAUSED 与 RunLimits 扩展位）。
- 测试零真实 LLM：ScriptedModel + httpx.MockTransport + 临时目录 SQLite。

## 6. 风险

| 风险 | 处置 |
|---|---|
| 真实 MiniClaw 源码稍后放入，与新建骨架冲突 | S3 前是纯新增，冲突面小；届时把 S1–S2 改为对齐现有实现的适配层 |
| sqlite 并发写（多线程测试） | WAL + 每操作独立连接 + 短事务；测试覆盖交错写入 |
| async/sync 边界混乱 | 模型与工具协议 async；SessionStore 同步、由 ConversationService 以 `asyncio.to_thread` 封装 |
