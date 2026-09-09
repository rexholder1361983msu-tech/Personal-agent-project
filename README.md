# MiniClaw

轻量、可恢复、可观测的 Python Agent Runtime。第一阶段：统一执行路径 + 会话隔离 + 持久化恢复。

## 架构

```text
CLI / FastAPI Web / Skill(子任务)
        |
ConversationService        # tenant/user/session 三键隔离：load → run → save
        |
AgentRuntime               # 无状态 ReAct 循环：RunState 显式传入传出
        |
LLM (OpenAI-compatible) + ToolRegistry + SkillRegistry
        |
SQLiteSessionStore         # WAL、三键主键、seq 有序消息日志
```

关键性质：

- **无共享会话状态**：Runtime 不持有任何历史；每个请求从 SessionStore 加载自己的 RunState，彻底消除"共享 history 串线"。
- **单一组装入口**：`build_runtime(config)` 返回 `RuntimeBundle`，CLI/Web/Skill 全部复用同一个 `AgentRuntime` 实例。
- **确定性可测**：模型协议为 async `ModelClient`，测试注入 `ScriptedModel`（脚本队列 + 请求记录），零真实 LLM、零网络。

## 安装与运行

```bash
pip install -e ".[dev,web]"

# CLI（REPL；一次性用 --message）
MINICLAW_API_KEY=sk-... miniclaw --message "你好" --session-id demo

# Web
uvicorn miniclaw.gateway.web:app  # 需在入口文件中 create_app(build_runtime(AgentConfig.from_env()))
```

配置经环境变量 `MINICLAW_API_BASE / API_KEY / MODEL / MAX_STEPS / DB / DEFAULT_TENANT`（见 `AgentConfig.from_env`）。

## Web API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/chat` | `{session_id, user_id, message, tenant_id?}`；租户解析顺序：body > `X-Tenant-ID` 头 > `default` |
| GET | `/v1/sessions/{id}/messages?user_id=&tenant_id=` | 读取会话历史（三键过滤） |
| DELETE | `/v1/sessions/{id}?user_id=&tenant_id=` | 清空指定会话 |

## 技能

在目录中放置 `SKILL.md`（front matter 需 `name`/`description`），`build_runtime(..., skill_dir=...)` 会把每个技能注册为 `skill_<name>` 工具。技能子任务走**同一个** AgentRuntime，子会话命名空间为 `<session_id>/skills/<name>`，子运行不可见其他技能工具（防递归）。

## 内置工具与安全边界

安全工具：`echo`、`current_time`、`calculator`（AST 白名单求值，非 eval）。

高风险工具（`shell` / `python_eval` / `file_read` / `file_write` / `web_fetch`）仅保留接口，默认**禁用**（执行即抛 `ToolDisabledError`）。字符串黑名单不作为安全边界；真正的 sandbox 与权限策略属于第二阶段。

## 失败语义

- 模型异常、超 `max_steps` / `max_tool_calls` → `RunStatus.FAILED` + 结构化 `error`（`model_error:` / `max_steps_exceeded:` / …）。
- 工具异常（含未知工具、参数解析失败、超时）→ 不终止运行，以 `tool_error:` 消息回填给模型。
- 失败运行的已产生消息仍会持久化。

## 测试

```bash
python -m pytest -q
```

覆盖：会话三键隔离（store/service/web 三层）、并发写、重启恢复、工具循环全路径（回填/异常/超时/限额）、事件序列、适配器协议形状（MockTransport）、CLI、技能统一路径，以及逐条映射交接文档验收标准的 `tests/test_acceptance_phase1.py`。

## 阶段状态

- 第一阶段（统一 Runtime / RunState / SQLite SessionStore / 会话隔离 / 确定性测试）：**已完成**，详见 `docs/PHASE1_PLAN.md`。
- 第二阶段候选：RunLimits 完整化与取消、事件 EventStore、RunState 持久化与审批恢复（PAUSED/同 run_id 恢复）、ToolPolicy 与审计、Docker sandbox、OpenTelemetry、长历史与记忆治理。
