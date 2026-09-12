"""build_runtime：唯一的运行时组装入口。

CLI/Web/Skill/后台任务都从这里获取 RuntimeBundle，保证
"同一条 Runtime 路径"。model/store 注入点专供测试
（ScriptedModel + 临时 SQLite 文件）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from miniclaw.config import AgentConfig
from miniclaw.llm.base import ModelClient
from miniclaw.llm.openai_compat import OpenAICompatModel
from miniclaw.runtime.events import EventListener
from miniclaw.runtime.limits import RunLimits
from miniclaw.runtime.loop import AgentRuntime, ApprovalGate
from miniclaw.session.event_store import SQLiteEventStore
from miniclaw.session.run_store import SQLiteRunStore
from miniclaw.session.service import ConversationService
from miniclaw.session.store import SessionStore, SQLiteSessionStore
from miniclaw.skills.loader import load_skills
from miniclaw.skills.skill_tool import SkillAsTool
from miniclaw.tools.builtin import default_tools
from miniclaw.tools.executor import LocalSubprocessExecutor
from miniclaw.tools.policy import ToolMode, ToolPolicy, ToolRule
from miniclaw.tools.registry import ToolRegistry
from miniclaw.tools.unsafe import PythonEvalTool, ShellTool


@dataclass
class RuntimeBundle:
    """一次组装的全部组件。网关只应使用 service 与 config。"""

    config: AgentConfig
    model: ModelClient
    tools: ToolRegistry
    runtime: AgentRuntime
    store: SessionStore
    service: ConversationService
    event_store: SQLiteEventStore | None = None
    run_store: SQLiteRunStore | None = None


def build_runtime(
    config: AgentConfig,
    *,
    model: ModelClient | None = None,
    store: SessionStore | None = None,
    skill_dir: str | Path | None = None,
    on_event: EventListener | None = None,
    event_store: SQLiteEventStore | None = None,
    run_store: SQLiteRunStore | None = None,
    approval_gate: ApprovalGate | None = None,
    tool_policy: ToolPolicy | None = None,
    redact_events: bool = False,
    enable_shell_tools: bool = False,
) -> RuntimeBundle:
    resolved_model = (
        model
        if model is not None
        else OpenAICompatModel(
            api_base=config.api_base,
            api_key=config.api_key,
            model=config.model,
            temperature=config.temperature,
        )
    )
    resolved_store = store if store is not None else SQLiteSessionStore(config.db_path)
    # 事件/运行元数据持久化默认只对生产组装（未注入 store）开启，与 store
    # 同库不同连接；测试注入 store 时保持无附加库，除非显式传入。
    resolved_event_store = (
        event_store
        if event_store is not None
        else (
            None
            if store is not None
            else SQLiteEventStore(config.db_path, redact=redact_events)
        )
    )
    resolved_run_store = (
        run_store
        if run_store is not None
        else (None if store is not None else SQLiteRunStore(config.db_path))
    )

    listeners: list[EventListener] = []
    if on_event is not None:
        listeners.append(on_event)
    if resolved_event_store is not None:
        listeners.append(resolved_event_store.listener())
    combined: EventListener | None = None
    if listeners:
        registered = tuple(listeners)

        def combined(event, _listeners=registered) -> None:
            for listener in _listeners:
                listener(event)

    registry = ToolRegistry()
    for tool in default_tools():
        registry.register(tool)
    resolved_tool_policy = tool_policy
    if enable_shell_tools:
        # 本地受限执行器：资源约束不是安全隔离——正因如此默认策略强制审批。
        # docker 就绪后换成 DockerSandboxExecutor（不可用时构造抛 ToolDisabledError）。
        executor = LocalSubprocessExecutor()
        registry.register(ShellTool(executor))
        registry.register(PythonEvalTool(executor))
        if resolved_tool_policy is None:
            resolved_tool_policy = ToolPolicy(
                {
                    "shell": ToolRule(mode=ToolMode.NEEDS_APPROVAL),
                    "python_eval": ToolRule(mode=ToolMode.NEEDS_APPROVAL),
                }
            )
    runtime = AgentRuntime(
        model=resolved_model,
        tools=registry,
        limits=RunLimits(max_steps=config.max_steps),
        on_event=combined,
        approval_gate=approval_gate,
        tool_policy=resolved_tool_policy,
    )

    if skill_dir is not None:
        for skill in load_skills(Path(skill_dir)):
            registry.register(SkillAsTool(skill, lambda: runtime))

    service = ConversationService(resolved_store, runtime=runtime, run_store=resolved_run_store)
    return RuntimeBundle(
        config=config,
        model=resolved_model,
        tools=registry,
        runtime=runtime,
        store=resolved_store,
        service=service,
        event_store=resolved_event_store,
        run_store=resolved_run_store,
    )
