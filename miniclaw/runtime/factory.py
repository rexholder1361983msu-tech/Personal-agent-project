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
from miniclaw.runtime.loop import AgentRuntime
from miniclaw.session.service import ConversationService
from miniclaw.session.store import SessionStore, SQLiteSessionStore
from miniclaw.skills.loader import load_skills
from miniclaw.skills.skill_tool import SkillAsTool
from miniclaw.tools.builtin import default_tools
from miniclaw.tools.registry import ToolRegistry


@dataclass
class RuntimeBundle:
    """一次组装的全部组件。网关只应使用 service 与 config。"""

    config: AgentConfig
    model: ModelClient
    tools: ToolRegistry
    runtime: AgentRuntime
    store: SessionStore
    service: ConversationService


def build_runtime(
    config: AgentConfig,
    *,
    model: ModelClient | None = None,
    store: SessionStore | None = None,
    skill_dir: str | Path | None = None,
    on_event: EventListener | None = None,
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

    registry = ToolRegistry()
    for tool in default_tools():
        registry.register(tool)
    runtime = AgentRuntime(
        model=resolved_model,
        tools=registry,
        limits=RunLimits(max_steps=config.max_steps),
        on_event=on_event,
    )

    if skill_dir is not None:
        for skill in load_skills(Path(skill_dir)):
            registry.register(SkillAsTool(skill, lambda: runtime))

    service = ConversationService(resolved_store, runtime=runtime)
    return RuntimeBundle(
        config=config,
        model=resolved_model,
        tools=registry,
        runtime=runtime,
        store=resolved_store,
        service=service,
    )
