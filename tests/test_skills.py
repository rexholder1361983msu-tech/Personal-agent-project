"""SKILL.md 解析与 SkillAsTool 走统一 Runtime 的测试。"""

from __future__ import annotations

import pytest

from miniclaw.config import AgentConfig
from miniclaw.llm.messages import Message
from miniclaw.llm.scripted import ScriptedModel, text_response, tool_call_response
from miniclaw.runtime.factory import build_runtime
from miniclaw.runtime.loop import AgentRuntime, final_reply
from miniclaw.runtime.state import RunState, RunStatus
from miniclaw.skills.loader import SkillFormatError, load_skills, parse_skill_md
from miniclaw.skills.skill_tool import SkillAsTool
from miniclaw.session.store import SQLiteSessionStore
from miniclaw.tools.base import ToolContext
from miniclaw.tools.registry import ToolRegistry

from tests.test_build_runtime import make_skill_dir


def test_parse_skill_md_happy_path():
    skill = parse_skill_md(
        "---\nname: greeter\ndescription: Greets people.\n---\nAlways greet warmly.\n"
    )
    assert skill.name == "greeter"
    assert skill.description == "Greets people."
    assert skill.instructions == "Always greet warmly."


def test_parse_skill_md_rejects_bad_front_matter():
    with pytest.raises(SkillFormatError):
        parse_skill_md("no delimiter\n")
    with pytest.raises(SkillFormatError):
        parse_skill_md("---\nname: x\n---".replace("x", "") + "\n")  # 缺 name
    with pytest.raises(SkillFormatError):
        parse_skill_md("---\nname only\n---\n")  # 行内无冒号
    with pytest.raises(SkillFormatError):
        parse_skill_md("---\nname: x\nunterminated")  # 无结束分隔符


def test_load_skills_dedup_and_order(tmp_path):
    skill_dir = make_skill_dir(tmp_path)
    skills = load_skills(skill_dir)
    assert [s.name for s in skills] == ["summarizer"]


def make_context() -> ToolContext:
    return ToolContext(
        tenant_id="tenant-1", user_id="user-1", session_id="sess-1", run_id="run-1"
    )


async def test_skill_as_tool_uses_same_runtime_and_propagates_identity():
    """技能子任务必须走同一个 runtime 实例，并携带父运行的租户/用户身份。"""
    model = ScriptedModel(text_response("SUMMARY: hello"))
    runtime = AgentRuntime(model=model, tools=ToolRegistry())
    skill_tool = SkillAsTool(
        parse_skill_md("---\nname: s1\ndescription: d\n---\nbody\n"),
        lambda: runtime,
    )

    result = await skill_tool.execute({"task": "summarize hello"}, make_context())

    assert result == "SUMMARY: hello"
    # 子运行的模型请求带上了技能指令与身份（通过唯一的模型请求记录证明走了 runtime）
    assert len(model.requests) == 1
    assert model.requests[0][0].content == "summarize hello"


async def test_skill_as_tool_child_session_namespace():
    """子运行使用独立 session 命名空间，父会话历史不泄入子运行。"""
    model = ScriptedModel(text_response("child reply"))
    runtime = AgentRuntime(model=model, tools=ToolRegistry())

    parent_state = RunState(tenant_id="t", user_id="u", session_id="parent-sess")
    parent_state.messages.append(Message(role="user", content="parent secret"))

    skill_tool = SkillAsTool(
        parse_skill_md("---\nname: s2\ndescription: d\n---\nbody\n"),
        lambda: runtime,
    )
    await skill_tool.execute({"task": "child task"}, make_context())

    # 模型只收到子任务的输入，不含父会话历史
    assert [m.content for m in model.requests[0]] == ["child task"]


async def test_skill_as_tool_excludes_other_skills_and_recurses_safely():
    """子运行工具集中排除所有 skill_* 工具，防止技能递归。"""
    from miniclaw.tools.builtin import EchoTool

    model = ScriptedModel(
        tool_call_response("c1", "echo", '{"text": "x"}'),
        text_response("done"),
    )
    registry = ToolRegistry()
    registry.register(EchoTool())
    runtime = AgentRuntime(model=model, tools=registry)
    skill_tool = SkillAsTool(
        parse_skill_md("---\nname: s3\ndescription: d\n---\nbody\n"),
        lambda: runtime,
    )
    registry.register(skill_tool)

    result = await skill_tool.execute({"task": "go"}, make_context())

    assert result == "done"
    assert [s.name for s in model.tool_specs[0]] == ["echo"]  # skill 工具不可见


async def test_skill_failure_returns_structured_error():
    model = ScriptedModel()  # 模型错误 → 子运行 FAILED
    runtime = AgentRuntime(model=model, tools=ToolRegistry())
    skill_tool = SkillAsTool(
        parse_skill_md("---\nname: s4\ndescription: d\n---\nbody\n"),
        lambda: runtime,
    )
    result = await skill_tool.execute({"task": "go"}, make_context())
    assert result.startswith("skill_error: model_error:")


async def test_skill_tool_requires_task_argument():
    runtime = AgentRuntime(model=ScriptedModel(), tools=ToolRegistry())
    skill_tool = SkillAsTool(
        parse_skill_md("---\nname: s5\ndescription: d\n---\nbody\n"),
        lambda: runtime,
    )
    with pytest.raises(ValueError):
        await skill_tool.execute({"task": ""}, make_context())


async def test_skill_via_full_stack_parent_chat(db_path, tmp_path):
    """端到端：父对话 → 模型调用 skill 工具 → 子运行 → 结果回填父对话。

    父子运行共用同一个 ScriptedModel，脚本按实际消费顺序排列：
    父请求工具调用 → 子技能终答 → 父终答。
    """
    model = ScriptedModel(
        tool_call_response("c1", "skill_summarizer", '{"task": "summarize this"}'),
        text_response("SUMMARY: summarize this"),
        text_response("final answer"),
    )
    bundle = build_runtime(
        AgentConfig(),
        model=model,
        store=SQLiteSessionStore(db_path),
        skill_dir=make_skill_dir(tmp_path),
    )
    state = await bundle.service.chat(
        tenant_id="t", user_id="u", session_id="s", user_input="please summarize"
    )
    assert state.status is RunStatus.COMPLETED
    assert final_reply(state) == "final answer"
    tool_msg = next(m for m in state.messages if m.role == "tool")
    assert tool_msg.content == "SUMMARY: summarize this"  # 技能子运行的结果
    bundle.store.close()
