"""ToolPolicy 测试：判定收敛（deny/needs_approval）、执行约束、审计脱敏。

验收：无权限 / 未审批的工具一次都不执行（ScriptedModel + 计数工具断言
calls == 0）；审计链默认保留负载，redact_events 开启后不留敏感字段。
"""

from __future__ import annotations

import pytest

from miniclaw.config import AgentConfig
from miniclaw.llm.scripted import ScriptedModel, text_response, tool_call_response
from miniclaw.runtime.events import RunEventType
from miniclaw.runtime.limits import RunLimits
from miniclaw.runtime.loop import AgentRuntime, final_reply
from miniclaw.runtime.state import RunState, RunStatus
from miniclaw.runtime.factory import build_runtime
from miniclaw.session.run_store import SQLiteRunStore
from miniclaw.session.store import SQLiteSessionStore
from miniclaw.tools.policy import ToolMode, ToolPolicy, ToolRule
from miniclaw.tools.registry import ToolRegistry
from tests.test_runtime_tool_loop import RecordingTool


def make_runtime(model, tools=(), policy=None, gate=None, on_event=None, limits=None) -> AgentRuntime:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return AgentRuntime(
        model=model,
        tools=registry,
        limits=limits,
        on_event=on_event,
        approval_gate=gate,
        tool_policy=policy,
    )


def named_tool(name: str, **kwargs) -> RecordingTool:
    tool = RecordingTool(**kwargs)
    tool.name = name
    return tool


def make_state() -> RunState:
    return RunState(tenant_id="t", user_id="u", session_id="s")


# ── 判定收敛：mode 决定执行 / 暂停 / 拒绝 ─────────────────────────────


async def test_default_policy_allows_and_executes():
    echo = named_tool("echo")
    model = ScriptedModel(tool_call_response("c1", "echo", "{}"), text_response("done"))
    runtime = make_runtime(model, [echo], policy=ToolPolicy())

    state = await runtime.run(make_state(), "go")

    assert state.status is RunStatus.COMPLETED
    assert len(echo.calls) == 1
    assert final_reply(state) == "done"


async def test_deny_backfills_error_without_execution():
    dangerous = named_tool("dangerous")
    model = ScriptedModel(tool_call_response("c1", "dangerous", "{}"), text_response("ok"))
    policy = ToolPolicy({"dangerous": ToolRule(mode=ToolMode.DENY)})
    runtime = make_runtime(model, [dangerous], policy=policy)

    state = await runtime.run(make_state(), "go")

    assert state.status is RunStatus.COMPLETED
    assert dangerous.calls == []  # 验收：无权限的工具一次都不执行
    assert state.messages[2].content == "tool_error: tool 'dangerous' is not permitted"
    assert state.messages[2].tool_call_id == "c1"
    assert final_reply(state) == "ok"


async def test_needs_approval_pauses_via_policy_and_resumes():
    dangerous = named_tool("dangerous")
    model = ScriptedModel(tool_call_response("c1", "dangerous", "{}"), text_response("done"))
    policy = ToolPolicy({"dangerous": ToolRule(mode=ToolMode.NEEDS_APPROVAL)})
    runtime = make_runtime(model, [dangerous], policy=policy)

    state = await runtime.run(make_state(), "go")
    assert state.status is RunStatus.PAUSED
    assert dangerous.calls == []  # 验收：未审批的工具一次都不执行
    run_id = state.run_id

    state = await runtime.run(state, None, approval=True)

    assert state.status is RunStatus.COMPLETED
    assert state.run_id == run_id
    assert len(dangerous.calls) == 1
    assert final_reply(state) == "done"


async def test_deny_takes_precedence_over_approval_gate():
    dangerous = named_tool("dangerous")
    model = ScriptedModel(tool_call_response("c1", "dangerous", "{}"), text_response("ok"))
    policy = ToolPolicy({"dangerous": ToolRule(mode=ToolMode.DENY)})
    runtime = make_runtime(model, [dangerous], policy=policy, gate=lambda *_: True)

    state = await runtime.run(make_state(), "go")

    assert state.status is RunStatus.COMPLETED  # 不是 PAUSED：deny 短路在审批之前
    assert dangerous.calls == []
    assert "not permitted" in state.messages[2].content


async def test_policy_allow_does_not_shadow_gate():
    # policy 未列名（allow）但 gate 命中 → 仍暂停：任一判定需审批即暂停
    echo = named_tool("echo")
    model = ScriptedModel(tool_call_response("c1", "echo", "{}"), text_response("done"))
    runtime = make_runtime(model, [echo], policy=ToolPolicy(), gate=lambda *_: True)

    state = await runtime.run(make_state(), "go")

    assert state.status is RunStatus.PAUSED
    assert echo.calls == []


async def test_default_deny_acts_as_allowlist():
    # default 收紧为 deny，规则表即白名单：只放行列名工具
    echo = named_tool("echo")
    blocked = named_tool("blocked")
    model = ScriptedModel(
        tool_call_response("c1", "blocked", "{}"),
        tool_call_response("c2", "echo", "{}"),
        text_response("done"),
    )
    policy = ToolPolicy({"echo": ToolRule()}, default=ToolRule(mode=ToolMode.DENY))
    runtime = make_runtime(model, [blocked, echo], policy=policy)

    state = await runtime.run(make_state(), "go")

    assert state.status is RunStatus.COMPLETED
    assert blocked.calls == []
    assert "not permitted" in state.messages[2].content
    assert len(echo.calls) == 1


# ── 执行约束：每工具 timeout_s / output_limit ────────────────────────


async def test_rule_timeout_tightens_global_default():
    slow = named_tool("slow", delay_s=0.05)
    model = ScriptedModel(tool_call_response("c1", "slow", "{}"), text_response("ok"))
    policy = ToolPolicy({"slow": ToolRule(timeout_s=0.01)})
    runtime = make_runtime(model, [slow], policy=policy)  # 全局默认 10s

    state = await runtime.run(make_state(), "go")

    assert state.status is RunStatus.COMPLETED
    assert state.messages[2].content == "tool_error: timeout after 0.01s"


async def test_global_timeout_wins_when_rule_is_looser():
    slow = named_tool("slow", delay_s=0.05)
    model = ScriptedModel(tool_call_response("c1", "slow", "{}"), text_response("ok"))
    policy = ToolPolicy({"slow": ToolRule(timeout_s=30.0)})
    runtime = make_runtime(
        model, [slow], policy=policy, limits=RunLimits(tool_timeout_s=0.01)
    )

    state = await runtime.run(make_state(), "go")

    assert state.status is RunStatus.COMPLETED
    assert state.messages[2].content == "tool_error: timeout after 0.01s"


async def test_output_limit_truncates_with_marker():
    verbose = named_tool("verbose", result="x" * 50)
    events = []
    model = ScriptedModel(tool_call_response("c1", "verbose", "{}"), text_response("ok"))
    policy = ToolPolicy({"verbose": ToolRule(output_limit=10)})
    runtime = make_runtime(model, [verbose], policy=policy, on_event=events.append)

    state = await runtime.run(make_state(), "go")

    expected = "x" * 10 + "...[truncated]"
    assert state.messages[2].content == expected
    returned = next(e for e in events if e.type is RunEventType.TOOL_RETURNED)
    assert returned.data["content"] == expected  # 审计看到的就是截断后内容
    assert returned.data["ok"] is True


def test_rule_rejects_invalid_constraint_values():
    with pytest.raises(ValueError):
        ToolRule(timeout_s=0)
    with pytest.raises(ValueError):
        ToolRule(output_limit=0)


# ── 组装入口与审计脱敏 ───────────────────────────────────────────────


async def test_shell_tools_absent_by_default(db_path):
    bundle = build_runtime(
        AgentConfig(), model=ScriptedModel(text_response("x")), store=SQLiteSessionStore(db_path)
    )
    assert "shell" not in bundle.tools
    assert "python_eval" not in bundle.tools
    bundle.store.close()


async def test_enable_shell_tools_forces_default_approval_policy(db_path):
    model = ScriptedModel(
        tool_call_response("c1", "python_eval", '{"code": "print(21+21)"}'),
        text_response("done"),
    )
    bundle = build_runtime(
        AgentConfig(),
        model=model,
        store=SQLiteSessionStore(db_path),
        run_store=SQLiteRunStore(db_path),
        enable_shell_tools=True,
    )
    assert "python_eval" in bundle.tools and "shell" in bundle.tools

    state1 = await bundle.service.chat(
        tenant_id="t", user_id="u", session_id="s", user_input="calc"
    )
    assert state1.status is RunStatus.PAUSED  # 默认策略强制审批：未批准不执行

    state2 = await bundle.service.resume(
        tenant_id="t", user_id="u", session_id="s", run_id=state1.run_id, approval=True
    )
    assert state2.status is RunStatus.COMPLETED
    tool_msg = next(m for m in state2.messages if m.tool_call_id == "c1")
    assert tool_msg.content.strip() == "42"  # 真实本地 python 子进程的输出
    assert final_reply(state2) == "done"
    bundle.store.close()
    bundle.run_store.close()


async def test_explicit_policy_overrides_enable_default(db_path):
    model = ScriptedModel(
        tool_call_response("c1", "python_eval", '{"code": "print(1)"}'), text_response("ok")
    )
    bundle = build_runtime(
        AgentConfig(),
        model=model,
        store=SQLiteSessionStore(db_path),
        enable_shell_tools=True,
        tool_policy=ToolPolicy({"python_eval": ToolRule(mode=ToolMode.DENY)}),
    )

    state = await bundle.service.chat(
        tenant_id="t", user_id="u", session_id="s", user_input="go"
    )

    assert state.status is RunStatus.COMPLETED  # 显式策略生效：deny 直接拒绝，不暂停
    assert "not permitted" in state.messages[2].content
    bundle.store.close()


async def test_policy_pauses_via_full_stack_and_resumes(db_path):
    model = ScriptedModel(
        tool_call_response("c1", "echo", '{"text": "hi"}'), text_response("done")
    )
    bundle = build_runtime(
        AgentConfig(),
        model=model,
        store=SQLiteSessionStore(db_path),
        run_store=SQLiteRunStore(db_path),
        tool_policy=ToolPolicy({"echo": ToolRule(mode=ToolMode.NEEDS_APPROVAL)}),
    )

    state1 = await bundle.service.chat(
        tenant_id="t", user_id="u", session_id="s", user_input="hello"
    )
    assert state1.status is RunStatus.PAUSED

    state2 = await bundle.service.resume(
        tenant_id="t", user_id="u", session_id="s", run_id=state1.run_id, approval=True
    )
    assert state2.status is RunStatus.COMPLETED
    assert final_reply(state2) == "done"
    bundle.store.close()
    bundle.run_store.close()


async def test_redact_events_flag_sanitizes_audit_trail(tmp_path):
    model = ScriptedModel(
        tool_call_response("c1", "echo", '{"text": "secret"}'), text_response("done")
    )
    bundle = build_runtime(
        AgentConfig(db_path=str(tmp_path / "audit.db")), model=model, redact_events=True
    )

    state = await bundle.service.chat(
        tenant_id="t", user_id="u", session_id="s", user_input="hello"
    )
    assert state.status is RunStatus.COMPLETED
    assert bundle.event_store is not None
    assert bundle.event_store.flush()

    events = bundle.event_store.get_events("t", "u", "s", state.run_id)
    called = next(e for e in events if e.type is RunEventType.TOOL_CALLED)
    returned = next(e for e in events if e.type is RunEventType.TOOL_RETURNED)
    assert called.data == {"name": "echo", "call_id": "c1"}  # arguments 已移除
    assert returned.data == {"name": "echo", "call_id": "c1", "ok": True}  # content 已移除
    bundle.store.close()
    bundle.event_store.close()
    bundle.run_store.close()


async def test_events_keep_payload_by_default(tmp_path):
    model = ScriptedModel(
        tool_call_response("c1", "echo", '{"text": "hi"}'), text_response("done")
    )
    bundle = build_runtime(AgentConfig(db_path=str(tmp_path / "audit.db")), model=model)

    state = await bundle.service.chat(
        tenant_id="t", user_id="u", session_id="s", user_input="hello"
    )
    assert bundle.event_store is not None
    assert bundle.event_store.flush()

    events = bundle.event_store.get_events("t", "u", "s", state.run_id)
    called = next(e for e in events if e.type is RunEventType.TOOL_CALLED)
    assert called.data == {
        "name": "echo",
        "call_id": "c1",
        "arguments": '{"text": "hi"}',
    }
    bundle.store.close()
    bundle.event_store.close()
    bundle.run_store.close()
