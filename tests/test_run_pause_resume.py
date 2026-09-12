"""审批暂停/恢复测试：Runtime 分支、跨进程恢复与全栈验收。

验收路径：暂停 → （模拟进程重启）→ 批准 → 同 run_id 继续 → 完成；
拒绝路径以 tool_error 回填后完成，工具一次都不执行。
"""

from __future__ import annotations

import pytest

from miniclaw.config import AgentConfig
from miniclaw.llm.base import ModelResponse
from miniclaw.llm.messages import Message, ToolCall, Usage
from miniclaw.llm.scripted import ScriptedModel, text_response, tool_call_response
from miniclaw.runtime.events import RunEventType
from miniclaw.runtime.factory import build_runtime
from miniclaw.runtime.loop import AgentRuntime, final_reply
from miniclaw.runtime.state import RunState, RunStatus
from miniclaw.session.event_store import SQLiteEventStore
from miniclaw.session.run_store import SQLiteRunStore
from miniclaw.session.store import SQLiteSessionStore
from miniclaw.tools.registry import ToolRegistry
from tests.test_build_runtime import make_skill_dir
from tests.test_runtime_tool_loop import RecordingTool


def make_runtime(model, tools=(), gate=None, on_event=None) -> AgentRuntime:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return AgentRuntime(
        model=model, tools=registry, on_event=on_event, approval_gate=gate
    )


def named_tool(name: str, **kwargs) -> RecordingTool:
    tool = RecordingTool(**kwargs)
    tool.name = name
    return tool


def multi_call_response(*calls: ToolCall) -> ModelResponse:
    return ModelResponse(
        message=Message(role="assistant", content=None, tool_calls=list(calls)),
        usage=Usage(prompt_tokens=1, completion_tokens=1),
        finish_reason="tool_calls",
    )


def make_state() -> RunState:
    return RunState(tenant_id="t", user_id="u", session_id="s")


# ── Runtime 层：暂停与恢复分支 ────────────────────────────────────────


async def test_pause_returns_paused_state_with_pending_call():
    dangerous = named_tool("dangerous")
    model = ScriptedModel(
        tool_call_response("c1", "dangerous", '{"cmd": "ls"}'), text_response("never")
    )
    gate_calls = []

    def gate(name, arguments, context):
        gate_calls.append((name, arguments, context.session_id))
        return True

    runtime = make_runtime(model, [dangerous], gate=gate)

    state = await runtime.run(make_state(), "go")

    assert state.status is RunStatus.PAUSED
    assert state.error is None
    assert dangerous.calls == []  # 未审批的工具一次都不执行
    assert [m.role for m in state.messages] == ["user", "assistant"]
    assert state.pending_approval == {
        "call": {"id": "c1", "name": "dangerous", "arguments": '{"cmd": "ls"}'},
        "remaining": [],
    }
    assert gate_calls == [("dangerous", '{"cmd": "ls"}', "s")]
    assert state.tool_calls_used == 1


async def test_resume_approved_executes_tool_and_completes():
    dangerous = named_tool("dangerous")
    model = ScriptedModel(
        tool_call_response("c1", "dangerous", '{"cmd": "ls"}'), text_response("done")
    )
    runtime = make_runtime(model, [dangerous], gate=lambda *_: True)

    state = await runtime.run(make_state(), "go")
    assert state.status is RunStatus.PAUSED
    run_id = state.run_id

    state = await runtime.run(state, None, approval=True)

    assert state.status is RunStatus.COMPLETED
    assert state.run_id == run_id
    assert dangerous.calls == [({"cmd": "ls"}, "t")]
    assert [m.role for m in state.messages] == ["user", "assistant", "tool", "assistant"]
    assert state.messages[2].content == "ECHOED"
    assert state.messages[2].tool_call_id == "c1"
    assert final_reply(state) == "done"


async def test_resume_rejected_backfills_tool_error():
    dangerous = named_tool("dangerous")
    model = ScriptedModel(
        tool_call_response("c1", "dangerous", "{}"), text_response("understood")
    )
    runtime = make_runtime(model, [dangerous], gate=lambda *_: True)

    state = await runtime.run(make_state(), "go")
    state = await runtime.run(state, None, approval=False)

    assert state.status is RunStatus.COMPLETED
    assert dangerous.calls == []  # 拒绝路径工具一次都不执行
    assert state.messages[2].content == "tool_error: tool 'dangerous' was not approved"
    assert state.messages[2].tool_call_id == "c1"
    assert final_reply(state) == "understood"


async def test_resume_without_decision_or_with_new_input_is_rejected():
    model = ScriptedModel(
        tool_call_response("c1", "dangerous", "{}"), text_response("x")
    )
    runtime = make_runtime(model, [named_tool("dangerous")], gate=lambda *_: True)
    state = await runtime.run(make_state(), "go")
    assert state.pending_approval is not None

    with pytest.raises(ValueError):
        await runtime.run(state, None)  # 缺 approval 决定
    with pytest.raises(ValueError):
        await runtime.run(state, "new input", approval=True)  # 挂起期间禁止新输入


async def test_approval_without_pending_is_rejected():
    runtime = make_runtime(ScriptedModel(text_response("ok")), [], gate=lambda *_: True)
    with pytest.raises(ValueError):
        await runtime.run(make_state(), "go", approval=True)


async def test_pause_after_partial_execution_defers_remaining_calls():
    echo = named_tool("echo", result="ECHOED")
    dangerous = named_tool("dangerous")
    model = ScriptedModel(
        multi_call_response(
            ToolCall(id="c1", name="echo", arguments='{"text": "hi"}'),
            ToolCall(id="c2", name="dangerous", arguments="{}"),
        ),
        text_response("done"),
    )
    runtime = make_runtime(
        model, [echo, dangerous], gate=lambda name, *_: name == "dangerous"
    )

    state = await runtime.run(make_state(), "go")

    assert state.status is RunStatus.PAUSED
    assert len(echo.calls) == 1  # 前面的普通调用已执行
    assert dangerous.calls == []
    assert [m.role for m in state.messages] == ["user", "assistant", "tool"]
    assert state.messages[2].tool_call_id == "c1"
    assert state.pending_approval == {
        "call": {"id": "c2", "name": "dangerous", "arguments": "{}"},
        "remaining": [],
    }

    state = await runtime.run(state, None, approval=True)

    assert state.status is RunStatus.COMPLETED
    assert len(echo.calls) == 1 and len(dangerous.calls) == 1
    # 每个 tool_call 都有结果回填
    assert [m.role for m in state.messages] == [
        "user", "assistant", "tool", "tool", "assistant",
    ]
    assert state.messages[3].tool_call_id == "c2"
    assert state.step == 2
    assert state.tool_calls_used == 2


async def test_event_chain_across_pause_and_resume():
    dangerous = named_tool("dangerous")
    events = []
    model = ScriptedModel(
        tool_call_response("c1", "dangerous", "{}"), text_response("done")
    )
    runtime = make_runtime(model, [dangerous], gate=lambda *_: True, on_event=events.append)

    state = await runtime.run(make_state(), "go")
    state = await runtime.run(state, None, approval=True)

    assert [e.type for e in events] == [
        RunEventType.RUN_STARTED,
        RunEventType.MODEL_REQUESTED,
        RunEventType.TOOL_CALLED,
        RunEventType.RUN_PAUSED,
        RunEventType.TOOL_RETURNED,
        RunEventType.MODEL_REQUESTED,
        RunEventType.RUN_COMPLETED,
    ]
    assert all(e.run_id == state.run_id for e in events)
    # 挂起时即发 TOOL_CALLED（P1 语义：未执行的调用也发），恢复只补 TOOL_RETURNED；
    # RUN_PAUSED 携带挂起的调用供跨进程接链
    assert events[2].data == {"name": "dangerous", "call_id": "c1", "arguments": "{}"}
    assert events[3].data == {"name": "dangerous", "call_id": "c1", "arguments": "{}"}
    assert events[4].data["ok"] is True


# ── ConversationService / 全栈：跨进程恢复验收 ────────────────────────


def make_bundle(db_path, model, event_store=None, gate=lambda *_: True):
    return build_runtime(
        AgentConfig(),
        model=model,
        store=SQLiteSessionStore(db_path),
        event_store=event_store,
        run_store=SQLiteRunStore(db_path),
        approval_gate=gate,
    )


async def test_full_stack_pause_restart_approve_resume(db_path):
    model = ScriptedModel(
        tool_call_response("c1", "echo", '{"text": "hi"}'), text_response("done")
    )
    es1 = SQLiteEventStore(db_path)
    bundle1 = make_bundle(db_path, model, event_store=es1)

    state1 = await bundle1.service.chat(
        tenant_id="t", user_id="u", session_id="s", user_input="hello"
    )
    assert state1.status is RunStatus.PAUSED
    run_id = state1.run_id
    assert bundle1.store.count_messages("t", "u", "s") == 2  # 暂停态消息已落库
    assert es1.flush()

    # ── 进程重启：全部换新连接 ──
    bundle1.store.close()
    bundle1.run_store.close()
    es1.close()
    es2 = SQLiteEventStore(db_path)
    bundle2 = make_bundle(db_path, model, event_store=es2)

    # 暂停未决期间同会话新消息被拒绝
    with pytest.raises(ValueError):
        await bundle2.service.chat(
            tenant_id="t", user_id="u", session_id="s", user_input="again"
        )

    state2 = await bundle2.service.resume(
        tenant_id="t", user_id="u", session_id="s", run_id=run_id, approval=True
    )

    assert state2.status is RunStatus.COMPLETED
    assert state2.run_id == run_id  # 全程一个 run_id
    assert final_reply(state2) == "done"
    # 增量持久化：恢复不重复写入，共 4 条消息
    assert bundle2.store.count_messages("t", "u", "s") == 4
    history = await bundle2.service.history(tenant_id="t", user_id="u", session_id="s")
    assert [m.role for m in history] == ["user", "assistant", "tool", "assistant"]
    # 运行快照 paused → completed，step/usage 跨恢复累计而非重置
    row = bundle2.run_store.get_run("t", "u", "s", run_id)
    assert row.status is RunStatus.COMPLETED
    assert row.pending_approval is None
    assert row.usage == Usage(prompt_tokens=2, completion_tokens=2)
    assert row.step == 2
    # 事件链跨重启完整，seq 续接
    assert es2.flush()
    types = [e.type for e in es2.get_events("t", "u", "s", run_id)]
    assert types == [
        RunEventType.RUN_STARTED,
        RunEventType.MODEL_REQUESTED,
        RunEventType.TOOL_CALLED,
        RunEventType.RUN_PAUSED,
        RunEventType.TOOL_RETURNED,
        RunEventType.MODEL_REQUESTED,
        RunEventType.RUN_COMPLETED,
    ]
    bundle2.store.close()
    bundle2.run_store.close()
    es2.close()


async def test_full_stack_rejected_resume_completes_without_executing(db_path):
    model = ScriptedModel(
        tool_call_response("c1", "echo", '{"text": "hi"}'), text_response("ok, skipping")
    )
    bundle = make_bundle(db_path, model)

    state1 = await bundle.service.chat(
        tenant_id="t", user_id="u", session_id="s", user_input="hello"
    )
    assert state1.status is RunStatus.PAUSED

    state2 = await bundle.service.resume(
        tenant_id="t", user_id="u", session_id="s", run_id=state1.run_id, approval=False
    )

    assert state2.status is RunStatus.COMPLETED
    assert final_reply(state2) == "ok, skipping"
    history = await bundle.service.history(tenant_id="t", user_id="u", session_id="s")
    assert history[2].content == "tool_error: tool 'echo' was not approved"
    bundle.store.close()
    bundle.run_store.close()


async def test_resume_rejects_unknown_or_not_paused_run(db_path):
    model = ScriptedModel(text_response("ok"))
    bundle = make_bundle(db_path, model, gate=lambda *_: False)

    state = await bundle.service.chat(
        tenant_id="t", user_id="u", session_id="s", user_input="hi"
    )
    assert state.status is RunStatus.COMPLETED

    with pytest.raises(ValueError):
        await bundle.service.resume(
            tenant_id="t", user_id="u", session_id="s", run_id="missing", approval=True
        )
    with pytest.raises(ValueError):
        await bundle.service.resume(
            tenant_id="t", user_id="u", session_id="s",
            run_id=state.run_id, approval=True,  # 已完成，非暂停
        )
    bundle.store.close()
    bundle.run_store.close()


async def test_resume_requires_run_store(db_path):
    bundle = build_runtime(
        AgentConfig(),
        model=ScriptedModel(text_response("x")),
        store=SQLiteSessionStore(db_path),
    )
    with pytest.raises(RuntimeError):
        await bundle.service.resume(
            tenant_id="t", user_id="u", session_id="s", run_id="r", approval=True
        )
    bundle.store.close()


async def test_skill_child_paused_reports_skill_error(db_path, tmp_path):
    model = ScriptedModel(
        tool_call_response("p1", "skill_summarizer", '{"task": "sum"}'),
        tool_call_response("c1", "echo", '{"text": "x"}'),  # 子运行请求 echo → 被门拦下
        text_response("parent done"),
    )
    bundle = build_runtime(
        AgentConfig(),
        model=model,
        store=SQLiteSessionStore(db_path),
        run_store=SQLiteRunStore(db_path),
        skill_dir=make_skill_dir(tmp_path),
        approval_gate=lambda name, *_: name == "echo",
    )

    state = await bundle.service.chat(
        tenant_id="t", user_id="u", session_id="s", user_input="use skill"
    )

    assert state.status is RunStatus.COMPLETED
    tool_msg = next(m for m in state.messages if m.tool_call_id == "p1")
    assert tool_msg.content == "skill_error: sub-run paused awaiting tool approval"
    assert final_reply(state) == "parent done"
    bundle.store.close()
    bundle.run_store.close()
