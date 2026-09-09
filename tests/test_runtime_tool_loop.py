"""AgentRuntime 工具循环的确定性测试（ScriptedModel，无真实 LLM）。"""

from __future__ import annotations

import asyncio

from miniclaw.llm.base import ModelResponse
from miniclaw.llm.messages import Message, ToolCall, Usage
from miniclaw.llm.scripted import ScriptedModel, text_response, tool_call_response
from miniclaw.runtime.events import RunEventType
from miniclaw.runtime.limits import RunLimits
from miniclaw.runtime.loop import AgentRuntime, final_reply
from miniclaw.runtime.state import RunState, RunStatus
from miniclaw.tools.registry import ToolRegistry


class RecordingTool:
    name = "echo"
    description = "test double echo"
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def __init__(self, result: str = "ECHOED", error: Exception | None = None, delay_s: float = 0.0):
        self.result = result
        self.error = error
        self.delay_s = delay_s
        self.calls: list[tuple[dict, str]] = []

    async def execute(self, arguments, context):
        self.calls.append((arguments, context.tenant_id))
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.error:
            raise self.error
        return self.result


def make_runtime(model, tool=None, limits=None, on_event=None) -> AgentRuntime:
    registry = ToolRegistry()
    if tool is not None:
        registry.register(tool)
    return AgentRuntime(model=model, tools=registry, limits=limits, on_event=on_event)


def make_state() -> RunState:
    return RunState(tenant_id="t", user_id="u", session_id="s")


async def test_tool_call_then_final_answer():
    tool = RecordingTool()
    model = ScriptedModel(
        tool_call_response("call-1", "echo", '{"text": "hi"}'),
        text_response("done"),
    )
    runtime = make_runtime(model, tool)

    state = await runtime.run(make_state(), "hello")

    assert state.status is RunStatus.COMPLETED
    assert state.error is None
    assert tool.calls == [({"text": "hi"}, "t")]
    assert [m.role for m in state.messages] == ["user", "assistant", "tool", "assistant"]
    assert state.messages[1].tool_calls[0].id == "call-1"
    assert state.messages[2].content == "ECHOED"
    assert state.messages[2].tool_call_id == "call-1"
    assert state.messages[3].content == "done"
    assert final_reply(state) == "done"
    assert state.step == 2
    assert state.tool_calls_used == 1
    assert state.usage == Usage(prompt_tokens=2, completion_tokens=2)
    # 第二次模型请求带上了工具结果
    assert [m.role for m in model.requests[1]] == ["user", "assistant", "tool"]


async def test_direct_answer_without_tools():
    model = ScriptedModel(text_response("hi there"))
    state = await make_runtime(model).run(make_state(), "hello")
    assert state.status is RunStatus.COMPLETED
    assert state.step == 1
    assert state.tool_calls_used == 0
    assert final_reply(state) == "hi there"
    assert model.tool_specs[0] == []  # 无工具时不给模型传工具


async def test_tool_error_is_fed_back_not_fatal():
    tool = RecordingTool(error=RuntimeError("boom"))
    model = ScriptedModel(tool_call_response("c1", "echo", "{}"), text_response("recovered"))
    state = await make_runtime(model, tool).run(make_state(), "go")
    assert state.status is RunStatus.COMPLETED
    assert "tool_error" in state.messages[2].content
    assert "boom" in state.messages[2].content
    assert state.messages[2].tool_call_id == "c1"
    assert final_reply(state) == "recovered"


async def test_unknown_tool_is_fed_back():
    model = ScriptedModel(tool_call_response("c1", "nope", "{}"), text_response("ok"))
    state = await make_runtime(model).run(make_state(), "go")
    assert state.status is RunStatus.COMPLETED
    assert state.messages[2].content == "tool_error: unknown tool 'nope'"


async def test_tool_timeout_is_fed_back():
    tool = RecordingTool(delay_s=0.05)
    model = ScriptedModel(tool_call_response("c1", "echo", "{}"), text_response("ok"))
    limits = RunLimits(max_steps=4, tool_timeout_s=0.01)
    state = await make_runtime(model, tool, limits).run(make_state(), "go")
    assert state.status is RunStatus.COMPLETED
    assert state.messages[2].content == "tool_error: timeout after 0.01s"


async def test_invalid_json_arguments_are_fed_back():
    model = ScriptedModel(tool_call_response("c1", "echo", "not-json"), text_response("ok"))
    state = await make_runtime(model, RecordingTool()).run(make_state(), "go")
    assert "invalid JSON arguments" in state.messages[2].content

    model2 = ScriptedModel(tool_call_response("c1", "echo", '"just a string"'), text_response("ok"))
    state2 = await make_runtime(model2, RecordingTool()).run(make_state(), "go")
    assert "arguments must be a JSON object" in state2.messages[2].content


async def test_max_steps_exceeded_fails_run():
    model = ScriptedModel(tool_call_response("c", "echo", "{}"), repeat_last=True)
    limits = RunLimits(max_steps=3, max_tool_calls=100)
    state = await make_runtime(model, RecordingTool(), limits).run(make_state(), "go")
    assert state.status is RunStatus.FAILED
    assert state.error.startswith("max_steps_exceeded")
    assert state.step == 3
    assert state.tool_calls_used == 3
    assert state.messages[-1].role == "tool"


async def test_max_tool_calls_exceeded_fails_run():
    two_calls = ModelResponse(
        message=Message(
            role="assistant",
            content=None,
            tool_calls=[
                ToolCall(id="c1", name="echo", arguments="{}"),
                ToolCall(id="c2", name="echo", arguments="{}"),
            ],
        ),
        usage=Usage(prompt_tokens=1, completion_tokens=1),
        finish_reason="tool_calls",
    )
    model = ScriptedModel(two_calls, text_response("never reached"))
    limits = RunLimits(max_steps=5, max_tool_calls=1)
    tool = RecordingTool()
    state = await make_runtime(model, tool, limits).run(make_state(), "go")
    assert state.status is RunStatus.FAILED
    assert state.error.startswith("max_tool_calls_exceeded")
    assert state.tool_calls_used == 1
    assert len(tool.calls) == 1


async def test_model_error_fails_run():
    model = ScriptedModel()  # 空脚本：complete 抛 AssertionError
    state = await make_runtime(model).run(make_state(), "go")
    assert state.status is RunStatus.FAILED
    assert state.error.startswith("model_error:")


async def test_event_sequence_for_tool_roundtrip():
    events = []
    model = ScriptedModel(tool_call_response("c1", "echo", '{"text": "x"}'), text_response("done"))
    state = await make_runtime(model, RecordingTool(), on_event=events.append).run(make_state(), "go")
    assert [e.type for e in events] == [
        RunEventType.RUN_STARTED,
        RunEventType.MODEL_REQUESTED,
        RunEventType.TOOL_CALLED,
        RunEventType.TOOL_RETURNED,
        RunEventType.MODEL_REQUESTED,
        RunEventType.RUN_COMPLETED,
    ]
    assert all(e.run_id == state.run_id for e in events)
    assert events[2].data == {"name": "echo", "call_id": "c1", "arguments": '{"text": "x"}'}
    assert events[3].data["ok"] is True


async def test_run_without_input_and_history_is_rejected():
    import pytest
    with pytest.raises(ValueError):
        await make_runtime(ScriptedModel()).run(make_state())
