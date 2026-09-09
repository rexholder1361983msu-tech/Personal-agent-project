"""第二阶段切片 P2-A：RunLimits 完整化（token 预算/总耗时）与协作式取消。

全部确定性：ScriptedModel 驱动，截止时间用注入 clock（迭代器供时间点），
取消用 CancelToken；无真实等待。
"""

from __future__ import annotations

import pytest

from miniclaw.llm.base import ModelResponse
from miniclaw.llm.messages import Message, ToolCall, Usage
from miniclaw.llm.scripted import ScriptedModel, text_response, tool_call_response
from miniclaw.runtime.cancel import CancelToken
from miniclaw.runtime.limits import RunLimits
from miniclaw.runtime.loop import AgentRuntime
from miniclaw.runtime.state import RunState, RunStatus
from miniclaw.tools.base import ToolContext
from miniclaw.tools.registry import ToolRegistry


class StubTool:
    name = "stub"
    description = "stub"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, result: str = "ok"):
        self.result = result
        self.calls = 0

    async def execute(self, arguments, context: ToolContext) -> str:
        self.calls += 1
        return self.result


class CancellingTool:
    name = "canceller"
    description = "cancels the run from inside a tool"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, token: CancelToken):
        self.token = token
        self.calls = 0

    async def execute(self, arguments, context: ToolContext) -> str:
        self.calls += 1
        self.token.cancel("stop requested by tool")
        return "cancelled, please stop"


def make_runtime(model, tool=None, limits=None, clock=None) -> AgentRuntime:
    registry = ToolRegistry()
    if tool is not None:
        registry.register(tool)
    return AgentRuntime(model=model, tools=registry, limits=limits, clock=clock)


def make_state() -> RunState:
    return RunState(tenant_id="t", user_id="u", session_id="s")


def make_clock(ticks: list[float]):
    iterator = iter(ticks)
    fallback = ticks[-1] if ticks else 0.0

    def clock() -> float:
        try:
            return next(iterator)
        except StopIteration:
            return fallback + 1_000.0  # 耗尽后推到远未来，deadline 必然触发

    return clock


# ---------- RunLimits 校验 ----------


def test_limits_validation_for_new_fields():
    with pytest.raises(ValueError):
        RunLimits(max_total_tokens=0)
    with pytest.raises(ValueError):
        RunLimits(max_run_duration_s=-0.1)
    limits = RunLimits(max_total_tokens=1, max_run_duration_s=0.0)
    assert limits.max_total_tokens == 1
    assert limits.max_run_duration_s == 0.0


# ---------- token 预算 ----------


async def test_token_budget_exceeded_fails_strictly():
    """累计 usage 超预算即失败，即使该响应本可作为终答（严格语义）。"""
    model = ScriptedModel(
        ModelResponse(
            message=Message(role="assistant", content=None,
                            tool_calls=[ToolCall(id="c1", name="stub", arguments="{}")]),
            usage=Usage(prompt_tokens=60, completion_tokens=0),
            finish_reason="tool_calls",
        ),
        ModelResponse(
            message=Message(role="assistant", content="final"),
            usage=Usage(prompt_tokens=60, completion_tokens=0),
            finish_reason="stop",
        ),
    )
    limits = RunLimits(max_total_tokens=100)
    state = await make_runtime(model, StubTool(), limits).run(make_state(), "go")

    assert state.status is RunStatus.FAILED
    assert state.error_kind == "budget_exceeded"
    assert state.error == "budget_exceeded: token usage 120 > 100"
    assert state.usage.prompt_tokens == 120


async def test_token_budget_not_exceeded_completes():
    model = ScriptedModel(
        ModelResponse(
            message=Message(role="assistant", content=None,
                            tool_calls=[ToolCall(id="c1", name="stub", arguments="{}")]),
            usage=Usage(prompt_tokens=60, completion_tokens=0),
            finish_reason="tool_calls",
        ),
        text_response("final", usage=Usage(prompt_tokens=30, completion_tokens=10)),
    )
    state = await make_runtime(model, StubTool(), RunLimits(max_total_tokens=100)).run(make_state(), "go")
    assert state.status is RunStatus.COMPLETED
    assert state.error_kind is None


# ---------- 截止时间（注入 clock，无真实等待） ----------


async def test_deadline_exceeded_before_first_model_request():
    clock = make_clock([0.0, 1.0])  # start=0.0；首个检查点 elapsed=1.0 > 0
    model = ScriptedModel(text_response("never"))
    state = await make_runtime(
        model, StubTool(), RunLimits(max_run_duration_s=0.0), clock=clock
    ).run(make_state(), "go")

    assert state.status is RunStatus.FAILED
    assert state.error_kind == "deadline_exceeded"
    assert state.error.startswith("deadline_exceeded:")
    assert model.requests == []  # 模型一次都没被调用


async def test_deadline_exceeded_after_tool_round():
    clock = make_clock([0.0, 0.1, 0.2, 5.0])  # start、检查点1、工具前检查点、第二轮检查点
    model = ScriptedModel(
        tool_call_response("c1", "stub", "{}"),
        text_response("never reached"),
    )
    tool = StubTool()
    state = await make_runtime(
        model, tool, RunLimits(max_run_duration_s=1.0), clock=clock
    ).run(make_state(), "go")

    assert state.status is RunStatus.FAILED
    assert state.error_kind == "deadline_exceeded"
    assert len(model.requests) == 1  # 第二次模型请求被检查点拦下
    assert tool.calls == 1
    assert state.messages[-1].role == "tool"  # 工具结果仍在，便于恢复


# ---------- 协作式取消 ----------


async def test_cancel_before_run_skips_model_entirely():
    token = CancelToken()
    token.cancel("stopped by user")
    model = ScriptedModel(text_response("never"))
    state = await make_runtime(model, None, None).run(make_state(), "go", cancel=token)

    assert state.status is RunStatus.FAILED
    assert state.error_kind == "cancelled"
    assert state.error == "cancelled: stopped by user"
    assert model.requests == []


async def test_cancel_from_inside_tool_stops_before_next_model_request():
    token = CancelToken()
    model = ScriptedModel(
        tool_call_response("c1", "canceller", "{}"),
        text_response("never reached"),
    )
    tool = CancellingTool(token)
    state = await make_runtime(model, tool).run(make_state(), "go", cancel=token)

    assert state.status is RunStatus.FAILED
    assert state.error_kind == "cancelled"
    assert state.error == "cancelled: stop requested by tool"
    assert tool.calls == 1
    assert len(model.requests) == 1
    # 工具的结果消息已回填（运行在下一个检查点停止）
    assert state.messages[-1].content == "cancelled, please stop"


async def test_uncancelled_token_has_no_effect():
    token = CancelToken()
    assert token.cancelled is False
    assert token.reason == ""
    model = ScriptedModel(text_response("done"))
    state = await make_runtime(model).run(make_state(), "go", cancel=token)
    assert state.status is RunStatus.COMPLETED


# ---------- error_kind 与既有失败路径的映射 ----------


async def test_error_kind_model_error():
    state = await make_runtime(ScriptedModel()).run(make_state(), "go")
    assert state.error_kind == "model_error"
    assert state.error.startswith("model_error:")


async def test_error_kind_budget_for_max_steps():
    model = ScriptedModel(tool_call_response("c", "stub", "{}"), repeat_last=True)
    tool = StubTool()
    state = await make_runtime(model, tool, RunLimits(max_steps=2, max_tool_calls=100)).run(make_state(), "go")
    assert state.error_kind == "budget_exceeded"
    assert state.error.startswith("max_steps_exceeded:")


async def test_failed_state_resets_error_between_runs():
    """同一 RunState 复用（恢复场景）：新一轮运行开始时清空旧错误。"""
    state = make_state()
    state.status = RunStatus.FAILED
    state.error = "model_error: old"
    state.error_kind = "model_error"

    model = ScriptedModel(text_response("fresh"))
    state = await make_runtime(model).run(state, "again")

    assert state.status is RunStatus.COMPLETED
    assert state.error is None
    assert state.error_kind is None
