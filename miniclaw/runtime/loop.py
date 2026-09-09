"""AgentRuntime：统一的 ReAct 执行路径。

Runtime 无状态：RunState 显式传入传出，CLI/Web/Skill/后台任务复用同一个循环。

    用户消息 → [模型请求 → (工具调用 → 结果回填)* → 终答]

资源与取消检查点（本阶段语义）：
- 取消（CancelToken）：每次模型请求前、每次工具执行前检查；
- 截止时间（max_run_duration_s）：同上检查点；
- token 预算（max_total_tokens）：每次累计 usage 后立即检查，即使该响应
  本可作为终答，超预算也判定运行失败（严格语义）。

失败语义（error 前缀 / error_kind）：
- 模型异常 → model_error: …            / model_error
- 步数、工具次数、token 超限 → …exceeded: … / budget_exceeded
- 总耗时超限 → deadline_exceeded: …     / deadline_exceeded
- 取消 → cancelled: <reason>            / cancelled
- 未预期异常 → runtime_error: …         / runtime_error
- 工具异常（含参数解析失败、未知工具、超时、被过滤）不终止运行，
  错误以 tool_error 消息回填给模型，由模型决定下一步，事件中记录 ok=False。
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Callable, Collection

from miniclaw.llm.base import ModelClient
from miniclaw.llm.messages import Message
from miniclaw.runtime.cancel import CancelToken
from miniclaw.runtime.events import EventListener, RunEvent, RunEventType
from miniclaw.runtime.limits import RunLimits
from miniclaw.runtime.state import RunState, RunStatus
from miniclaw.tools.base import ToolContext
from miniclaw.tools.registry import ToolRegistry


def final_reply(state: RunState) -> str | None:
    """返回最后一条非空 assistant 消息，作为本次运行的答复。"""
    for message in reversed(state.messages):
        if message.role == "assistant" and message.content is not None:
            return message.content
    return None


class AgentRuntime:
    """一次运行的状态机。传入 RunState，返回推进后的 RunState。"""

    def __init__(
        self,
        model: ModelClient,
        tools: ToolRegistry,
        limits: RunLimits | None = None,
        on_event: EventListener | None = None,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.limits = limits or RunLimits()
        self.on_event = on_event
        self._clock = clock or time.monotonic

    def _emit(self, state: RunState, event_type: RunEventType, **data) -> None:
        if self.on_event is None:
            return
        self.on_event(
            RunEvent(type=event_type, run_id=state.run_id, step=state.step, data=data)
        )

    async def run(
        self,
        state: RunState,
        user_input: str | None = None,
        *,
        tool_names: Collection[str] | None = None,
        cancel: CancelToken | None = None,
    ) -> RunState:
        """执行一次运行。

        user_input 非空时作为 user 消息追加；为 None 时基于已有消息继续
        （为暂停/恢复预留）。tool_names 限定本次运行可见的工具
        （技能子运行用它排除其他技能，防止递归）。cancel 为协作式取消令牌，
        在检查点响应，不做抢占。
        """
        if user_input is None and not state.messages:
            raise ValueError("user_input is required to start a run")
        allowed = set(tool_names) if tool_names is not None else None
        specs = (
            self.tools.specs()
            if allowed is None
            else [s for s in self.tools.specs() if s.name in allowed]
        )
        started = self._clock()

        state.status = RunStatus.RUNNING
        state.error = None
        state.error_kind = None
        state.updated_at = time.time()
        if user_input is not None:
            state.messages.append(Message(role="user", content=user_input))
        self._emit(state, RunEventType.RUN_STARTED)

        try:
            while True:
                if (failure := self._check_guards(state, cancel, started)) is not None:
                    return failure
                if state.step >= self.limits.max_steps:
                    return self._fail(
                        state,
                        f"max_steps_exceeded: limit is {self.limits.max_steps}",
                        "budget_exceeded",
                    )
                state.step += 1
                self._emit(state, RunEventType.MODEL_REQUESTED, step=state.step)
                try:
                    response = await self.model.complete(state.messages, specs)
                except Exception as exc:
                    return self._fail(
                        state, f"model_error: {type(exc).__name__}: {exc}", "model_error"
                    )
                state.usage.add(response.usage)
                token_budget = self.limits.max_total_tokens
                if token_budget is not None:
                    total = state.usage.prompt_tokens + state.usage.completion_tokens
                    if total > token_budget:
                        return self._fail(
                            state,
                            f"budget_exceeded: token usage {total} > {token_budget}",
                            "budget_exceeded",
                        )
                state.messages.append(response.message)

                calls = response.message.tool_calls or []
                if not calls:
                    state.status = RunStatus.COMPLETED
                    state.updated_at = time.time()
                    self._emit(state, RunEventType.RUN_COMPLETED, steps=state.step)
                    return state

                context = ToolContext(
                    tenant_id=state.tenant_id,
                    user_id=state.user_id,
                    session_id=state.session_id,
                    run_id=state.run_id,
                    agent_id=state.agent_id,
                )
                for call in calls:
                    if state.tool_calls_used >= self.limits.max_tool_calls:
                        return self._fail(
                            state,
                            f"max_tool_calls_exceeded: limit is {self.limits.max_tool_calls}",
                            "budget_exceeded",
                        )
                    if (failure := self._check_guards(state, cancel, started)) is not None:
                        return failure
                    state.tool_calls_used += 1
                    self._emit(
                        state,
                        RunEventType.TOOL_CALLED,
                        name=call.name,
                        call_id=call.id,
                        arguments=call.arguments,
                    )
                    if allowed is not None and call.name not in allowed:
                        content, ok = (
                            f"tool_error: tool '{call.name}' is not allowed in this context",
                            False,
                        )
                    else:
                        tool = self.tools.get(call.name)
                        if tool is None:
                            content, ok = (
                                f"tool_error: unknown tool '{call.name}'",
                                False,
                            )
                        else:
                            content, ok = await self._execute_tool(tool, call, context)
                    self._emit(
                        state,
                        RunEventType.TOOL_RETURNED,
                        name=call.name,
                        call_id=call.id,
                        ok=ok,
                        content=content,
                    )
                    state.messages.append(
                        Message(role="tool", content=content, tool_call_id=call.id)
                    )
        except Exception as exc:  # 防御性兜底：未预期异常落成结构化失败
            return self._fail(state, f"runtime_error: {type(exc).__name__}: {exc}", "runtime_error")

    def _check_guards(
        self, state: RunState, cancel: CancelToken | None, started: float
    ) -> RunState | None:
        """检查取消与截止时间；触发时返回失败态，否则返回 None。"""
        if cancel is not None and cancel.cancelled:
            return self._fail(state, f"cancelled: {cancel.reason}", "cancelled")
        deadline = self.limits.max_run_duration_s
        if deadline is not None:
            elapsed = self._clock() - started
            if elapsed > deadline:
                return self._fail(
                    state,
                    f"deadline_exceeded: {elapsed:.3f}s > {deadline}s",
                    "deadline_exceeded",
                )
        return None

    async def _execute_tool(self, tool, call, context: ToolContext) -> tuple[str, bool]:
        try:
            try:
                arguments = json.loads(call.arguments) if call.arguments else {}
            except json.JSONDecodeError as exc:
                return f"tool_error: invalid JSON arguments: {exc}", False
            if not isinstance(arguments, dict):
                return "tool_error: arguments must be a JSON object", False
            result = await asyncio.wait_for(
                tool.execute(arguments, context), timeout=self.limits.tool_timeout_s
            )
            return result if isinstance(result, str) else str(result), True
        except asyncio.TimeoutError:
            return f"tool_error: timeout after {self.limits.tool_timeout_s}s", False
        except Exception as exc:
            return f"tool_error: {type(exc).__name__}: {exc}", False

    def _fail(self, state: RunState, error: str, kind: str) -> RunState:
        state.status = RunStatus.FAILED
        state.error = error
        state.error_kind = kind
        state.updated_at = time.time()
        self._emit(state, RunEventType.RUN_FAILED, error=error, error_kind=kind)
        return state
