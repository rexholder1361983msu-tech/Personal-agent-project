"""AgentRuntime：统一的 ReAct 执行路径。

Runtime 无状态：RunState 显式传入传出，CLI/Web/Skill/后台任务复用同一个循环。

    用户消息 → [模型请求 → (工具调用 → 结果回填)* → 终答]

资源与取消检查点（本阶段语义）：
- 取消（CancelToken）：每次模型请求前、每次工具执行前检查；
- 截止时间（max_run_duration_s）：同上检查点；恢复后的运行从恢复时刻重新起算；
- token 预算（max_total_tokens）：每次累计 usage 后立即检查，即使该响应
  本可作为终答，超预算也判定运行失败（严格语义）。

审批暂停/恢复（P2-C）：
- approval_gate(name, arguments, context) 返回 True 的工具调用不执行，
  运行置 PAUSED：pending_approval 记录该调用与同一响应中尚未处理的后续
  调用（保证恢复后每个 tool_call 都有结果回填）；
- 恢复：run(state, approval=True/False)（user_input 保持 None，同 run_id），
  批准则执行挂起的调用后继续；拒绝则以 tool_error 消息回填后继续；
- 挂起未决时禁止发送新的 user 输入（协议上 tool_call 必须先有结果）；
- 事件：挂起时发 TOOL_CALLED（沿用 P1 语义，未执行的调用也发）+ RUN_PAUSED，
  恢复后补 TOOL_RETURNED，跨进程由 EventStore 按 seq 接成一条链；
  RUN_STARTED 每个 run_id 只发一次（恢复不重发）。

工具策略（P2-D）：
- tool_policy 按工具名声明 mode：deny → tool_error 回填、一次都不执行；
  needs_approval → 与 approval_gate 任一命中即走暂停路径，deny 分支优先短路；
- 规则级 timeout_s / output_limit 覆盖全局默认（超时取更严格者），截断结果
  追加 ...[truncated] 标记；恢复执行的挂起调用沿用执行约束，但不重判 mode
  （审批决定已做出）。

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
from miniclaw.llm.messages import Message, ToolCall
from miniclaw.runtime.cancel import CancelToken
from miniclaw.runtime.events import EventListener, RunEvent, RunEventType
from miniclaw.runtime.limits import RunLimits
from miniclaw.runtime.state import RunState, RunStatus
from miniclaw.tools.base import ToolContext
from miniclaw.tools.policy import ToolMode, ToolPolicy, ToolRule
from miniclaw.tools.registry import ToolRegistry

# (tool_name, arguments_json, context) -> 该调用是否需要人工审批
ApprovalGate = Callable[[str, str, ToolContext], bool]


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
        approval_gate: ApprovalGate | None = None,
        tool_policy: ToolPolicy | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.limits = limits or RunLimits()
        self.on_event = on_event
        self.approval_gate = approval_gate
        self.tool_policy = tool_policy
        self._clock = clock or time.monotonic

    def _emit(self, state: RunState, event_type: RunEventType, **data) -> None:
        if self.on_event is None:
            return
        self.on_event(
            RunEvent(
                type=event_type,
                run_id=state.run_id,
                step=state.step,
                data=data,
                tenant_id=state.tenant_id,
                user_id=state.user_id,
                session_id=state.session_id,
            )
        )

    async def run(
        self,
        state: RunState,
        user_input: str | None = None,
        *,
        tool_names: Collection[str] | None = None,
        cancel: CancelToken | None = None,
        approval: bool | None = None,
    ) -> RunState:
        """执行一次运行。

        user_input 非空时作为 user 消息追加；为 None 时基于已有消息继续。
        tool_names 限定本次运行可见的工具（技能子运行用它排除其他技能，
        防止递归）。cancel 为协作式取消令牌，在检查点响应，不做抢占。

        审批恢复：state.pending_approval 非空时，approval 必须给出
        （True 执行挂起的调用，False 以 tool_error 回填），user_input 必须为
        None；step/usage/tool_calls_used 沿用传入状态，run_id 不变。
        """
        if user_input is None and not state.messages:
            raise ValueError("user_input is required to start a run")
        if state.pending_approval is not None:
            if user_input is not None:
                raise ValueError(
                    "a tool call is awaiting approval; resolve it with run(state, approval=...) first"
                )
            if approval is None:
                raise ValueError(
                    "approval decision (True/False) is required to resume a paused run"
                )
        elif approval is not None:
            raise ValueError("approval decision given but no tool call is awaiting approval")
        pending = state.pending_approval
        allowed = set(tool_names) if tool_names is not None else None
        specs = (
            self.tools.specs()
            if allowed is None
            else [s for s in self.tools.specs() if s.name in allowed]
        )
        started = self._clock()
        context = ToolContext(
            tenant_id=state.tenant_id,
            user_id=state.user_id,
            session_id=state.session_id,
            run_id=state.run_id,
            agent_id=state.agent_id,
        )

        state.status = RunStatus.RUNNING
        state.error = None
        state.error_kind = None
        state.updated_at = time.time()
        if user_input is not None:
            state.messages.append(Message(role="user", content=user_input))
        if pending is None:
            # RUN_STARTED 每个 run_id 只发一次；恢复运行不重发。
            self._emit(state, RunEventType.RUN_STARTED)

        try:
            if pending is not None:
                state.pending_approval = None
                head = ToolCall(
                    id=pending["call"]["id"],
                    name=pending["call"]["name"],
                    arguments=pending["call"].get("arguments", "{}"),
                )
                remaining = [
                    ToolCall(id=c["id"], name=c["name"], arguments=c.get("arguments", "{}"))
                    for c in pending.get("remaining", [])
                ]
                if (failure := self._check_guards(state, cancel, started)) is not None:
                    return failure
                await self._resolve_pending(state, head, context, approval)
                outcome = await self._process_calls(
                    state, remaining, context, allowed, cancel, started
                )
                if outcome != "done":
                    return state

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

                outcome = await self._process_calls(
                    state, calls, context, allowed, cancel, started
                )
                if outcome != "done":
                    return state
        except Exception as exc:  # 防御性兜底：未预期异常落成结构化失败
            return self._fail(state, f"runtime_error: {type(exc).__name__}: {exc}", "runtime_error")

    async def _process_calls(
        self,
        state: RunState,
        calls: list[ToolCall],
        context: ToolContext,
        allowed: set[str] | None,
        cancel: CancelToken | None,
        started: float,
    ) -> str:
        """依次处理一批工具调用。

        返回 "done"（全部处理完）或 "paused"/"failed"——后两种情况下 state
        已置为 PAUSED/FAILED，调用方应直接返回。暂停时 pending_approval
        记录待审批调用与同一响应中尚未处理的后续调用。
        """
        for index, call in enumerate(calls):
            if state.tool_calls_used >= self.limits.max_tool_calls:
                self._fail(
                    state,
                    f"max_tool_calls_exceeded: limit is {self.limits.max_tool_calls}",
                    "budget_exceeded",
                )
                return "failed"
            if (failure := self._check_guards(state, cancel, started)) is not None:
                return "failed"
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
            elif (tool := self.tools.get(call.name)) is None:
                content, ok = (
                    f"tool_error: unknown tool '{call.name}'",
                    False,
                )
            elif self._is_denied(call.name):
                content, ok = (
                    f"tool_error: tool '{call.name}' is not permitted",
                    False,
                )
            elif self._needs_approval(call, context):
                state.pending_approval = {
                    "call": {"id": call.id, "name": call.name, "arguments": call.arguments},
                    "remaining": [
                        {"id": c.id, "name": c.name, "arguments": c.arguments}
                        for c in calls[index + 1 :]
                    ],
                }
                state.status = RunStatus.PAUSED
                state.updated_at = time.time()
                self._emit(
                    state,
                    RunEventType.RUN_PAUSED,
                    name=call.name,
                    call_id=call.id,
                    arguments=call.arguments,
                )
                return "paused"
            else:
                content, ok = await self._execute_tool(
                    tool, call, context, self._rule_for(call.name)
                )
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
        return "done"

    async def _resolve_pending(
        self, state: RunState, head: ToolCall, context: ToolContext, approval: bool
    ) -> None:
        """恢复时处理挂起的审批调用：批准则执行，拒绝则回填 tool_error。

        TOOL_CALLED 已在挂起时发出（每次调用只发一次），此处只补
        TOOL_RETURNED；EventStore 按 run_id + seq 把跨进程的两段接成一条链。
        审批决定已做出，不重判 policy mode；规则级执行约束
        （timeout/output_limit）仍然生效。
        """
        if approval:
            tool = self.tools.get(head.name)
            if tool is None:
                content, ok = (f"tool_error: unknown tool '{head.name}'", False)
            else:
                content, ok = await self._execute_tool(
                    tool, head, context, self._rule_for(head.name)
                )
        else:
            content, ok = (f"tool_error: tool '{head.name}' was not approved", False)
        self._emit(
            state,
            RunEventType.TOOL_RETURNED,
            name=head.name,
            call_id=head.id,
            ok=ok,
            content=content,
        )
        state.messages.append(
            Message(role="tool", content=content, tool_call_id=head.id)
        )

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

    def _is_denied(self, name: str) -> bool:
        return (
            self.tool_policy is not None and self.tool_policy.decide(name) is ToolMode.DENY
        )

    def _needs_approval(self, call: ToolCall, context: ToolContext) -> bool:
        """policy 与 approval_gate 任一判定需审批即暂停（deny 已在更早分支短路）。"""
        if (
            self.tool_policy is not None
            and self.tool_policy.decide(call.name) is ToolMode.NEEDS_APPROVAL
        ):
            return True
        return self.approval_gate is not None and self.approval_gate(
            call.name, call.arguments, context
        )

    def _rule_for(self, name: str) -> ToolRule | None:
        return self.tool_policy.rule_for(name) if self.tool_policy is not None else None

    async def _execute_tool(
        self, tool, call, context: ToolContext, rule: ToolRule | None = None
    ) -> tuple[str, bool]:
        timeout = (
            rule.effective_timeout(self.limits.tool_timeout_s)
            if rule is not None
            else self.limits.tool_timeout_s
        )
        try:
            try:
                arguments = json.loads(call.arguments) if call.arguments else {}
            except json.JSONDecodeError as exc:
                return f"tool_error: invalid JSON arguments: {exc}", False
            if not isinstance(arguments, dict):
                return "tool_error: arguments must be a JSON object", False
            result = await asyncio.wait_for(
                tool.execute(arguments, context), timeout=timeout
            )
            content = result if isinstance(result, str) else str(result)
            if rule is not None:
                content = rule.truncate(content)
            return content, True
        except asyncio.TimeoutError:
            return f"tool_error: timeout after {timeout}s", False
        except Exception as exc:
            return f"tool_error: {type(exc).__name__}: {exc}", False

    def _fail(self, state: RunState, error: str, kind: str) -> RunState:
        state.status = RunStatus.FAILED
        state.error = error
        state.error_kind = kind
        state.updated_at = time.time()
        self._emit(state, RunEventType.RUN_FAILED, error=error, error_kind=kind)
        return state
