"""ConversationService：网关（CLI/Web/Skill）使用的会话门面。

职责：
- 以三键加载或创建 RunState，把持久化消息映射回 Message 对象；
- 保存时只追加未持久化的消息（幂等；重启恢复后不会重复写入历史）；
- 运行元数据快照到 RunStore（status/step/usage/pending_approval），
  支撑审批暂停后的跨进程恢复（resume）；
- 强制隔离：所有存储访问都显式携带 tenant_id/user_id/session_id。

并发约定：Phase 1 中每个会话同一时刻只有一个写入方（一次运行），
不同会话之间可以并发。同一会话的多写者协调属于后续阶段。
"""

from __future__ import annotations

import asyncio
import time

from miniclaw.llm.messages import Message
from miniclaw.runtime.loop import AgentRuntime
from miniclaw.runtime.state import RunState, RunStatus
from miniclaw.session.run_store import SQLiteRunStore
from miniclaw.session.store import SessionStore


class ConversationService:
    def __init__(
        self,
        store: SessionStore,
        runtime: AgentRuntime | None = None,
        run_store: SQLiteRunStore | None = None,
    ) -> None:
        self._store = store
        self._runtime = runtime
        self._run_store = run_store

    @property
    def runtime(self) -> AgentRuntime | None:
        """本服务使用的 AgentRuntime（未注入时为 None）。"""
        return self._runtime

    async def chat(
        self, *, tenant_id: str, user_id: str, session_id: str, user_input: str
    ) -> RunState:
        """一次完整对话：加载历史 → Runtime 执行 → 持久化。

        运行可能以 PAUSED 结束（工具调用等待审批）；此后同会话的新消息
        会被拒绝，须先经 resume() 处理挂起的审批。
        """
        if self._runtime is None:
            raise RuntimeError(
                "ConversationService requires a runtime for chat(); "
                "pass it to __init__ or use build_runtime()"
            )
        paused = await self._get_paused(tenant_id, user_id, session_id)
        if paused is not None:
            raise ValueError(
                f"session has a paused run ({paused.run_id}) awaiting approval; "
                "call resume() first"
            )
        state = await self.load_or_create(
            tenant_id=tenant_id, user_id=user_id, session_id=session_id
        )
        state = await self._runtime.run(state, user_input)
        await self.save(state)
        await self._persist_run(state)
        return state

    async def resume(
        self,
        *,
        tenant_id: str,
        user_id: str,
        session_id: str,
        run_id: str,
        approval: bool,
    ) -> RunState:
        """恢复一次 PAUSED 的运行（同 run_id）。

        快照取自 RunStore，消息取自 SessionStore（重启后依然可用）；
        approval=True 执行挂起的工具调用，False 以 tool_error 回填。
        """
        if self._runtime is None:
            raise RuntimeError(
                "ConversationService requires a runtime for resume(); "
                "pass it to __init__ or use build_runtime()"
            )
        if self._run_store is None:
            raise RuntimeError(
                "resume() requires a run_store; "
                "pass run_store to ConversationService or build_runtime"
            )
        state = await asyncio.to_thread(
            self._run_store.get_run, tenant_id, user_id, session_id, run_id
        )
        if state is None:
            raise ValueError(f"run '{run_id}' not found in session '{session_id}'")
        if state.status is not RunStatus.PAUSED:
            raise ValueError(f"run '{run_id}' is {state.status.value}, not paused")
        messages = await asyncio.to_thread(
            self._store.get_messages, tenant_id, user_id, session_id
        )
        state.messages = list(messages)
        state.persisted_messages = len(messages)
        state = await self._runtime.run(state, None, approval=approval)
        await self.save(state)
        await self._persist_run(state)
        return state

    async def _get_paused(
        self, tenant_id: str, user_id: str, session_id: str
    ) -> RunState | None:
        if self._run_store is None:
            return None
        return await asyncio.to_thread(
            self._run_store.get_paused_run, tenant_id, user_id, session_id
        )

    async def _persist_run(self, state: RunState) -> None:
        if self._run_store is None:
            return
        await asyncio.to_thread(self._run_store.upsert_run, state)

    async def load_or_create(
        self, *, tenant_id: str, user_id: str, session_id: str
    ) -> RunState:
        """加载会话历史为一次新运行的初始状态；不存在则为空会话。"""
        messages = await asyncio.to_thread(
            self._store.get_messages, tenant_id, user_id, session_id
        )
        state = RunState(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            messages=list(messages),
        )
        state.persisted_messages = len(messages)
        return state

    async def save(self, state: RunState) -> RunState:
        """追加 state 中尚未持久化的消息。无新消息时为幂等空操作。"""
        new_messages = state.messages[state.persisted_messages:]
        if new_messages:
            await asyncio.to_thread(
                self._store.append_messages,
                state.tenant_id,
                state.user_id,
                state.session_id,
                new_messages,
            )
            state.persisted_messages = len(state.messages)
        state.updated_at = time.time()
        return state

    async def history(
        self, *, tenant_id: str, user_id: str, session_id: str
    ) -> list[Message]:
        return await asyncio.to_thread(
            self._store.get_messages, tenant_id, user_id, session_id
        )

    async def clear(self, *, tenant_id: str, user_id: str, session_id: str) -> int:
        return await asyncio.to_thread(
            self._store.clear, tenant_id, user_id, session_id
        )
