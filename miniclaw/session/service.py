"""ConversationService：网关（CLI/Web/Skill）使用的会话门面。

职责：
- 以三键加载或创建 RunState，把持久化消息映射回 Message 对象；
- 保存时只追加未持久化的消息（幂等；重启恢复后不会重复写入历史）；
- 强制隔离：所有存储访问都显式携带 tenant_id/user_id/session_id。

并发约定：Phase 1 中每个会话同一时刻只有一个写入方（一次运行），
不同会话之间可以并发。同一会话的多写者协调属于第二阶段。

RunState 的 status/step/usage 跨进程恢复属于第二阶段；
本阶段持久化的是对话消息本身。
"""

from __future__ import annotations

import asyncio
import time

from miniclaw.llm.messages import Message
from miniclaw.runtime.state import RunState
from miniclaw.session.store import SessionStore


class ConversationService:
    def __init__(self, store: SessionStore) -> None:
        self._store = store

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
