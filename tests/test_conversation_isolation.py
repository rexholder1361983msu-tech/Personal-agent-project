"""ConversationService 级别的会话隔离与恢复测试。

不使用真实模型：会话内容直接以 Message 追加到 RunState，
模拟后续 AgentRuntime 循环将要做的事情。
"""

from __future__ import annotations

import asyncio

from miniclaw.llm.messages import Message
from miniclaw.session.service import ConversationService
from miniclaw.session.store import SQLiteSessionStore
from miniclaw.runtime.state import RunStatus


async def test_interleaved_sessions_do_not_crosstalk(db_path):
    """三个身份共用同一个 session_id 字符串，交错对话后互不可见对方历史。"""
    store = SQLiteSessionStore(db_path)
    service = ConversationService(store)

    async def converse(tenant: str, user: str, session: str, turns: list[str]) -> None:
        for turn in turns:
            state = await service.load_or_create(tenant_id=tenant, user_id=user, session_id=session)
            state.messages.append(Message(role="user", content=turn))
            state.messages.append(Message(role="assistant", content=f"echo: {turn}"))
            await service.save(state)

    await asyncio.gather(
        converse("t1", "u1", "shared", ["a1", "a2"]),
        converse("t1", "u2", "shared", ["b1", "b2"]),
        converse("t2", "u1", "shared", ["c1", "c2"]),
    )

    a = [m.content for m in await service.history(tenant_id="t1", user_id="u1", session_id="shared")]
    b = [m.content for m in await service.history(tenant_id="t1", user_id="u2", session_id="shared")]
    c = [m.content for m in await service.history(tenant_id="t2", user_id="u1", session_id="shared")]
    assert a == ["a1", "echo: a1", "a2", "echo: a2"]
    assert b == ["b1", "echo: b1", "b2", "echo: b2"]
    assert c == ["c1", "echo: c1", "c2", "echo: c2"]
    store.close()


async def test_restart_recovery_and_no_duplicate(db_path):
    """模拟进程重启：重开存储后历史完整，继续对话不重复写入。"""
    store = SQLiteSessionStore(db_path)
    service = ConversationService(store)

    state = await service.load_or_create(tenant_id="t", user_id="u", session_id="s")
    state.messages.append(Message(role="user", content="hi"))
    state.messages.append(Message(role="assistant", content="hello"))
    await service.save(state)

    # 旧的内存对象全部丢弃，重新打开存储与服务。
    store.close()
    store2 = SQLiteSessionStore(db_path)
    service2 = ConversationService(store2)

    resumed = await service2.load_or_create(tenant_id="t", user_id="u", session_id="s")
    assert [m.content for m in resumed.messages] == ["hi", "hello"]
    assert resumed.persisted_messages == 2
    assert resumed.status is RunStatus.RUNNING
    assert resumed.run_id != state.run_id  # 每次加载是一次新运行

    resumed.messages.append(Message(role="user", content="again"))
    await service2.save(resumed)

    contents = [m.content for m in await service2.history(tenant_id="t", user_id="u", session_id="s")]
    assert contents == ["hi", "hello", "again"]
    store2.close()


async def test_save_is_idempotent_without_new_messages(db_path):
    store = SQLiteSessionStore(db_path)
    service = ConversationService(store)
    state = await service.load_or_create(tenant_id="t", user_id="u", session_id="s")
    state.messages.append(Message(role="user", content="hi"))
    await service.save(state)
    await service.save(state)
    await service.save(state)
    assert await service.history(tenant_id="t", user_id="u", session_id="s") == state.messages
    assert len(state.messages) == 1
    store.close()


async def test_loaded_state_keeps_identity_fields(db_path):
    store = SQLiteSessionStore(db_path)
    service = ConversationService(store)
    state = await service.load_or_create(
        tenant_id="tenant-x", user_id="user-x", session_id="session-x"
    )
    assert (state.tenant_id, state.user_id, state.session_id) == (
        "tenant-x",
        "user-x",
        "session-x",
    )
    store.close()
