"""ConversationService.chat() 路径：历史回灌、隔离与失败持久化。"""

from __future__ import annotations

import pytest

from miniclaw.llm.scripted import ScriptedModel, text_response
from miniclaw.runtime.loop import AgentRuntime, final_reply
from miniclaw.runtime.state import RunStatus
from miniclaw.session.service import ConversationService
from miniclaw.session.store import SQLiteSessionStore
from miniclaw.tools.registry import ToolRegistry


def make_service(db_path, model) -> ConversationService:
    return ConversationService(
        SQLiteSessionStore(db_path), runtime=AgentRuntime(model=model, tools=ToolRegistry())
    )


async def test_chat_persists_and_feeds_history_back(db_path):
    model = ScriptedModel(text_response("hi A"), text_response("more A"))
    service = make_service(db_path, model)

    run1 = await service.chat(tenant_id="t", user_id="u", session_id="s", user_input="hello")
    run2 = await service.chat(tenant_id="t", user_id="u", session_id="s", user_input="more")

    assert run1.status is RunStatus.COMPLETED
    assert final_reply(run1) == "hi A"
    # 第二次请求必须带上第一轮完整历史
    assert [m.content for m in model.requests[1]] == ["hello", "hi A", "more"]
    history = await service.history(tenant_id="t", user_id="u", session_id="s")
    assert [m.content for m in history] == ["hello", "hi A", "more", "more A"]


async def test_chat_is_isolated_between_users_with_same_session_id(db_path):
    model = ScriptedModel(
        text_response("a1"), text_response("b1"), text_response("a2"), text_response("b2")
    )
    service = make_service(db_path, model)

    await service.chat(tenant_id="t", user_id="u1", session_id="shared", user_input="a1")
    await service.chat(tenant_id="t", user_id="u2", session_id="shared", user_input="b1")
    await service.chat(tenant_id="t", user_id="u1", session_id="shared", user_input="a2")
    await service.chat(tenant_id="t", user_id="u2", session_id="shared", user_input="b2")

    # u2 的首轮请求里绝不能看到 u1 的历史
    assert [m.content for m in model.requests[1]] == ["b1"]
    assert [m.content for m in model.requests[2]] == ["a1", "a1", "a2"]
    assert [m.content for m in model.requests[3]] == ["b1", "b1", "b2"]

    h1 = await service.history(tenant_id="t", user_id="u1", session_id="shared")
    h2 = await service.history(tenant_id="t", user_id="u2", session_id="shared")
    assert [m.content for m in h1] == ["a1", "a1", "a2", "a2"]
    assert [m.content for m in h2] == ["b1", "b1", "b2", "b2"]


async def test_chat_failure_still_persists_messages(db_path):
    model = ScriptedModel()  # 无脚本 → 模型错误 → FAILED
    service = make_service(db_path, model)

    state = await service.chat(tenant_id="t", user_id="u", session_id="s", user_input="hello")

    assert state.status is RunStatus.FAILED
    assert state.error.startswith("model_error:")
    history = await service.history(tenant_id="t", user_id="u", session_id="s")
    assert [m.content for m in history] == ["hello"]


async def test_chat_without_runtime_is_rejected(db_path):
    service = ConversationService(SQLiteSessionStore(db_path))
    with pytest.raises(RuntimeError):
        await service.chat(tenant_id="t", user_id="u", session_id="s", user_input="hello")
