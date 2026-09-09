"""SessionStore 的三键隔离、顺序、持久化与并发写测试。"""

from __future__ import annotations

import threading

from miniclaw.llm.messages import Message, ToolCall
from miniclaw.session.store import SQLiteSessionStore


def test_three_key_isolation(db_path):
    store = SQLiteSessionStore(db_path)
    store.append_messages("tenant-a", "user-1", "shared-session", [Message(role="user", content="from a1")])
    store.append_messages("tenant-a", "user-2", "shared-session", [Message(role="user", content="from a2")])
    store.append_messages("tenant-b", "user-1", "shared-session", [Message(role="user", content="from b1")])

    assert [m.content for m in store.get_messages("tenant-a", "user-1", "shared-session")] == ["from a1"]
    assert [m.content for m in store.get_messages("tenant-a", "user-2", "shared-session")] == ["from a2"]
    assert [m.content for m in store.get_messages("tenant-b", "user-1", "shared-session")] == ["from b1"]
    store.close()


def test_message_ordering_across_appends(db_path):
    store = SQLiteSessionStore(db_path)
    store.append_messages(
        "t", "u", "s",
        [Message(role="user", content="1"), Message(role="assistant", content="2")],
    )
    store.append_messages("t", "u", "s", [Message(role="user", content="3")])
    assert [m.content for m in store.get_messages("t", "u", "s")] == ["1", "2", "3"]
    store.close()


def test_persistence_across_reopen(db_path):
    store = SQLiteSessionStore(db_path)
    store.append_messages(
        "t", "u", "s",
        [Message(role="user", content="hello"), Message(role="assistant", content="hi")],
    )
    store.close()

    reopened = SQLiteSessionStore(db_path)
    messages = reopened.get_messages("t", "u", "s")
    assert [(m.role, m.content) for m in messages] == [("user", "hello"), ("assistant", "hi")]
    reopened.close()


def test_tool_messages_roundtrip(db_path):
    store = SQLiteSessionStore(db_path)
    call = ToolCall(id="call-1", name="echo", arguments='{"text": "hi"}')
    store.append_messages(
        "t", "u", "s",
        [
            Message(role="user", content="hi"),
            Message(role="assistant", content=None, tool_calls=[call]),
            Message(role="tool", content="hi", tool_call_id="call-1"),
        ],
    )
    store.close()

    messages = SQLiteSessionStore(db_path).get_messages("t", "u", "s")
    assert messages[0].content == "hi"
    assert messages[1].tool_calls == [call]
    assert messages[1].tool_call_id is None
    assert messages[2].content == "hi"
    assert messages[2].tool_call_id == "call-1"
    assert messages[2].tool_calls is None


def test_list_sessions_scoped_by_tenant_and_user(db_path):
    store = SQLiteSessionStore(db_path)
    store.append_messages("t", "u1", "s1", [Message(role="user", content="a")])
    store.append_messages(
        "t", "u1", "s2",
        [Message(role="user", content="b"), Message(role="assistant", content="c")],
    )
    store.append_messages("t", "u2", "s1", [Message(role="user", content="d")])
    store.append_messages("other-tenant", "u1", "s3", [Message(role="user", content="e")])

    metas = {m.session_id: m for m in store.list_sessions("t", "u1")}
    assert set(metas) == {"s1", "s2"}
    assert metas["s1"].message_count == 1
    assert metas["s2"].message_count == 2
    assert [m.session_id for m in store.list_sessions("t", "u2")] == ["s1"]
    store.close()


def test_clear_removes_only_target_session(db_path):
    store = SQLiteSessionStore(db_path)
    store.append_messages("t", "u1", "s1", [Message(role="user", content="a")])
    store.append_messages("t", "u1", "s2", [Message(role="user", content="b")])

    assert store.clear("t", "u1", "s1") == 1
    assert store.get_messages("t", "u1", "s1") == []
    assert [m.content for m in store.get_messages("t", "u1", "s2")] == ["b"]
    store.close()


def test_empty_append_is_noop(db_path):
    store = SQLiteSessionStore(db_path)
    assert store.append_messages("t", "u", "s", []) == 0
    assert store.list_sessions("t", "u") == []
    store.close()


def test_empty_scope_keys_are_rejected(db_path):
    store = SQLiteSessionStore(db_path)
    for keys in [("", "u", "s"), ("t", "", "s"), ("t", "u", "")]:
        try:
            store.append_messages(*keys, [Message(role="user", content="x")])
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for keys {keys}")
    store.close()


def test_concurrent_appends_to_different_sessions(db_path):
    """多线程同时写不同会话：写入完整、互不串线。"""
    store = SQLiteSessionStore(db_path)
    errors: list[Exception] = []

    def worker(tenant: str, user: str, session: str) -> None:
        try:
            for i in range(20):
                store.append_messages(
                    tenant, user, session,
                    [Message(role="user", content=f"{tenant}/{user}/{session}/{i}")],
                )
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=("t1", "u1", "s1")),
        threading.Thread(target=worker, args=("t1", "u1", "s2")),
        threading.Thread(target=worker, args=("t1", "u2", "s1")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    own = store.get_messages("t1", "u1", "s1")
    assert len(own) == 20
    assert all(m.content.startswith("t1/u1/s1/") for m in own)
    assert store.get_messages("t1", "u2", "s1")[0].content == "t1/u2/s1/0"
    store.close()


def test_concurrent_appends_to_same_session(db_path):
    """多线程同时写同一会话：全部落库、无丢失、无重复。"""
    store = SQLiteSessionStore(db_path)

    def worker(tag: str) -> None:
        for i in range(20):
            store.append_messages("t", "u", "s", [Message(role="user", content=f"{tag}-{i}")])

    threads = [threading.Thread(target=worker, args=("a",)), threading.Thread(target=worker, args=("b",))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    messages = store.get_messages("t", "u", "s")
    assert len(messages) == 40
    assert len({m.content for m in messages}) == 40
    store.close()
