"""SQLite EventStore 测试：顺序、隔离、重启、脱敏与 runtime 桥接。"""

from __future__ import annotations

import pytest

from miniclaw.config import AgentConfig
from miniclaw.llm.scripted import ScriptedModel, text_response, tool_call_response
from miniclaw.runtime.events import RunEvent, RunEventType
from miniclaw.runtime.factory import build_runtime
from miniclaw.runtime.state import RunStatus
from miniclaw.session.event_store import SQLiteEventStore
from miniclaw.session.store import SQLiteSessionStore


def make_event(event_type: RunEventType, run_id: str, step: int, data: dict | None = None) -> RunEvent:
    return RunEvent(
        type=event_type,
        run_id=run_id,
        step=step,
        data=data or {},
        tenant_id="t",
        user_id="u",
        session_id="s",
    )


def append_chain(store: SQLiteEventStore, run_id: str = "r1") -> None:
    store.append_event("t", "u", "s", run_id, make_event(RunEventType.RUN_STARTED, run_id, 0))
    store.append_event("t", "u", "s", run_id, make_event(RunEventType.MODEL_REQUESTED, run_id, 1, {"step": 1}))
    store.append_event(
        "t", "u", "s", run_id,
        make_event(
            RunEventType.TOOL_CALLED, run_id, 1,
            {"name": "echo", "call_id": "c1", "arguments": '{"text": "héllo"}'},
        ),
    )
    store.append_event(
        "t", "u", "s", run_id,
        make_event(
            RunEventType.TOOL_RETURNED, run_id, 1,
            {"name": "echo", "call_id": "c1", "ok": True, "content": "ECHOED"},
        ),
    )
    store.append_event("t", "u", "s", run_id, make_event(RunEventType.RUN_COMPLETED, run_id, 2, {"steps": 2}))


def test_roundtrip_preserves_order_and_payload(tmp_path):
    store = SQLiteEventStore(tmp_path / "events.db")
    append_chain(store)

    events = store.get_events("t", "u", "s", "r1")

    assert [e.type for e in events] == [
        RunEventType.RUN_STARTED,
        RunEventType.MODEL_REQUESTED,
        RunEventType.TOOL_CALLED,
        RunEventType.TOOL_RETURNED,
        RunEventType.RUN_COMPLETED,
    ]
    assert all(e.run_id == "r1" for e in events)
    assert [e.step for e in events] == [0, 1, 1, 1, 2]
    # tool_called / tool_returned 参数完整往返（含非 ASCII）
    assert events[2].data == {"name": "echo", "call_id": "c1", "arguments": '{"text": "héllo"}'}
    assert events[3].data == {"name": "echo", "call_id": "c1", "ok": True, "content": "ECHOED"}
    # 还原后的事件带全三键
    assert (events[0].tenant_id, events[0].user_id, events[0].session_id) == ("t", "u", "s")
    store.close()


def test_seq_continues_across_restart(tmp_path):
    path = tmp_path / "events.db"
    store = SQLiteEventStore(path)
    store.append_event("t", "u", "s", "r1", make_event(RunEventType.RUN_STARTED, "r1", 0))
    store.append_event("t", "u", "s", "r1", make_event(RunEventType.MODEL_REQUESTED, "r1", 1))
    store.close()

    reopened = SQLiteEventStore(path)
    reopened.append_event("t", "u", "s", "r1", make_event(RunEventType.RUN_COMPLETED, "r1", 1))
    events = reopened.get_events("t", "u", "s", "r1")
    assert [e.type for e in events] == [
        RunEventType.RUN_STARTED,
        RunEventType.MODEL_REQUESTED,
        RunEventType.RUN_COMPLETED,
    ]
    reopened.close()


def test_scope_and_run_id_isolation(tmp_path):
    store = SQLiteEventStore(tmp_path / "events.db")
    append_chain(store, run_id="run-a")
    append_chain(store, run_id="run-b")
    store.append_event(
        "t2", "u", "s", "run-a", make_event(RunEventType.RUN_STARTED, "run-a", 0)
    )

    assert len(store.get_events("t", "u", "s", "run-a")) == 5
    assert len(store.get_events("t", "u", "s", "run-b")) == 5
    # 相同 run_id 落在不同租户下互不可见：查询必须带全三键
    assert len(store.get_events("t2", "u", "s", "run-a")) == 1
    store.close()


def test_empty_scope_keys_are_rejected(tmp_path):
    store = SQLiteEventStore(tmp_path / "events.db")
    with pytest.raises(ValueError):
        store.append_event("", "u", "s", "r1", make_event(RunEventType.RUN_STARTED, "r1", 0))
    with pytest.raises(ValueError):
        store.get_events("t", "u", "s", "")
    store.close()


def test_redact_drops_arguments_and_content(tmp_path):
    store = SQLiteEventStore(tmp_path / "events.db", redact=True)
    append_chain(store)

    events = store.get_events("t", "u", "s", "r1")

    assert events[2].data == {"name": "echo", "call_id": "c1"}
    assert events[3].data == {"name": "echo", "call_id": "c1", "ok": True}
    # 非敏感事件不受影响
    assert events[4].data == {"steps": 2}
    store.close()


def test_listener_enqueues_and_flush_persists(tmp_path):
    store = SQLiteEventStore(tmp_path / "events.db")
    listener = store.listener()
    # listener 也可在无事件循环的上下文中直接调用（入队即返回）
    listener(make_event(RunEventType.RUN_STARTED, "r1", 0))
    listener(make_event(
        RunEventType.RUN_FAILED, "r1", 0,
        {"error": "model_error: x", "error_kind": "model_error"},
    ))

    assert store.flush()
    events = store.get_events("t", "u", "s", "r1")
    assert [e.type for e in events] == [RunEventType.RUN_STARTED, RunEventType.RUN_FAILED]
    assert events[1].data["error_kind"] == "model_error"
    store.close()


async def test_bridge_full_stack_records_call_chain(db_path, tmp_path):
    # store 与 event_store 同库不同连接（生产默认布局）
    event_store = SQLiteEventStore(db_path)
    model = ScriptedModel(
        tool_call_response("c1", "echo", '{"text": "hi"}'),
        text_response("done"),
    )
    bundle = build_runtime(
        AgentConfig(),
        model=model,
        store=SQLiteSessionStore(db_path),
        event_store=event_store,
    )

    state = await bundle.service.chat(
        tenant_id="t", user_id="u", session_id="s", user_input="hello"
    )

    assert state.status is RunStatus.COMPLETED
    assert event_store.flush()
    events = event_store.get_events("t", "u", "s", state.run_id)
    assert [e.type for e in events] == [
        RunEventType.RUN_STARTED,
        RunEventType.MODEL_REQUESTED,
        RunEventType.TOOL_CALLED,
        RunEventType.TOOL_RETURNED,
        RunEventType.MODEL_REQUESTED,
        RunEventType.RUN_COMPLETED,
    ]
    assert all(e.run_id == state.run_id for e in events)
    assert events[2].data["name"] == "echo"
    assert events[2].data["arguments"] == '{"text": "hi"}'
    assert events[3].data["ok"] is True
    bundle.store.close()
    event_store.close()


async def test_on_event_callback_and_event_store_both_fire(db_path):
    received = []
    event_store = SQLiteEventStore(db_path)
    bundle = build_runtime(
        AgentConfig(),
        model=ScriptedModel(text_response("ok")),
        store=SQLiteSessionStore(db_path),
        on_event=received.append,
        event_store=event_store,
    )

    state = await bundle.service.chat(
        tenant_id="t", user_id="u", session_id="s", user_input="hello"
    )

    # 进程内回调同步收到（未脱敏）
    assert [e.type for e in received] == [
        RunEventType.RUN_STARTED,
        RunEventType.MODEL_REQUESTED,
        RunEventType.RUN_COMPLETED,
    ]
    assert event_store.flush()
    assert len(event_store.get_events("t", "u", "s", state.run_id)) == 3
    bundle.store.close()
    event_store.close()


async def test_run_failure_is_recorded(db_path):
    event_store = SQLiteEventStore(db_path)
    bundle = build_runtime(
        AgentConfig(),
        model=ScriptedModel(),  # 空脚本 → model_error
        store=SQLiteSessionStore(db_path),
        event_store=event_store,
    )

    state = await bundle.service.chat(
        tenant_id="t", user_id="u", session_id="s", user_input="hello"
    )

    assert event_store.flush()
    events = event_store.get_events("t", "u", "s", state.run_id)
    assert [e.type for e in events] == [
        RunEventType.RUN_STARTED,
        RunEventType.MODEL_REQUESTED,
        RunEventType.RUN_FAILED,
    ]
    assert events[2].data["error_kind"] == "model_error"
    bundle.store.close()
    event_store.close()
