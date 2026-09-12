"""SQLite RunStore 测试：快照往返、upsert 覆盖、暂停查询与键校验。"""

from __future__ import annotations

import pytest

from miniclaw.llm.messages import Usage
from miniclaw.runtime.state import RunState, RunStatus
from miniclaw.session.run_store import SQLiteRunStore


def make_state(**overrides) -> RunState:
    defaults = dict(
        tenant_id="t",
        user_id="u",
        session_id="s",
        run_id="run-1",
        status=RunStatus.PAUSED,
        step=2,
        tool_calls_used=1,
    )
    defaults.update(overrides)
    state = RunState(**defaults)
    state.usage = Usage(prompt_tokens=11, completion_tokens=7)
    state.error = None
    state.pending_approval = {
        "call": {"id": "c2", "name": "shell", "arguments": '{"cmd": "rm -rf /"}'},
        "remaining": [{"id": "c3", "name": "echo", "arguments": "{}"}],
    }
    return state


def test_roundtrip_preserves_all_run_metadata(tmp_path):
    store = SQLiteRunStore(tmp_path / "runs.db")
    state = make_state(agent_id="worker-1", error_kind=None)

    store.upsert_run(state)
    loaded = store.get_run("t", "u", "s", "run-1")

    assert loaded is not None
    assert (loaded.tenant_id, loaded.user_id, loaded.session_id) == ("t", "u", "s")
    assert loaded.run_id == "run-1"
    assert loaded.agent_id == "worker-1"
    assert loaded.status is RunStatus.PAUSED
    assert loaded.step == 2
    assert loaded.tool_calls_used == 1
    assert loaded.usage == Usage(prompt_tokens=11, completion_tokens=7)
    assert loaded.error is None and loaded.error_kind is None
    assert loaded.created_at == pytest.approx(state.created_at)
    # 消息不由 RunStore 持有：返回空列表，由调用方从 SessionStore 装载
    assert loaded.messages == []
    assert loaded.persisted_messages == 0
    assert loaded.pending_approval == {
        "call": {"id": "c2", "name": "shell", "arguments": '{"cmd": "rm -rf /"}'},
        "remaining": [{"id": "c3", "name": "echo", "arguments": "{}"}],
    }
    store.close()


def test_upsert_overwrites_status_and_clears_pending(tmp_path):
    store = SQLiteRunStore(tmp_path / "runs.db")
    store.upsert_run(make_state())
    resumed = make_state(status=RunStatus.COMPLETED, step=3, tool_calls_used=2)
    resumed.pending_approval = None
    resumed.error = None
    resumed.usage = Usage(prompt_tokens=15, completion_tokens=9)

    store.upsert_run(resumed)
    loaded = store.get_run("t", "u", "s", "run-1")

    assert loaded.status is RunStatus.COMPLETED
    assert loaded.step == 3
    assert loaded.tool_calls_used == 2
    assert loaded.usage == Usage(prompt_tokens=15, completion_tokens=9)
    assert loaded.pending_approval is None
    store.close()


def test_failed_run_metadata_roundtrip(tmp_path):
    store = SQLiteRunStore(tmp_path / "runs.db")
    failed = make_state(status=RunStatus.FAILED)
    failed.pending_approval = None
    failed.error = "model_error: Boom"
    failed.error_kind = "model_error"
    store.upsert_run(failed)

    loaded = store.get_run("t", "u", "s", "run-1")
    assert loaded.status is RunStatus.FAILED
    assert loaded.error == "model_error: Boom"
    assert loaded.error_kind == "model_error"
    store.close()


def test_scope_isolation_and_missing_run(tmp_path):
    store = SQLiteRunStore(tmp_path / "runs.db")
    store.upsert_run(make_state())

    assert store.get_run("t", "u", "s", "run-1") is not None
    assert store.get_run("t", "u", "s", "other") is None
    assert store.get_run("t2", "u", "s", "run-1") is None
    assert store.get_run("t", "u2", "s", "run-1") is None
    assert store.get_run("t", "u", "s2", "run-1") is None
    store.close()


def test_get_paused_run_returns_latest_only(tmp_path):
    store = SQLiteRunStore(tmp_path / "runs.db")
    assert store.get_paused_run("t", "u", "s") is None

    store.upsert_run(make_state(run_id="r1"))
    first = store.get_paused_run("t", "u", "s")
    assert first is not None and first.run_id == "r1"

    # 第二次暂停后返回最近的；旧运行转为完成后不再出现
    import time as _time

    _time.sleep(0.01)
    store.upsert_run(make_state(run_id="r2"))
    second = store.get_paused_run("t", "u", "s")
    assert second is not None and second.run_id == "r2"
    store.close()


def test_empty_scope_keys_are_rejected(tmp_path):
    store = SQLiteRunStore(tmp_path / "runs.db")
    with pytest.raises(ValueError):
        store.upsert_run(make_state(tenant_id=""))
    with pytest.raises(ValueError):
        store.get_run("t", "u", "s", "")
    with pytest.raises(ValueError):
        store.get_paused_run("t", "", "s")
    store.close()
