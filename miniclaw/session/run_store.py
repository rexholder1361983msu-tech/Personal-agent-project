"""SQLite RunStore：按 run_id 快照运行元数据，支撑暂停/审批/跨进程恢复。

设计要点（复用 SessionStore/EventStore 的三键模式）：
- 表 runs 主键为 (三键, run_id)；只存运行元数据（status/step/usage/
  error/pending_approval），消息仍由 SessionStore 唯一持有，两处不重复。
- upsert_run 整行覆盖写；消息的持久化游标（persisted_messages）是
  ConversationService 的内存态，不入库。
- get_run / get_paused_run 返回的 RunState 不含 messages（为空列表），
  由调用方从 SessionStore 装载后补齐；persisted_messages 由调用方设置。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from miniclaw.llm.messages import Usage
from miniclaw.runtime.state import RunState, RunStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    tenant_id       TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    run_id          TEXT NOT NULL,
    agent_id        TEXT NOT NULL,
    status          TEXT NOT NULL,
    step            INTEGER NOT NULL,
    tool_calls_used INTEGER NOT NULL,
    usage_json      TEXT NOT NULL,
    error           TEXT,
    error_kind      TEXT,
    pending_approval_json TEXT,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    PRIMARY KEY (tenant_id, user_id, session_id, run_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_scope_status
    ON runs (tenant_id, user_id, session_id, status);
"""


def _ensure_keys(*keys: str) -> None:
    if not all(keys):
        raise ValueError("scope keys (tenant_id, user_id, session_id) must be non-empty")


def _dump_pending(pending: dict | None) -> str | None:
    if pending is None:
        return None
    return json.dumps(pending, ensure_ascii=False)


def _load_pending(raw: str | None) -> dict | None:
    if raw is None:
        return None
    return json.loads(raw)


class SQLiteRunStore:
    """基于 SQLite 的 RunStore。path 为文件路径或 ":memory:"。"""

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA busy_timeout = 5000")
        if self._path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def upsert_run(self, state: RunState) -> None:
        """以 run_id 为粒度整行快照运行元数据。"""
        _ensure_keys(state.tenant_id, state.user_id, state.session_id, state.run_id)
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO runs"
                " (tenant_id, user_id, session_id, run_id, agent_id, status, step,"
                "  tool_calls_used, usage_json, error, error_kind,"
                "  pending_approval_json, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                " ON CONFLICT (tenant_id, user_id, session_id, run_id) DO UPDATE SET"
                "  agent_id = excluded.agent_id,"
                "  status = excluded.status,"
                "  step = excluded.step,"
                "  tool_calls_used = excluded.tool_calls_used,"
                "  usage_json = excluded.usage_json,"
                "  error = excluded.error,"
                "  error_kind = excluded.error_kind,"
                "  pending_approval_json = excluded.pending_approval_json,"
                "  updated_at = excluded.updated_at",
                (
                    state.tenant_id,
                    state.user_id,
                    state.session_id,
                    state.run_id,
                    state.agent_id,
                    state.status.value,
                    state.step,
                    state.tool_calls_used,
                    json.dumps(
                        {
                            "prompt_tokens": state.usage.prompt_tokens,
                            "completion_tokens": state.usage.completion_tokens,
                        }
                    ),
                    state.error,
                    state.error_kind,
                    _dump_pending(state.pending_approval),
                    state.created_at,
                    now,
                ),
            )

    def _row_to_state(self, row) -> RunState:
        (
            tenant_id,
            user_id,
            session_id,
            run_id,
            agent_id,
            status,
            step,
            tool_calls_used,
            usage_json,
            error,
            error_kind,
            pending_json,
            created_at,
            updated_at,
        ) = row
        usage = json.loads(usage_json)
        return RunState(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=session_id,
            run_id=run_id,
            agent_id=agent_id,
            status=RunStatus(status),
            step=step,
            tool_calls_used=tool_calls_used,
            usage=Usage(
                prompt_tokens=usage["prompt_tokens"],
                completion_tokens=usage["completion_tokens"],
            ),
            error=error,
            error_kind=error_kind,
            created_at=created_at,
            updated_at=updated_at,
            pending_approval=_load_pending(pending_json),
        )

    def get_run(
        self, tenant_id: str, user_id: str, session_id: str, run_id: str
    ) -> RunState | None:
        """按三键 + run_id 读取快照；messages 为空，由调用方从 SessionStore 补齐。"""
        _ensure_keys(tenant_id, user_id, session_id, run_id)
        with self._lock:
            row = self._conn.execute(
                "SELECT tenant_id, user_id, session_id, run_id, agent_id, status, step,"
                " tool_calls_used, usage_json, error, error_kind,"
                " pending_approval_json, created_at, updated_at"
                " FROM runs"
                " WHERE tenant_id = ? AND user_id = ? AND session_id = ? AND run_id = ?",
                (tenant_id, user_id, session_id, run_id),
            ).fetchone()
        return self._row_to_state(row) if row is not None else None

    def get_paused_run(
        self, tenant_id: str, user_id: str, session_id: str
    ) -> RunState | None:
        """返回该会话最近一次 PAUSED 的运行；没有则 None。"""
        _ensure_keys(tenant_id, user_id, session_id)
        with self._lock:
            row = self._conn.execute(
                "SELECT tenant_id, user_id, session_id, run_id, agent_id, status, step,"
                " tool_calls_used, usage_json, error, error_kind,"
                " pending_approval_json, created_at, updated_at"
                " FROM runs"
                " WHERE tenant_id = ? AND user_id = ? AND session_id = ?"
                "   AND status = ?"
                " ORDER BY updated_at DESC, rowid DESC LIMIT 1",
                (tenant_id, user_id, session_id, RunStatus.PAUSED.value),
            ).fetchone()
        return self._row_to_state(row) if row is not None else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
