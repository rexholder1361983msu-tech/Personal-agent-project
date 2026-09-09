"""SQLite SessionStore：按 tenant_id/user_id/session_id 三键隔离的会话持久化。

设计要点：
- 三键是会话的完整身份，所有查询都必须带全三键；不存在"只按 session_id 查"的路径。
- 单连接 + 进程内锁串行化写事务，配合 WAL 保证多线程下的确定性；
  跨进程并发留到后续换 PostgreSQL/Redis 时解决。
- 消息按 (三键, seq) 唯一，seq 在三键范围内单调递增，批量追加保持顺序。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from miniclaw.llm.messages import Message, ToolCall

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    tenant_id  TEXT NOT NULL,
    user_id    TEXT NOT NULL,
    session_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (tenant_id, user_id, session_id)
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id       TEXT NOT NULL,
    user_id         TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT,
    tool_call_id    TEXT,
    tool_calls_json TEXT,
    created_at      REAL NOT NULL,
    UNIQUE (tenant_id, user_id, session_id, seq),
    FOREIGN KEY (tenant_id, user_id, session_id)
        REFERENCES sessions (tenant_id, user_id, session_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_scope_seq
    ON messages (tenant_id, user_id, session_id, seq);
"""


@dataclass
class SessionMeta:
    tenant_id: str
    user_id: str
    session_id: str
    created_at: float
    updated_at: float
    message_count: int


class SessionStore(Protocol):
    """会话存储契约。实现必须保证三键隔离，空键视为编程错误。"""

    def append_messages(
        self, tenant_id: str, user_id: str, session_id: str, messages: Sequence[Message]
    ) -> int: ...

    def get_messages(self, tenant_id: str, user_id: str, session_id: str) -> list[Message]: ...

    def list_sessions(self, tenant_id: str, user_id: str) -> list[SessionMeta]: ...

    def count_messages(self, tenant_id: str, user_id: str, session_id: str) -> int: ...

    def clear(self, tenant_id: str, user_id: str, session_id: str) -> int: ...

    def close(self) -> None: ...


def _ensure_keys(*keys: str) -> None:
    """校验方法实际收到的三键（或 list_sessions 的两键）非空。"""
    if not all(keys):
        raise ValueError("scope keys (tenant_id, user_id, session_id) must be non-empty")


def _dump_tool_calls(tool_calls: list[ToolCall] | None) -> str | None:
    if not tool_calls:
        return None
    return json.dumps(
        [{"id": c.id, "name": c.name, "arguments": c.arguments} for c in tool_calls],
        ensure_ascii=False,
    )


def _load_tool_calls(raw: str | None) -> list[ToolCall] | None:
    if raw is None:
        return None
    return [
        ToolCall(id=d["id"], name=d["name"], arguments=d.get("arguments", "{}"))
        for d in json.loads(raw)
    ]


class SQLiteSessionStore:
    """基于 SQLite 的 SessionStore。path 为文件路径或 ":memory:"。"""

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

    def append_messages(
        self, tenant_id: str, user_id: str, session_id: str, messages: Sequence[Message]
    ) -> int:
        _ensure_keys(tenant_id, user_id, session_id)
        if not messages:
            return 0
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO sessions (tenant_id, user_id, session_id, created_at, updated_at)"
                " VALUES (?, ?, ?, ?, ?)"
                " ON CONFLICT (tenant_id, user_id, session_id)"
                " DO UPDATE SET updated_at = excluded.updated_at",
                (tenant_id, user_id, session_id, now, now),
            )
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), -1) FROM messages"
                " WHERE tenant_id = ? AND user_id = ? AND session_id = ?",
                (tenant_id, user_id, session_id),
            ).fetchone()
            seq = row[0] + 1
            for message in messages:
                self._conn.execute(
                    "INSERT INTO messages"
                    " (tenant_id, user_id, session_id, seq, role, content,"
                    "  tool_call_id, tool_calls_json, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        tenant_id,
                        user_id,
                        session_id,
                        seq,
                        message.role,
                        message.content,
                        message.tool_call_id,
                        _dump_tool_calls(message.tool_calls),
                        now,
                    ),
                )
                seq += 1
        return len(messages)

    def get_messages(self, tenant_id: str, user_id: str, session_id: str) -> list[Message]:
        _ensure_keys(tenant_id, user_id, session_id)
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content, tool_call_id, tool_calls_json FROM messages"
                " WHERE tenant_id = ? AND user_id = ? AND session_id = ?"
                " ORDER BY seq ASC",
                (tenant_id, user_id, session_id),
            ).fetchall()
        return [
            Message(
                role=role,
                content=content,
                tool_calls=_load_tool_calls(tool_calls_json),
                tool_call_id=tool_call_id,
            )
            for role, content, tool_call_id, tool_calls_json in rows
        ]

    def list_sessions(self, tenant_id: str, user_id: str) -> list[SessionMeta]:
        _ensure_keys(tenant_id, user_id)
        with self._lock:
            rows = self._conn.execute(
                "SELECT s.session_id, s.created_at, s.updated_at, COUNT(m.id)"
                " FROM sessions s"
                " LEFT JOIN messages m"
                "   ON m.tenant_id = s.tenant_id AND m.user_id = s.user_id"
                "  AND m.session_id = s.session_id"
                " WHERE s.tenant_id = ? AND s.user_id = ?"
                " GROUP BY s.session_id, s.created_at, s.updated_at"
                " ORDER BY s.updated_at DESC",
                (tenant_id, user_id),
            ).fetchall()
        return [
            SessionMeta(
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
                created_at=created_at,
                updated_at=updated_at,
                message_count=count,
            )
            for session_id, created_at, updated_at, count in rows
        ]

    def count_messages(self, tenant_id: str, user_id: str, session_id: str) -> int:
        _ensure_keys(tenant_id, user_id, session_id)
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM messages"
                " WHERE tenant_id = ? AND user_id = ? AND session_id = ?",
                (tenant_id, user_id, session_id),
            ).fetchone()
        return row[0]

    def clear(self, tenant_id: str, user_id: str, session_id: str) -> int:
        _ensure_keys(tenant_id, user_id, session_id)
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "DELETE FROM messages"
                " WHERE tenant_id = ? AND user_id = ? AND session_id = ?",
                (tenant_id, user_id, session_id),
            )
            self._conn.execute(
                "DELETE FROM sessions"
                " WHERE tenant_id = ? AND user_id = ? AND session_id = ?",
                (tenant_id, user_id, session_id),
            )
        return cursor.rowcount

    def close(self) -> None:
        with self._lock:
            self._conn.close()
