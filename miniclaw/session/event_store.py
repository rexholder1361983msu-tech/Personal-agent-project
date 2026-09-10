"""SQLite EventStore：按 run_id 记录运行事件，还原完整调用链。

设计要点（复用 SessionStore 的三键 + seq 模式）：
- 表 run_events 按 (三键, run_id, seq) 唯一，seq 在该范围内单调递增；
  run_id 理论上全局唯一，但查询仍带全三键，不提供"只按 run_id 查"的路径。
- 单连接 + 进程内锁串行化写事务，配合 WAL；与 SessionStore 同库时
  必须使用各自独立的连接实例。
- 异步写入走"专属写线程 + FIFO 队列"（listener()）：on_event 在事件
  循环内被同步调用，不能直接写 SQLite 阻塞循环，也不能用并发 to_thread
  ——并发写入会让 seq 分配顺序背离事件发射顺序。flush() 提供确定性
  排空点，close() 排空后停线程。
- redact=True 时落库前移除敏感字段（工具入参与返回内容），为 P2-F
  脱敏预留；进程内 on_event 回调仍拿到未脱敏事件，不受影响。
"""

from __future__ import annotations

import json
import queue
import sqlite3
import sys
import threading
import time
from pathlib import Path
from typing import Protocol

from miniclaw.runtime.events import EventListener, RunEvent, RunEventType

_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant_id   TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    run_id      TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    type        TEXT NOT NULL,
    step        INTEGER NOT NULL,
    data_json   TEXT NOT NULL,
    created_at  REAL NOT NULL,
    UNIQUE (tenant_id, user_id, session_id, run_id, seq)
);

CREATE INDEX IF NOT EXISTS idx_run_events_scope_run_seq
    ON run_events (tenant_id, user_id, session_id, run_id, seq);
"""

# 脱敏时从事件 data 中整体移除的键：工具入参与工具/模型输出内容。
_REDACTED_KEYS = frozenset({"arguments", "content"})

_STOP = object()
_FLUSH = object()


def _ensure_keys(*keys: str) -> None:
    if not all(keys):
        raise ValueError("scope keys (tenant_id, user_id, session_id) must be non-empty")


class EventStore(Protocol):
    """事件存储契约。实现必须保证三键隔离，空键视为编程错误。"""

    def append_event(
        self,
        tenant_id: str,
        user_id: str,
        session_id: str,
        run_id: str,
        event: RunEvent,
    ) -> None: ...

    def get_events(
        self, tenant_id: str, user_id: str, session_id: str, run_id: str
    ) -> list[RunEvent]: ...

    def close(self) -> None: ...


class SQLiteEventStore:
    """基于 SQLite 的 EventStore。path 为文件路径或 ":memory:"。"""

    def __init__(self, path: str | Path, *, redact: bool = False) -> None:
        self._path = str(path)
        self._redact = redact
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.execute("PRAGMA busy_timeout = 5000")
        if self._path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._flushed = threading.Event()
        self._flush_lock = threading.Lock()
        self._writer = threading.Thread(
            target=self._write_loop, name="miniclaw-event-writer", daemon=True
        )
        self._writer.start()

    # -- 同步 API（测试与外部直接写入） -------------------------------

    def append_event(
        self,
        tenant_id: str,
        user_id: str,
        session_id: str,
        run_id: str,
        event: RunEvent,
    ) -> None:
        _ensure_keys(tenant_id, user_id, session_id, run_id)
        data = dict(event.data)
        if self._redact:
            for key in _REDACTED_KEYS:
                data.pop(key, None)
        now = time.time()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(seq), -1) FROM run_events"
                " WHERE tenant_id = ? AND user_id = ? AND session_id = ? AND run_id = ?",
                (tenant_id, user_id, session_id, run_id),
            ).fetchone()
            self._conn.execute(
                "INSERT INTO run_events"
                " (tenant_id, user_id, session_id, run_id, seq, type, step, data_json, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    tenant_id,
                    user_id,
                    session_id,
                    run_id,
                    row[0] + 1,
                    event.type.value,
                    event.step,
                    json.dumps(data, ensure_ascii=False),
                    now,
                ),
            )

    def get_events(
        self, tenant_id: str, user_id: str, session_id: str, run_id: str
    ) -> list[RunEvent]:
        _ensure_keys(tenant_id, user_id, session_id, run_id)
        with self._lock:
            rows = self._conn.execute(
                "SELECT type, step, data_json FROM run_events"
                " WHERE tenant_id = ? AND user_id = ? AND session_id = ? AND run_id = ?"
                " ORDER BY seq ASC",
                (tenant_id, user_id, session_id, run_id),
            ).fetchall()
        return [
            RunEvent(
                type=RunEventType(event_type),
                run_id=run_id,
                step=step,
                data=json.loads(data_json),
                tenant_id=tenant_id,
                user_id=user_id,
                session_id=session_id,
            )
            for event_type, step, data_json in rows
        ]

    # -- 异步写入口（runtime on_event 桥接） ---------------------------

    def listener(self) -> EventListener:
        """返回可挂到 AgentRuntime.on_event 的回调：入队即返回，不阻塞事件循环。"""

        def on_event(event: RunEvent) -> None:
            self._queue.put(
                (event.tenant_id, event.user_id, event.session_id, event.run_id, event)
            )

        return on_event

    def _write_loop(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            if item is _FLUSH:
                self._flushed.set()
                continue
            tenant_id, user_id, session_id, run_id, event = item
            try:
                self.append_event(tenant_id, user_id, session_id, run_id, event)
            except Exception as exc:  # 写失败不拖垮运行，也不静默
                print(
                    f"event store write failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )

    def flush(self, timeout_s: float = 10.0) -> bool:
        """阻塞等待已入队事件全部落库。返回 False 表示超时。"""
        with self._flush_lock:
            self._flushed.clear()
            self._queue.put(_FLUSH)
            return self._flushed.wait(timeout_s)

    def close(self) -> None:
        self._queue.put(_STOP)
        self._writer.join(timeout=10.0)
        with self._lock:
            self._conn.close()


def make_store_listener(store: SQLiteEventStore) -> EventListener:
    """桥接辅助：等价于 store.listener()。"""
    return store.listener()
