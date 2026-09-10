"""轻量运行事件。

仅用于测试断言与最小观测；持久化 EventStore 与标准 tracing 属于第二阶段。
回调应快速返回且不抛异常。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class RunEventType(str, Enum):
    RUN_STARTED = "run_started"
    MODEL_REQUESTED = "model_requested"
    TOOL_CALLED = "tool_called"
    TOOL_RETURNED = "tool_returned"
    RUN_PAUSED = "run_paused"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


@dataclass
class RunEvent:
    type: RunEventType
    run_id: str
    step: int
    data: dict[str, Any] = field(default_factory=dict)
    # 三键在运行时由 _emit 从 RunState 填充；EventStore 落库与按 scope 查询依赖它们。
    tenant_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None


EventListener = Callable[[RunEvent], None]
