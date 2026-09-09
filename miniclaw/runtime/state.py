"""RunState：一次运行（run）的全部可变状态。

RunState 显式传入 AgentRuntime，Runtime 本身不持有任何会话状态。
持久化由 SessionStore 负责：本阶段只持久化对话消息，
status/step/usage 的跨进程恢复（暂停/审批/恢复）属于第二阶段。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from miniclaw.llm.messages import Message, Usage


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    # PAUSED 保留给第二阶段的审批/人工介入，本阶段不会产生。
    PAUSED = "paused"


@dataclass
class RunState:
    """一次运行的状态。由 ConversationService 创建与维护，Runtime 只读写。"""

    tenant_id: str
    user_id: str
    session_id: str
    run_id: str = field(default_factory=lambda: uuid4().hex)
    agent_id: str = "default"
    status: RunStatus = RunStatus.RUNNING
    step: int = 0
    tool_calls_used: int = 0
    messages: list[Message] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    error: str | None = None
    # 机器可读的错误类别：model_error / budget_exceeded / deadline_exceeded /
    # cancelled / runtime_error。error 前缀与之一致，保持人读兼容。
    error_kind: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    # 已写入 SessionStore 的消息数量，由 ConversationService 维护，
    # Runtime 与网关不应依赖它。save 时只追加其之后的新消息。
    persisted_messages: int = 0
