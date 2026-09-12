"""RunState：一次运行（run）的全部可变状态。

RunState 显式传入 AgentRuntime，Runtime 本身不持有任何会话状态。
持久化分工：对话消息由 SessionStore 负责（增量追加）；
status/step/usage/pending_approval 等运行元数据由 RunStore 以
run_id 为粒度快照（第二阶段 P2-C 起），用于暂停/审批/跨进程恢复。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from uuid import uuid4

from miniclaw.llm.messages import Message, Usage


class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    # 审批挂起：approval_gate 判定工具调用需人工审批时进入；
    # 恢复（批准/拒绝）后转回 RUNNING，直至 COMPLETED/FAILED。
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
    # 审批挂起信息：{"call": {"id","name","arguments"}, "remaining": [...]}。
    # AgentRuntime 暂停时写入、恢复时消费并清空；RunStore 落库为 JSON。
    # remaining 是同一响应中排在该调用之后、尚未处理的工具调用，
    # 保证恢复后每个 tool_call 都有结果回填。
    pending_approval: dict[str, Any] | None = None
