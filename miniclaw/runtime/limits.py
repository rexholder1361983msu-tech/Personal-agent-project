"""RunLimits：一次运行的资源上限。

token 预算、总耗时、并发数与运行取消属于第二阶段。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunLimits:
    max_steps: int = 8
    max_tool_calls: int = 24
    tool_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be >= 1")
        if self.tool_timeout_s <= 0:
            raise ValueError("tool_timeout_s must be > 0")
