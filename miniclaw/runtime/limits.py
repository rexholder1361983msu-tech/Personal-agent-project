"""RunLimits：一次运行的资源上限。

None 表示不限制。单工具超时由 tool_timeout_s 控制；
取消能力见 cancel.py 的 CancelToken。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RunLimits:
    max_steps: int = 8
    max_tool_calls: int = 24
    tool_timeout_s: float = 10.0
    max_total_tokens: int | None = None
    max_run_duration_s: float | None = None

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be >= 1")
        if self.tool_timeout_s <= 0:
            raise ValueError("tool_timeout_s must be > 0")
        if self.max_total_tokens is not None and self.max_total_tokens < 1:
            raise ValueError("max_total_tokens must be >= 1 when set")
        if self.max_run_duration_s is not None and self.max_run_duration_s < 0:
            raise ValueError("max_run_duration_s must be >= 0 when set")
