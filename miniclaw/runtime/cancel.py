"""协作式取消令牌。

cancel() 可从任意线程/任务调用；AgentRuntime 在检查点
（每次模型请求前、每次工具执行前）查看并终止运行。
不做抢占式取消：正在执行的工具会自然结束，之后运行才停止。
"""

from __future__ import annotations

import threading


class CancelToken:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._reason = ""

    def cancel(self, reason: str = "cancelled by caller") -> None:
        with self._lock:
            if not self._cancelled:
                self._cancelled = True
                self._reason = reason

    @property
    def cancelled(self) -> bool:
        with self._lock:
            return self._cancelled

    @property
    def reason(self) -> str:
        with self._lock:
            return self._reason
