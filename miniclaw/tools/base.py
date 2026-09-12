"""工具协议：ToolContext 携带运行身份，Tool 是执行契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ToolContext:
    """工具执行时的运行身份。权限与审计（第二阶段）也以此为输入。"""

    tenant_id: str
    user_id: str
    session_id: str
    run_id: str
    agent_id: str = "default"


class ToolDisabledError(RuntimeError):
    """工具存在但未启用（未配置执行器 / sandbox 不可用）。"""


class Tool(Protocol):
    """工具契约。

    - name/description/parameters：提交给模型的描述（parameters 为 JSON Schema）。
    - execute 返回字符串结果回填给模型；抛出的异常由 Runtime 转成
      tool_error 消息回填，不会终止运行。
    """

    name: str
    description: str
    parameters: dict

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str: ...
