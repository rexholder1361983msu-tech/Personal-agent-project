"""高风险工具的接口占位。

这些类只保留 Tool 协议形状，execute 一律抛出 ToolDisabledError。
真正的安全隔离（Docker sandbox、权限策略、审批）属于第二阶段；
在此之前它们不应该被注册进任何默认 Registry。
字符串黑名单不是安全边界，这里干脆不提供黑名单实现。
"""

from __future__ import annotations

from typing import Any

from miniclaw.tools.base import ToolContext


class ToolDisabledError(RuntimeError):
    """工具存在但未启用（等待第二阶段的 sandbox/策略层）。"""


class _DisabledTool:
    enabled = False
    parameters: dict = {"type": "object", "properties": {}}

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        raise ToolDisabledError(
            f"tool '{self.name}' is disabled: requires the phase-2 sandbox/policy layer"
        )


class ShellTool(_DisabledTool):
    name = "shell"
    description = "Execute a shell command. DISABLED until the phase-2 sandbox."


class PythonEvalTool(_DisabledTool):
    name = "python_eval"
    description = "Evaluate Python code. DISABLED until the phase-2 sandbox."


class FileReadTool(_DisabledTool):
    name = "file_read"
    description = "Read a file from disk. DISABLED until the phase-2 policy layer."


class FileWriteTool(_DisabledTool):
    name = "file_write"
    description = "Write a file to disk. DISABLED until the phase-2 policy layer."


class WebFetchTool(_DisabledTool):
    name = "web_fetch"
    description = "Fetch a URL. DISABLED until the phase-2 policy layer (SSRF checks)."
