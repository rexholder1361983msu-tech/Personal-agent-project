"""高风险工具：executor 支撑的 shell/python_eval 与仍禁用的占位。

- ShellTool / PythonEvalTool：构造时传入 ToolExecutor 即激活；裸构造保持
  禁用占位语义（execute 抛 ToolDisabledError），默认不注册进 Registry。
  激活时的隔离级别由传入的 executor 决定（本地受限 subprocess 或 Docker sandbox）。
- file_read / file_write / web_fetch：仍为禁用占位——路径圈禁与 SSRF 校验
  属后续切片，不提供半吊子实现。

字符串黑名单不是安全边界，隔离只由 executor（sandbox）提供；
ToolDisabledError 定义在 base.py，此处再导出保持旧导入路径可用。
"""

from __future__ import annotations

import os
from typing import Any

from miniclaw.tools.base import ToolContext, ToolDisabledError
from miniclaw.tools.executor import ExecutionSpec, ToolExecutor

__all__ = [
    "FileReadTool",
    "FileWriteTool",
    "PythonEvalTool",
    "ShellTool",
    "ToolDisabledError",
    "WebFetchTool",
]


class _DisabledTool:
    enabled = False
    parameters: dict = {"type": "object", "properties": {}}

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        raise ToolDisabledError(
            f"tool '{self.name}' is disabled: requires the phase-2 sandbox/policy layer"
        )


class FileReadTool(_DisabledTool):
    name = "file_read"
    description = "Read a file from disk. DISABLED until path confinement lands."


class FileWriteTool(_DisabledTool):
    name = "file_write"
    description = "Write a file to disk. DISABLED until path confinement lands."


class WebFetchTool(_DisabledTool):
    name = "web_fetch"
    description = "Fetch a URL. DISABLED until SSRF checks land."


class ExecutorTool:
    """把 ToolExecutor 包成 Tool 协议的基类。

    不传 executor 时保持禁用占位语义（兼容旧测试与"默认不启用"约定）；
    传了 executor 后，execute = 校验参数 → build_spec → executor.run。
    executor 抛出的异常（超时/非零退出/禁用）由 Runtime 转成 tool_error 回填。
    """

    name = "executor_tool"
    description = "Base class; override name/parameters/argument_name/build_spec."
    parameters: dict = {"type": "object", "properties": {}}
    argument_name = "input"

    def __init__(self, executor: ToolExecutor | None = None) -> None:
        self._executor = executor

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        if self._executor is None:
            raise ToolDisabledError(f"tool '{self.name}' is disabled: no executor configured")
        code = arguments.get(self.argument_name)
        if not isinstance(code, str) or not code.strip():
            raise ValueError(
                f"{self.name} requires a non-empty string '{self.argument_name}' argument"
            )
        return await self._executor.run(self.build_spec(code), context)

    def build_spec(self, code: str) -> ExecutionSpec:
        raise NotImplementedError


class ShellTool(ExecutorTool):
    name = "shell"
    description = (
        "Execute a shell command through the configured executor "
        "(restricted subprocess or sandbox). Requires approval by default."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run."}
        },
        "required": ["command"],
    }
    argument_name = "command"

    def build_spec(self, command: str) -> ExecutionSpec:
        if os.name == "nt":
            return ExecutionSpec(argv=("cmd", "/c", command))
        return ExecutionSpec(argv=("sh", "-c", command))


class PythonEvalTool(ExecutorTool):
    name = "python_eval"
    description = (
        "Run Python code (isolated mode, source on stdin) through the configured "
        "executor. Requires approval by default."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python source code to execute."}
        },
        "required": ["code"],
    }
    argument_name = "code"

    def build_spec(self, code: str) -> ExecutionSpec:
        return ExecutionSpec(argv=("python", "-I", "-"), stdin_text=code)
