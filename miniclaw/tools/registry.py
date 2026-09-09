"""ToolRegistry：工具注册表与模型侧 ToolSpec 的唯一来源。"""

from __future__ import annotations

from typing import Any

from miniclaw.llm.base import ToolSpec
from miniclaw.tools.base import Tool


class ToolRegistry:
    """按名称注册工具。specs() 按名称排序，保证模型看到的工具顺序确定。"""

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}

    def register(self, tool: Tool) -> None:
        name = tool.name
        if not name:
            raise ValueError("tool name must be non-empty")
        if name in self._tools:
            raise ValueError(f"duplicate tool name: {name}")
        self._tools[name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(name=tool.name, description=tool.description, parameters=tool.parameters)
            for _, tool in sorted(self._tools.items())
        ]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
