"""模型客户端协议与请求/响应类型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from miniclaw.llm.messages import Message, Usage


@dataclass
class ToolSpec:
    """提交给模型的工具描述（JSON Schema）。"""

    name: str
    description: str
    parameters: dict


@dataclass
class ModelResponse:
    """一次模型调用返回的助手消息与用量。"""

    message: Message
    usage: Usage = field(default_factory=Usage)
    finish_reason: str | None = None  # "stop" | "tool_calls" | ...


class ModelClient(Protocol):
    """模型协议。测试注入 ScriptedModel，生产使用 OpenAI-compatible 适配器。"""

    async def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> ModelResponse: ...
