"""模型协议的公共域对象。

Runtime、SessionStore 与模型适配器都以这些类型为契约，
不依赖任何具体模型 SDK。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToolCall:
    """模型发起的一次工具调用；arguments 是 JSON 编码的参数字符串。"""

    id: str
    name: str
    arguments: str = "{}"


@dataclass
class Message:
    """OpenAI 风格的会话消息。

    role 取值："system" | "user" | "assistant" | "tool"。
    tool_calls 只出现在 assistant 消息上；tool_call_id 只出现在 tool 消息上，
    用于把工具结果回填到对应的调用。
    """

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


@dataclass
class Usage:
    """一次运行的 token 用量累计。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
