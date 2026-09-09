"""ScriptedModel：确定性测试模型。

按脚本顺序返回预设响应，不访问任何真实 LLM；每次请求的
messages 与 tools 都被记录下来，供测试断言"模型实际看到了什么"。
"""

from __future__ import annotations

from typing import Sequence

from miniclaw.llm.base import ModelResponse, ToolSpec
from miniclaw.llm.messages import Message, ToolCall, Usage


def text_response(content: str, *, usage: Usage | None = None) -> ModelResponse:
    """构造一条纯文本终答响应。"""
    return ModelResponse(
        message=Message(role="assistant", content=content),
        usage=usage or Usage(prompt_tokens=1, completion_tokens=1),
        finish_reason="stop",
    )


def tool_call_response(
    call_id: str, name: str, arguments: str = "{}", *, usage: Usage | None = None
) -> ModelResponse:
    """构造一条请求单个工具调用的响应。"""
    return ModelResponse(
        message=Message(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id=call_id, name=name, arguments=arguments)],
        ),
        usage=usage or Usage(prompt_tokens=1, completion_tokens=1),
        finish_reason="tool_calls",
    )


class ScriptedModel:
    """脚本队列模型。队列耗尽且未开启 repeat_last 时抛出 AssertionError。"""

    def __init__(self, *responses: ModelResponse, repeat_last: bool = False) -> None:
        self._responses = list(responses)
        self._repeat_last = repeat_last
        self._last = responses[-1] if responses else None
        self.requests: list[list[Message]] = []
        self.tool_specs: list[list[ToolSpec]] = []

    async def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> ModelResponse:
        # 浅拷贝列表：Message 对象本身在创建后不再被修改。
        self.requests.append(list(messages))
        self.tool_specs.append(list(tools))
        if self._responses:
            return self._responses.pop(0)
        if self._repeat_last and self._last is not None:
            return self._last
        raise AssertionError("ScriptedModel: no scripted response left")
