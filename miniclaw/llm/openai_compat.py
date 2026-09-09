"""OpenAI-compatible Chat Completions 适配器。

只依赖 httpx；测试通过注入 httpx.AsyncClient（MockTransport）零网络运行。
"""

from __future__ import annotations

from typing import Any, Sequence

import httpx

from miniclaw.llm.base import ModelResponse, ToolSpec
from miniclaw.llm.messages import Message, ToolCall, Usage


def message_to_payload(message: Message) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role}
    if message.content is not None:
        payload["content"] = message.content
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    return payload


class OpenAICompatModel:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        *,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._client = client or httpx.AsyncClient(timeout=timeout_s)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "temperature": self._temperature,
            "messages": [message_to_payload(m) for m in messages],
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            payload["tool_choice"] = "auto"

        response = await self._client.post(
            f"{self._api_base}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        response.raise_for_status()
        data = response.json()

        choice = data["choices"][0]
        raw_message = choice["message"]
        tool_calls = [
            ToolCall(
                id=tc["id"],
                name=tc["function"]["name"],
                arguments=tc["function"].get("arguments") or "{}",
            )
            for tc in raw_message.get("tool_calls") or []
        ]
        usage = data.get("usage") or {}
        return ModelResponse(
            message=Message(
                role="assistant",
                content=raw_message.get("content"),
                tool_calls=tool_calls or None,
            ),
            usage=Usage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
            ),
            finish_reason=choice.get("finish_reason"),
        )
