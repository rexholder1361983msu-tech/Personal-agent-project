"""OpenAI-compatible 适配器测试：MockTransport 零网络，只验证协议形状。"""

from __future__ import annotations

import json

import httpx
import pytest

from miniclaw.llm.base import ToolSpec
from miniclaw.llm.messages import Message, ToolCall, Usage
from miniclaw.llm.openai_compat import OpenAICompatModel


def make_model(handler) -> tuple[OpenAICompatModel, list[dict]]:
    captured: list[dict] = []

    def capturing_handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(capturing_handler))
    return (
        OpenAICompatModel(
            api_base="https://example.test/v1",
            api_key="secret",
            model="m-1",
            client=client,
        ),
        captured,
    )


async def test_request_shape_and_response_parsing():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-9",
                                    "type": "function",
                                    "function": {"name": "echo", "arguments": '{"text": "hi"}'},
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 7},
            },
        )

    model, captured = make_model(handler)
    response = await model.complete(
        [Message(role="user", content="hi")],
        [ToolSpec(name="echo", description="d", parameters={"type": "object"})],
    )

    assert captured[0]["model"] == "m-1"
    assert captured[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert captured[0]["tools"] == [
        {"type": "function", "function": {"name": "echo", "description": "d", "parameters": {"type": "object"}}}
    ]
    assert captured[0]["tool_choice"] == "auto"
    assert response.finish_reason == "tool_calls"
    assert response.message.tool_calls == [ToolCall(id="call-9", name="echo", arguments='{"text": "hi"}')]
    assert response.usage == Usage(prompt_tokens=5, completion_tokens=7)


async def test_tool_history_is_serialized():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
                "usage": {},
            },
        )

    model, captured = make_model(handler)
    await model.complete(
        [
            Message(role="user", content="hi"),
            Message(role="assistant", content=None, tool_calls=[ToolCall(id="c1", name="echo", arguments="{}")]),
            Message(role="tool", content="ECHOED", tool_call_id="c1"),
        ],
        [],
    )

    messages = captured[0]["messages"]
    assert messages[0] == {"role": "user", "content": "hi"}
    assert messages[1]["tool_calls"] == [
        {"id": "c1", "type": "function", "function": {"name": "echo", "arguments": "{}"}}
    ]
    assert messages[2] == {"role": "tool", "content": "ECHOED", "tool_call_id": "c1"}
    assert "tools" not in captured[0]  # 无工具时不传 tools


async def test_http_error_raises():
    model, _ = make_model(lambda request: httpx.Response(500, json={"error": "boom"}))
    with pytest.raises(httpx.HTTPStatusError):
        await model.complete([Message(role="user", content="hi")], [])
