"""工具协议、注册表与内置工具的测试。"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from miniclaw.tools.base import ToolContext
from miniclaw.tools.builtin import CalculatorTool, CurrentTimeTool, EchoTool, default_tools
from miniclaw.tools.registry import ToolRegistry
from miniclaw.tools.unsafe import ShellTool, ToolDisabledError


def make_context() -> ToolContext:
    return ToolContext(
        tenant_id="t", user_id="u", session_id="s", run_id="r", agent_id="default"
    )


async def test_echo_tool():
    tool = EchoTool()
    assert await tool.execute({"text": "hello"}, make_context()) == "hello"
    with pytest.raises(ValueError):
        await tool.execute({}, make_context())


async def test_current_time_tool_with_injected_clock():
    fixed = datetime(2026, 9, 9, 12, 0, 0, tzinfo=timezone.utc)
    tool = CurrentTimeTool(clock=lambda: fixed)
    assert await tool.execute({}, make_context()) == "2026-09-09T12:00:00+00:00"


async def test_calculator_tool_arithmetic():
    tool = CalculatorTool()
    assert await tool.execute({"expression": "2 + 3 * 4"}, make_context()) == "14"
    assert await tool.execute({"expression": "(2 + 3) * 4"}, make_context()) == "20"
    assert await tool.execute({"expression": "-2 ** 2"}, make_context()) == "-4"
    assert await tool.execute({"expression": "7 // 2"}, make_context()) == "3"


async def test_calculator_rejects_non_arithmetic():
    tool = CalculatorTool()
    for expression in ["__import__('os').system('id')", "print(1)", "1 if True else 2", "x"]:
        with pytest.raises((ValueError, SyntaxError)):
            await tool.execute({"expression": expression}, make_context())


async def test_calculator_division_by_zero_propagates():
    tool = CalculatorTool()
    with pytest.raises(ZeroDivisionError):
        await tool.execute({"expression": "1 / 0"}, make_context())


def test_registry_register_and_specs_sorted():
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(CalculatorTool())
    registry.register(CurrentTimeTool())
    assert registry.names() == ["calculator", "current_time", "echo"]
    assert [s.name for s in registry.specs()] == ["calculator", "current_time", "echo"]
    spec = registry.get("echo") is not None
    assert spec
    assert "echo" in registry
    assert registry.get("missing") is None


def test_registry_rejects_duplicate_names():
    registry = ToolRegistry()
    registry.register(EchoTool())
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(EchoTool())


def test_default_tools_are_all_safe_names():
    names = {tool.name for tool in default_tools()}
    assert names == {"echo", "current_time", "calculator"}


async def test_unsafe_tools_are_disabled_by_default():
    tool = ShellTool()
    with pytest.raises(ToolDisabledError):
        await tool.execute({"command": "ls"}, make_context())
