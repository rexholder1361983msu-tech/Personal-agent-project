"""内置安全工具：纯函数、进程内执行、无副作用。

高风险工具（shell/python_eval/文件/网络）见 unsafe.py：
只保留接口且默认禁用，真正的隔离等待第二阶段的 sandbox。
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from typing import Any, Callable

from miniclaw.tools.base import ToolContext

_NUMBER_TYPES = (int, float)


class EchoTool:
    """原样返回 text 参数，用于连通性与测试。"""

    name = "echo"
    description = "Echo back the given text. Useful for connectivity checks."
    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Text to echo back."}},
        "required": ["text"],
    }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        text = arguments.get("text")
        if not isinstance(text, str):
            raise ValueError("echo requires a string 'text' argument")
        return text


class CurrentTimeTool:
    """返回当前 UTC 时间（ISO 8601）。clock 可注入以便确定性测试。"""

    name = "current_time"
    description = "Return the current UTC time in ISO 8601 format."
    parameters = {"type": "object", "properties": {}}

    def __init__(self, clock: Callable[[], datetime] | None = None) -> None:
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        return self._clock().isoformat()


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def _eval_arithmetic(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, _NUMBER_TYPES) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BINOPS):
        left, right = _eval_arithmetic(node.left), _eval_arithmetic(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left**right
        return left // right
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARYOPS):
        value = _eval_arithmetic(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


class CalculatorTool:
    """四则运算计算器。

    用 AST 白名单求值（仅数字与 + - * / % ** //、一元正负、括号），
    不是 eval；非算术语法一律拒绝。
    """

    name = "calculator"
    description = "Evaluate an arithmetic expression with + - * / % ** // and parentheses."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Arithmetic expression, e.g. '2 + 3 * 4'."}
        },
        "required": ["expression"],
    }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        expression = arguments.get("expression")
        if not isinstance(expression, str):
            raise ValueError("calculator requires a string 'expression' argument")
        tree = ast.parse(expression, mode="eval")
        result = _eval_arithmetic(tree.body)
        return str(result)


def default_tools() -> list:
    """默认注册进 Registry 的安全工具集合。"""
    return [EchoTool(), CurrentTimeTool(), CalculatorTool()]
