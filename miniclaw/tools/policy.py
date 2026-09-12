"""ToolPolicy：按工具名声明的权限与执行约束，独立于工具实现。

判定单一来源：mode 决定调用是否执行 / 暂停待审批 / 直接拒绝；
timeout_s 与 output_limit 是执行约束（覆盖全局默认，超时取更严格者）；
network 标记是 P2-E sandbox 的声明输入，P2-D 本身不消费。

默认规则全 allow：未注入 policy 或未列名的工具行为与此前完全一致。
default 也可收紧为 deny，把规则表用作白名单。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ToolMode(str, Enum):
    ALLOW = "allow"
    NEEDS_APPROVAL = "needs_approval"
    DENY = "deny"


@dataclass(frozen=True)
class ToolRule:
    """单个工具的规则。None 字段表示"沿用全局默认 / 不约束"。"""

    mode: ToolMode = ToolMode.ALLOW
    timeout_s: float | None = None
    output_limit: int | None = None
    network: bool = False

    def __post_init__(self) -> None:
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0 when set")
        if self.output_limit is not None and self.output_limit < 1:
            raise ValueError("output_limit must be >= 1 when set")

    def effective_timeout(self, default_s: float) -> float:
        """与全局默认取更严格者（较小值）。"""
        if self.timeout_s is None:
            return default_s
        return min(self.timeout_s, default_s)

    def truncate(self, content: str) -> str:
        """超出 output_limit 时截断并追加标记；模型与审计看到的都是截断后内容。"""
        if self.output_limit is not None and len(content) > self.output_limit:
            return content[: self.output_limit] + "...[truncated]"
        return content


DEFAULT_RULE = ToolRule()


class ToolPolicy:
    """按工具名的规则表。未列名的工具回落 default（默认全 allow）。

    rules 由组装方（build_runtime）一次性给出，运行期只读。
    """

    def __init__(
        self, rules: dict[str, ToolRule] | None = None, *, default: ToolRule = DEFAULT_RULE
    ) -> None:
        self.rules: dict[str, ToolRule] = dict(rules or {})
        self.default = default

    def rule_for(self, name: str) -> ToolRule:
        return self.rules.get(name, self.default)

    def decide(self, name: str) -> ToolMode:
        return self.rule_for(name).mode
