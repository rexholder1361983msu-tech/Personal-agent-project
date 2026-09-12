"""SkillAsTool：技能包装成工具，子任务走同一个 AgentRuntime。

设计约束（对应调研结论）：技能执行不得复制第二套 Agent 循环。
- 子运行使用注册时的同一个 runtime 实例（通过 runtime_ref 晚绑定）；
- 子会话 scope 为 session_id + "/skills/<name>"，tenant/user 沿用当前运行；
- 子运行可见工具排除所有 skill_* 工具，防止技能递归调用；
- 子运行的对话仅用于得出工具结果，不单独持久化（父会话保存工具结果）。
"""

from __future__ import annotations

from typing import Any, Callable

from miniclaw.runtime.loop import AgentRuntime, final_reply
from miniclaw.runtime.state import RunState, RunStatus
from miniclaw.skills.loader import Skill
from miniclaw.tools.base import ToolContext


class SkillAsTool:
    def __init__(self, skill: Skill, runtime_ref: Callable[[], AgentRuntime]) -> None:
        self._skill = skill
        self._runtime_ref = runtime_ref

    @property
    def name(self) -> str:
        return f"skill_{self._skill.name}"

    @property
    def description(self) -> str:
        return self._skill.description

    parameters: dict = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "交给该技能完成的任务描述。",
            }
        },
        "required": ["task"],
    }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        task = arguments.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("skill tool requires a non-empty 'task' string argument")

        runtime = self._runtime_ref()
        allowed = [n for n in runtime.tools.names() if not n.startswith("skill_")]
        child_state = RunState(
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            session_id=f"{context.session_id}/skills/{self._skill.name}",
        )
        child_state = await runtime.run(child_state, task, tool_names=allowed)
        if child_state.status is RunStatus.FAILED:
            return f"skill_error: {child_state.error}"
        if child_state.status is RunStatus.PAUSED:
            # 子运行等待审批时没有终答；如实上报，父运行可决定下一步。
            return "skill_error: sub-run paused awaiting tool approval"
        return final_reply(child_state) or ""
