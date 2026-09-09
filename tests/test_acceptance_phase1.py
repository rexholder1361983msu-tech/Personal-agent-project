"""第一阶段验收测试：逐条映射 docs/AGENT_BUILD_HANDOFF.md 的验收标准。

AC1 两个用户并发请求不会互相看到历史、工具结果或记忆
AC2 进程重启后可以从 SQLite session 继续对话
AC3 CLI、Web、Skill 使用同一个 AgentRuntime，而不是各自复制循环
AC4 工具调用、工具结果和失败状态可以通过测试稳定复现
AC5 测试不访问真实 LLM（全部 ScriptedModel；长期记忆属后续阶段，AC1 的记忆部分暂不适用）

所有测试零网络：模型为 ScriptedModel，存储为临时目录 SQLite。
"""

from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient

from miniclaw.config import AgentConfig
from miniclaw.gateway import cli
from miniclaw.gateway.web import create_app
from miniclaw.llm.scripted import ScriptedModel, text_response, tool_call_response
from miniclaw.runtime.factory import build_runtime
from miniclaw.runtime.loop import AgentRuntime, final_reply
from miniclaw.runtime.state import RunState, RunStatus
from miniclaw.session.store import SQLiteSessionStore
from miniclaw.tools.registry import ToolRegistry

from tests.test_build_runtime import make_skill_dir


# ---------- AC1：并发隔离（历史与工具结果） ----------


async def test_ac1_concurrent_users_never_see_each_others_history_or_tool_results(db_path):
    """两个用户真实并发（asyncio.gather），各自会话单写入者。

    无论事件循环如何交错，断言：
    - 每个用户的历史只含自己的消息与答复；
    - 模型收到的每个请求里，user 消息绝不混入另一用户的内容
      （覆盖历史与工具结果——工具结果同样以消息形式进入上下文）。
    """
    model = ScriptedModel(text_response("reply"), repeat_last=True)
    bundle = build_runtime(AgentConfig(), model=model, store=SQLiteSessionStore(db_path))

    async def session_flow(user: str, session: str, turns: list[str]) -> None:
        for turn in turns:
            await bundle.service.chat(
                tenant_id="t", user_id=user, session_id=session, user_input=turn
            )

    await asyncio.gather(
        session_flow("u1", "s1", ["a1", "a2"]),
        session_flow("u2", "s2", ["b1", "b2"]),
    )

    h1 = [m.content for m in await bundle.service.history(tenant_id="t", user_id="u1", session_id="s1")]
    h2 = [m.content for m in await bundle.service.history(tenant_id="t", user_id="u2", session_id="s2")]
    assert h1 == ["a1", "reply", "a2", "reply"]
    assert h2 == ["b1", "reply", "b2", "reply"]

    for request in model.requests:
        user_texts = [m.content for m in request if m.role == "user"]
        owners = {"a" if text.startswith("a") else "b" for text in user_texts}
        assert len(owners) == 1, f"crosstalk in request: {user_texts}"
    bundle.store.close()


async def test_ac1_tool_results_stay_in_owner_session(db_path):
    """工具结果只进入发起者的上下文（顺序执行，脚本完全确定）。"""
    model = ScriptedModel(
        tool_call_response("c1", "echo", '{"text": "secret-of-u1"}'),  # u1 的工具调用
        text_response("u1 done"),
        text_response("u2 reply"),  # u2 的请求
    )
    bundle = build_runtime(AgentConfig(), model=model, store=SQLiteSessionStore(db_path))

    await bundle.service.chat(tenant_id="t", user_id="u1", session_id="s", user_input="use echo")
    await bundle.service.chat(tenant_id="t", user_id="u2", session_id="s", user_input="hello")

    # u2 的那次模型请求中不能出现 u1 的任何内容（含工具结果）
    u2_request = model.requests[-1]
    assert [m.content for m in u2_request] == ["hello"]
    for message in u2_request:
        assert "secret-of-u1" not in (message.content or "")
    bundle.store.close()


# ---------- AC2：进程重启恢复 ----------


async def test_ac2_resume_conversation_after_simulated_restart(db_path):
    model = ScriptedModel(text_response("first reply"))
    bundle = build_runtime(AgentConfig(), model=model, store=SQLiteSessionStore(db_path))
    await bundle.service.chat(tenant_id="t", user_id="u", session_id="s", user_input="hello")
    bundle.store.close()  # 模拟进程结束，丢弃全部内存对象

    model2 = ScriptedModel(text_response("second reply"))
    bundle2 = build_runtime(AgentConfig(), model=model2, store=SQLiteSessionStore(db_path))
    state = await bundle2.service.chat(tenant_id="t", user_id="u", session_id="s", user_input="continue")

    assert state.status is RunStatus.COMPLETED
    assert final_reply(state) == "second reply"
    # 重启后模型收到了重启前的完整历史
    assert [m.content for m in model2.requests[0]] == ["hello", "first reply", "continue"]
    history = [m.content for m in await bundle2.service.history(
        tenant_id="t", user_id="u", session_id="s"
    )]
    assert history == ["hello", "first reply", "continue", "second reply"]
    bundle2.store.close()


# ---------- AC3：CLI / Web / Skill 同一条 Runtime 路径 ----------


def test_ac3_cli_web_and_skill_share_one_runtime(db_path, tmp_path, capsys):
    """一个 bundle、一个 runtime、一个 ScriptedModel 记录全部请求。

    依次走 CLI 一次性消息、Web /v1/chat、技能子运行——
    所有模型调用必须经由同一个 runtime 实例（以共享模型请求记录为证）。
    """
    model = ScriptedModel(
        text_response("cli reply"),                       # 1. CLI 一次性消息
        text_response("web reply"),                       # 2. Web chat
        tool_call_response("c1", "skill_summarizer", '{"task": "summarize"}'),  # 3. 父请求技能
        text_response("SUMMARY: summarize"),              # 4. 技能子运行（同一 runtime）
        text_response("final answer"),                    # 5. 父终答
    )
    bundle = build_runtime(
        AgentConfig(),
        model=model,
        store=SQLiteSessionStore(db_path),
        skill_dir=make_skill_dir(tmp_path),
    )
    assert bundle.service.runtime is bundle.runtime

    # CLI 路径
    assert cli.main(["--message", "via cli", "--session-id", "cli-s"], bundle=bundle) == 0
    assert "cli reply" in capsys.readouterr().out

    # Web 路径（同一个 bundle）
    client = TestClient(create_app(bundle))
    body = client.post(
        "/v1/chat",
        json={"session_id": "web-s", "user_id": "u", "message": "via web"},
    ).json()
    assert body["reply"] == "web reply"

    # Skill 路径：父对话触发技能，子运行复用同一 runtime
    state = asyncio.run(
        bundle.service.chat(
            tenant_id="t", user_id="u", session_id="skill-s", user_input="please summarize"
        )
    )
    assert final_reply(state) == "final answer"
    tool_msg = next(m for m in state.messages if m.role == "tool")
    assert tool_msg.content == "SUMMARY: summarize"

    # 五次模型调用全部记录在同一个模型实例上 = 没有第二套循环
    assert len(model.requests) == 5
    assert [m.content for m in model.requests[0] if m.role == "user"] == ["via cli"]
    assert [m.content for m in model.requests[1] if m.role == "user"] == ["via web"]
    bundle.store.close()


# ---------- AC4：工具调用 / 结果 / 失败的稳定复现 ----------


class _FixedTool:
    name = "echo"
    description = "echo"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, outcome: str):
        self._outcome = outcome

    async def execute(self, arguments, context):
        if self._outcome == "ok":
            return "TOOL_RESULT_42"
        raise RuntimeError("TOOL_BOOM")


def make_state() -> RunState:
    return RunState(tenant_id="t", user_id="u", session_id="s")


async def test_ac4_tool_roundtrip_and_failures_are_deterministic(db_path):
    from miniclaw.runtime.loop import AgentRuntime

    # (a) 工具调用与结果回填
    model = ScriptedModel(tool_call_response("c1", "echo", "{}"), text_response("done"))
    registry = ToolRegistry()
    registry.register(_FixedTool("ok"))
    state = await AgentRuntime(model=model, tools=registry).run(make_state(), "go")
    assert state.status is RunStatus.COMPLETED
    tool_msg = next(m for m in state.messages if m.role == "tool")
    assert tool_msg.content == "TOOL_RESULT_42"
    assert tool_msg.tool_call_id == "c1"

    # (b) 工具异常回填给模型，运行不失败
    model2 = ScriptedModel(tool_call_response("c1", "echo", "{}"), text_response("done"))
    registry2 = ToolRegistry()
    registry2.register(_FixedTool("boom"))
    state2 = await AgentRuntime(model=model2, tools=registry2).run(make_state(), "go")
    assert state2.status is RunStatus.COMPLETED
    assert "TOOL_BOOM" in next(m for m in state2.messages if m.role == "tool").content

    # (c) 模型异常 → FAILED + 结构化 error
    model3 = ScriptedModel()
    state3 = await AgentRuntime(model=model3, tools=ToolRegistry()).run(make_state(), "go")
    assert state3.status is RunStatus.FAILED
    assert state3.error.startswith("model_error:")


# ---------- AC5：无真实 LLM ----------


def test_ac5_acceptance_uses_scripted_model_only():
    """验收测试模块不导入任何真实适配器进行调用（导入 openai_compat 仅为类型无副作用）。"""
    import miniclaw.llm.scripted as scripted

    assert hasattr(scripted, "ScriptedModel")
    # ScriptedModel 不持有网络客户端
    model = ScriptedModel(text_response("x"))
    assert not hasattr(model, "_client")
