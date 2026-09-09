"""CLI 入口测试：注入 ScriptedModel bundle，验证与 Web 相同的 Runtime 路径。"""

from __future__ import annotations

from miniclaw.config import AgentConfig
from miniclaw.gateway import cli
from miniclaw.llm.scripted import ScriptedModel, text_response
from miniclaw.runtime.factory import build_runtime
from miniclaw.session.store import SQLiteSessionStore


def make_bundle(db_path, model):
    return build_runtime(AgentConfig(), model=model, store=SQLiteSessionStore(db_path))


def test_one_shot_message_prints_reply(db_path, capsys):
    model = ScriptedModel(text_response("cli reply"))
    bundle = make_bundle(db_path, model)

    code = cli.main(["--message", "hello", "--session-id", "s1"], bundle=bundle)

    assert code == 0
    assert "cli reply" in capsys.readouterr().out
    bundle.store.close()


def test_one_shot_failure_returns_nonzero(db_path, capsys):
    bundle = make_bundle(db_path, ScriptedModel())  # 模型错误 → FAILED

    code = cli.main(["--message", "hello"], bundle=bundle)

    assert code == 1
    assert "model_error:" in capsys.readouterr().err
    bundle.store.close()


def test_repl_reads_until_empty_line(db_path, capsys):
    model = ScriptedModel(text_response("r1"), text_response("r2"))
    bundle = make_bundle(db_path, model)
    inputs = iter(["first", "second", ""])

    # 直接调用 run_repl 以注入 input_fn（main 的 argv 路径不接收输入函数）
    import asyncio

    code = asyncio.run(
        cli.run_repl(
            bundle,
            tenant_id="default",
            user_id="local",
            session_id="repl",
            input_fn=lambda prompt="": next(inputs),
        )
    )

    assert code == 0
    out = capsys.readouterr().out
    assert "r1" in out and "r2" in out
    # 两轮共用同一会话：第二次请求带上了第一轮历史
    assert [m.content for m in model.requests[1]] == ["first", "r1", "second"]
    bundle.store.close()


def test_repl_failure_returns_nonzero(db_path, capsys):
    bundle = make_bundle(db_path, ScriptedModel())
    import asyncio

    code = asyncio.run(
        cli.run_repl(
            bundle,
            tenant_id="default",
            user_id="local",
            session_id="repl",
            input_fn=lambda prompt="": "boom",
        )
    )
    assert code == 1
    assert "model_error:" in capsys.readouterr().out
    bundle.store.close()


def test_cli_and_web_share_one_runtime_path(db_path):
    """CLI 注入的 bundle 与 Web 使用同一个 service.chat 入口。"""
    model = ScriptedModel(text_response("same path"))
    bundle = make_bundle(db_path, model)
    assert bundle.service.runtime is bundle.runtime

    code = cli.main(["--message", "hi", "--session-id", "s"], bundle=bundle)
    assert code == 0
    # 模型请求经由 runtime → service → store，可从存储断言落库
    history = bundle.store.get_messages("default", "local", "s")
    assert [m.content for m in history] == ["hi", "same path"]
    bundle.store.close()
