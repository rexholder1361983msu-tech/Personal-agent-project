"""命令行入口。与 Web 共用 build_runtime + ConversationService 同一条路径。

用法：
    miniclaw --message "你好" --session-id demo      # 一次性
    miniclaw                                          # REPL
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from miniclaw.config import AgentConfig
from miniclaw.runtime.factory import RuntimeBundle, build_runtime
from miniclaw.runtime.loop import final_reply
from miniclaw.runtime.state import RunState, RunStatus


async def chat_once(
    bundle: RuntimeBundle,
    tenant_id: str,
    user_id: str,
    session_id: str,
    message: str,
) -> RunState:
    return await bundle.service.chat(
        tenant_id=tenant_id, user_id=user_id, session_id=session_id, user_input=message
    )


async def run_repl(
    bundle: RuntimeBundle,
    tenant_id: str,
    user_id: str,
    session_id: str,
    *,
    input_fn=input,
    output_fn=print,
) -> int:
    output_fn(
        f"miniclaw REPL | tenant={tenant_id} user={user_id} session={session_id} | 空行或 Ctrl-D 退出"
    )
    while True:
        try:
            line = await asyncio.to_thread(input_fn, "you> ")
        except EOFError:
            break
        if not line.strip():
            break
        state = await bundle.service.chat(
            tenant_id=tenant_id, user_id=user_id, session_id=session_id, user_input=line
        )
        reply = final_reply(state)
        output_fn(f"agent> {reply if reply is not None else '(no reply)'}")
        if state.status is RunStatus.FAILED:
            output_fn(f"[run failed] {state.error}")
            return 1
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="miniclaw", description="MiniClaw agent CLI")
    parser.add_argument("--message", help="一次性消息；省略则进入 REPL")
    parser.add_argument("--session-id", default="cli", help="会话 ID（默认 cli）")
    parser.add_argument("--user-id", default="local", help="用户 ID（默认 local）")
    parser.add_argument("--tenant-id", default=None, help="租户 ID（默认取配置）")
    parser.add_argument("--db", default=None, help="SQLite 路径（默认取配置）")
    parser.add_argument("--skill-dir", default=None, help="技能目录（含 SKILL.md）")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    return parser


def main(argv: list[str] | None = None, *, bundle: RuntimeBundle | None = None) -> int:
    """bundle 参数仅供测试注入；生产路径总是由 build_runtime 构造。"""
    args = build_arg_parser().parse_args(argv)
    if bundle is None:
        config = AgentConfig.from_env()
        if args.db:
            config.db_path = args.db
        if args.api_base:
            config.api_base = args.api_base
        if args.model:
            config.model = args.model
        if args.max_steps:
            config.max_steps = args.max_steps
        bundle = build_runtime(config, skill_dir=args.skill_dir)

    tenant_id = args.tenant_id or bundle.config.default_tenant
    try:
        if args.message:
            state = asyncio.run(
                chat_once(bundle, tenant_id, args.user_id, args.session_id, args.message)
            )
            reply = final_reply(state)
            print(reply if reply is not None else "(no reply)")
            if state.status is RunStatus.FAILED:
                print(f"[run failed] {state.error}", file=sys.stderr)
                return 1
            return 0
        return asyncio.run(
            run_repl(bundle, tenant_id, args.user_id, args.session_id)
        )
    except KeyboardInterrupt:
        print("\nbye")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
