"""FastAPI Web 网关。

每个请求独立走 ConversationService → AgentRuntime，按
tenant/user/session 三键隔离；应用内没有任何共享的 Agent 会话状态，
彻底消除"共享 history 串线"问题。身份来源：请求体 > X-Tenant-ID 头
> 配置的 default_tenant。认证属于后续阶段。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, Header
from pydantic import BaseModel

from miniclaw.runtime.factory import RuntimeBundle
from miniclaw.runtime.loop import final_reply


class ChatRequest(BaseModel):
    session_id: str
    user_id: str
    message: str
    tenant_id: str | None = None


def create_app(bundle: RuntimeBundle) -> FastAPI:
    app = FastAPI(title="miniclaw", version="0.1.0")
    app.state.bundle = bundle

    def resolve_tenant(body_tenant: str | None, header_tenant: str | None) -> str:
        return body_tenant or header_tenant or bundle.config.default_tenant

    @app.post("/v1/chat")
    async def chat(req: ChatRequest, x_tenant_id: Annotated[str | None, Header()] = None):
        state = await bundle.service.chat(
            tenant_id=resolve_tenant(req.tenant_id, x_tenant_id),
            user_id=req.user_id,
            session_id=req.session_id,
            user_input=req.message,
        )
        return {
            "run_id": state.run_id,
            "status": state.status.value,
            "reply": final_reply(state),
            "error": state.error,
            "usage": {
                "prompt_tokens": state.usage.prompt_tokens,
                "completion_tokens": state.usage.completion_tokens,
            },
        }

    @app.get("/v1/sessions/{session_id}/messages")
    async def history(
        session_id: str,
        user_id: str,
        tenant_id: str | None = None,
        x_tenant_id: Annotated[str | None, Header()] = None,
    ):
        messages = await bundle.service.history(
            tenant_id=resolve_tenant(tenant_id, x_tenant_id),
            user_id=user_id,
            session_id=session_id,
        )
        return {
            "messages": [
                {"role": m.role, "content": m.content, "tool_call_id": m.tool_call_id}
                for m in messages
            ]
        }

    @app.delete("/v1/sessions/{session_id}")
    async def clear_session(
        session_id: str,
        user_id: str,
        tenant_id: str | None = None,
        x_tenant_id: Annotated[str | None, Header()] = None,
    ):
        deleted = await bundle.service.clear(
            tenant_id=resolve_tenant(tenant_id, x_tenant_id),
            user_id=user_id,
            session_id=session_id,
        )
        return {"deleted_messages": deleted}

    return app
