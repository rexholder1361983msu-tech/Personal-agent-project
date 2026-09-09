"""FastAPI Web 网关测试：会话隔离、恢复、身份解析（TestClient，零网络、零真实 LLM）。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from miniclaw.config import AgentConfig
from miniclaw.gateway.web import create_app
from miniclaw.llm.scripted import ScriptedModel, text_response
from miniclaw.runtime.factory import build_runtime
from miniclaw.session.store import SQLiteSessionStore


def make_client(db_path, model) -> TestClient:
    bundle = build_runtime(AgentConfig(), model=model, store=SQLiteSessionStore(db_path))
    return TestClient(create_app(bundle))


def chat(client, session_id, user_id, message, *, tenant_id=None, headers=None):
    return client.post(
        "/v1/chat",
        json={
            "session_id": session_id,
            "user_id": user_id,
            "message": message,
            **({"tenant_id": tenant_id} if tenant_id else {}),
        },
        headers=headers or {},
    )


def test_chat_returns_reply_and_usage(db_path):
    model = ScriptedModel(text_response("hello from agent"))
    client = make_client(db_path, model)

    response = chat(client, "s1", "u1", "hi")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["reply"] == "hello from agent"
    assert body["error"] is None
    assert body["usage"]["prompt_tokens"] == 1


def test_two_users_same_session_id_do_not_crosstalk(db_path):
    """验收标准的 Web 层验证：共享 session_id 字符串的不同身份互不可见。"""
    model = ScriptedModel(text_response("r1"), text_response("r2"))
    client = make_client(db_path, model)

    assert chat(client, "shared", "u1", "from u1").json()["reply"] == "r1"
    assert chat(client, "shared", "u2", "from u2").json()["reply"] == "r2"

    # u2 的请求绝不能带上 u1 的历史（模型第二请求只含 u2 自己的消息）
    assert [m.content for m in model.requests[1]] == ["from u2"]

    h1 = client.get("/v1/sessions/shared/messages", params={"user_id": "u1"}).json()["messages"]
    h2 = client.get("/v1/sessions/shared/messages", params={"user_id": "u2"}).json()["messages"]
    assert [m["content"] for m in h1] == ["from u1", "r1"]
    assert [m["content"] for m in h2] == ["from u2", "r2"]


def test_tenant_resolution_body_header_default(db_path):
    model = ScriptedModel(text_response("a"), text_response("b"), text_response("c"))
    client = make_client(db_path, model)

    # body 显式 tenant
    assert chat(client, "s", "u", "m1", tenant_id="t-body").status_code == 200
    # header tenant（无 body 时生效）
    assert chat(client, "s", "u", "m2", headers={"X-Tenant-ID": "t-header"}).status_code == 200
    # 都缺省 → default tenant，且与显式租户隔离
    assert chat(client, "s", "u", "m3").status_code == 200

    body_tenant = client.get(
        "/v1/sessions/s/messages", params={"user_id": "u", "tenant_id": "t-body"}
    ).json()["messages"]
    header_tenant = client.get(
        "/v1/sessions/s/messages", params={"user_id": "u", "tenant_id": "t-header"}
    ).json()["messages"]
    default_tenant = client.get(
        "/v1/sessions/s/messages", params={"user_id": "u", "tenant_id": "default"}
    ).json()["messages"]

    assert [m["content"] for m in body_tenant] == ["m1", "a"]
    assert [m["content"] for m in header_tenant] == ["m2", "b"]
    assert [m["content"] for m in default_tenant] == ["m3", "c"]


def test_history_survives_app_restart(db_path):
    """验收标准的恢复验证：同库重开 app 后历史仍在且可继续对话。"""
    model = ScriptedModel(text_response("first"))
    client = make_client(db_path, model)
    chat(client, "s1", "u1", "hello")

    model2 = ScriptedModel(text_response("second"))
    client2 = make_client(db_path, model2)
    chat(client2, "s1", "u1", "again")

    history = client2.get("/v1/sessions/s1/messages", params={"user_id": "u1"}).json()["messages"]
    assert [m["content"] for m in history] == ["hello", "first", "again", "second"]
    # 继续对话时模型收到了重启前的完整历史
    assert [m.content for m in model2.requests[0]] == ["hello", "first", "again"]


def test_failed_run_reports_error(db_path):
    model = ScriptedModel()  # 无脚本 → model_error
    client = make_client(db_path, model)
    body = chat(client, "s1", "u1", "hello").json()
    assert body["status"] == "failed"
    assert body["error"].startswith("model_error:")
    assert body["reply"] is None


def test_clear_session_removes_only_target(db_path):
    model = ScriptedModel(text_response("a"), text_response("b"))
    client = make_client(db_path, model)
    chat(client, "s1", "u1", "m1")
    chat(client, "s2", "u1", "m2")

    response = client.delete("/v1/sessions/s1", params={"user_id": "u1"})

    assert response.status_code == 200
    assert response.json()["deleted_messages"] == 2
    assert client.get("/v1/sessions/s1/messages", params={"user_id": "u1"}).json()["messages"] == []
    assert len(
        client.get("/v1/sessions/s2/messages", params={"user_id": "u1"}).json()["messages"]
    ) == 2


def test_validation_error_rejected(db_path):
    client = make_client(db_path, ScriptedModel(text_response("x")))
    response = client.post("/v1/chat", json={"session_id": "s", "user_id": "u"})  # 缺 message
    assert response.status_code == 422
