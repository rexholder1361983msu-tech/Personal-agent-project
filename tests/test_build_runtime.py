"""build_runtime 组装入口测试。"""

from __future__ import annotations

import pytest

from miniclaw.config import AgentConfig
from miniclaw.llm.scripted import ScriptedModel, text_response
from miniclaw.runtime.factory import build_runtime
from miniclaw.runtime.loop import final_reply
from miniclaw.runtime.state import RunStatus
from miniclaw.session.store import SQLiteSessionStore
from miniclaw.skills.loader import SkillFormatError


def test_bundle_wires_single_runtime_path(tmp_path):
    config = AgentConfig(db_path=str(tmp_path / "s.db"))
    bundle = build_runtime(config)
    assert bundle.service.runtime is bundle.runtime
    assert bundle.runtime.model is bundle.model
    assert bundle.tools.names() == ["calculator", "current_time", "echo"]
    assert isinstance(bundle.store, SQLiteSessionStore)
    bundle.store.close()


async def test_build_runtime_end_to_end_with_injected_model(db_path):
    model = ScriptedModel(text_response("ok"))
    bundle = build_runtime(AgentConfig(), model=model, store=SQLiteSessionStore(db_path))
    state = await bundle.service.chat(
        tenant_id="t", user_id="u", session_id="s", user_input="hello"
    )
    assert state.status is RunStatus.COMPLETED
    assert final_reply(state) == "ok"
    assert [m.content for m in await bundle.service.history(
        tenant_id="t", user_id="u", session_id="s"
    )] == ["hello", "ok"]
    bundle.store.close()


def make_skill_dir(tmp_path):
    skill_dir = tmp_path / "skills" / "summarizer"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: summarizer\ndescription: Summarize the given text.\n---\n"
        "Always reply with: SUMMARY: <input>\n",
        encoding="utf-8",
    )
    return tmp_path / "skills"


def test_skill_dir_registers_skill_tools(tmp_path):
    bundle = build_runtime(
        AgentConfig(),
        model=ScriptedModel(),
        store=SQLiteSessionStore(tmp_path / "s.db"),
        skill_dir=make_skill_dir(tmp_path),
    )
    assert "skill_summarizer" in bundle.tools.names()
    spec = next(s for s in bundle.tools.specs() if s.name == "skill_summarizer")
    assert spec.description == "Summarize the given text."
    bundle.store.close()


def test_duplicate_skill_names_are_rejected(tmp_path):
    for name in ("a", "b"):
        skill_dir = tmp_path / "skills" / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: same\ndescription: dup\n---\nbody\n", encoding="utf-8"
        )
    with pytest.raises(ValueError):
        build_runtime(
            AgentConfig(),
            model=ScriptedModel(),
            store=SQLiteSessionStore(tmp_path / "s.db"),
            skill_dir=tmp_path / "skills",
        )


def test_invalid_skill_md_is_rejected(tmp_path):
    skill_dir = tmp_path / "skills" / "broken"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("no front matter here\n", encoding="utf-8")
    with pytest.raises(SkillFormatError):
        build_runtime(
            AgentConfig(),
            model=ScriptedModel(),
            store=SQLiteSessionStore(tmp_path / "s.db"),
            skill_dir=tmp_path / "skills",
        )


def test_config_from_env():
    config = AgentConfig.from_env(
        {
            "MINICLAW_MODEL": "m-2",
            "MINICLAW_MAX_STEPS": "3",
            "MINICLAW_DB": "/tmp/x.db",
            "MINICLAW_API_KEY": "k",
        }
    )
    assert config.model == "m-2"
    assert config.max_steps == 3
    assert config.db_path == "/tmp/x.db"
    assert config.api_key == "k"
    assert config.default_tenant == "default"
