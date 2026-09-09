from miniclaw.llm.messages import Message, Usage
from miniclaw.runtime.state import RunState, RunStatus


def test_new_state_defaults():
    state = RunState(tenant_id="t", user_id="u", session_id="s")
    assert state.status is RunStatus.RUNNING
    assert state.step == 0
    assert state.tool_calls_used == 0
    assert state.messages == []
    assert state.usage == Usage()
    assert state.error is None
    assert state.persisted_messages == 0


def test_run_ids_are_unique_per_run():
    a = RunState(tenant_id="t", user_id="u", session_id="s")
    b = RunState(tenant_id="t", user_id="u", session_id="s")
    assert a.run_id != b.run_id


def test_usage_accumulates():
    usage = Usage()
    usage.add(Usage(prompt_tokens=10, completion_tokens=5))
    usage.add(Usage(prompt_tokens=1, completion_tokens=2))
    assert usage == Usage(prompt_tokens=11, completion_tokens=7)
