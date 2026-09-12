"""ToolExecutor 测试：本地受限执行器、工具包装、Docker 命令行形状（假 runner）。

不依赖真实 docker：命令行断言用注入的 fake runner；docker_available 探测
只测"无 CLI"分支。真实容器冒烟待 docker 就绪后手动执行。
"""

from __future__ import annotations

import pytest

from miniclaw.tools.base import ToolContext, ToolDisabledError
from miniclaw.tools.executor import (
    DockerSandboxExecutor,
    ExecutionSpec,
    LocalSubprocessExecutor,
    ToolExecutionError,
    docker_available,
    truncate_output,
)
from miniclaw.tools.unsafe import PythonEvalTool, ShellTool


def make_context() -> ToolContext:
    return ToolContext(tenant_id="t", user_id="u", session_id="s", run_id="r")


def py_spec(code: str) -> ExecutionSpec:
    return ExecutionSpec(argv=("python", "-I", "-c", code))


# ── 本地受限执行器：资源约束（不是隔离） ─────────────────────────────


async def test_local_runs_python_and_returns_stdout():
    executor = LocalSubprocessExecutor()
    result = await executor.run(py_spec("print('hi')"), make_context())
    assert result.strip() == "hi"  # Windows 子进程输出是 CRLF 行尾，不硬编码


async def test_local_timeout_kills_and_raises():
    executor = LocalSubprocessExecutor(timeout_s=0.2)
    with pytest.raises(ToolExecutionError, match="timed out"):
        await executor.run(py_spec("import time; time.sleep(5)"), make_context())


async def test_local_output_limit_truncates():
    executor = LocalSubprocessExecutor(output_limit=10)
    result = await executor.run(py_spec("print('x' * 100)"), make_context())
    assert result == "x" * 10 + "...[truncated]"


async def test_local_nonzero_exit_raises_with_output():
    executor = LocalSubprocessExecutor()
    spec = py_spec("import sys; print('boom'); sys.exit(3)")
    with pytest.raises(ToolExecutionError) as exc_info:
        await executor.run(spec, make_context())
    assert "exit code 3" in str(exc_info.value)
    assert "boom" in str(exc_info.value)


async def test_local_runs_in_fresh_working_directory():
    executor = LocalSubprocessExecutor()
    result = await executor.run(py_spec("import os; print(os.listdir('.'))"), make_context())
    assert result.strip() == "[]"  # 每次执行独立空临时目录，不落在项目目录


def test_truncate_output_noop_within_limit():
    assert truncate_output("short", 10) == "short"


# ── 工具包装：ExecutorTool ───────────────────────────────────────────


class FakeExecutor:
    def __init__(self, output: str = "FAKE") -> None:
        self.output = output
        self.specs: list[ExecutionSpec] = []

    async def run(self, spec: ExecutionSpec, context: ToolContext) -> str:
        self.specs.append(spec)
        return self.output


async def test_shell_tool_builds_os_shell_spec_and_delegates():
    fake = FakeExecutor()
    tool = ShellTool(fake)
    result = await tool.execute({"command": "ls"}, make_context())
    assert result == "FAKE"
    (spec,) = fake.specs
    assert spec.stdin_text is None
    assert spec.argv[-1] == "ls"  # Windows: cmd /c ls；POSIX: sh -c ls
    assert spec.argv[0] in ("cmd", "sh")


async def test_python_eval_sends_code_on_stdin():
    fake = FakeExecutor()
    tool = PythonEvalTool(fake)
    await tool.execute({"code": "print(1)"}, make_context())
    (spec,) = fake.specs
    assert spec.argv == ("python", "-I", "-")
    assert spec.stdin_text == "print(1)"


async def test_executor_tool_rejects_empty_payload():
    tool = PythonEvalTool(FakeExecutor())
    with pytest.raises(ValueError, match="non-empty string"):
        await tool.execute({"code": "  "}, make_context())


async def test_executor_tool_without_executor_stays_disabled():
    tool = ShellTool()
    with pytest.raises(ToolDisabledError):
        await tool.execute({"command": "ls"}, make_context())


# ── Docker sandbox 执行器：命令行形状 + 降级（假 runner） ────────────


def make_fake_runner(returncode: int = 0, output: str = "ok"):
    seen: list[tuple[tuple[str, ...], str | None]] = []

    async def runner(argv, stdin_text, timeout_s):
        seen.append((argv, stdin_text))
        return returncode, output

    return runner, seen


async def test_docker_command_line_shape_with_stdin():
    runner, seen = make_fake_runner()
    executor = DockerSandboxExecutor(runner=runner, available=lambda: True)
    spec = ExecutionSpec(argv=("python", "-I", "-"), stdin_text="print(1)")

    result = await executor.run(spec, make_context())

    assert result == "ok"
    (argv, stdin_text), = seen
    assert argv[:3] == ("docker", "run", "--rm")
    for flag, value in [
        ("--network", "none"),
        ("--user", "1000:1000"),
        ("--pids-limit", "128"),
        ("--memory", "512m"),
        ("--cpus", "1.0"),
        ("--tmpfs", "/tmp:rw,size=64m"),
    ]:
        assert argv[argv.index(flag) + 1] == value, flag
    assert "--read-only" in argv
    assert "-i" in argv  # 有 stdin 才加 -i
    assert argv[-4:] == ("python:3.11-slim", "python", "-I", "-")  # 镜像后是载荷
    assert stdin_text == "print(1)"


async def test_docker_command_line_without_stdin_omits_dash_i():
    runner, seen = make_fake_runner()
    executor = DockerSandboxExecutor(runner=runner, available=lambda: True)
    await executor.run(ExecutionSpec(argv=("ls",)), make_context())
    (argv, _), = seen
    assert "-i" not in argv


async def test_docker_nonzero_exit_raises():
    runner, _ = make_fake_runner(returncode=1, output="bad")
    executor = DockerSandboxExecutor(runner=runner, available=lambda: True)
    with pytest.raises(ToolExecutionError, match="exit code 1: bad"):
        await executor.run(ExecutionSpec(argv=("ls",)), make_context())


async def test_docker_timeout_raises():
    async def slow_runner(argv, stdin_text, timeout_s):
        raise TimeoutError("simulated")  # asyncio.TimeoutError 的基类

    executor = DockerSandboxExecutor(runner=slow_runner, available=lambda: True)
    with pytest.raises(ToolExecutionError, match="timed out"):
        await executor.run(ExecutionSpec(argv=("ls",)), make_context())


def test_docker_unavailable_degrades_to_disabled():
    with pytest.raises(ToolDisabledError, match="unavailable"):
        DockerSandboxExecutor(available=lambda: False)


def test_docker_available_false_without_cli(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert docker_available() is False
