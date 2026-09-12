"""ToolExecutor：把"跑一段外部命令/代码"从工具逻辑中抽出的执行后端。

两个实现，隔离级别截然不同：
- LocalSubprocessExecutor：本机受限 subprocess（超时、输出上限、独立临时
  工作目录）。**这只是资源约束，不是安全隔离**：子进程与宿主同权限，
  可访问文件系统与网络。仅用于开发验证与受信任环境。
- DockerSandboxExecutor：容器内执行（非 root、只读根 FS、无网络、
  PIDs/内存/CPU 上限）。docker 不可用时构造抛 ToolDisabledError——
  "工具不可用"是诚实的降级，不是假隔离。

黑名单与参数校验只作为前置快速失败，安全边界只由 sandbox 提供。
测试通过注入 runner / available 探测函数避免依赖真实 docker。
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from miniclaw.tools.base import ToolContext, ToolDisabledError

_TRUNCATION_MARKER = "...[truncated]"


@dataclass(frozen=True)
class ExecutionSpec:
    """一次受限执行的描述：完整命令行 + 可选 stdin。"""

    argv: tuple[str, ...]
    stdin_text: str | None = None


class ToolExecutionError(RuntimeError):
    """执行完成但没有可用结果（超时 / 非零退出码）。"""


class ToolExecutor(Protocol):
    """执行后端契约。隔离级别由实现决定，调用方按风险选择。"""

    async def run(self, spec: ExecutionSpec, context: ToolContext) -> str: ...


def truncate_output(text: str, limit: int) -> str:
    if len(text) > limit:
        return text[:limit] + _TRUNCATION_MARKER
    return text


def docker_available() -> bool:
    """docker CLI 存在且守护进程响应；任何异常都返回 False，不抛出。"""
    if shutil.which("docker") is None:
        return False
    try:
        proc = subprocess.run(("docker", "info"), capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


# 异步命令运行器：(argv, stdin, timeout) -> (退出码, 输出)。测试可注入假实现。
CommandRunner = Callable[[tuple[str, ...], str | None, float], Awaitable[tuple[int, str]]]


async def _subprocess_runner(
    argv: tuple[str, ...],
    stdin_text: str | None,
    timeout_s: float,
    *,
    cwd: str | None = None,
) -> tuple[int, str]:
    """跑一个子进程，合并 stdout/stderr，统一 UTF-8 解码（errors=replace，
    Windows 控制台 GBK 输出不会炸）。超时杀进程后向上抛 TimeoutError。"""
    proc = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        stdin=(
            asyncio.subprocess.PIPE
            if stdin_text is not None
            else asyncio.subprocess.DEVNULL
        ),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    data = stdin_text.encode("utf-8") if stdin_text is not None else None
    try:
        out, _ = await asyncio.wait_for(proc.communicate(data), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return proc.returncode or 0, out.decode("utf-8", errors="replace")


def _check_exit(returncode: int, output: str, output_limit: int) -> str:
    truncated = truncate_output(output, output_limit)
    if returncode != 0:
        raise ToolExecutionError(f"exit code {returncode}: {truncated}")
    return truncated


class LocalSubprocessExecutor:
    """本机受限 subprocess：超时 + 输出上限 + 每次执行独立临时工作目录。

    ⚠ 资源约束不是安全隔离：子进程与宿主同权限。只用于开发验证 /
    受信任环境；生产高风险工具用 DockerSandboxExecutor。
    """

    def __init__(self, *, timeout_s: float = 10.0, output_limit: int = 8000) -> None:
        self.timeout_s = timeout_s
        self.output_limit = output_limit

    async def run(self, spec: ExecutionSpec, context: ToolContext) -> str:
        with tempfile.TemporaryDirectory(prefix="miniclaw-exec-") as cwd:
            try:
                returncode, output = await _subprocess_runner(
                    spec.argv, spec.stdin_text, self.timeout_s, cwd=cwd
                )
            except asyncio.TimeoutError:
                raise ToolExecutionError(
                    f"execution timed out after {self.timeout_s}s"
                ) from None
        return _check_exit(returncode, output, self.output_limit)


class DockerSandboxExecutor:
    """容器内执行：非 root、只读根 FS、无网络、PIDs/内存/CPU 上限、tmpfs /tmp。

    加固与载荷无关：`--network none`、`--read-only`、`--user 1000:1000`、
    `--pids-limit`、`--memory`、`--cpus`；有 stdin 时自动加 `-i`。
    docker 不可用（未安装 / 引擎未运行）时构造抛 ToolDisabledError；
    镜像需预先 pull（默认不会静默下载）。
    """

    def __init__(
        self,
        *,
        image: str = "python:3.11-slim",
        timeout_s: float = 10.0,
        output_limit: int = 8000,
        memory: str = "512m",
        cpus: str = "1.0",
        pids_limit: int = 128,
        runner: CommandRunner | None = None,
        available: Callable[[], bool] | None = None,
    ) -> None:
        probe = available if available is not None else docker_available
        if not probe():
            raise ToolDisabledError(
                "docker sandbox unavailable: docker CLI missing or engine not running"
            )
        self.image = image
        self.timeout_s = timeout_s
        self.output_limit = output_limit
        self.memory = memory
        self.cpus = cpus
        self.pids_limit = pids_limit
        self._runner = runner if runner is not None else _subprocess_runner

    def command_line(self, spec: ExecutionSpec) -> tuple[str, ...]:
        argv: list[str] = [
            "docker", "run", "--rm",
            "--network", "none",
            "--read-only",
            "--user", "1000:1000",
            "--pids-limit", str(self.pids_limit),
            "--memory", self.memory,
            "--cpus", self.cpus,
            "--tmpfs", "/tmp:rw,size=64m",
        ]
        if spec.stdin_text is not None:
            argv.append("-i")
        argv.append(self.image)
        argv.extend(spec.argv)
        return tuple(argv)

    async def run(self, spec: ExecutionSpec, context: ToolContext) -> str:
        try:
            returncode, output = await self._runner(
                self.command_line(spec), spec.stdin_text, self.timeout_s
            )
        except asyncio.TimeoutError:
            raise ToolExecutionError(
                f"sandbox execution timed out after {self.timeout_s}s"
            ) from None
        return _check_exit(returncode, output, self.output_limit)
