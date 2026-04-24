"""Sandbox 抽象接口.

设计目标
--------
1. 所有执行类工具都通过 Sandbox 接口，而不是直接调 subprocess / docker。
2. 未来加 DockerSandbox / E2BSandbox 时，上层代码零改动。
3. 接口只暴露 agent 真正需要的能力（命令执行 + 文件读写 + 生命周期），
   不泄漏后端实现细节（不要出现 container_id / e2b_session 之类的字段）。

命名约定
--------
- exec  : 执行一条命令，返回 ExecResult（包含 exit_code, stdout, stderr, duration）
- read_file / write_file : 读写沙盒内文件
- close : 显式销毁沙盒（LocalSandbox 里是清理临时目录，DockerSandbox 里是 docker rm）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ExecResult:
    """一次命令执行的结果.

    不可变（frozen）避免上层误改。所有字段必须有值，
    失败场景用 exit_code != 0 表示，而不是抛异常（除非是沙盒本身故障）。
    """

    exit_code: int
    """进程退出码. 0 表示成功. 超时/被 kill 通常是非零值."""

    stdout: str
    """标准输出. 已经 decode 成 str（utf-8, errors='replace'）."""

    stderr: str
    """标准错误. 同上."""

    duration_s: float
    """实际耗时（秒）. 用于日志和超时分析."""

    timed_out: bool = False
    """是否因超时被 kill. True 时 exit_code 一般是 -9 / 124 之类."""

    @property
    def ok(self) -> bool:
        """方便上层写 `if result.ok: ...`"""
        return self.exit_code == 0 and not self.timed_out

    def summary(self, max_chars: int = 500) -> str:
        """给 LLM 看的压缩版. 避免把几 MB 的输出塞进上下文."""
        parts = [f"exit_code={self.exit_code}", f"duration={self.duration_s:.2f}s"]
        if self.timed_out:
            parts.append("TIMED_OUT")

        def _clip(s: str, label: str) -> str:
            s = s.strip()
            if not s:
                return ""
            if len(s) > max_chars:
                s = s[: max_chars // 2] + f"\n...[truncated {len(s) - max_chars} chars]...\n" + s[-max_chars // 2 :]
            return f"--- {label} ---\n{s}"

        body = "\n".join(filter(None, [_clip(self.stdout, "stdout"), _clip(self.stderr, "stderr")]))
        header = " ".join(parts)
        return f"{header}\n{body}" if body else header


@runtime_checkable
class Sandbox(Protocol):
    """沙盒的统一接口.

    使用模式（推荐用 async context manager）:
        async with LocalSandbox() as sb:
            r = await sb.exec("python --version")
            if r.ok:
                ...

    生命周期:
        start() -> 多次 exec/read/write -> close()
    """

    async def start(self) -> None:
        """初始化沙盒. 创建工作目录、启动容器、连接 E2B 等."""
        ...

    async def close(self) -> None:
        """销毁沙盒. 清理临时目录、停止容器、断开 E2B 等. 幂等."""
        ...

    async def exec(
        self,
        cmd: str | list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float = 60.0,
    ) -> ExecResult:
        """执行一条命令.

        Parameters
        ----------
        cmd
            字符串时走 shell（/bin/bash -c），列表时走 exec（更安全，不走 shell 解析）。
        cwd
            工作目录. None 表示用沙盒默认工作目录（通常是 workspace 根）。
        env
            额外环境变量. 会和沙盒默认 env 合并.
        timeout_s
            命令级超时. 超时会 kill 整个进程组并返回 timed_out=True.

        Returns
        -------
        ExecResult
            命令失败（非零退出）不抛异常，返回结果让上层决定怎么办.
            只有沙盒自身故障才抛异常（见 errors.py）.
        """
        ...

    async def read_file(self, path: str) -> bytes:
        """读取沙盒内文件. path 可以是相对（相对 workspace）或绝对."""
        ...

    async def write_file(self, path: str, content: bytes | str) -> None:
        """写入文件. 会自动创建父目录. str 会按 utf-8 编码."""
        ...

    @property
    def workspace(self) -> str:
        """沙盒内的工作目录路径. agent 应该在这里干活.

        对 LocalSandbox 来说是宿主机的实际路径.
        对 DockerSandbox 来说是容器内的路径（比如 /workspace）.
        """
        ...

    async def __aenter__(self) -> Sandbox:
        ...

    async def __aexit__(self, *exc_info: object) -> None:
        ...