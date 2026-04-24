"""LocalSandbox: 基于 subprocess 的本地沙盒.

这是 Day 1 的阶段性实现：
- 不是真正的隔离（和宿主共享一切），但提供"工作目录隔离 + 超时 + 黑名单"
- 足以防御"agent 手滑删了你的 git 仓库"这类事故
- 后续会被 DockerSandbox 取代，但接口不变

明确不做的事:
- 不限 CPU/内存（subprocess 层面做不了）
- 不限网络（同上）
- 黑名单不是安全边界，只是愚蠢拦截
- 不防 fork bomb（要用 cgroup，已经超出 local 的范围）

这些留给 DockerSandbox 阶段做.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import signal
import time
import uuid
from pathlib import Path
from typing import ClassVar

from .base import ExecResult, Sandbox
from .errors import (
    CommandBlockedError,
    PathEscapeError,
    SandboxClosedError,
    SandboxStartupError,
)

log = logging.getLogger(__name__)


# --- 黑名单模式 ---------------------------------------------------------------
# 不是安全边界. 只拦截"明显有害且 agent 不应该做"的命令.
# 真正的安全要靠容器隔离. 这些模式是为了 Day 1 在宿主机裸跑时止血.

_BLOCKLIST_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # 递归删除根目录 / home / 整个盘
    (re.compile(r"\brm\s+(-[rRfF]+\s+)+(/|~|\$HOME|/\*|/\s)", re.IGNORECASE), "rm on root/home"),
    # 对特殊目录的破坏
    (re.compile(r"\brm\s+(-[rRfF]+\s+)+(/etc|/usr|/bin|/sbin|/var|/System|/Library)\b"), "rm on system dir"),
    # 提权
    (re.compile(r"(^|\s|;|&&|\|\|)\s*sudo\b"), "sudo is forbidden"),
    (re.compile(r"(^|\s|;|&&|\|\|)\s*su\s"), "su is forbidden"),
    # 远程脚本执行（典型的 curl | sh 模式）
    (re.compile(r"(curl|wget)\b.+\|\s*(bash|sh|zsh|python|perl)\b"), "remote script execution"),
    # 磁盘直写
    (re.compile(r"\b(dd|mkfs|fdisk|parted)\b"), "disk write tool"),
    # fork bomb 的经典写法
    (re.compile(r":\(\)\s*\{\s*:\|:&\s*\};?\s*:"), "fork bomb"),
    # 改 shell rc / 关键 dotfile
    (re.compile(r">\s*~?/?\.?(bashrc|zshrc|profile|bash_profile|ssh/authorized_keys)\b"), "overwriting dotfile"),
)


def _check_blocklist(cmd: str) -> None:
    """匹配到任一模式就 raise. 只在 shell 模式（str cmd）下检查.

    list 模式的 cmd 不过 shell 解析，上面那些模式基本攻击不了，不检查.
    """
    for pat, reason in _BLOCKLIST_PATTERNS:
        if pat.search(cmd):
            raise CommandBlockedError(cmd, reason)


# --- LocalSandbox -------------------------------------------------------------


class LocalSandbox(Sandbox):
    """subprocess 实现的沙盒.

    一个实例对应一个 session 的工作区. 不要跨 session 复用.
    """

    # 默认工作区根目录. 所有 sandbox 实例在这下面各开一个子目录.
    DEFAULT_ROOT: ClassVar[Path] = Path.home() / ".bugstar" / "workspaces"

    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
        default_timeout_s: float = 60.0,
        default_env: dict[str, str] | None = None,
        keep_on_close: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        workspace_root
            所有 session 工作区的父目录. 默认 ~/.bugstar/workspaces/.
        default_timeout_s
            exec() 不传 timeout_s 时用这个.
        default_env
            每次 exec 都会注入的环境变量. 会被 exec() 的 env 参数覆盖.
        keep_on_close
            close() 时是否保留工作目录. 调试时开，正常场景不开.
        """
        self._root = workspace_root or self.DEFAULT_ROOT
        self._default_timeout_s = default_timeout_s
        self._default_env = default_env or {}
        self._keep_on_close = keep_on_close

        self._workspace: Path | None = None
        self._closed = False
        # 给每个 sandbox 一个短 id，方便日志对齐
        self._id = uuid.uuid4().hex[:8]

    # --- 生命周期 --------------------------------------------------------

    async def start(self) -> None:
        if self._workspace is not None:
            return  # 幂等
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            ws = self._root / f"sb-{self._id}"
            ws.mkdir(parents=True, exist_ok=False)
            self._workspace = ws
            log.info("[sandbox %s] started at %s", self._id, ws)
        except OSError as e:
            raise SandboxStartupError(f"Failed to create workspace: {e}") from e

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._workspace and self._workspace.exists() and not self._keep_on_close:
            try:
                shutil.rmtree(self._workspace)
                log.info("[sandbox %s] closed, workspace removed", self._id)
            except OSError as e:
                log.warning("[sandbox %s] failed to clean workspace: %s", self._id, e)
        else:
            log.info("[sandbox %s] closed, workspace kept at %s", self._id, self._workspace)

    async def __aenter__(self) -> LocalSandbox:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    # --- 属性 ------------------------------------------------------------

    @property
    def workspace(self) -> str:
        self._ensure_ready()
        assert self._workspace is not None
        return str(self._workspace)

    # --- 命令执行 --------------------------------------------------------

    async def exec(
        self,
        cmd: str | list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        self._ensure_ready()
        timeout = timeout_s if timeout_s is not None else self._default_timeout_s

        # 解析 cwd
        cwd_path = self._resolve_cwd(cwd)

        # 合并环境变量. 以当前进程环境为基底，叠 default_env，再叠本次 env.
        merged_env = {**os.environ, **self._default_env, **(env or {})}

        # shell vs exec
        if isinstance(cmd, str):
            _check_blocklist(cmd)
            argv: list[str] = ["/bin/bash", "-c", cmd]
            display_cmd = cmd
        else:
            argv = list(cmd)
            display_cmd = " ".join(argv)

        log.info("[sandbox %s] exec: %s (cwd=%s, timeout=%.1fs)", self._id, display_cmd, cwd_path, timeout)

        start = time.monotonic()
        # start_new_session=True 让子进程成为独立进程组的组长，
        # 超时时 killpg 才能干掉它派生的所有孙进程.
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(cwd_path),
            env=merged_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )

        timed_out = False
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            timed_out = True
            self._kill_process_group(proc)
            # 再等一下收尾，避免 zombie
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=2.0)
            except TimeoutError:
                stdout_b, stderr_b = b"", b""

        duration = time.monotonic() - start
        # 超时时 returncode 可能是 None（如果上面第二次 wait_for 也挂了）
        exit_code = proc.returncode if proc.returncode is not None else -signal.SIGKILL

        result = ExecResult(
            exit_code=exit_code,
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
            duration_s=duration,
            timed_out=timed_out,
        )
        log.info(
            "[sandbox %s] exec done: exit=%d duration=%.2fs timed_out=%s",
            self._id,
            result.exit_code,
            result.duration_s,
            result.timed_out,
        )
        return result

    @staticmethod
    def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
        """超时时杀整个进程组（包括子进程派生的孙进程）."""
        if proc.returncode is not None:
            return
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            # 进程已经没了，或者没权限（不该发生在同用户场景）
            pass

    # --- 文件操作 --------------------------------------------------------

    async def read_file(self, path: str) -> bytes:
        self._ensure_ready()
        abs_path = self._resolve_and_guard(path)
        return await asyncio.to_thread(abs_path.read_bytes)

    async def write_file(self, path: str, content: bytes | str) -> None:
        self._ensure_ready()
        abs_path = self._resolve_and_guard(path)

        def _write() -> None:
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, str):
                abs_path.write_text(content, encoding="utf-8")
            else:
                abs_path.write_bytes(content)

        await asyncio.to_thread(_write)

    # --- 内部工具 --------------------------------------------------------

    def _ensure_ready(self) -> None:
        if self._closed:
            raise SandboxClosedError("Sandbox has been closed")
        if self._workspace is None:
            raise SandboxStartupError("Sandbox not started. Call start() or use 'async with'.")

    def _resolve_cwd(self, cwd: str | None) -> Path:
        assert self._workspace is not None
        if cwd is None:
            return self._workspace
        p = Path(cwd)
        if not p.is_absolute():
            p = self._workspace / p
        p = p.resolve()
        self._guard_path(p)
        if not p.exists():
            # cwd 必须存在（subprocess 不会自动创建）
            raise FileNotFoundError(f"cwd does not exist: {p}")
        return p

    def _resolve_and_guard(self, path: str) -> Path:
        """解析相对路径并检查是否在 workspace 内."""
        assert self._workspace is not None
        p = Path(path)
        if not p.is_absolute():
            p = self._workspace / p
        p = p.resolve()
        self._guard_path(p)
        return p

    def _guard_path(self, abs_path: Path) -> None:
        """确保路径在 workspace 内部. 防 ../ 逃逸."""
        assert self._workspace is not None
        ws = self._workspace.resolve()
        try:
            abs_path.relative_to(ws)
        except ValueError:
            raise PathEscapeError(str(abs_path), str(ws)) from None
