"""Sandbox 抽象层.

用法::

    from bugstar.sandbox import LocalSandbox

    async with LocalSandbox() as sb:
        r = await sb.exec("python --version")
        print(r.summary())
"""

from .base import ExecResult, Sandbox
from .errors import (
    CommandBlockedError,
    PathEscapeError,
    SandboxClosedError,
    SandboxError,
    SandboxStartupError,
)
from .local import LocalSandbox

__all__ = [
    "CommandBlockedError",
    "ExecResult",
    "LocalSandbox",
    "PathEscapeError",
    "Sandbox",
    "SandboxClosedError",
    "SandboxError",
    "SandboxStartupError",
]
