"""沙盒自身故障的异常类型.

注意：命令执行失败（exit_code != 0）不是异常，是 ExecResult 的正常返回值.
只有沙盒基础设施坏掉（容器起不来、目录没权限、资源耗尽）才抛这些.
"""


class SandboxError(Exception):
    """所有沙盒异常的基类."""


class SandboxStartupError(SandboxError):
    """沙盒启动失败. 比如 docker daemon 没跑、E2B 账号额度用完、临时目录创建失败."""


class SandboxClosedError(SandboxError):
    """试图在已关闭的沙盒上做操作."""


class CommandBlockedError(SandboxError):
    """命令被黑名单拦截. 这是 Day 1 保命用的，Docker/E2B 后端不需要这个."""

    def __init__(self, cmd: str, reason: str) -> None:
        super().__init__(f"Command blocked: {cmd!r}. Reason: {reason}")
        self.cmd = cmd
        self.reason = reason


class PathEscapeError(SandboxError):
    """试图读写 workspace 以外的路径. 防止 agent 用 ../../../../etc/passwd 逃出去."""

    def __init__(self, path: str, workspace: str) -> None:
        super().__init__(f"Path {path!r} escapes workspace {workspace!r}")
        self.path = path
        self.workspace = workspace
