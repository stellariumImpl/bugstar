"""terminal 工具: 基于 Sandbox 的命令执行.

设计要点
--------
- 工具函数本身无状态，通过闭包注入 sandbox 实例.
- 返回值是 str（LLM 要看），用 ExecResult.summary() 压缩.
- 沙盒异常（被黑名单拦、路径逃逸、沙盒挂了）统一包装成字符串返回，
  让 LLM 能看到原因并调整策略，而不是让整个 agent loop 崩掉.
"""

from __future__ import annotations

from langchain_core.tools import tool

from bugstar.sandbox import Sandbox, SandboxError


def make_terminal_tool(sandbox: Sandbox, *, default_timeout_s: float = 60.0):
    """构造绑定到特定 sandbox 的 terminal 工具.

    使用闭包避免把 sandbox 实例通过参数暴露给 LLM.
    """

    @tool
    async def terminal(command: str) -> str:
        """在 macOS 沙盒工作目录里执行 shell 命令.

        用途:
        - 列文件、读文件内容、跑脚本、装包、跑测试等所有需要 shell 的场景.
        - 工作目录已经隔离，不会影响你的系统文件. 可以放心执行.

        约束:
        - 单条命令超时 60s. 需要更长时间的任务请拆分.
        - 禁止 sudo / rm -rf 系统目录 / 管道远程脚本等危险操作.
        - 文件路径建议用相对路径（相对于工作目录）.

        Args:
            command: 要执行的 shell 命令. 支持管道、重定向、&&、|| 等.

        Returns:
            包含 exit_code、耗时、stdout、stderr 的文本. exit_code=0 表示成功.
        """
        try:
            result = await sandbox.exec(command, timeout_s=default_timeout_s)
            return result.summary()
        except SandboxError as e:
            # 沙盒级别的故障（黑名单拦截、沙盒挂了）要让 LLM 看到，
            # 它能据此调整策略（比如改个不触发黑名单的等价写法）.
            return f"SANDBOX_ERROR: {type(e).__name__}: {e}"

    return terminal
