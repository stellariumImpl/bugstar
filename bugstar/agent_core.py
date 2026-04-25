"""可复用的 agent loop（兼容层）.

历史: Day 1/2 这里是完整的 agent 循环实现.
3.1 起: 真正的实现在 bugstar.agent.Executor, 这里只是兼容 wrapper,
       让 main.py / runner.py 不必改太多 import 就能用上 Executor.

新代码请直接用 bugstar.agent.Executor.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from bugstar.agent import Budget, Executor
from bugstar.agent import ToolCallRecord as _ExecToolCallRecord


@dataclass
class ToolCallRecord:
    name: str
    args: dict[str, Any]
    result: str
    duration_s: float
    call_id: str = ""


@dataclass
class AgentRunResult:
    final_reply: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    messages: list[Any] = field(default_factory=list)
    total_duration_s: float = 0.0
    stopped_reason: str = "ok"
    error: str | None = None
    # 3.1 新增: budget 快照. 老调用方可以无视.
    budget_snapshot: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.stopped_reason == "ok"


DEFAULT_SYSTEM_PROMPT = """你是 BugStar，一个运行在 macOS 上的专业工程助手。

你可以根据需要使用 terminal 执行命令。

【执行环境】
- 所有 terminal 命令都在隔离的工作目录内运行: {workspace}
- 工作目录之外的文件无法读写，请使用相对路径或指定此目录下的绝对路径
- 单条命令超时 60 秒

【已知上下文（长期记忆）】:
{memory_context}
"""


def _adapt_record(r: _ExecToolCallRecord) -> ToolCallRecord:
    """把 Executor 的 record 适配回旧字段集（丢掉 success/deferred）."""
    return ToolCallRecord(
        name=r.name,
        args=r.args,
        result=r.result,
        duration_s=r.duration_s,
        call_id=r.call_id,
    )


async def run_agent_once(
    *,
    user_input: str,
    llm: Any,
    tools_by_name: dict[str, Any],
    workspace: str,
    memory_context: str = "（长期记忆已禁用）",
    system_prompt_template: str = DEFAULT_SYSTEM_PROMPT,
    max_iterations: int = 20,
    on_tool_call: Callable[[ToolCallRecord], None] | None = None,
    budget: Budget | None = None,
    sandbox: Any = None,
) -> AgentRunResult:
    """跑一轮 agent. 内部委托给 Executor.

    sandbox 参数 (3.1 起必传): Executor 需要直接拿到 sandbox 引用以做收尾验证.
    """
    forwarder: Callable[[_ExecToolCallRecord], None] | None = None
    if on_tool_call:

        def forwarder(rec: _ExecToolCallRecord) -> None:
            on_tool_call(_adapt_record(rec))

    if sandbox is None:
        raise RuntimeError(
            "run_agent_once now requires `sandbox` param. "
            "Please pass the LocalSandbox instance you used to build the terminal tool."
        )

    executor = Executor(
        llm=llm,
        tools_by_name=tools_by_name,
        sandbox=sandbox,
        budget=budget,
        system_prompt_template=system_prompt_template,
        memory_context=memory_context,
        max_iterations=max_iterations,
        on_tool_call=forwarder,
        verify_artifacts=True,
    )
    result = await executor.run(user_input)

    return AgentRunResult(
        final_reply=result.final_reply,
        tool_calls=[_adapt_record(r) for r in result.tool_calls],
        messages=result.messages,
        total_duration_s=result.total_duration_s,
        stopped_reason=result.stopped_reason,
        error=result.error,
        budget_snapshot=result.budget_snapshot,
    )