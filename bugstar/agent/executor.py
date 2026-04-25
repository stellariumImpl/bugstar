"""Executor: 有边界的 agent 执行循环.

替代 Day 1/2 的 run_agent_once. 关键差异:

1. **预算追踪**: Budget 对象记录每一步的 LLM/tool 调用、token、墙上时间.
2. **失败计数**: 连续 N 次工具失败 (exit_code != 0) 触发停止.
3. **强制串行**: 一轮里 LLM 下了多个 tool_call 时, 只执行**第一个**,
   其余的明确告诉 LLM"已被推迟到下一轮，请等结果再决定".
   这避免了 t009 那种"一次下两个相互依赖的命令" 的 race.
4. **产物收尾验证**: 自然结束前自动跑 `ls workspace`, 让 LLM 看到
   它实际产出了什么 (而不是凭印象汇报"已完成").

不做的事 (留给后续 milestone):
- planning (3.3)
- reflection on failure (3.2)
- task complexity profiling (3.3)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from bugstar.sandbox import Sandbox

from .budget import Budget

log = logging.getLogger(__name__)


# --- Result types ---------------------------------------------------------


@dataclass
class ToolCallRecord:
    """单次 tool_call 的可序列化记录."""

    name: str
    args: dict[str, Any]
    result: str
    duration_s: float
    success: bool  # 来自 ExecResult.ok / 工具的 SANDBOX_ERROR
    call_id: str = ""
    deferred: bool = False  # True 表示这次 tool_call 被强制串行推迟了


@dataclass
class ExecutorResult:
    final_reply: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    messages: list[Any] = field(default_factory=list)
    total_duration_s: float = 0.0
    stopped_reason: str = "ok"
    error: str | None = None
    budget_snapshot: dict = field(default_factory=dict)
    artifacts_listing: str = ""  # 收尾时 ls 工作区的输出, 给 trace 看
    deferred_tool_calls: int = 0  # 被强制串行推迟的次数（统计指标）

    @property
    def ok(self) -> bool:
        return self.stopped_reason == "ok"


# --- Default system prompt ------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """你是 BugStar，一个运行在 macOS 上的专业工程助手。

【执行环境】
- 所有 terminal 命令都在隔离的工作目录内运行: {workspace}
- 工作目录之外的文件无法读写
- 单条命令超时 60 秒

【行为约束 — 重要】
1. 当一个命令的结果决定你的回答（计数、读数、状态等），不要只跑一次就下结论.
   命令工具有歧义时（例如 `wc -l` 数的是换行符不是行数），用第二种方式交叉验证.
2. 写完文件 / 产出结果后，必须读回来核对，再向用户汇报.
   不要凭印象说"已完成".
3. 同一轮里如果你需要多个命令，**只下一个**:
   - 如果命令之间有依赖（B 需要 A 的结果），等 A 跑完再决定 B.
   - 系统会强制只执行你下的第一个命令，其余被忽略.
4. 失败时不要硬撑. 如果同一类操作连续失败 2 次，停下来汇报问题给用户.

【已知上下文（长期记忆）】:
{memory_context}
"""


# --- Executor -------------------------------------------------------------


class Executor:
    """有边界的 agent 执行器.

    一个 Executor 实例对应一次任务执行. 不要复用.
    """

    def __init__(
        self,
        *,
        llm: Any,
        tools_by_name: dict[str, Any],
        sandbox: Sandbox,
        budget: Budget | None = None,
        system_prompt_template: str = DEFAULT_SYSTEM_PROMPT,
        memory_context: str = "（长期记忆已禁用）",
        max_iterations: int = 20,
        on_tool_call: Callable[[ToolCallRecord], None] | None = None,
        verify_artifacts: bool = True,
    ) -> None:
        self.llm = llm
        self.tools_by_name = tools_by_name
        self.sandbox = sandbox
        self.budget = budget or Budget()
        self.system_prompt_template = system_prompt_template
        self.memory_context = memory_context
        self.max_iterations = max_iterations
        self.on_tool_call = on_tool_call
        self.verify_artifacts = verify_artifacts

    # --- main entry ---------------------------------------------------

    async def run(self, user_input: str) -> ExecutorResult:
        start = time.monotonic()

        messages: list[Any] = [
            SystemMessage(
                content=self.system_prompt_template.format(
                    workspace=self.sandbox.workspace,
                    memory_context=self.memory_context,
                )
            ),
            HumanMessage(content=user_input),
        ]
        tool_records: list[ToolCallRecord] = []
        deferred_count = 0
        final_reply = ""
        stopped_reason = "ok"
        error: str | None = None

        try:
            for _ in range(self.max_iterations):
                # --- 边界检查 -------------------------------------------
                br = self.budget.check()
                if br is not None:
                    stopped_reason = br
                    break

                # --- LLM call -------------------------------------------
                res = await self.llm.ainvoke(messages)
                tokens = self._estimate_tokens(res)
                self.budget.record_llm_call(tokens=tokens)

                # 防御 stop token 续写
                if isinstance(res.content, str) and "(User) >" in res.content:
                    res.content = res.content.split("(User) >")[0].strip()
                messages.append(res)

                # --- 自然结束 -------------------------------------------
                if not res.tool_calls:
                    final_reply = (res.content or "").strip()
                    # 收尾时强制 list 工作区, 给一个"产物落地"的事实快照.
                    if self.verify_artifacts:
                        artifacts = await self._snapshot_artifacts()
                    else:
                        artifacts = ""
                    return self._finish(
                        start=start,
                        messages=messages,
                        tool_records=tool_records,
                        final_reply=final_reply,
                        stopped_reason="ok",
                        error=None,
                        artifacts=artifacts,
                        deferred_count=deferred_count,
                    )

                # --- 强制串行: 多 tool_call 只执行第一个 ----------------
                executed_call = res.tool_calls[0]
                deferred_calls = res.tool_calls[1:]

                # 给被推迟的 tool_call 都返回一个明确的 ToolMessage,
                # 否则 OpenAI API 会抱怨"AIMessage 里有 N 个 tool_calls 但只回了 1 个 ToolMessage".
                for dc in deferred_calls:
                    deferred_count += 1
                    deferred_msg = (
                        "DEFERRED: Executor 强制串行. 这个 tool_call 没有执行. "
                        "请等第一个 tool_call 的结果出来后, 再决定是否需要这一步."
                    )
                    messages.append(
                        ToolMessage(content=deferred_msg, tool_call_id=dc.get("id", ""))
                    )
                    rec = ToolCallRecord(
                        name=dc.get("name", ""),
                        args=dc.get("args", {}) or {},
                        result=deferred_msg,
                        duration_s=0.0,
                        success=False,
                        call_id=dc.get("id", ""),
                        deferred=True,
                    )
                    tool_records.append(rec)
                    if self.on_tool_call:
                        self.on_tool_call(rec)

                # --- 执行被选中的 tool_call ------------------------------
                rec = await self._invoke_tool(executed_call)
                tool_records.append(rec)
                self.budget.record_tool_call(success=rec.success)
                messages.append(ToolMessage(content=rec.result, tool_call_id=rec.call_id))
                if self.on_tool_call:
                    self.on_tool_call(rec)
            else:
                stopped_reason = "max_iterations"
        except Exception as e:  # noqa: BLE001
            stopped_reason = "exception"
            error = f"{type(e).__name__}: {e}"

        # 走到这里说明被 break 或 for-else 触发, 不是自然结束
        artifacts = ""
        if self.verify_artifacts and stopped_reason != "exception":
            try:
                artifacts = await self._snapshot_artifacts()
            except Exception:  # noqa: BLE001
                artifacts = ""
        return self._finish(
            start=start,
            messages=messages,
            tool_records=tool_records,
            final_reply=final_reply,
            stopped_reason=stopped_reason,
            error=error,
            artifacts=artifacts,
            deferred_count=deferred_count,
        )

    # --- helpers ------------------------------------------------------

    async def _invoke_tool(self, tool_call: dict) -> ToolCallRecord:
        name = tool_call["name"]
        args = tool_call.get("args", {}) or {}
        call_id = tool_call.get("id", "")

        if name not in self.tools_by_name:
            return ToolCallRecord(
                name=name,
                args=args,
                result=f"错误：未找到工具 {name}",
                duration_s=0.0,
                success=False,
                call_id=call_id,
            )

        t0 = time.monotonic()
        try:
            observation = await self.tools_by_name[name].ainvoke(args)
        except Exception as e:  # noqa: BLE001
            observation = f"工具执行异常: {e}"
            return ToolCallRecord(
                name=name,
                args=args,
                result=observation,
                duration_s=time.monotonic() - t0,
                success=False,
                call_id=call_id,
            )

        obs_str = str(observation)
        # 用 terminal 工具返回的字符串里我们能看到 "exit_code=0" 或 "SANDBOX_ERROR".
        # 这里粗略判定 success: 没有 SANDBOX_ERROR 且 exit_code=0 才算成功.
        success = self._infer_success(obs_str)
        return ToolCallRecord(
            name=name,
            args=args,
            result=obs_str,
            duration_s=time.monotonic() - t0,
            success=success,
            call_id=call_id,
        )

    @staticmethod
    def _infer_success(obs_str: str) -> bool:
        """从工具返回字符串里推断成功与否.

        当前只支持 terminal 工具的格式. 加新工具时这里可能需要扩展.
        """
        if "SANDBOX_ERROR" in obs_str:
            return False
        if "工具执行异常" in obs_str:
            return False
        # ExecResult.summary 输出 "exit_code=N ..."
        # exit_code=0 => 成功; 其他 => 失败
        if "exit_code=" in obs_str:
            line = obs_str.split("\n", 1)[0]
            return "exit_code=0" in line
        # 没有 exit_code 信息（比如自定义工具），保守认为成功
        return True

    @staticmethod
    def _estimate_tokens(ai_message: Any) -> int:
        """从 LLM 返回估算 token 用量.

        OpenAI 的 langchain wrapper 会在 response_metadata.token_usage 里给值.
        拿不到就给 0, 不让它崩.
        """
        meta = getattr(ai_message, "response_metadata", None) or {}
        usage = meta.get("token_usage", {}) or {}
        return int(usage.get("total_tokens", 0) or 0)

    async def _snapshot_artifacts(self) -> str:
        """收尾时 list 工作区. 给 trace / debug 一个事实参考."""
        try:
            r = await self.sandbox.exec("ls -la", timeout_s=5.0)
            return r.stdout or ""
        except Exception:  # noqa: BLE001
            return ""

    def _finish(
        self,
        *,
        start: float,
        messages: list[Any],
        tool_records: list[ToolCallRecord],
        final_reply: str,
        stopped_reason: str,
        error: str | None,
        artifacts: str,
        deferred_count: int,
    ) -> ExecutorResult:
        # 如果 final_reply 还是空但有 messages, 试着从最后一个 AIMessage 抽
        if not final_reply:
            for m in reversed(messages):
                if isinstance(m, AIMessage) and m.content:
                    final_reply = str(m.content).strip()
                    break

        return ExecutorResult(
            final_reply=final_reply,
            tool_calls=tool_records,
            messages=messages,
            total_duration_s=time.monotonic() - start,
            stopped_reason=stopped_reason,
            error=error,
            budget_snapshot=self.budget.snapshot(),
            artifacts_listing=artifacts,
            deferred_tool_calls=deferred_count,
        )