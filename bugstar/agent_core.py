"""可复用的 agent loop.

把 agent 的核心循环从 CLI 解耦：
- 不依赖全局变量（llm / sandbox / memo 都通过参数传入）
- 不 print，而是把所有事件塞进 trace
- 返回一个结构化的 AgentRunResult，bench 和 CLI 都能用

CLI 在 main.py 里 wrap 一层：从 trace 打印，顺带处理 mem0 的回写.
Bench 在 runner.py 里 wrap 一层：从 trace 判定任务是否成功.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage


@dataclass
class ToolCallRecord:
    """一次 tool_call 的记录."""

    name: str
    args: dict[str, Any]
    result: str
    duration_s: float
    # 原始 tool_call_id，用于对齐 ToolMessage
    call_id: str = ""


@dataclass
class AgentRunResult:
    """一次 agent run 的完整痕迹.

    设计要点：
    - 保留完整 messages 序列（system/human/ai/tool），可以重放整个对话.
    - tool_calls 单独列出，方便统计"agent 用了几次 terminal、都跑了啥"。
    - final_reply 是最后一次非空 AI 回复的 content，这是直接判断"agent 最后说了什么"的地方.
    - stopped_reason 区分 "自然结束" vs "循环上限" vs "异常"，bench 需要这个.
    """

    final_reply: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    messages: list[Any] = field(default_factory=list)
    total_duration_s: float = 0.0
    stopped_reason: str = "ok"  # ok | max_iterations | exception
    error: str | None = None

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


async def run_agent_once(
    *,
    user_input: str,
    llm: Any,
    tools_by_name: dict[str, Any],
    workspace: str,
    memory_context: str = "（长期记忆已禁用）",
    system_prompt_template: str = DEFAULT_SYSTEM_PROMPT,
    max_iterations: int = 20,
    on_tool_call: Any = None,  # callable(ToolCallRecord) -> None, 给 CLI 实时打印用
) -> AgentRunResult:
    """跑一轮 agent，直到 LLM 不再调工具或达到迭代上限.

    这个函数不知道 CLI 的存在、不知道 mem0 的存在。
    唯一的副作用就是调 llm / 调 tool，然后把过程记进 trace.

    Parameters
    ----------
    user_input : 本轮用户输入
    llm        : 已经 bind_tools 的 ChatModel
    tools_by_name : 工具名 -> 工具对象 的字典（必须支持 ainvoke）
    workspace  : 沙盒工作目录路径，用于注入 system prompt
    memory_context : 长期记忆内容，关闭记忆时传"（长期记忆已禁用）"
    system_prompt_template : 可自定义，方便 bench 测不同 prompt
    max_iterations : 防止 LLM 无限 tool_call 循环的兜底
    on_tool_call : 每次工具调用完成后的回调，CLI 用来实时打印
    """
    start = time.monotonic()

    messages: list[Any] = [
        SystemMessage(
            content=system_prompt_template.format(
                workspace=workspace,
                memory_context=memory_context,
            )
        ),
        HumanMessage(content=user_input),
    ]

    tool_call_records: list[ToolCallRecord] = []
    final_reply = ""
    stopped_reason = "ok"
    error: str | None = None

    try:
        for _ in range(max_iterations):
            res = await llm.ainvoke(messages)

            # 防御：模型可能续写 "(User) >"，截断
            if isinstance(res.content, str) and "(User) >" in res.content:
                res.content = res.content.split("(User) >")[0].strip()

            messages.append(res)

            if not res.tool_calls:
                # 自然结束
                final_reply = (res.content or "").strip()
                stopped_reason = "ok"
                break

            # 有 tool_calls，依次执行
            for tc in res.tool_calls:
                tool_name = tc["name"]
                tool_args = tc.get("args", {}) or {}
                call_id = tc.get("id", "")

                if tool_name not in tools_by_name:
                    obs = f"错误：未找到工具 {tool_name}"
                    messages.append(ToolMessage(content=obs, tool_call_id=call_id))
                    rec = ToolCallRecord(
                        name=tool_name, args=tool_args, result=obs, duration_s=0.0, call_id=call_id
                    )
                    tool_call_records.append(rec)
                    if on_tool_call:
                        on_tool_call(rec)
                    continue

                t0 = time.monotonic()
                try:
                    observation = await tools_by_name[tool_name].ainvoke(tool_args)
                except Exception as e:  # noqa: BLE001
                    observation = f"工具执行异常: {e}"
                dt = time.monotonic() - t0

                obs_str = str(observation)
                messages.append(ToolMessage(content=obs_str, tool_call_id=call_id))

                rec = ToolCallRecord(
                    name=tool_name,
                    args=tool_args,
                    result=obs_str,
                    duration_s=dt,
                    call_id=call_id,
                )
                tool_call_records.append(rec)
                if on_tool_call:
                    on_tool_call(rec)
        else:
            # for 循环正常结束（没 break），说明到达迭代上限
            stopped_reason = "max_iterations"
    except Exception as e:  # noqa: BLE001
        stopped_reason = "exception"
        error = f"{type(e).__name__}: {e}"

    return AgentRunResult(
        final_reply=final_reply,
        tool_calls=tool_call_records,
        messages=messages,
        total_duration_s=time.monotonic() - start,
        stopped_reason=stopped_reason,
        error=error,
    )