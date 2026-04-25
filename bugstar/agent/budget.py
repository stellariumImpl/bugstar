"""Budget: 给 agent 一次执行设置硬边界.

设计原则:
- 硬边界（hard limits）—— 超了直接停, 不商量
- 多维度独立计数 —— 任何一个维度爆了就停, 不用所有都爆
- 不依赖 LLM 自觉 —— LLM 永远没有"算力意识", 必须从外部约束

为什么不让 LLM 自己看预算?
  实验证明: 把"你还剩 3 次 tool_call" 塞进 prompt, LLM 大概率会
  (a) 忽略它继续按惯性来  或  (b) 过度焦虑提前收尾.
  所以预算是外部判定 + 强制中断, 不是 prompt 里的提醒.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Budget:
    """一次 agent 执行的预算上限.

    每个字段都是"上限". 超了 over_budget() 返回 True.
    """

    # --- 硬上限（外部传入）-------------------------------------------------

    max_llm_calls: int = 15
    """LLM 调用上限. 主要兜底, 避免 max_iterations 失效时的死循环."""

    max_tool_calls: int = 30
    """工具调用上限. 简单任务用不了几次, 多了通常是 agent 在挣扎."""

    max_wall_time_s: float = 120.0
    """整个执行的墙上时间上限. 注意不等于 LLM 时间总和."""

    max_consecutive_failures: int = 3
    """连续失败的 tool_call 数. 超了说明 agent 卡在死胡同."""

    max_tokens: int = 100_000
    """估算的 token 总量上限. None 表示不限."""

    # --- 实际累计（运行时维护）---------------------------------------------

    llm_calls: int = 0
    tool_calls: int = 0
    consecutive_failures: int = 0
    total_tokens: int = 0
    started_at: float = field(default_factory=time.monotonic)

    # 终止原因. 一旦设置就不再改变.
    stop_reason: str | None = None

    # --- 操作 ---------------------------------------------------------------

    def record_llm_call(self, tokens: int = 0) -> None:
        self.llm_calls += 1
        self.total_tokens += tokens

    def record_tool_call(self, *, success: bool) -> None:
        self.tool_calls += 1
        if success:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1

    def wall_time_s(self) -> float:
        return time.monotonic() - self.started_at

    def check(self) -> str | None:
        """返回 stop_reason 字符串如果已超限, 否则 None.

        每轮循环开始时调用. 一旦返回非 None, 调用方应立即停止.
        """
        if self.stop_reason is not None:
            return self.stop_reason
        if self.llm_calls >= self.max_llm_calls:
            self.stop_reason = "budget:max_llm_calls"
        elif self.tool_calls >= self.max_tool_calls:
            self.stop_reason = "budget:max_tool_calls"
        elif self.consecutive_failures >= self.max_consecutive_failures:
            self.stop_reason = "budget:too_many_failures"
        elif self.wall_time_s() >= self.max_wall_time_s:
            self.stop_reason = "budget:wall_time"
        elif self.total_tokens >= self.max_tokens:
            self.stop_reason = "budget:max_tokens"
        return self.stop_reason

    def snapshot(self) -> dict:
        """给日志/trace用的快照."""
        return {
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "consecutive_failures": self.consecutive_failures,
            "total_tokens": self.total_tokens,
            "wall_time_s": round(self.wall_time_s(), 2),
            "stop_reason": self.stop_reason,
        }