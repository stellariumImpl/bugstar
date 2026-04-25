"""bugstar.agent: agent 的"思考结构".

当前包含:
- Budget: 预算与硬边界
- Executor: 有边界的执行循环（替代 Day 1/2 的 run_agent_once）

后续会加:
- Reflector (Milestone 3.2)
- Profiler / Planner (Milestone 3.3)
"""

from .budget import Budget
from .executor import Executor, ExecutorResult, ToolCallRecord

__all__ = [
    "Budget",
    "Executor",
    "ExecutorResult",
    "ToolCallRecord",
]