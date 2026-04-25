"""BugStar bench: 任务集 + runner + checks.

用法见项目根目录的 run_bench.py.
"""

from .checks import CheckResult, run_check
from .runner import TaskResult, format_summary, run_all, run_task

__all__ = [
    "CheckResult",
    "TaskResult",
    "format_summary",
    "run_all",
    "run_check",
    "run_task",
]