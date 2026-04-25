"""Bench 运行引擎.

一个任务的完整生命周期：
    1. 读 YAML 任务定义
    2. 起一个独立 LocalSandbox（keep_on_close 依环境变量）
    3. 把 fixtures 拷进 workspace
    4. 构造 llm + tools + agent_core.run_agent_once
    5. 给 agent 一个整体 timeout
    6. 跑所有 checks
    7. 存 trace 到 runs/<timestamp>/<task_id>.json
    8. 返回 TaskResult

关键设计：
- bench 和 CLI 完全独立。bench 不 import main.py、不依赖 mem0、不依赖任何全局状态.
- 每个任务一个独立沙盒。任务之间零干扰.
- 失败的 check 不阻止后续 check 继续跑，全部 check 完才汇总（一次调试修多个问题）.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml
from langchain_openai import ChatOpenAI

from bugstar.agent_core import run_agent_once
from bugstar.sandbox import LocalSandbox
from bugstar.tools import make_terminal_tool

from .checks import CheckResult, run_check


@dataclass
class TaskResult:
    task_id: str
    description: str
    passed: bool
    check_results: list[CheckResult] = field(default_factory=list)
    agent_duration_s: float = 0.0
    agent_tool_call_count: int = 0
    agent_stopped_reason: str = ""
    agent_error: str | None = None
    trace_path: str = ""


def _load_task(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _copy_fixtures(fixtures: list[dict[str, str]] | None, workspace: str, fixtures_root: Path) -> None:
    """把 task.yaml 里声明的 fixtures 拷到 sandbox workspace."""
    if not fixtures:
        return
    for fx in fixtures:
        src_rel = fx["from"]
        dst_rel = fx.get("to", src_rel)
        src = fixtures_root / src_rel
        dst = Path(workspace) / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


async def run_task(
    task_path: Path,
    *,
    fixtures_root: Path,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    runs_dir: Path,
    keep_workspace: bool = False,
) -> TaskResult:
    """跑一个任务，返回结构化结果."""
    task = _load_task(task_path)
    task_id = task["id"]
    description = task.get("description", "")
    prompt = task["prompt"]
    task_timeout = task.get("timeout_s", 120.0)
    fixtures = task.get("fixtures")
    checks = task.get("checks", [])

    # 独立沙盒（不复用任何全局状态）.
    sandbox = LocalSandbox(keep_on_close=keep_workspace)
    await sandbox.start()

    # 拷 fixtures
    if fixtures:
        _copy_fixtures(fixtures, sandbox.workspace, fixtures_root)

    # LLM + 工具（每个任务一个新实例，避免交叉污染）.
    terminal = make_terminal_tool(sandbox)
    tools_by_name: dict[str, Any] = {"terminal": terminal}
    llm = ChatOpenAI(model=model, temperature=temperature).bind_tools([terminal])

    # 跑 agent.
    try:
        agent_result = await asyncio.wait_for(
            run_agent_once(
                user_input=prompt,
                llm=llm,
                tools_by_name=tools_by_name,
                workspace=sandbox.workspace,
                memory_context="（bench 模式，无长期记忆）",
            ),
            timeout=task_timeout,
        )
    except TimeoutError:
        # 任务整体超时. 生成一个代表超时的"假"结果，依然跑 checks（通常会全 fail）.
        from bugstar.agent_core import AgentRunResult

        agent_result = AgentRunResult(
            final_reply="",
            tool_calls=[],
            messages=[],
            total_duration_s=task_timeout,
            stopped_reason="task_timeout",
            error=f"exceeded task timeout {task_timeout}s",
        )

    # 跑 checks. 逐个跑，不短路.
    check_results: list[CheckResult] = []
    for check in checks:
        try:
            cr = await run_check(check, sandbox=sandbox, agent_result=agent_result)
        except Exception as e:  # noqa: BLE001
            cr = CheckResult(False, f"check raised: {type(e).__name__}: {e}", str(check.get("type")))
        check_results.append(cr)

    passed = all(cr.passed for cr in check_results) if check_results else False

    # 存 trace
    runs_dir.mkdir(parents=True, exist_ok=True)
    trace_path = runs_dir / f"{task_id}.json"
    _dump_trace(trace_path, task=task, agent_result=agent_result, check_results=check_results)

    # 清理
    await sandbox.close()

    return TaskResult(
        task_id=task_id,
        description=description,
        passed=passed,
        check_results=check_results,
        agent_duration_s=agent_result.total_duration_s,
        agent_tool_call_count=len(agent_result.tool_calls),
        agent_stopped_reason=agent_result.stopped_reason,
        agent_error=agent_result.error,
        trace_path=str(trace_path),
    )


def _dump_trace(path: Path, *, task: dict, agent_result: Any, check_results: list[CheckResult]) -> None:
    """把一次任务的完整痕迹存成 JSON. 用于事后 debug。"""

    # messages 里的 langchain 对象直接 json dump 不行，需要转
    def _msg_to_dict(m: Any) -> dict:
        d: dict[str, Any] = {"type": type(m).__name__}
        content = getattr(m, "content", "")
        if isinstance(content, list):
            # content 可能是多模态列表
            d["content"] = [str(x) for x in content]
        else:
            d["content"] = str(content)
        tcs = getattr(m, "tool_calls", None)
        if tcs:
            d["tool_calls"] = [
                {"id": t.get("id"), "name": t.get("name"), "args": t.get("args")}
                for t in tcs
            ]
        tool_call_id = getattr(m, "tool_call_id", None)
        if tool_call_id:
            d["tool_call_id"] = tool_call_id
        return d

    payload = {
        "task": task,
        "agent": {
            "final_reply": agent_result.final_reply,
            "total_duration_s": agent_result.total_duration_s,
            "stopped_reason": agent_result.stopped_reason,
            "error": agent_result.error,
            "tool_calls": [asdict(t) for t in agent_result.tool_calls],
            "messages": [_msg_to_dict(m) for m in agent_result.messages],
        },
        "checks": [asdict(c) for c in check_results],
        "passed": all(c.passed for c in check_results) if check_results else False,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def run_all(
    tasks_dir: Path,
    *,
    fixtures_root: Path,
    runs_root: Path,
    task_filter: str | None = None,
    concurrency: int = 3,
    model: str = "gpt-4o",
    keep_workspace: bool = False,
) -> list[TaskResult]:
    """跑 tasks_dir 下所有 YAML 任务，返回结果列表.

    concurrency 控制并发（每个任务一个沙盒，不会互相污染）.
    太高会撞 OpenAI rate limit，3-5 通常合适.
    """
    ts = time.strftime("%Y%m%d-%H%M%S")
    runs_dir = runs_root / ts
    runs_dir.mkdir(parents=True, exist_ok=True)

    task_files = sorted(tasks_dir.glob("*.yaml"))
    if task_filter:
        task_files = [p for p in task_files if task_filter in p.name]
    if not task_files:
        raise FileNotFoundError(f"No task .yaml under {tasks_dir} (filter={task_filter!r})")

    sem = asyncio.Semaphore(concurrency)

    async def _run_one(p: Path) -> TaskResult:
        async with sem:
            return await run_task(
                p,
                fixtures_root=fixtures_root,
                model=model,
                runs_dir=runs_dir,
                keep_workspace=keep_workspace,
            )

    return await asyncio.gather(*[_run_one(p) for p in task_files])


def format_summary(results: list[TaskResult]) -> str:
    """给终端看的结果表格."""
    total = len(results)
    passed = sum(1 for r in results if r.passed)

    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append(f"Bench: {passed}/{total} passed")
    lines.append(f"{'=' * 70}")
    for r in results:
        icon = "✅" if r.passed else "❌"
        lines.append(
            f"{icon} {r.task_id:<35} "
            f"tools={r.agent_tool_call_count:<3} "
            f"dur={r.agent_duration_s:>5.1f}s "
            f"stop={r.agent_stopped_reason}"
        )
        if not r.passed:
            for cr in r.check_results:
                if not cr.passed:
                    lines.append(f"    ↳ [{cr.check_type}] {cr.detail}")
            if r.agent_error:
                lines.append(f"    ↳ agent error: {r.agent_error}")
    lines.append(f"{'=' * 70}")
    lines.append(f"Traces: {os.path.dirname(results[0].trace_path) if results else '(none)'}")
    return "\n".join(lines)