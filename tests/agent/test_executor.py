"""Executor 单元测试.

不依赖真 LLM. 用一个可编程的 FakeLLM, 你预设它每次该出什么响应.
重点验证:
- 预算 (budget) 真的硬停
- 连续失败真的触发 too_many_failures
- 多 tool_call 真的被强制串行（只跑第一个）
- 自然结束时真的做了产物快照
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from bugstar.agent import Budget, Executor
from bugstar.sandbox import LocalSandbox

# --- Fake LLM ---------------------------------------------------------------


@dataclass
class FakeAIMessage:
    """简化版的 langchain AIMessage."""

    content: str
    tool_calls: list[dict]  # 每个 dict: {id, name, args}
    response_metadata: dict


class FakeLLM:
    """可编程 fake LLM: 每次 ainvoke 按顺序返回预设响应."""

    def __init__(self, responses: list[FakeAIMessage]) -> None:
        self.responses = deque(responses)
        self.invocations: list[Any] = []

    async def ainvoke(self, messages: list[Any]) -> FakeAIMessage:
        self.invocations.append(list(messages))
        if not self.responses:
            raise RuntimeError("FakeLLM ran out of canned responses")
        return self.responses.popleft()


# --- Fake terminal tool -----------------------------------------------------


class FakeTerminalTool:
    """fake terminal: 按预设结果返回, 不真的跑 shell."""

    name = "terminal"

    def __init__(self, results: list[str]) -> None:
        self.results = deque(results)
        self.call_args: list[dict] = []

    async def ainvoke(self, args: dict) -> str:
        self.call_args.append(args)
        if not self.results:
            return "exit_code=0 duration=0.00s"
        return self.results.popleft()


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
async def sb(tmp_path: Path):
    sandbox = LocalSandbox(workspace_root=tmp_path / "ws")
    await sandbox.start()
    try:
        yield sandbox
    finally:
        await sandbox.close()


def _ai(content: str = "", tool_calls: list[dict] | None = None) -> FakeAIMessage:
    return FakeAIMessage(content=content, tool_calls=tool_calls or [], response_metadata={})


def _tc(call_id: str, name: str = "terminal", args: dict | None = None) -> dict:
    return {"id": call_id, "name": name, "args": args or {"command": "echo hi"}}


# --- 1. 自然结束（最基本路径）-----------------------------------------------


async def test_natural_finish_records_artifacts(sb: LocalSandbox) -> None:
    # agent 一轮就回了文本, 没调工具.
    llm = FakeLLM([_ai(content="done")])
    tool = FakeTerminalTool([])
    ex = Executor(llm=llm, tools_by_name={"terminal": tool}, sandbox=sb)

    result = await ex.run("hi")

    assert result.ok
    assert result.final_reply == "done"
    assert result.stopped_reason == "ok"
    # 产物快照应该被记录（即便是空目录, 至少有 . 和 ..）
    assert "total" in result.artifacts_listing or "." in result.artifacts_listing


# --- 2. 强制串行（多 tool_call 只执行第一个）---------------------------------


async def test_serial_enforcement_only_first_tool_runs(sb: LocalSandbox) -> None:
    # 第 1 轮: LLM 同时下两个 tool_call. Executor 只能执行第一个.
    # 第 2 轮: LLM 看完第一个的结果, 收尾.
    llm = FakeLLM([
        _ai(tool_calls=[
            _tc("c1", args={"command": "echo first"}),
            _tc("c2", args={"command": "echo second"}),
        ]),
        _ai(content="finished"),
    ])
    tool = FakeTerminalTool(["exit_code=0 duration=0.01s\n--- stdout ---\nfirst"])
    ex = Executor(llm=llm, tools_by_name={"terminal": tool}, sandbox=sb)

    result = await ex.run("do two things")

    # 真正调到 terminal 的只有 1 次（第二个被推迟了）
    assert len(tool.call_args) == 1
    assert tool.call_args[0]["command"] == "echo first"
    # 但 records 里两个都在: 一个执行, 一个 deferred
    executed = [r for r in result.tool_calls if not r.deferred]
    deferred = [r for r in result.tool_calls if r.deferred]
    assert len(executed) == 1
    assert len(deferred) == 1
    assert result.deferred_tool_calls == 1


# --- 3. 失败计数（连续失败超限触发停止）-------------------------------------


async def test_consecutive_failures_trigger_stop(sb: LocalSandbox) -> None:
    # LLM 连续下三个会失败的 tool_call. budget.max_consecutive_failures=2,
    # 所以第 2 次失败后 budget.check() 应返回 too_many_failures, 第 3 轮不会再调 LLM.
    llm = FakeLLM([
        _ai(tool_calls=[_tc("c1", args={"command": "false"})]),
        _ai(tool_calls=[_tc("c2", args={"command": "false"})]),
        _ai(tool_calls=[_tc("c3", args={"command": "false"})]),  # 不应该被消费
    ])
    tool = FakeTerminalTool([
        "exit_code=1 duration=0.00s\n--- stderr ---\nfail",
        "exit_code=1 duration=0.00s\n--- stderr ---\nfail",
        "exit_code=1 duration=0.00s\n--- stderr ---\nfail",
    ])
    budget = Budget(max_consecutive_failures=2)
    ex = Executor(llm=llm, tools_by_name={"terminal": tool}, sandbox=sb, budget=budget)

    result = await ex.run("keep failing")

    assert result.stopped_reason == "budget:too_many_failures"
    # 应该只调用 LLM 2 次（第 3 轮一开始 budget.check 就拒绝了）
    assert budget.llm_calls == 2
    assert budget.consecutive_failures == 2


async def test_success_resets_consecutive_failures(sb: LocalSandbox) -> None:
    # 失败 → 成功 → 失败 → 不应触发 too_many_failures（不连续）
    llm = FakeLLM([
        _ai(tool_calls=[_tc("c1", args={"command": "false"})]),
        _ai(tool_calls=[_tc("c2", args={"command": "true"})]),
        _ai(tool_calls=[_tc("c3", args={"command": "false"})]),
        _ai(content="finished"),
    ])
    tool = FakeTerminalTool([
        "exit_code=1 duration=0.00s",
        "exit_code=0 duration=0.00s",
        "exit_code=1 duration=0.00s",
    ])
    budget = Budget(max_consecutive_failures=2)
    ex = Executor(llm=llm, tools_by_name={"terminal": tool}, sandbox=sb, budget=budget)

    result = await ex.run("mixed")

    assert result.ok, f"expected ok, got {result.stopped_reason}"
    # 最后一次失败后再成功收尾, 连续失败计数应该是 1（最后那个 false）
    assert budget.consecutive_failures == 1


# --- 4. 预算硬上限 -----------------------------------------------------------


async def test_max_llm_calls_stops(sb: LocalSandbox) -> None:
    # 预算只允许 2 次 LLM 调用, agent 想搞 5 轮也搞不成
    llm = FakeLLM([
        _ai(tool_calls=[_tc("c1")]),
        _ai(tool_calls=[_tc("c2")]),
        _ai(tool_calls=[_tc("c3")]),  # 不应被消费
    ])
    tool = FakeTerminalTool(["exit_code=0", "exit_code=0", "exit_code=0"])
    budget = Budget(max_llm_calls=2)
    ex = Executor(llm=llm, tools_by_name={"terminal": tool}, sandbox=sb, budget=budget)

    result = await ex.run("loop forever")

    assert result.stopped_reason == "budget:max_llm_calls"
    assert budget.llm_calls == 2


async def test_max_tool_calls_stops(sb: LocalSandbox) -> None:
    llm = FakeLLM([
        _ai(tool_calls=[_tc("c1")]),
        _ai(tool_calls=[_tc("c2")]),
        _ai(tool_calls=[_tc("c3")]),
    ])
    tool = FakeTerminalTool(["exit_code=0", "exit_code=0", "exit_code=0"])
    budget = Budget(max_tool_calls=2)
    ex = Executor(llm=llm, tools_by_name={"terminal": tool}, sandbox=sb, budget=budget)

    result = await ex.run("call lots of tools")

    assert result.stopped_reason == "budget:max_tool_calls"
    assert budget.tool_calls == 2


# --- 5. 未知工具 ------------------------------------------------------------


async def test_unknown_tool_recorded_as_failure(sb: LocalSandbox) -> None:
    llm = FakeLLM([
        _ai(tool_calls=[_tc("c1", name="nonexistent")]),
        _ai(content="ok"),
    ])
    ex = Executor(llm=llm, tools_by_name={"terminal": FakeTerminalTool([])}, sandbox=sb)

    result = await ex.run("call ghost")

    assert result.ok  # 自然结束
    assert len(result.tool_calls) == 1
    assert not result.tool_calls[0].success
    assert "未找到工具" in result.tool_calls[0].result


# --- 6. budget snapshot 进入结果 --------------------------------------------


async def test_budget_snapshot_in_result(sb: LocalSandbox) -> None:
    llm = FakeLLM([_ai(content="immediate")])
    ex = Executor(llm=llm, tools_by_name={"terminal": FakeTerminalTool([])}, sandbox=sb)

    result = await ex.run("hi")

    snap = result.budget_snapshot
    assert snap["llm_calls"] == 1
    assert snap["tool_calls"] == 0
    assert "wall_time_s" in snap
    assert json.dumps(snap)  # 可序列化, 给 trace 用