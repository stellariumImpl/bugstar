"""Bench 任务的成功判定.

每个 check 返回 (passed: bool, detail: str).
detail 用于失败时告诉你"为什么没过"，方便 debug.

Check 类型:
    file_exists     检查沙盒工作目录里某个文件存在
    file_contains   检查文件内容包含某字符串
    shell           跑一段 shell，按 exit_code / stdout / stderr 判定
    reply_contains  最终 agent 回复包含某子串
    reply_not_contains 最终 agent 回复不包含某子串
    tool_called     是否调用了某个工具
    tool_not_called 是否没调用某个工具（用来验证"agent 应该拒绝"的任务）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bugstar.agent_core import AgentRunResult
from bugstar.sandbox import Sandbox


@dataclass
class CheckResult:
    passed: bool
    detail: str
    check_type: str


async def run_check(
    check: dict[str, Any],
    *,
    sandbox: Sandbox,
    agent_result: AgentRunResult,
) -> CheckResult:
    """分发 check. 不认识的类型直接 fail，不要静默放过."""
    t = check.get("type")
    if t == "file_exists":
        return await _check_file_exists(check, sandbox=sandbox)
    if t == "file_contains":
        return await _check_file_contains(check, sandbox=sandbox)
    if t == "shell":
        return await _check_shell(check, sandbox=sandbox)
    if t == "reply_contains":
        return _check_reply_contains(check, agent_result=agent_result)
    if t == "reply_not_contains":
        return _check_reply_not_contains(check, agent_result=agent_result)
    if t == "tool_called":
        return _check_tool_called(check, agent_result=agent_result)
    if t == "tool_not_called":
        return _check_tool_not_called(check, agent_result=agent_result)
    return CheckResult(False, f"unknown check type: {t!r}", str(t))


# --- 文件系统 check ---------------------------------------------------------


async def _check_file_exists(check: dict[str, Any], *, sandbox: Sandbox) -> CheckResult:
    path = check["path"]
    # 用 test -e，这样软链/目录/普通文件都算
    r = await sandbox.exec(f"test -e {_q(path)}")
    if r.ok:
        return CheckResult(True, f"{path} exists", "file_exists")
    return CheckResult(False, f"{path} does not exist in workspace", "file_exists")


async def _check_file_contains(check: dict[str, Any], *, sandbox: Sandbox) -> CheckResult:
    path = check["path"]
    needle = check["contains"]
    try:
        content = await sandbox.read_file(path)
        text = content.decode("utf-8", errors="replace")
    except FileNotFoundError:
        return CheckResult(False, f"{path} not found", "file_contains")
    except Exception as e:  # noqa: BLE001
        return CheckResult(False, f"failed to read {path}: {e}", "file_contains")

    if needle in text:
        return CheckResult(True, f"{path} contains {needle!r}", "file_contains")
    return CheckResult(
        False,
        f"{path} does not contain {needle!r}. actual head: {text[:200]!r}",
        "file_contains",
    )


# --- shell check ------------------------------------------------------------


async def _check_shell(check: dict[str, Any], *, sandbox: Sandbox) -> CheckResult:
    cmd = check["cmd"]
    expect_exit = check.get("expect_exit_code", 0)
    expect_out = check.get("expect_stdout_contains")
    expect_out_not = check.get("expect_stdout_not_contains")
    timeout = check.get("timeout_s", 30.0)

    r = await sandbox.exec(cmd, timeout_s=timeout)

    if r.exit_code != expect_exit:
        return CheckResult(
            False,
            f"exit_code={r.exit_code} expected {expect_exit}. stderr={r.stderr[:200]!r}",
            "shell",
        )
    if expect_out is not None and expect_out not in r.stdout:
        return CheckResult(
            False,
            f"stdout missing {expect_out!r}. got: {r.stdout[:200]!r}",
            "shell",
        )
    if expect_out_not is not None and expect_out_not in r.stdout:
        return CheckResult(
            False,
            f"stdout contains {expect_out_not!r} but should not",
            "shell",
        )
    return CheckResult(True, f"shell ok: {cmd}", "shell")


# --- reply check ------------------------------------------------------------


def _check_reply_contains(check: dict[str, Any], *, agent_result: AgentRunResult) -> CheckResult:
    needles = check["needles"] if "needles" in check else [check["needle"]]
    case_insensitive = check.get("case_insensitive", False)
    reply = agent_result.final_reply
    if case_insensitive:
        reply_cmp = reply.lower()
        needles_cmp = [n.lower() for n in needles]
    else:
        reply_cmp = reply
        needles_cmp = needles
    missing = [n for n, nc in zip(needles, needles_cmp, strict=True) if nc not in reply_cmp]
    if missing:
        return CheckResult(
            False,
            f"reply missing: {missing}. reply head: {reply[:200]!r}",
            "reply_contains",
        )
    return CheckResult(True, f"reply contains all of {needles}", "reply_contains")


def _check_reply_not_contains(
    check: dict[str, Any], *, agent_result: AgentRunResult
) -> CheckResult:
    needles = check["needles"] if "needles" in check else [check["needle"]]
    case_insensitive = check.get("case_insensitive", False)
    reply = agent_result.final_reply
    reply_cmp = reply.lower() if case_insensitive else reply
    needles_cmp = [n.lower() for n in needles] if case_insensitive else needles
    hits = [n for n, nc in zip(needles, needles_cmp, strict=True) if nc in reply_cmp]
    if hits:
        return CheckResult(
            False,
            f"reply unexpectedly contains: {hits}",
            "reply_not_contains",
        )
    return CheckResult(True, f"reply has none of {needles}", "reply_not_contains")


# --- tool usage check -------------------------------------------------------


def _check_tool_called(check: dict[str, Any], *, agent_result: AgentRunResult) -> CheckResult:
    name = check["name"]
    calls = [c for c in agent_result.tool_calls if c.name == name]
    min_times = check.get("min_times", 1)
    if len(calls) >= min_times:
        return CheckResult(True, f"{name} called {len(calls)} times", "tool_called")
    return CheckResult(
        False,
        f"{name} called {len(calls)} times, expected >= {min_times}",
        "tool_called",
    )


def _check_tool_not_called(
    check: dict[str, Any], *, agent_result: AgentRunResult
) -> CheckResult:
    name = check["name"]
    calls = [c for c in agent_result.tool_calls if c.name == name]
    if not calls:
        return CheckResult(True, f"{name} not called (as expected)", "tool_not_called")
    return CheckResult(
        False,
        f"{name} was called {len(calls)} times, expected 0. first args={calls[0].args}",
        "tool_not_called",
    )


# --- 小工具 -----------------------------------------------------------------


def _q(s: str) -> str:
    """给 shell 命令里的路径加单引号转义."""
    return "'" + s.replace("'", "'\\''") + "'"