"""Bench 离线 smoke 测试. 不需要 OpenAI key.

用假的 AgentRunResult 走完 check 引擎，验证:
- fixture 能被拷进 sandbox
- 每种 check 类型都能判定
- trace 文件能生成且可重读
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bugstar.agent_core import AgentRunResult, ToolCallRecord
from bugstar.sandbox import LocalSandbox
from tests.bench.checks import CheckResult, run_check
from tests.bench.runner import _copy_fixtures, _dump_trace


async def main() -> int:
    failures = []

    # --- 1. file_exists + file_contains + shell --------------------------
    async with LocalSandbox() as sb:
        # 预置一个文件
        await sb.write_file("greeting.txt", "Hello, BugStar!")

        fake_result = AgentRunResult(
            final_reply="已经写入 greeting.txt",
            tool_calls=[ToolCallRecord(name="terminal", args={"command": "ls"}, result="ok", duration_s=0.01)],
            messages=[],
            total_duration_s=0.5,
            stopped_reason="ok",
        )

        checks_to_pass = [
            {"type": "file_exists", "path": "greeting.txt"},
            {"type": "file_contains", "path": "greeting.txt", "contains": "BugStar"},
            {"type": "shell", "cmd": "cat greeting.txt", "expect_stdout_contains": "Hello"},
            {"type": "reply_contains", "needle": "greeting.txt"},
            {"type": "tool_called", "name": "terminal"},
            {"type": "tool_not_called", "name": "manage_memories"},
        ]
        for c in checks_to_pass:
            r = await run_check(c, sandbox=sb, agent_result=fake_result)
            if not r.passed:
                failures.append(f"[pass-expected] {c}: {r.detail}")
            else:
                print(f"  ✓ {c['type']}: {r.detail}")

        checks_to_fail = [
            {"type": "file_exists", "path": "does_not_exist.txt"},
            {"type": "file_contains", "path": "greeting.txt", "contains": "NOT_HERE"},
            {"type": "shell", "cmd": "false"},
            {"type": "reply_contains", "needle": "unrelated_substring_xyz"},
            {"type": "tool_called", "name": "nonexistent_tool"},
            {"type": "tool_not_called", "name": "terminal"},
        ]
        for c in checks_to_fail:
            r = await run_check(c, sandbox=sb, agent_result=fake_result)
            if r.passed:
                failures.append(f"[fail-expected] {c}: unexpectedly passed")
            else:
                print(f"  ✓ correctly failed: {c['type']}: {r.detail[:60]}")

    # --- 2. fixture 拷贝 --------------------------------------------------
    async with LocalSandbox() as sb:
        fixtures_root = Path(__file__).parent / "fixtures"
        _copy_fixtures(
            [{"from": "people.csv", "to": "people.csv"}],
            sb.workspace,
            fixtures_root,
        )
        # 用 awk 数行比 wc -l 可靠: wc -l 数的是换行符，最后一行没 \n 会少算一行.
        # awk END{print NR} 无论有没有尾 \n 都正确.
        r = await sb.exec("awk 'END{print NR}' people.csv")
        line_count = r.stdout.strip() if r.ok else ""
        if line_count != "6":  # 5 data rows + 1 header
            failures.append(f"fixture copy failed: got {line_count!r} lines (expected 6). stderr={r.stderr!r}")
        else:
            print(f"  ✓ fixture copied, line count = {line_count}")

    # --- 3. trace dump ---------------------------------------------------
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        trace_path = Path(tmp) / "x.json"
        fake_task = {"id": "t999_smoke", "prompt": "test", "checks": []}
        fake_agent = AgentRunResult(
            final_reply="done",
            tool_calls=[ToolCallRecord(name="terminal", args={"command": "ls"}, result="ok", duration_s=0.01)],
            messages=[],
            total_duration_s=0.5,
            stopped_reason="ok",
        )
        fake_checks = [CheckResult(passed=True, detail="ok", check_type="shell")]
        _dump_trace(trace_path, task=fake_task, agent_result=fake_agent, check_results=fake_checks)
        data = json.loads(trace_path.read_text(encoding="utf-8"))
        assert data["task"]["id"] == "t999_smoke"
        assert data["passed"] is True
        assert data["agent"]["final_reply"] == "done"
        assert len(data["agent"]["tool_calls"]) == 1
        print(f"  ✓ trace dump: {len(data['agent']['tool_calls'])} tool_call, passed={data['passed']}")

    if failures:
        print("\n❌ FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\n✅ All bench infrastructure smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))