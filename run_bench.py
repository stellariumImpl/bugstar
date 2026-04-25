"""Bench 命令行入口.

用法:
    uv run python run_bench.py                        # 跑所有任务
    uv run python run_bench.py --filter t001          # 只跑名字含 t001 的任务
    uv run python run_bench.py --concurrency 1        # 串行跑（方便看日志）
    uv run python run_bench.py --keep-workspace       # 失败时保留沙盒工作目录方便检查
    uv run python run_bench.py --model gpt-4o-mini    # 换模型

所有运行 trace 会存到 tests/bench/runs/<timestamp>/<task_id>.json,
任务失败时强烈建议先看对应 trace.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from tests.bench import format_summary, run_all

# 项目根
ROOT = Path(__file__).parent
TASKS_DIR = ROOT / "tests" / "bench" / "tasks"
FIXTURES_DIR = ROOT / "tests" / "bench" / "fixtures"
RUNS_ROOT = ROOT / "tests" / "bench" / "runs"


def _setup_logging() -> None:
    """bench 默认安静模式：只看 WARN+. 噪声少，汇总清晰."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # sandbox 的 info 日志在 bench 里没意义，压掉
    logging.getLogger("bugstar.sandbox.local").setLevel(logging.WARNING)
    # httpx / openai 同样
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> int:
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY 未设置. 请在 .env 里配置后再跑 bench.", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser(description="BugStar bench runner")
    parser.add_argument("--filter", default=None, help="只跑名字中含该子串的任务（比如 t001）")
    parser.add_argument("--concurrency", type=int, default=3, help="并发任务数（默认 3）")
    parser.add_argument("--model", default="gpt-4o", help="LLM 型号（默认 gpt-4o）")
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="任务完成后不删 sandbox 工作目录. 方便失败调试.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="开启详细日志（含 sandbox 命令执行）",
    )
    args = parser.parse_args()

    _setup_logging()
    if args.verbose:
        logging.getLogger("bugstar").setLevel(logging.INFO)

    print("🧪 BugStar bench")
    print(f"   tasks_dir    : {TASKS_DIR}")
    print(f"   model        : {args.model}")
    print(f"   concurrency  : {args.concurrency}")
    if args.filter:
        print(f"   filter       : {args.filter}")
    print()

    results = asyncio.run(
        run_all(
            tasks_dir=TASKS_DIR,
            fixtures_root=FIXTURES_DIR,
            runs_root=RUNS_ROOT,
            task_filter=args.filter,
            concurrency=args.concurrency,
            model=args.model,
            keep_workspace=args.keep_workspace,
        )
    )

    print(format_summary(results))

    passed = sum(1 for r in results if r.passed)
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())