import asyncio
import logging
import os
import shutil
import uuid

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from bugstar import run_agent_once
from bugstar.agent_core import ToolCallRecord
from bugstar.sandbox import LocalSandbox
from bugstar.tools import make_terminal_tool

load_dotenv()

# 日志: 默认保留 warning+，避免内部执行细节刷屏到交互界面。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
# 压掉 httpx / httpcore / openai 的请求日志噪声. 这些在开发初期看着热闹，
# 实际信息量接近零，还会插进 input() 提示符后面导致错觉"用户说了日志".
for noisy in ("httpx", "httpcore", "openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# sandbox 的命令级日志对开发排查有帮助，但日常交互里基本是噪声，
# 尤其像内部快照脚本这类实现细节不应直接打到用户界面。
logging.getLogger("bugstar.sandbox.local").setLevel(logging.WARNING)

# --- 固定配置 ---
USER_ID = "felix"
MEM0_LOCAL_DIR = os.path.join(os.getcwd(), ".mem0")  # Qdrant 向量库
MEM0_HOME_DIR = os.path.expanduser("~/.mem0")  # mem0 默认的 history sqlite

# 开关：默认启用 mem0. 开发阶段调 agent 本体时可以 `BUGSTAR_MEMORY=0` 关掉，
# 省掉每轮多一次 LLM 调用 + 多次 embedding 的开销.
MEMORY_ENABLED = os.getenv("BUGSTAR_MEMORY", "1") != "0"

# 开关：是否保留工作目录. 调试期间建议开 (BUGSTAR_KEEP_WORKSPACE=1),
# 这样 session 结束后 agent 创建/修改的文件还在，你能进去看.
# 注意: 开着的话 ~/.bugstar/workspaces/ 下会累积，偶尔手动清一下.
KEEP_WORKSPACE = os.getenv("BUGSTAR_KEEP_WORKSPACE", "0") == "1"

# --- 1. 核心组件初始化 ---

# 沙盒在 main() 里异步启动，这里只声明变量.
sandbox: LocalSandbox | None = None

# mem0 也懒初始化：关闭时完全不 import，避免无谓的 spaCy 下载.
memo = None
if MEMORY_ENABLED:
    from mem0 import Memory

    memo = Memory.from_config({
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "path": MEM0_LOCAL_DIR,
            }
        }
    })


def _make_memory_tool(user_id: str):
    """用闭包固定 user_id，不把它暴露给 LLM，避免模型编造身份。"""

    @tool
    def manage_memories(action: str) -> str:
        """管理用户的长期记忆事实。

        - action='list': 查看所有事实
        - action='reset': 清空记忆（会同时清理本地向量库与 history sqlite）
        """
        global memo
        if not MEMORY_ENABLED or memo is None:
            return "长期记忆当前已禁用（BUGSTAR_MEMORY=0）。"

        if action == "list":
            memories = memo.get_all(filters={"user_id": user_id})
            if not memories:
                return "目前记忆库中没有任何记录。"
            results = memories.get("results", []) if isinstance(memories, dict) else memories
            if not results:
                return "目前记忆库中没有任何记录。"
            facts = [f"- {m.get('memory', str(m))}" for m in results]
            return "\n".join(facts)
        elif action == "reset":
            try:
                memo.reset()
            except Exception as e:
                print(f"[!] memo.reset() 异常（忽略继续物理清理）: {e}")

            for path in (MEM0_LOCAL_DIR, MEM0_HOME_DIR):
                if os.path.exists(path):
                    try:
                        shutil.rmtree(path)
                        print(f"[*] 已删除 {path}")
                    except Exception as e:
                        print(f"[!] 删除 {path} 失败: {e}")

            from mem0 import Memory as _Memory  # 局部 import，避免 reset 前就挂了
            memo = _Memory.from_config({
                "vector_store": {
                    "provider": "qdrant",
                    "config": {"path": MEM0_LOCAL_DIR},
                }
            })
            return "记忆库已成功清空（含本地向量库与 history）。本轮对话不会被写回。"
        return f"未知操作: {action}。支持的 action: 'list' | 'reset'"

    return manage_memories


manage_memories = _make_memory_tool(USER_ID)

# terminal 工具在 main() 里等 sandbox 启动后再构造.
terminal_tool = None
tools: list = []
llm = None


def _build_llm_and_tools():
    """在 sandbox 准备好之后调用，构造 llm 和工具列表."""
    global terminal_tool, tools, llm
    assert sandbox is not None, "sandbox 必须先 start()"
    terminal_tool = make_terminal_tool(sandbox)

    # 记忆关闭时不把 manage_memories 暴露给 LLM，避免它幻觉去用一个废工具.
    tools = [terminal_tool]
    if MEMORY_ENABLED:
        tools.append(manage_memories)

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        stop=["(User) >", "User:", "\n(User)"]
    ).bind_tools(tools)


# --- 2. 调度层 ---

async def run_bugstar(user_input: str, session_id: str, user_id: str = USER_ID):
    assert llm is not None and terminal_tool is not None, "请先调用 _build_llm_and_tools()"

    # A. 检索长期事实（记忆关闭时跳过）
    if MEMORY_ENABLED and memo is not None:
        search_results = memo.search(user_input, filters={"user_id": user_id})
        mem_strings = []
        results_list = (
            search_results.get("results", [])
            if isinstance(search_results, dict)
            else search_results
        )
        for m in results_list:
            if isinstance(m, dict):
                content = m.get("memory") or m.get("text") or m.get("content")
                if content:
                    mem_strings.append(content)
            elif isinstance(m, str):
                mem_strings.append(m)
            else:
                mem_strings.append(getattr(m, "memory", str(m)))
        memory_context = "\n".join(mem_strings) if mem_strings else "尚无长期记忆。"
    else:
        memory_context = "（长期记忆已禁用）"

    assert sandbox is not None

    # B. 组装工具字典. manage_memories 只有开了 mem0 才放.
    tools_by_name: dict = {"terminal": terminal_tool}
    if MEMORY_ENABLED:
        tools_by_name["manage_memories"] = manage_memories

    # C. 实时打印每次 tool_call（CLI 体验）.
    skip_memory_write = False

    def _on_tool_call(rec: ToolCallRecord) -> None:
        nonlocal skip_memory_write
        print(f"[*] 调用工具 [{rec.name}]: {rec.args}")
        if rec.name == "manage_memories" and rec.args.get("action") == "reset":
            skip_memory_write = True

    # D. 跑 agent.
    result = await run_agent_once(
        user_input=user_input,
        llm=llm,
        tools_by_name=tools_by_name,
        workspace=sandbox.workspace,
        memory_context=memory_context,
        on_tool_call=_on_tool_call,
        sandbox=sandbox,
    )

    # E. 打印最终回复 + 处理异常/超迭代.
    if result.stopped_reason == "exception":
        print(f"[!] 本轮 agent 执行异常: {result.error}")
        return
    if result.stopped_reason == "max_iterations":
        print("[!] 本轮 agent 达到迭代上限，已停止.")

    if result.final_reply:
        print(f"\n[BugStar]: {result.final_reply}")

        # F. 回写长期记忆（只在开启、未触发 reset、回复非空时）.
        if MEMORY_ENABLED and memo is not None and not skip_memory_write:
            try:
                memo.add(
                    f"User: {user_input}\nAssistant: {result.final_reply}",
                    user_id=user_id,
                )
            except Exception as e:
                print(f"[!] 写入长期记忆失败: {e}")
        elif skip_memory_write:
            print("[*] 本轮包含 reset 操作，跳过记忆回写。")


async def main():
    global sandbox

    mem_status = "ON" if MEMORY_ENABLED else "OFF (BUGSTAR_MEMORY=0)"
    print(f"--- BugStar 已启动 (Memory={mem_status}, LocalSandbox) ---")

    sandbox = LocalSandbox(keep_on_close=KEEP_WORKSPACE)
    await sandbox.start()
    _build_llm_and_tools()

    print(f"[*] 工作目录: {sandbox.workspace}")
    if KEEP_WORKSPACE:
        print("[*] BUGSTAR_KEEP_WORKSPACE=1: 退出后目录保留，可随时查看产物.")

    try:
        while True:
            try:
                q = input("\n(User) > ")
            except (EOFError, KeyboardInterrupt):
                print("\n[*] 收到退出信号。")
                break

            if q.lower().strip() in ["q", "exit", "quit"]:
                break
            if not q.strip():
                continue

            try:
                await run_bugstar(q, str(uuid.uuid4())[:8])
            except Exception as e:
                print(f"[!] 本轮执行异常: {e}")
    finally:
        print("\n[*] 正在安全关闭 BugStar...")
        ws_path = sandbox.workspace if sandbox else None
        await sandbox.close()
        if KEEP_WORKSPACE and ws_path:
            print(f"[*] 工作目录已保留: {ws_path}")
            print(f"[*] 查看: open {ws_path}")


if __name__ == "__main__":
    asyncio.run(main())
