import os
import shutil
import asyncio
import uuid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_community.tools.shell.tool import ShellTool
from langchain_core.tools import tool
from mem0 import Memory

load_dotenv()

# --- 固定配置 ---
USER_ID = "felix"
MEM0_LOCAL_DIR = os.path.join(os.getcwd(), ".mem0")   # Qdrant 向量库
MEM0_HOME_DIR = os.path.expanduser("~/.mem0")          # mem0 默认的 history sqlite

# --- 1. 核心组件初始化 ---
shell_tool = ShellTool(allow_dangerous_requests=True)

memo = Memory.from_config({
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "path": MEM0_LOCAL_DIR,
        }
    }
})


def _make_memory_tool(user_id: str):
    """
    用闭包固定 user_id，不把它暴露给 LLM，避免模型编造身份。
    """
    @tool
    def manage_memories(action: str) -> str:
        """
        管理用户的长期记忆事实。
        - action='list':  查看所有事实
        - action='reset': 清空记忆（会同时清理本地向量库与 history sqlite）
        """
        global memo  # 必须在函数顶部声明，且先于任何 memo 的使用

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
            # 1) mem0 自己的 reset
            try:
                memo.reset()
            except Exception as e:
                print(f"[!] memo.reset() 异常（忽略继续物理清理）: {e}")

            # 2) 物理删除两份存储，兜底清干净
            for path in (MEM0_LOCAL_DIR, MEM0_HOME_DIR):
                if os.path.exists(path):
                    try:
                        shutil.rmtree(path)
                        print(f"[*] 已删除 {path}")
                    except Exception as e:
                        print(f"[!] 删除 {path} 失败: {e}")

            # 3) 重新初始化 memo，保证后续调用不会崩
            memo = Memory.from_config({
                "vector_store": {
                    "provider": "qdrant",
                    "config": {"path": MEM0_LOCAL_DIR},
                }
            })

            return "记忆库已成功清空（含本地向量库与 history）。本轮对话不会被写回。"

        return f"未知操作: {action}。支持的 action: 'list' | 'reset'"

    return manage_memories


manage_memories = _make_memory_tool(USER_ID)

tools = [shell_tool, manage_memories]

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    stop=["(User) >", "User:", "\n(User)"]
).bind_tools(tools)


# --- 2. 调度层 ---
async def run_bugstar(user_input: str, session_id: str, user_id: str = USER_ID):
    # A. 检索长期事实
    search_results = memo.search(user_input, filters={"user_id": user_id})
    mem_strings = []
    results_list = search_results.get("results", []) if isinstance(search_results, dict) else search_results
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

    messages = [
        SystemMessage(content=f"""你是 BugStar，一个运行在 macOS 上的专业工程助手。

你可以根据需要使用 terminal 执行命令，或使用 manage_memories 管理你的知识。

【已知上下文（长期记忆）】:
{memory_context}
"""),
        HumanMessage(content=user_input)
    ]

    # 关键防御变量：本轮是否调用了 reset？是的话跳过记忆回写
    skip_memory_write = False

    while True:
        res = await llm.ainvoke(messages)
        if "(User) >" in res.content:
            res.content = res.content.split("(User) >")[0].strip()
        messages.append(res)

        # 扫描本轮 tool_calls 里是否有 reset
        if res.tool_calls:
            for tc in res.tool_calls:
                if tc["name"] == "manage_memories" and tc["args"].get("action") == "reset":
                    skip_memory_write = True

        if not res.tool_calls:
            if res.content.strip():
                print(f"\n[BugStar]: {res.content}")

            # 只在未触发 reset 且回复非空时，才写回长期记忆
            if not skip_memory_write and res.content.strip():
                try:
                    memo.add(
                        f"User: {user_input}\nAssistant: {res.content}",
                        user_id=user_id,
                    )
                except Exception as e:
                    print(f"[!] 写入长期记忆失败: {e}")
            elif skip_memory_write:
                print("[*] 本轮包含 reset 操作，跳过记忆回写。")
            break

        # 执行工具调用
        for tool_call in res.tool_calls:
            tool_name = tool_call["name"]
            available_tools = {
                "terminal": shell_tool,
                "manage_memories": manage_memories,
            }

            if tool_name in available_tools:
                print(f"[*] 调用工具 [{tool_name}]: {tool_call['args']}")
                try:
                    observation = await available_tools[tool_name].ainvoke(tool_call["args"])
                except Exception as e:
                    observation = f"工具执行异常: {e}"
                messages.append(ToolMessage(
                    content=str(observation),
                    tool_call_id=tool_call["id"],
                ))
            else:
                messages.append(ToolMessage(
                    content=f"错误：未找到工具 {tool_name}",
                    tool_call_id=tool_call["id"],
                ))


async def main():
    print("--- BugStar (记忆自省模式) 已启动 ---")
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

    print("\n[*] 正在安全关闭 BugStar...")


if __name__ == "__main__":
    asyncio.run(main())