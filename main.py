import os
import asyncio
import uuid
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_community.tools.shell.tool import ShellTool
from langchain_core.tools import tool  # 导入 tool 装饰器
from mem0 import Memory

load_dotenv()

# --- 1. 核心组件初始化 ---
shell_tool = ShellTool(allow_dangerous_requests=True)

# 默认本地模式，数据存储在本地文件
# 指定本地存储路径为项目根目录下的 .mem0 文件夹
memo = Memory.from_config({
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "path": os.path.join(os.getcwd(), ".mem0"), # 强制存放在当前目录
        }
    }
})

@tool
def manage_memories(action: str, user_id: str = "felix") -> str:
    """
    管理用户的长期记忆事实。
    - action='list': 查看所有事实
    - action='reset': 清空记忆
    """
    if action == "list":
        # 修复点：这里必须使用 filters 传参
        memories = memo.get_all(filters={"user_id": user_id})
        if not memories:
            return "目前记忆库中没有任何记录。"
        # 兼容性提取：有些版本返回列表，有些返回字典下的 results
        results = memories.get("results", []) if isinstance(memories, dict) else memories
        facts = [f"- {m.get('memory', str(m))}" for m in results]
        return "\n".join(facts)

    elif action == "reset":
        memo.reset()
        return "记忆库已成功清空。"

    return "未知操作。"


# 更新工具集
tools = [shell_tool, manage_memories]

# 初始化 LLM，确保 stop 参数防止幻觉
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    stop=["(User) >", "User:", "\n(User)"]
).bind_tools(tools)


# --- 2. 调度层：带语义记忆与防御机制的循环 ---
async def run_bugstar(user_input, session_id, user_id="felix"):
    # A. 检索长期事实
    search_results = memo.search(user_input, filters={"user_id": user_id})
    mem_strings = []
    results_list = search_results.get("results", []) if isinstance(search_results, dict) else search_results
    for m in results_list:
        if isinstance(m, dict):
            content = m.get("memory") or m.get("text") or m.get("content")
            if content: mem_strings.append(content)
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

    while True:
        res = await llm.ainvoke(messages)
        if "(User) >" in res.content:
            res.content = res.content.split("(User) >")[0].strip()
        messages.append(res)

        if not res.tool_calls:
            if res.content.strip():
                print(f"\n[BugStar]: {res.content}")
            # 对话结束，提炼记忆
            memo.add(f"User: {user_input}\nAssistant: {res.content}", user_id=user_id)
            break

        for tool_call in res.tool_calls:
            # 动态匹配工具
            tool_name = tool_call["name"]
            available_tools = {"terminal": shell_tool, "manage_memories": manage_memories}

            if tool_name in available_tools:
                print(f"[*] 调用工具 [{tool_name}]: {tool_call['args']}")
                observation = await available_tools[tool_name].ainvoke(tool_call["args"])
                messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
            else:
                messages.append(ToolMessage(content=f"错误：未找到工具 {tool_name}", tool_call_id=tool_call["id"]))


async def main():
    print("--- BugStar (记忆自省模式) 已启动 ---")
    while True:
        try:
            q = input("\n(User) > ")
            if q.lower() in ['q', 'exit', 'quit']: break
            if not q.strip(): continue
            await run_bugstar(q, str(uuid.uuid4())[:8])

        finally:
            # 显式关闭循环，减少垃圾回收时的异常
            print("\n[*] 正在安全关闭 BugStar...")


if __name__ == "__main__":
    asyncio.run(main())