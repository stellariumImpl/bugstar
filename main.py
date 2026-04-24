import asyncio
import os
import shutil
import sqlite3
import sys
import uuid

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from mem0 import Memory

from sandbox import ensure_sandbox, ensure_task_workspace, reset_sandbox, sandbox_shell

load_dotenv()

# --- 固定配置 ---
USER_ID = "felix"
MEM0_LOCAL_DIR = os.path.join(os.getcwd(), ".mem0")
MEM0_HOME_DIR = os.path.expanduser("~/.mem0")
MEM0_HISTORY_DB = os.path.join(MEM0_HOME_DIR, "history.db")
FRONTEND_KEYWORDS = (
    "frontend", "front-end", "web", "html", "css", "javascript", "typescript",
    "react", "vue", "vite", "next", "nuxt", "node", "npm", "pnpm", "yarn",
    "前端", "网页", "可视化",
)


def sanitize_text(value: object) -> str:
    """将任意值清洗为可安全输出/存储的 UTF-8 文本。"""
    text = value if isinstance(value, str) else str(value)
    return text.encode("utf-8", errors="replace").decode("utf-8")


def safe_print(*values: object, sep: str = " ", end: str = "\n") -> None:
    """打印前先做 UTF-8 清洗,避免 surrogate 导致崩溃。"""
    text = sep.join(sanitize_text(v) for v in values)
    print(text, end=end)


def ensure_mem0_history_schema() -> None:
    """确保 mem0 的 history sqlite 存在且包含所需表。"""
    os.makedirs(MEM0_HOME_DIR, exist_ok=True)
    conn = sqlite3.connect(MEM0_HISTORY_DB)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id TEXT PRIMARY KEY,
                memory_id TEXT,
                old_memory TEXT,
                new_memory TEXT,
                event TEXT,
                created_at DATETIME,
                updated_at DATETIME,
                is_deleted INTEGER,
                actor_id TEXT,
                role TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                session_scope TEXT,
                role TEXT,
                content TEXT,
                name TEXT,
                created_at DATETIME
            )
        """)
        conn.commit()
    finally:
        conn.close()


def is_frontend_request(user_input: str) -> bool:
    """根据关键词判断当前任务是否为前端相关请求。"""
    normalized = user_input.lower()
    return any(keyword in normalized for keyword in FRONTEND_KEYWORDS)


def build_frontend_runtime_probe() -> str:
    """前端任务运行时自检:先探测 node/npm,缺失时自动安装后复检。"""
    probe_result = sandbox_shell.invoke({"command": "node -v && npm -v"})
    if "[exit code:" not in probe_result:
        return probe_result

    install_result = sandbox_shell.invoke({
        "command": (
            "apt-get update && "
            "apt-get install -y --no-install-recommends nodejs npm && "
            "node -v && npm -v"
        )
    })
    if "[exit code:" in install_result:
        return (
            "初次探测失败:\n"
            f"{probe_result}\n\n"
            "自动安装并复检失败:\n"
            f"{install_result}"
        )
    return (
        "初次探测失败,已自动安装并复检成功:\n"
        f"{install_result}"
    )


try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    # 某些运行环境(如被重定向的流)可能不支持 reconfigure
    pass


# --- 1. 核心组件初始化 ---

memo = Memory.from_config({
    "vector_store": {
        "provider": "qdrant",
        "config": {"path": MEM0_LOCAL_DIR},
    }
})
ensure_mem0_history_schema()


def _make_memory_tool(user_id: str):
    """用闭包固定 user_id,避免 LLM 编造身份。"""
    @tool
    def manage_memories(action: str) -> str:
        """管理长期记忆。action='list' 查看,action='reset' 清空。"""
        global memo

        if action == "list":
            memories = memo.get_all(filters={"user_id": user_id})
            if not memories:
                return "目前记忆库中没有任何记录。"
            results = memories.get("results", []) if isinstance(memories, dict) else memories
            if not results:
                return "目前记忆库中没有任何记录。"
            facts = []
            for m in results:
                content = ""
                if isinstance(m, dict):
                    content = m.get("memory") or m.get("text") or m.get("content") or str(m)
                elif isinstance(m, str):
                    content = m
                else:
                    content = getattr(m, "memory", str(m))
                facts.append(f"- {sanitize_text(content)}")
            return "\n".join(facts)

        elif action == "reset":
            try:
                memo.reset()
            except Exception as e:
                safe_print(f"[!] memo.reset() 异常(忽略继续物理清理): {e}")

            # 只清理当前项目向量库 + BugStar 使用的 history.db,避免误删 ~/.mem0 下其他项目数据。
            if os.path.exists(MEM0_LOCAL_DIR):
                try:
                    shutil.rmtree(MEM0_LOCAL_DIR)
                    safe_print(f"[*] 已删除 {MEM0_LOCAL_DIR}")
                except Exception as e:
                    safe_print(f"[!] 删除 {MEM0_LOCAL_DIR} 失败: {e}")
            if os.path.exists(MEM0_HISTORY_DB):
                try:
                    os.remove(MEM0_HISTORY_DB)
                    safe_print(f"[*] 已删除 {MEM0_HISTORY_DB}")
                except Exception as e:
                    safe_print(f"[!] 删除 {MEM0_HISTORY_DB} 失败: {e}")

            # mem0 初始化要求目录存在,重建前先 mkdir
            os.makedirs(MEM0_LOCAL_DIR, exist_ok=True)
            os.makedirs(MEM0_HOME_DIR, exist_ok=True)

            memo = Memory.from_config({
                "vector_store": {
                    "provider": "qdrant",
                    "config": {"path": MEM0_LOCAL_DIR},
                }
            })
            ensure_mem0_history_schema()
            return "记忆库已成功清空。本轮对话不会被写回。"

        return f"未知操作: {action}。支持: 'list' | 'reset'"

    return manage_memories


manage_memories = _make_memory_tool(USER_ID)


@tool
def write_file(path: str, content: str) -> str:
    """写入或覆盖文件。内容通过 base64 编码传输,避免 shell 转义问题。

    path 是容器内绝对路径(通常以 /workspace 开头)。
    默认优先写入 /workspace/workspaces/<task_id>/ 任务目录。
    """
    import base64
    import binascii
    import re

    safe_path = sanitize_text(path)
    safe_content = sanitize_text(content)
    decoded_from_nested_base64 = False

    # 防御: 有些模型会把内容先自行 base64 一次。这里做通用纠错:
    # 仅在“长 base64 文本”且解码后是高可打印率 UTF-8 文本时,才自动还原。
    # 该判定不依赖任务类型(前端/后端/ML/脚本),避免场景硬编码。
    compact = safe_content.strip().replace("\n", "").replace("\r", "")
    looks_like_base64 = (
        len(compact) >= 64
        and len(compact) % 4 == 0
        and re.fullmatch(r"[A-Za-z0-9+/=]+", compact) is not None
    )
    if looks_like_base64:
        try:
            candidate_bytes = base64.b64decode(compact, validate=True)
            # 二进制内容通常含 NUL;文本文件一般不应包含。
            if b"\x00" in candidate_bytes:
                raise ValueError("decoded content contains NUL")

            candidate = candidate_bytes.decode("utf-8")
            if candidate:
                printable_count = sum(ch.isprintable() or ch in "\n\r\t" for ch in candidate)
                printable_ratio = printable_count / len(candidate)
                has_text_separators = any(ch in candidate for ch in (" ", "\n", "\t"))

                if printable_ratio >= 0.98 and has_text_separators:
                    safe_content = candidate
                    decoded_from_nested_base64 = True
        except (binascii.Error, UnicodeDecodeError, ValueError):
            pass

    encoded = base64.b64encode(safe_content.encode("utf-8")).decode("ascii")
    cmd = (
        f"mkdir -p $(dirname {safe_path!r}) && "
        f"echo {encoded} | base64 -d > {safe_path!r}"
    )
    result = sandbox_shell.invoke({"command": cmd})
    if "[exit code:" in result:
        return f"写入失败: {result}"
    extra = "；已自动解码模型误传的 base64 内容" if decoded_from_nested_base64 else ""
    return f"已写入 {safe_path} ({len(safe_content)} 字符){extra}"


tools = [sandbox_shell, reset_sandbox, manage_memories, write_file]

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    stop=["(User) >", "User:", "\n(User)"],
).bind_tools(tools)


# --- 2. 调度层 ---
async def run_bugstar(
    user_input: str,
    session_id: str,
    user_id: str = USER_ID,
    chat_history: list[HumanMessage | AIMessage] | None = None,
) -> str:
    user_input = sanitize_text(user_input)
    task_workspace = ensure_task_workspace(session_id)
    frontend_runtime_context = ""
    if is_frontend_request(user_input):
        safe_print("[*] 前端任务检测:正在自检 node/npm 运行时...")
        frontend_probe = build_frontend_runtime_probe()
        frontend_runtime_context = sanitize_text(frontend_probe)
        preview = (
            frontend_runtime_context
            if len(frontend_runtime_context) <= 1200
            else frontend_runtime_context[:1200] + f"\n...(已截断,共 {len(frontend_runtime_context)} 字符)"
        )
        safe_print(f"[结果]:\n{preview}\n")

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
        SystemMessage(content=f"""你是 BugStar,一个运行在 macOS 上的工程助手。

【执行环境】
你通过 sandbox_shell 在 Docker 容器里执行命令,不是直接操作用户的 Mac。
- 容器是 Python 3.11 + uv + ruff + pytest
- 项目目录映射在 /workspace (这是主项目根)
- 当前任务目录: {task_workspace}
- pip install / 修改系统文件不会影响宿主机
- 宿主机可访问容器的这些端口: 3000, 5000, 5173, 8000, 8080, 8888
- 每次命令执行上限 120 秒

【工作原则】
你带着这五条原则做每一个决策。遇到本节没明说的情况,回到这五条推理。

一、动手而不空谈。
用户说"做 X",你就做 X,不给"开发步骤 1-5"的规划书。
需求模糊时问一两句关键的,问完就动手。
除非用户明确要求,不要把工作外包给用户(例如让用户本地新建项目、手动上传文件、自己安装依赖)。

二、保持工作区整洁。
/workspace 是 BugStar 主项目。任何不属于主项目的产出 ——
用户让你新建的小项目、demo、验证脚本、临时文件 —— 都要在一个独立子目录里完成,
默认放到 {task_workspace}。不污染主项目结构。动手前先想清楚"这次要产生的东西属于哪里"。

三、明确性优于便利。
shell 命令、重定向、管道有歧义时,停下来想一秒再执行。
写完代码不确定对不对,跑一下验证。别让用户替你兜底。

四、只改你该改的。
任务相关的代码动手,不相关的位置发现问题就说一声,不擅自修。
跨范围重构之前先问用户。

五、规范不是清单,是判断力。
遇到这段原则没明说的情况,回到前四条推理。推不出来就问。

【会话连续性】
- 当前会话 ID: {session_id}
- 用户输入若是"可以/继续/按你说的做/开始"这类短确认,默认承接上一轮未完成任务直接执行,不要退回兜底问句。
- 只有在目标确实不明确且无法安全假设时,才追问一个关键问题。

【技术约定】
- Python 代码:snake_case 函数变量、PascalCase 类、公开函数带类型注解和 docstring
- Python 修改后跑 `ruff check .`,与本次无关的既存告警说明一下即可
- 创建/覆盖文件用 write_file,不用 echo / cat heredoc。传给 write_file 的 content 必须是源码原文,不要再自行 base64 编码
- 启动 web 服务用 nohup 后台运行,端口从上面列表选,告诉用户访问地址
- 当前任务是前端项目时,优先直接在任务目录里创建可运行版本并验证;若缺依赖先尝试在沙盒内安装,失败后再给最小替代方案(如无构建工具的静态 HTML/CSS/JS)。
- 前端请求必须先根据【前端运行时自检】结果行动。不得跳过探测直接给“环境不支持”结论。

【已知上下文(长期记忆)】:
{memory_context}

【前端运行时自检】:
{frontend_runtime_context or "本轮未触发前端自检。"}
"""),
    ]
    if chat_history:
        messages.extend(chat_history[-12:])
    messages.append(HumanMessage(content=user_input))

    skip_memory_write = False

    final_response = ""
    while True:
        res = await llm.ainvoke(messages)
        res.content = sanitize_text(res.content)
        if "(User) >" in res.content:
            res.content = res.content.split("(User) >")[0].strip()
        messages.append(res)

        if res.tool_calls and res.content.strip():
            safe_print(f"\n[BugStar 思考]: {res.content.strip()}")

        if res.tool_calls:
            for tc in res.tool_calls:
                if tc["name"] == "manage_memories" and tc["args"].get("action") == "reset":
                    skip_memory_write = True

        if not res.tool_calls:
            if res.content.strip():
                safe_print(f"\n[BugStar]: {res.content}")
                final_response = res.content

            if not skip_memory_write and res.content.strip():
                try:
                    memo.add(
                        f"User: {user_input}\nAssistant: {res.content}",
                        user_id=user_id,
                    )
                except Exception as e:
                    safe_print(f"[!] 写入长期记忆失败: {e}")
            elif skip_memory_write:
                safe_print("[*] 本轮包含 reset 操作,跳过记忆回写。")
            break

        available_tools = {
            "sandbox_shell": sandbox_shell,
            "reset_sandbox": reset_sandbox,
            "manage_memories": manage_memories,
            "write_file": write_file,
        }

        for tool_call in res.tool_calls:
            tool_name = tool_call["name"]
            if tool_name in available_tools:
                safe_args = sanitize_text(tool_call["args"])
                safe_print(f"[*] 调用工具 [{tool_name}]: {safe_args}")
                try:
                    observation = await available_tools[tool_name].ainvoke(tool_call["args"])
                except Exception as e:
                    observation = f"工具执行异常: {e}"

                obs_str = sanitize_text(observation).rstrip()
                preview = obs_str if len(obs_str) <= 2000 else obs_str[:2000] + f"\n...(已截断,共 {len(obs_str)} 字符)"
                safe_print(f"[结果]:\n{preview}\n")

                messages.append(ToolMessage(
                    content=obs_str,
                    tool_call_id=tool_call["id"],
                ))
            else:
                messages.append(ToolMessage(
                    content=f"错误:未找到工具 {tool_name}",
                    tool_call_id=tool_call["id"],
                ))

    if chat_history is not None and final_response.strip():
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=final_response))
        if len(chat_history) > 24:
            del chat_history[:-24]

    return final_response


async def main():
    safe_print("--- BugStar (沙盒模式) 正在启动 ---")
    try:
        ensure_sandbox()
    except RuntimeError as e:
        safe_print(f"[!] {e}")
        return

    safe_print("--- BugStar 已就绪 ---")

    session_id = str(uuid.uuid4())[:8]
    chat_history: list[HumanMessage | AIMessage] = []

    while True:
        try:
            q = input("\n(User) > ")
        except (EOFError, KeyboardInterrupt):
            safe_print("\n[*] 收到退出信号。")
            break

        q = sanitize_text(q)
        if q.lower().strip() in ["q", "exit", "quit"]:
            break
        if not q.strip():
            continue

        try:
            await run_bugstar(q, session_id, chat_history=chat_history)
        except Exception as e:
            safe_print(f"[!] 本轮执行异常: {e}")

    safe_print("\n[*] 正在安全关闭 BugStar...")

    # 主动关闭 qdrant client,避免解释器 shutdown 阶段的 ImportError 噪音
    try:
        client = getattr(memo, "vector_store", None)
        client = getattr(client, "client", None)
        if client is not None:
            client.close()
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
