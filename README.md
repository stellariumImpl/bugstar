# BugStar

一个运行在 macOS 上的轻量工程助手 Agent，具备 **Shell 执行能力** 和 **长期语义记忆**。

基于 LangChain + GPT-4o + [mem0](https://github.com/mem0ai/mem0) 构建，采用调度循环（scheduler loop）架构，支持 Agent 在单轮对话中多次调用工具直至任务完成。

---

## ✨ 特性

- **终端工具**：通过 `ShellTool` 直接执行 shell 命令，适合做本地开发辅助、项目巡检、自动化诊断。
- **长期记忆**：使用 mem0 做事实提炼（fact extraction）+ 向量检索，对话结束后自动沉淀关键信息到本地 Qdrant。
- **记忆自省**:Agent 可以通过 `manage_memories` 工具自主查看 / 清空自己的记忆。
- **本地优先**:向量库存储在项目目录下的 `.mem0/`,不依赖云端服务(除 OpenAI API 外)。
- **幻觉防御**:通过 `stop` 参数约束模型不自行伪造用户回合。

---

## 🧱 架构

```
┌──────────────────────────────────────────────────┐
│                    User Input                    │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │  mem0.search(user_input)    │  ← 检索相关长期记忆
         └──────────────┬──────────────┘
                        │
                        ▼
         ┌─────────────────────────────┐
         │   System Prompt + History   │
         │   + Retrieved Memory        │
         └──────────────┬──────────────┘
                        │
                        ▼
                ┌──────────────┐
                │  GPT-4o LLM  │◀──────┐
                └──────┬───────┘       │
                       │               │
              ┌────────┴────────┐      │
              │  tool_calls?    │      │
              └───┬─────────┬───┘      │
                 Yes        No         │
                  │         │          │
                  ▼         ▼          │
        ┌─────────────┐  ┌───────────┐ │
        │ Shell /     │  │ 输出回复  │ │
        │ Memory Tool │  │ + mem0.add│ │
        └──────┬──────┘  └───────────┘ │
               │                       │
               └───────────────────────┘
                  (ToolMessage 回填)
```

---

## 📦 环境要求

- Python 3.11
- macOS(ShellTool 基于本机 shell)
- OpenAI API Key

---

## 🚀 安装与运行

使用 [uv](https://github.com/astral-sh/uv) 管理依赖(推荐):

```bash
# 克隆项目
git clone <your-repo-url> bugstar
cd bugstar

# 安装依赖
uv sync

# 配置环境变量
echo "OPENAI_API_KEY=sk-..." > .env

# 启动
uv run python main.py
```

启动后进入交互 REPL:

```text
--- BugStar (记忆自省模式) 已启动 ---

(User) > 当前目录下都有什么文件?
[*] 调用工具 [terminal]: {'commands': 'ls -al'}
[BugStar]: ...

(User) > 展示你关于我的所有认知
[*] 调用工具 [manage_memories]: {'action': 'list'}
[BugStar]: ...
```

输入 `q` / `exit` / `quit` 退出。

---

## 🛠 可用工具

| 工具名              | 作用                                     | 示例调用                                  |
| ------------------- | ---------------------------------------- | ----------------------------------------- |
| `terminal`          | 执行任意 shell 命令                      | `ls -al`、`git status`、`pytest`          |
| `manage_memories`   | 查看 / 清空长期记忆                      | `action='list'`、`action='reset'`         |

> ⚠️ **安全提示**:`ShellTool` 初始化时使用了 `allow_dangerous_requests=True`,没有命令白名单。请勿在生产环境或不受信的对话场景中使用。

---

## 🧠 记忆系统说明

### 存储位置

mem0 在本地使用两份存储:

| 路径               | 内容                       | 来源                        |
| ------------------ | -------------------------- | --------------------------- |
| `./.mem0/`         | Qdrant 向量库(事实嵌入)    | 由 `Memory.from_config` 指定 |
| `~/.mem0/`         | SQLite history(对话历史)   | mem0 默认路径,不受配置控制   |

### 写入时机

每轮对话在 Agent 输出最终回复(无 tool_calls)后,调度层会调用:

```python
memo.add(f"User: {user_input}\nAssistant: {res.content}", user_id=user_id)
```

mem0 会用内置 LLM 从这段对话中提炼"事实"并向量化入库。

### ⚠️ 已知问题:Reset 残影

调用 `manage_memories(action='reset')` 后,调度循环仍会在本轮结束时执行 `memo.add(...)`,把"用户要求清空记忆库 / 助手已清空"这段对话重新提炼为事实写回库中。表现为:

```text
(User) > 清空你的记忆库
[BugStar]: 我已经清空了我的记忆库。

(User) > 我们现在有哪些历史记忆
[BugStar]: 1. 用户要求助手清空记忆库...
          2. 助手确认已清空记忆库...
          （还会意外浮现清空前的旧事实——mem0_entities 集合未完全清理）
```

**绕过方式**(代码层修复见 [TODO](#-todo)):

1. 手动删除本地存储:
   ```bash
   rm -rf ./.mem0 ~/.mem0
   ```
2. 重启 BugStar,记忆库会以空状态重建。

---

## 📁 项目结构

```
bugstar/
├── .env                  # OpenAI API Key(自行创建)
├── .mem0/                # 本地 Qdrant 向量库(运行后生成)
├── .python-version       # 3.11
├── main.py               # 主入口:调度循环 + 工具定义
├── notebooks/            # 实验性 notebook
├── pyproject.toml        # 依赖声明
├── uv.lock
└── README.md
```

---

## 🔧 关键配置

### LLM

```python
ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    stop=["(User) >", "User:", "\n(User)"]  # 防止模型自问自答
)
```

### mem0

```python
Memory.from_config({
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "path": os.path.join(os.getcwd(), ".mem0"),
        }
    }
})
```

默认使用 OpenAI 作为 mem0 的事实提炼 LLM,消耗额外 token。

---

## 📝 TODO

- [ ] **修复 reset 残影**:在调度循环中标记"本轮调用过 reset",跳过对应 `memo.add`
- [ ] **彻底清理 history**:reset 时同步删除 `~/.mem0/history.db`
- [ ] **收敛 `user_id`**:把 `manage_memories` 的 `user_id` 参数从 tool 签名中移除,改用闭包固定,防止 LLM 编造用户
- [ ] **Shell 沙箱**:引入命令白名单 / 黑名单,降低 `allow_dangerous_requests` 风险
- [ ] **多会话支持**:当前 `session_id` 生成了但未用于记忆隔离,可接入 mem0 的 `run_id` 维度

---

## 📄 License

MIT(或自行选择)