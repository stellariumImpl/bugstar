# BugStar

macOS 上的轻量工程助手 Agent。具备 Shell 执行和长期语义记忆。

LangChain + GPT-4o + [mem0](https://github.com/mem0ai/mem0)。

## 运行

```bash
uv sync
echo "OPENAI_API_KEY=sk-..." > .env
uv run python main.py
```

输入 `q` 退出。

## 工具

- `terminal` — 执行 shell 命令
- `manage_memories` — `action='list'` 查看长期记忆,`action='reset'` 清空

## 记忆

mem0 在本地维护两份存储:

- `./.mem0/` — Qdrant 向量库
- `~/.mem0/` — history sqlite(mem0 默认路径,不受 config 控制)

每轮对话结束后,调度层会把 `user_input + assistant_reply` 交给 mem0 提炼事实入库。`reset` 会物理删除以上两个目录后重建。

## 注意

`ShellTool` 启用了 `allow_dangerous_requests=True`,无命令白名单。仅供本地使用。