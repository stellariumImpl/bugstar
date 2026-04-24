# BugStar

macOS 上的轻量工程助手 Agent。具备沙盒化 Shell 执行和长期语义记忆。

LangChain + GPT-4o + [mem0](https://github.com/mem0ai/mem0) + Docker。

## 前置要求

- Python 3.11
- [uv](https://github.com/astral-sh/uv)
- Docker Desktop(需在 BugStar 启动前运行)
- OpenAI API Key

## 运行

```bash
uv sync
echo "OPENAI_API_KEY=sk-..." > .env
uv run python main.py
```

首次启动会自动构建沙盒镜像(`bugstar-sandbox:latest`),耗时 1-2 分钟。

输入 `q` 退出,容器会自动清理。

## 工具

- `sandbox_shell` — 在容器内执行 shell 命令
- `reset_sandbox` — 重建容器,丢弃容器内状态
- `write_file` — 安全写入文件(base64 编码传输,避免转义问题;对临时文件路径有启发式提醒)
- `manage_memories` — 查看/清空长期记忆

## 架构

```
┌─────────────────────────────────┐
│  宿主机                          │
│  ┌──────────┐  tool   ┌───────┐ │
│  │ BugStar  │────────→│Docker │ │
│  │ main.py  │         │容器    │ │
│  │          │←────────│       │ │
│  └──────────┘  result └───┬───┘ │
│                           │     │
│  项目目录 ←── bind mount ─┘     │
└─────────────────────────────────┘
```

- 容器 = 长驻(混合生命周期),命令之间保持状态
- 网络 = 开放(可 pip install / curl)
- 文件映射 = 只映射项目目录

## 目录约定

- `/workspace/` 就是宿主机的项目根(通过 bind mount)
- `/workspace/workspaces/<task_id>/` — Agent 的任务工作区(推荐默认写入位置)
- 任务相关 demo、验证脚本、临时文件建议写入对应 task_id 子目录,避免污染项目根

## 记忆

mem0 维护两份本地存储:

- `./.mem0/` — Qdrant 向量库
- `~/.mem0/` — history sqlite

`manage_memories action='reset'` 会重置当前项目向量库并清理 BugStar 的 history.db。触发 reset 的对话本身不会被回写。
