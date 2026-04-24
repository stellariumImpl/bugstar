# BugStar

一个运行在 macOS 上的轻量工程助手 Agent。基于 LangChain + OpenAI，支持通过自然语言调用 shell 执行任务，所有命令在隔离的工作目录中运行。

**当前阶段：Day 1 —— Sandbox 抽象层已就位。**

---

## 特性

- **隔离执行**：agent 的所有 shell 命令都在独立的工作目录中运行（`~/.bugstar/workspaces/sb-<id>/`），不会污染你的系统、git 仓库或 dotfiles。
- **命令超时**：单条命令默认 60 秒超时，超时会 kill 整个进程组（包括派生的孙进程）。
- **路径逃逸防御**：`../../../etc/passwd` 这类尝试会抛 `PathEscapeError`。
- **危险命令拦截**：`rm -rf /`、`sudo`、`curl | sh` 等明显有害的模式会被拦截（非安全边界，只防愚蠢事故）。
- **可切换后端**：执行层通过 `Sandbox` Protocol 抽象，未来可以无痛切到 Docker / E2B。
- **可选长期记忆**：基于 mem0 + Qdrant 的长期记忆。开发阶段可通过环境变量关闭。
- **自我纠错**：agent 能看到命令的真实 `exit_code` 和 stderr，从而基于真实反馈修正自己的行为。

---

## 目录结构

```
bugstar/
├── main.py                      # 入口：agent 主循环 + CLI
├── pyproject.toml
├── pytest.ini
├── bugstar/
│   ├── sandbox/
│   │   ├── base.py              # Sandbox Protocol + ExecResult
│   │   ├── local.py             # LocalSandbox 实现（subprocess）
│   │   └── errors.py            # 沙盒异常类型
│   └── tools/
│       └── terminal.py          # 基于 Sandbox 的 terminal 工具
└── tests/
    └── sandbox/
        └── test_local.py        # Sandbox 契约测试（30 个）
```

---

## 快速开始

### 1. 环境准备

- macOS / Linux
- Python 3.11+
- `uv`（推荐）或 pip

### 2. 安装依赖

```bash
uv sync --group dev
```

### 3. 配置 OpenAI API Key

在项目根目录创建 `.env` 文件：

```bash
OPENAI_API_KEY=sk-...
```

### 4. 跑契约测试（验证 Sandbox 工作正常）

```bash
uv run pytest tests/sandbox/ -v
```

应看到 `30 passed`。

### 5. 启动 BugStar

最常用的开发模式（关掉 mem0 + 保留工作目录方便查看产物）：

```bash
BUGSTAR_MEMORY=0 BUGSTAR_KEEP_WORKSPACE=1 uv run python main.py
```

启动后会看到：

```
--- BugStar 已启动 (Memory=OFF (BUGSTAR_MEMORY=0), LocalSandbox) ---
[*] 工作目录: /Users/<you>/.bugstar/workspaces/sb-xxxxxxxx
[*] BUGSTAR_KEEP_WORKSPACE=1: 退出后目录保留，可随时查看产物.

(User) > 
```

---

## 使用示例

### 基本对话

```
(User) > 你好
[BugStar]: 你好！有什么我可以帮助你的吗？
```

### 让 agent 写代码并执行

```
(User) > 在当前目录新建 a.py，写一个函数返回 1+1，跑一下看结果
[*] 调用工具 [terminal]: {'command': "echo 'def add(): return 1 + 1' > a.py"}
[*] 调用工具 [terminal]: {'command': 'python3 -c "from a import add; print(add())"'}
[BugStar]: 函数执行成功，返回结果是 2。
```

退出后你可以查看产物：

```bash
ls ~/.bugstar/workspaces/sb-xxxxxxxx/
cat ~/.bugstar/workspaces/sb-xxxxxxxx/a.py
```

### 危险命令会被拦截

```
(User) > 帮我把 /etc/hosts 清空
[BugStar]: 抱歉，我无法协助修改系统目录中的文件...
```

### 超时会被强制终止

```
(User) > 跑一个 sleep 120
# 60 秒后被 kill，agent 看到 timed_out=True，会告诉你任务超时
```

### 退出

按 `Ctrl+D` / `Ctrl+C`，或输入 `q` / `exit` / `quit`。

---

## 环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `OPENAI_API_KEY` | 必须 | OpenAI API key |
| `BUGSTAR_MEMORY` | `1` | `0` 关闭 mem0 长期记忆。关闭后启动快、每轮只调 1 次 LLM。 |
| `BUGSTAR_KEEP_WORKSPACE` | `0` | `1` 退出时保留工作目录。调试期推荐开。 |

**开发阶段推荐组合**：

```bash
BUGSTAR_MEMORY=0 BUGSTAR_KEEP_WORKSPACE=1 uv run python main.py
```

**生产/演示模式**（带记忆、自动清理）：

```bash
uv run python main.py
```

---

## Sandbox 架构

BugStar 的执行层通过 `Sandbox` Protocol 抽象，上层代码不依赖具体后端：

```python
from bugstar.sandbox import LocalSandbox

async with LocalSandbox() as sb:
    r = await sb.exec("python --version")
    if r.ok:
        print(r.stdout)
    await sb.write_file("hello.py", "print('hi')")
    await sb.exec("python hello.py")
```

### 核心类型

- **`Sandbox`** (Protocol): 统一接口，包含 `exec` / `read_file` / `write_file` / `start` / `close`。
- **`ExecResult`** (frozen dataclass): `exit_code` / `stdout` / `stderr` / `duration_s` / `timed_out`。
- **`LocalSandbox`**: 基于 subprocess 的实现，工作目录隔离到 `~/.bugstar/workspaces/sb-<id>/`。

### 异常

- `CommandBlockedError`: 命令被黑名单拦截。
- `PathEscapeError`: 路径超出 workspace。
- `SandboxStartupError`: 沙盒启动失败（比如工作目录创建失败）。
- `SandboxClosedError`: 在已关闭的沙盒上做操作。

### 当前 LocalSandbox 的保护范围

✅ **已提供**：
- 工作目录隔离（固定 `~/.bugstar/workspaces/sb-<id>/`）
- 命令级超时 + 进程组 kill（避免孙进程泄漏）
- 路径逃逸防御
- 简单黑名单（8 类危险模式）

❌ **未提供**（需要升级到 DockerSandbox）：
- CPU / 内存限额
- 网络隔离
- 文件系统真隔离（LocalSandbox 下 agent 理论上仍能 cd 出去，只是常规路径操作会被拦）
- 多租户并发安全

**黑名单不是安全边界**，只拦截"agent 手滑"级别的事故。真正的隔离在容器阶段。

---

## 扩展：加一个新后端

未来加 DockerSandbox / E2BSandbox 只需要实现 `Sandbox` Protocol：

```python
from bugstar.sandbox import Sandbox, ExecResult

class DockerSandbox(Sandbox):
    async def start(self) -> None: ...
    async def close(self) -> None: ...
    async def exec(self, cmd, cwd=None, env=None, timeout_s=60.0) -> ExecResult: ...
    async def read_file(self, path: str) -> bytes: ...
    async def write_file(self, path: str, content) -> None: ...
    @property
    def workspace(self) -> str: ...
```

同一套契约测试（`tests/sandbox/test_local.py`）会自然成为新后端的验收标准 —— 通过 parametrize 可以同时在多个后端上跑。

---

## 测试

```bash
# 跑 Sandbox 契约测试
uv run pytest tests/sandbox/ -v

# 代码质量检查
uv run ruff check .
```

契约测试覆盖：
- `exec` 基本行为（stdout/stderr/exit_code/env/shell vs list）
- 工作目录隔离（pwd、相对 cwd、逃逸拦截）
- 超时 + 进程组 kill
- 文件读写 + 路径防御
- 黑名单（8 种危险命令 + 日常命令不误伤）
- 生命周期（幂等 close、关闭后调用拒绝、context manager 清理）
- 沙盒间隔离
- 同沙盒内并发

---

## 开发路线

BugStar 按阶段推进，每阶段有明确的出口标准。

- **Day 1 ✅**：Sandbox 抽象 + LocalSandbox + 契约测试（**当前阶段**）
- **Day 2**：Bench 体系 —— 5-10 个代表性任务 + `run_bench.py`，用于评估每次改动。
- **Day 3**：结构化日志（jsonl）—— 每次 LLM 调用和 tool_call 持久化，出 bug 能回放。
- **Day 4**：验证闸门 —— 实现类任务必须跑过 pytest / ruff / build 才算完成。
- **Day 5+**：DockerSandbox、E2B 混合路由、任务画像、分层镜像等（按真实数据驱动）。

详见项目维护者的架构笔记。

---

## 常见问题

### Q: 第一次启动特别慢？

如果 `BUGSTAR_MEMORY=1`（默认），mem0 会下载 spaCy 模型（~12MB）并初始化 Qdrant。**只会发生一次**，之后启动秒级。想完全避免，用 `BUGSTAR_MEMORY=0`。

### Q: agent 写的文件我看不到？

默认 session 结束时工作目录自动清理。调试期设 `BUGSTAR_KEEP_WORKSPACE=1`，退出时会打印路径，`open <path>` 直接看。

### Q: `(User) >` 提示符后面混入了 httpx 日志？

这是 Day 1 前期的现象，当前版本已压掉 httpx / httpcore / openai 的 INFO 日志。如果还能看到，检查 `logging.basicConfig` 后面是否有那三行 `setLevel(WARNING)`。

### Q: 启动后有一堆 mem0 / Qdrant 日志？

开了 `BUGSTAR_MEMORY=1` 就会有。正常。不想看开 `BUGSTAR_MEMORY=0`。

### Q: 可以跑在 Linux 上吗？

可以。Sandbox 层只用 stdlib + subprocess + posix 信号，跨平台。system prompt 里提到了 macOS，Linux 上 agent 的行为会一致。

### Q: 可以跑在 Windows 上吗？

`LocalSandbox` 用了 `os.killpg` + `start_new_session`（POSIX 专有），当前版本**不支持 Windows**。未来加 DockerSandbox 后 Windows 通过 Docker Desktop 就能跑。

---

## 设计笔记

### 为什么命令失败不抛异常？

`exit 1` 是命令的正常返回值之一。agent 经常故意跑"可能失败"的命令（`grep foo || echo not found`）。异常会让整个 agent loop 崩，agent 也看不到 stderr 没法修。**只有沙盒基础设施坏了**（目录没权限、容器起不来）才抛。

### 为什么 `exec` 同时支持字符串和列表？

字符串走 `/bin/bash -c`，支持管道、重定向、环境变量展开 —— agent 写 shell 片段最自然的方式。列表走纯 exec，避免 shell 解析，用于需要精确控制参数的场景。

### 为什么 `summary()` 要截断？

LLM 上下文是钱。`find /` 之类的命令可能输出几十 MB。截断头尾各 250 字符并标注 `[truncated ...]`，既让 LLM 知道"输出被截了"，又不爆 context。

### 为什么黑名单这么粗？

LLM 总能绕过细粒度的规则（`rm   -rf   /`、`$(echo rm) -rf /`）。黑名单不是安全边界，只是最后一道防傻事的网。**真正的安全在容器/E2B**，现在拦傻事够用。

---

## License

MIT