"""
BugStar 沙盒模块(按项目隔离)

- 每个项目目录对应一个独立容器,容器名从路径哈希派生
- 同一项目多次启动会复用同一容器,跨项目互不干扰
- 启动时确保镜像存在,不在就构建
- 启动时拉起一个长驻容器,映射当前项目目录到 /workspace
- 提供 sandbox_shell 工具供 Agent 调用
- 提供 reset_sandbox 工具供 Agent/用户重建容器
- 退出时自动停止容器(不删除,下次直接复用)
"""
import atexit
import hashlib
import os
import subprocess
from pathlib import Path

from langchain_core.tools import tool

# --- 常量 ---
IMAGE_NAME = "bugstar-sandbox:latest"
CONTAINER_PREFIX = "bugstar-sandbox"
WORKSPACE_IN_CONTAINER = "/workspace"
TASK_WORKSPACES_ROOT = f"{WORKSPACE_IN_CONTAINER}/workspaces"
HOST_PROJECT_DIR = os.path.abspath(os.getcwd())
DOCKERFILE_DIR = HOST_PROJECT_DIR  # Dockerfile 放在项目根
DOCKERFILE_PATH = os.path.join(DOCKERFILE_DIR, "Dockerfile")
IMAGE_FINGERPRINT_FILE = os.path.join(HOST_PROJECT_DIR, ".bugstar_image_fingerprint")


def _container_name_for_project() -> str:
    """按项目绝对路径生成唯一容器名。同路径总是映射到同一容器。

    短哈希(前 10 位)发生碰撞的概率 ~1e-12,对个人使用场景足够。
    """
    digest = hashlib.sha1(HOST_PROJECT_DIR.encode("utf-8")).hexdigest()[:10]
    return f"{CONTAINER_PREFIX}-{digest}"


CONTAINER_NAME = _container_name_for_project()


def _run(cmd: list[str], check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    """执行 docker 命令,默认捕获输出。"""
    return subprocess.run(
        cmd,
        check=check,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _image_exists() -> bool:
    result = _run(["docker", "images", "-q", IMAGE_NAME], check=False)
    return bool(result.stdout.strip())


def _file_sha256(path: str) -> str:
    """计算文件 sha256。文件不存在时返回空串。"""
    file_path = Path(path)
    if not file_path.exists():
        return ""
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def _read_image_fingerprint() -> str:
    """读取上次构建时记录的 Dockerfile 指纹。"""
    file_path = Path(IMAGE_FINGERPRINT_FILE)
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8", errors="replace").strip()


def _write_image_fingerprint(fingerprint: str) -> None:
    """写入当前镜像对应的 Dockerfile 指纹。"""
    Path(IMAGE_FINGERPRINT_FILE).write_text(fingerprint, encoding="utf-8")


def _container_running() -> bool:
    result = _run(
        ["docker", "ps", "-q", "-f", f"name=^{CONTAINER_NAME}$"],
        check=False,
    )
    return bool(result.stdout.strip())


def _container_exists() -> bool:
    result = _run(
        ["docker", "ps", "-aq", "-f", f"name=^{CONTAINER_NAME}$"],
        check=False,
    )
    return bool(result.stdout.strip())


def _ensure_dir_in_container(path: str) -> None:
    """在容器内确保目录存在。容器未运行时静默跳过。"""
    if not _container_running():
        return
    subprocess.run(
        ["docker", "exec", CONTAINER_NAME, "mkdir", "-p", path],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def build_image() -> None:
    """构建沙盒镜像。"""
    print(f"[sandbox] 构建镜像 {IMAGE_NAME} (首次会慢一点)...")
    try:
        _run(
            ["docker", "build", "-t", IMAGE_NAME, DOCKERFILE_DIR],
            capture=False,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "镜像构建失败。常见原因:\n"
            "  1. 无法访问 Docker Hub(国内网络常见)\n"
            "     → Docker Desktop → Settings → Docker Engine 配置 registry-mirrors\n"
            "  2. 代理未配置\n"
            "     → Docker Desktop → Settings → Resources → Proxies\n"
            "  3. 磁盘空间不足\n"
            "     → docker system prune"
        ) from e
    print("[sandbox] 镜像构建完成。")


# 预留的端口映射:容器内 → 宿主机(一一对应)
# 覆盖主流前后端开发框架的默认端口
FORWARDED_PORTS = [
    3000,   # React / Next.js / Node 常用
    5173,   # Vite 默认
    5000,   # Flask 默认
    8000,   # Django / python -m http.server
    8080,   # 通用后端
    8888,   # Jupyter
]


def _port_in_use(port: int) -> bool:
    """粗略判断宿主机端口是否已被占用。用 docker port 检查比较靠谱,
    但这里直接用 Python socket bind 测试,简单有效。"""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", port))
        return False
    except OSError:
        return True
    finally:
        s.close()


def start_container() -> None:
    """启动本项目的长驻容器。

    三种情况:
    - 容器正在跑  → 直接复用
    - 容器存在但停着 → docker start 秒级唤醒
    - 容器不存在 → docker run 新建
    """
    if _container_running():
        print(f"[sandbox] 容器 {CONTAINER_NAME} 已在运行,复用。")
        return

    if _container_exists():
        # 停止的容器,直接 start 唤醒,保留里面装好的依赖
        print(f"[sandbox] 发现已停止的容器 {CONTAINER_NAME},唤醒中...")
        result = subprocess.run(
            ["docker", "start", CONTAINER_NAME],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            print("[sandbox] 容器已唤醒,项目环境保留。")
            return
        # 唤醒失败(比如容器状态坏了),兜底删掉重建
        print(f"[sandbox] 唤醒失败({result.stderr.strip()}),删除重建...")
        _run(["docker", "rm", "-f", CONTAINER_NAME], check=False)

    # 过滤出当前可用的端口
    available_ports = []
    skipped_ports = []
    for port in FORWARDED_PORTS:
        if _port_in_use(port):
            skipped_ports.append(port)
        else:
            available_ports.append(port)

    if skipped_ports:
        print(f"[sandbox] 以下端口被宿主机占用,不映射: {skipped_ports}")

    print(f"[sandbox] 新建容器 {CONTAINER_NAME}...")

    cmd = [
        "docker", "run",
        "-d",
        "--name", CONTAINER_NAME,
        # 项目隔离标记,list_sandboxes 用 label 筛选
        "--label", "bugstar.project=1",
        "--label", f"bugstar.host_path={HOST_PROJECT_DIR}",
        "-v", f"{HOST_PROJECT_DIR}:{WORKSPACE_IN_CONTAINER}",
        "-w", WORKSPACE_IN_CONTAINER,
    ]
    for port in available_ports:
        cmd += ["-p", f"{port}:{port}"]
    cmd.append(IMAGE_NAME)

    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"容器启动失败(exit {result.returncode}):\n{result.stderr.strip()}\n\n"
            f"常见原因:\n"
            f"  - 端口冲突: 检查宿主机是否有进程占用 {available_ports}\n"
            f"    (macOS 上 AirPlay Receiver 会占 5000,可在 系统设置→通用→隔空投送与接力 关闭)\n"
            f"  - Docker Desktop 未启动或资源不足\n"
            f"  - 镜像损坏: 可尝试 docker rmi {IMAGE_NAME} 后重启 BugStar 重建"
        )

    print(
        f"[sandbox] 容器已就绪。\n"
        f"         项目目录映射: {HOST_PROJECT_DIR} → {WORKSPACE_IN_CONTAINER}\n"
        f"         端口透传: {available_ports or '(无)'}"
    )
    _ensure_dir_in_container(TASK_WORKSPACES_ROOT)


def stop_container() -> None:
    """停止(但不删除)容器。保留容器以便下次秒级唤醒。

    如需彻底删除,使用 destroy_container()。
    """
    if _container_running():
        print(f"[sandbox] 停止容器 {CONTAINER_NAME}(保留以便复用)...")
        _run(["docker", "stop", CONTAINER_NAME], check=False)


def destroy_container() -> None:
    """彻底删除容器(清空容器内所有状态)。供 reset_sandbox 使用。"""
    if _container_exists():
        print(f"[sandbox] 销毁容器 {CONTAINER_NAME}...")
        _run(["docker", "rm", "-f", CONTAINER_NAME], check=False)


def list_sandboxes() -> list[dict]:
    """列出所有 BugStar 创建的容器,附带项目路径和状态。供用户手动管理。"""
    result = _run(
        [
            "docker", "ps", "-a",
            "--filter", "label=bugstar.project=1",
            "--format", "{{.Names}}|{{.Status}}|{{.Label \"bugstar.host_path\"}}",
        ],
        check=False,
    )
    sandboxes = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3:
            sandboxes.append({
                "name": parts[0],
                "status": parts[1],
                "host_path": parts[2],
            })
    return sandboxes


def ensure_sandbox() -> None:
    """启动时调用:确保镜像和容器都就绪。"""
    try:
        _run(["docker", "version"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(
            "Docker 不可用。请确认 Docker Desktop 已启动。"
        ) from e

    dockerfile_fingerprint = _file_sha256(DOCKERFILE_PATH)
    cached_fingerprint = _read_image_fingerprint()
    image_exists = _image_exists()

    should_rebuild = (not image_exists) or (dockerfile_fingerprint != cached_fingerprint)
    if should_rebuild:
        if not image_exists:
            print("[sandbox] 未检测到沙盒镜像,将执行首次构建。")
        elif dockerfile_fingerprint != cached_fingerprint:
            print("[sandbox] 检测到 Dockerfile 变更,将重建镜像并替换旧容器。")
        # 容器绑定镜像版本,重建前先销毁旧容器避免继续复用旧环境
        destroy_container()
        build_image()
        _write_image_fingerprint(dockerfile_fingerprint)

    start_container()
    _ensure_dir_in_container(TASK_WORKSPACES_ROOT)

    # 退出时停止(不删除),下次启动时秒级唤醒
    atexit.register(stop_container)


# --- 2. Tools ---

@tool
def sandbox_shell(command: str) -> str:
    """在沙盒容器内执行 shell 命令。

    容器里是 Python 3.11 + uv + ruff + pytest,项目目录映射到 /workspace。
    容器网络开放,可以 pip install / curl / git clone。
    命令在隔离环境中运行,不会污染宿主机。
    """
    if not _container_running():
        return "错误:沙盒容器未运行。请先调用 reset_sandbox 或重启 BugStar。"

    try:
        result = subprocess.run(
            ["docker", "exec", CONTAINER_NAME, "bash", "-c", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return "错误:命令执行超过 120 秒被终止。"

    # 拼装 stdout + stderr + exit code
    parts = []
    if result.stdout:
        parts.append(result.stdout.rstrip())
    if result.stderr:
        parts.append(f"[stderr]\n{result.stderr.rstrip()}")
    if result.returncode != 0:
        parts.append(f"[exit code: {result.returncode}]")

    return "\n".join(parts) if parts else "(命令无输出)"


@tool
def reset_sandbox() -> str:
    """重建本项目的沙盒容器,丢弃容器内所有状态(已装的包、修改过的系统文件等)。

    项目目录中的文件不受影响(因为是宿主映射)。
    当容器被装了一堆垃圾、或进入异常状态时调用。
    不会影响其他项目的容器。
    """
    try:
        destroy_container()
        start_container()
        _ensure_dir_in_container(TASK_WORKSPACES_ROOT)
        return "沙盒已重建。容器内状态已清空,/workspace 保持不变。"
    except Exception as e:
        return f"重建失败: {e}"


def task_workspace_path(task_id: str) -> str:
    """返回会话任务目录绝对路径(/workspace/workspaces/<task_id>)。"""
    safe_task_id = "".join(ch for ch in task_id if ch.isalnum() or ch in ("-", "_"))
    if not safe_task_id:
        safe_task_id = "default"
    return f"{TASK_WORKSPACES_ROOT}/{safe_task_id}"


def ensure_task_workspace(task_id: str) -> str:
    """确保指定 task_id 的任务目录存在,返回目录绝对路径。"""
    path = task_workspace_path(task_id)
    _ensure_dir_in_container(path)
    return path
