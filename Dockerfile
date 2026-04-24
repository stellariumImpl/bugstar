# BugStar 沙盒镜像
# 给 Agent 一个干净的 Python 3.11 环境,装了基础工具链
FROM python:3.11-slim

# 常用工具 + git(很多 pip 包依赖 git) + Node.js/npm(前端任务常用)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# 装 uv(跟你宿主一致的包管理工具)
RUN pip install --no-cache-dir uv ruff pytest

# 工作目录:宿主的项目会映射到这里
WORKDIR /workspace

# 保持容器长驻(默认进程不退出)
CMD ["sleep", "infinity"]
