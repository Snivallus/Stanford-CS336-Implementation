# CS336: Language Modeling from Scratch

本仓库包含 [Stanford CS336 2025 Spring](https://stanford-cs336.github.io/spring2025/) 所有作业的实现.

## 环境配置

- **编程语言:** Python 3.11+
- **深度学习:** PyTorch
- **依赖管理:** [uv](https://github.com/astral-sh/uv)
- **测试:** pytest

```bash
source download_datasets.sh     # 下载数据集
pip install uv                  # 通过 pip 安装 uv
cd assignment1-basics           # 进入作业目录 (以 Assignment 1 为例)
uv run pytest                   # 运行测试
uv run python scripts/train.py  # 运行训练脚本
```

依赖项会根据每个作业的 `pyproject.toml` 自动安装.