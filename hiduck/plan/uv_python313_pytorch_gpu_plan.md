# uv + Python 3.13 + PyTorch GPU 环境搭建思路与落地计划

## 1. 当前环境与项目现状核查

| 检查项 | 现状 |
| --- | --- |
| 项目路径 | `/home/hiyoungman/CodeProject/RidgewalkerWebServiceProject/ridgewalker-deeplearning-notebook` |
| 操作系统 | Linux WSL2 (`x86_64`) |
| GPU 硬件 | **NVIDIA GeForce RTX 4060 Laptop GPU**（8 GB 显存，Ada Lovelace 架构，`sm_89`） |
| 显卡驱动 | **595.79**，`nvidia-smi` 显示最高支持 **CUDA 13.2** |
| 系统 `nvcc` | 未安装（**无需安装**，原因见下文） |
| `uv` 版本 | `0.11.4`（已就绪） |
| Python 3.13 | 本机 `uv` 已缓存 **`cpython-3.13.12-linux-x86_64-gnu`**（无需重新下载 Python） |
| 项目环境文件 | 当前无 `pyproject.toml`、`uv.lock`、`.python-version`、`.venv`、`.gitignore` |
| 现有代码/笔记依赖 | 扫描 [check_cuda.py](../../check_cuda.py) 与全部 `.ipynb`，当前导入：`torch`、`numpy`、`pandas`、`matplotlib` |

---

## 2. 核心思路梳理

### 2.1 驱动版本与 PyTorch CUDA 版本的对应关系
- `nvidia-smi` 中显示的 `CUDA Version: 13.2` 代表**当前显卡驱动支持的最高 CUDA 运行时上限**（向下兼容更低的 CUDA 12.x / 13.0）。
- **无需在 WSL2 里手动安装系统级 CUDA Toolkit（`nvcc`）**：PyTorch 官方提供的 GPU wheel 已内置配套的 CUDA Runtime 和 cuDNN 共享库；只要驱动满足最低版本要求即可直接运行 GPU 训练/推理（仅在后续需要本地编译自定义 CUDA C++ 算子时才需要 `nvcc`）。
- **CUDA 变体选择**：
  - **`cu130`（推荐）**：PyTorch 当前主流稳定 CUDA 渠道，支持 Python 3.13，完美契合驱动上限（13.2）和 RTX 4060（`sm_89`）。
  - **`cu128`（备选）**：CUDA 12.8 渠道，若后续计划引入某些尚未适配 CUDA 13 的第三方预编译扩展库（如部分老版本 `flash-attn`、`xformers`、`bitsandbytes` 等），可选 `cu128`。

### 2.2 为什么用 `uv` 装 GPU 版 PyTorch 需要显式配置索引
- 直接执行 `uv add torch` 会从默认 PyPI 解析，容易装到 CPU 版或不匹配的 CUDA 变体。
- 按 `uv` 官方推荐方式，在 `pyproject.toml` 中组合使用 **`[[tool.uv.index]]`（设置 `explicit = true`）** 与 **`[tool.uv.sources]`**：
  1. `explicit = true` 保证该 PyTorch 索引**只服务于显式绑定的包**（如 `torch`、`torchvision`），避免 `numpy`、`pandas` 等普通依赖误去 PyTorch 源解析导致版本冲突或解析变慢。
  2. 通过环境 marker（`sys_platform == 'linux' or sys_platform == 'win32'`）将 Linux/Windows 指向 CUDA 索引，而其他平台（如 macOS）自动回退到 PyPI，保证 `uv.lock` 跨平台解析不报错。

### 2.3 项目定位与 Python 版本锁定
- 当前仓库是**学习笔记与脚本工程**（非待发布的 Python 包），在 `pyproject.toml` 中显式设置 `[tool.uv] package = false`，避免 `uv sync` 尝试构建安装当前目录本身。
- 通过 `.python-version`（`3.13`）与 `requires-python = ">=3.13,<3.14"` 锁定 Python 3.13 系列，防止后续意外漂移到 3.14 引起深度学习依赖轮子缺失。
- 仓库目前缺少 `.gitignore`，落地时建议一并补齐，防止 `.venv/`、`__pycache__/`、`.ipynb_checkpoints/` 等产物进入 Git 版本库。

---

## 3. 拟落地配置预览

> 以下以默认推荐配置（`cu130` + 现有笔记常用依赖 + `torchvision` + `ipykernel`）为例；若希望调整 CUDA 变体或精简依赖，确认时说明即可。

### `pyproject.toml`（新增）
```toml
[project]
name = "ridgewalker-deeplearning-notebook"
version = "0.1.0"
description = "Deep learning study notebooks with PyTorch GPU environment"
requires-python = ">=3.13,<3.14"
dependencies = [
    "torch",
    "torchvision",
    "numpy",
    "pandas",
    "matplotlib",
    "ipykernel",
]

[tool.uv]
package = false

[tool.uv.sources]
torch = [
    { index = "pytorch-cu130", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
    { index = "pytorch-cu130", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true
```

### `.python-version`（新增）
```text
3.13
```

### `.gitignore`（新增）
忽略 `.venv/`、`__pycache__/`、`*.py[cod]`、`.ipynb_checkpoints/`、`.uv-cache/` 等常见运行与缓存目录（保留 `pyproject.toml`、`.python-version`、`uv.lock` 供版本控制复现）。

---

## 4. 落地执行步骤（确认后执行）

1. **创建项目配置文件**：在项目根目录创建 `pyproject.toml`、`.python-version` 与 `.gitignore`。
2. **校验 PyTorch 索引与锁依赖**：使用 `uv lock` 解析 Python 3.13 + 目标 CUDA 渠道的精确依赖版本，确认解析出的 `torch` 携带对应 `+cu130`（或 `+cu128`）构建标签。
3. **创建虚拟环境并同步依赖**：执行 `uv sync` 创建 `.venv`（基于本机 `cpython-3.13.12`）并下载安装 GPU 版 PyTorch 及其他依赖。
4. **端到端 GPU 与环境验证**：
   - 运行 `uv run python --version` 确认 Python 版本为 `3.13.x` 且指向 `.venv`。
   - 运行 `uv run python check_cuda.py`，验证：
     - `torch.__version__` 含预期 CUDA 后缀；
     - `torch.cuda.is_available()` 为 `True`；
     - 识别到 `RTX 4060 Laptop GPU`；
     - cuDNN 可用且 GPU 张量运算 + `Conv2d` 卷积测试通过。
   - 批量验证 `numpy`、`pandas`、`matplotlib`、`ipykernel`（及 `torchvision`）均可正常导入。
5. **汇报结果与日常使用指引**：输出验证结果及在终端 / Jupyter Notebook 中选择该 `.venv` 内核的方法。

---

## 5. 待确认项

1. **CUDA 渠道**：默认用 **`cu130`**（推荐），还是改为 **`cu128`**？
2. **依赖清单**：默认一次配齐 `torch` + `torchvision` + 现有笔记用到的 `numpy`、`pandas`、`matplotlib`、`ipykernel`，还是先只装最精简的 `torch`？
3. **`.gitignore`**：是否一并新建 `.gitignore` 以忽略 `.venv/` 等目录？（默认：**是**）
4. **镜像源偏好**：是否使用默认 PyPI 与 PyTorch 官方源，或需要配置指定国内镜像源？
