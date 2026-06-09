# openoperator-start-kit

OpenOperator 竞赛模板仓库 —— 为 Cambricon MLU370 加速卡编写 BANG C 算子。

竞赛官网: https://openoperator.cn

## 项目简介

本仓库是 OpenOperator 竞赛的起点模板。参赛者 Fork 此仓库后，编写 BANG C 算子内核（`.mlu` 文件），推送至 `main` 分支，远程评测服务器会自动评分并更新排行榜。

- **硬件目标**: Cambricon MLU370
- **编程语言**: BANG C (类似 CUDA 的 C 方言)
- **SDK**: Neuware SDK (CNToolkit + CNRT)
- **Python 运行时**: Cambricon 定制版 PyTorch 2.1.0 + torch_mlu

## 目录结构

```
.
├── config            # 指定需要评测的题目 ID（三位数编码）
├── Makefile          # 使用 cncc 编译 .mlu 文件
├── test_ops.py       # 本地测试脚本
├── requirements.txt  # 依赖说明文档（非 pip 安装）
├── .gitignore
├── .vscode/
│   └── settings.json # VS Code 配置：将 .mlu 识别为 C++
├── LeakyReLU.mlu     # 001 LeakyReLU 算子
├── 070_Sqrt.mlu      # 070 Sqrt 算子
└── 103_MSE_Loss.mlu  # 103 MSE Loss 算子
```

## 环境要求

安装 Cambricon Neuware SDK 后，还需安装定制版 PyTorch:

1. Neuware SDK (CNToolkit) —— 系统级安装，提供 `cncc` 编译器
2. Cambricon 定制 PyTorch wheel: `torch-2.1.0-cp310-linux_x86_64.whl`
3. Cambricon 定制 torch_mlu wheel: `torch_mlu-*.whl`

## 快速开始

### 编译

```bash
make          # 编译 config 中列出的 .mlu 文件
make all      # 编译全部 .mlu 文件
make check    # 验证编译环境（检查 cncc 和 MLU 设备）
make clean    # 清除编译产物 (*.o)
```

### 本地测试

```bash
python3 test_ops.py              # 测试 config 中的题目
python3 test_ops.py --all        # 测试所有已注册的算子
python3 test_ops.py LeakyReLU    # 按名称测试
python3 test_ops.py 001 070      # 按题目编号测试
```

测试脚本会将 BANG C 算子的输出与 PyTorch CPU 参考实现对比，报告最大绝对误差和加速比。

### 远程评测

1. Fork 本仓库并设为 **私有**
2. 将竞赛评测机器人添加为协作者
3. 配置 GitHub Webhook 指向 `http://152.136.18.42:8000/webhook`
4. 编写并推送代码到 `main` 分支
5. 约 1-3 分钟后，评测结果将以评论形式出现在对应 commit 上
6. 排行榜每 5 分钟更新一次，取每位选手的最高得分

## 编写算子

每个 `.mlu` 文件需包含:

1. **`__mlu_entry__` 内核函数** —— 运行在 MLU 核心上的 BANG C 代码
2. **`bang_func(...)` 函数** —— 供 PyTorch 调用的 C++ 入口

核心编程模式:

```cpp
#include <bang.h>
#include <torch/extension.h>
#include <cnrt.h>

// 多核任务分发：taskId 标识当前核心，taskDim 为 {4,1,1}
// NRAM 分块：CHUNK_SIZE = 4096
// 数据搬运：__memcpy 实现 GDRAM <-> NRAM 传输
// 内核启动：通过 cnrtQueue_t 流启动
```

### BANG C 常用 API

| 功能 | API |
|---|---|
| 元素级运算 | `__bang_add`, `__bang_mul`, `__bang_active_relu` 等 |
| 数据搬运 | `__memcpy` (GDRAM ↔ NRAM) |
| 规约操作 | `__bang_reduce_sum` 等 (配合 SRAM) |
| 类型转换 | `__bang_float2half`, `__bang_half2float` 等 |

## 自定义评测范围

修改 `config` 文件，每行一个三位数题目 ID:

```
001
070
103
```

若 `config` 文件不存在，远程评测服务器将评测所有题目。

## License

本项目仅供 OpenOperator 竞赛使用。
