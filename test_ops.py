#!/usr/bin/env python3
"""
Cambricon MLU370 BANG C 算子本地测试脚本

用法:
    python3 test_ops.py              # 测试 config 中列出的所有题目
    python3 test_ops.py --all        # 测试所有 .mlu 文件
    python3 test_ops.py LeakyReLU    # 测试指定算子

依赖: torch, torch_mlu (寒武纪定制版)
"""

import os
import re
import sys
import time
import sysconfig
import shutil
import argparse
import pathlib
import subprocess

import torch

try:
    import torch_mlu  # noqa: F401
except ImportError:
    print("ERROR: torch_mlu 未安装。请先安装寒武纪版 PyTorch 和 torch_mlu。")
    sys.exit(1)

if torch.mlu.device_count() == 0:
    print("ERROR: 未检测到 MLU 设备。请在 MLU370 服务器上运行此脚本。")
    sys.exit(1)

print(f"MLU device: {torch.mlu.get_device_name(0)}")


# ============================================================
# 算子注册表: name -> (mlu_file, arg_names, ref_fn, shape, extra_kwargs)
# ============================================================
OPS_META = {
    "LeakyReLU": {
        "file": "LeakyReLU.mlu",
        "args": ["input", "negative_slope"],
        "ref": lambda x, ns=0.01: torch.nn.functional.leaky_relu(x, ns),
        "shape": (1024, 256),
        "extra": {"negative_slope": 0.01},
    },
    "Sqrt": {
        "file": "Sqrt.mlu",
        "args": ["x"],
        "ref": lambda x: torch.sqrt(torch.abs(x)),
        "shape": (1024, 256),
        "extra": {},
    },
    "MSE_Loss": {
        "file": "103_MSE_Loss.mlu",
        "args": ["predictions", "targets"],
        "ref": lambda pred, targ: torch.nn.functional.mse_loss(pred, targ),
        "shape": (1024, 256),
        "extra": {},
    },
    "Scatter_add": {
        "file": "105_Scatter_add.mlu",
        "args": ["src", "index", "dim_size"],
        "ref": lambda src, idx, ds: torch.zeros(ds, src.size(1))
        .index_add_(0, idx.to(torch.int32) % ds, src),
        "shape": [(1024, 256), (1024,)],
        "extra": {"dim_size": 512},
    },
    "PointwiseConv2d": {
        "file": "104_PointwiseConv2d.mlu",
        "args": ["x", "weight"],
        "ref": lambda x, w, bias=None: torch.nn.functional.conv2d(x, w, bias),
        "shape": [(2, 64, 32, 32), (128, 64, 1, 1)],
        "extra": {},
    },
}

# config 中三位编号 -> 算子名的映射
NUM_TO_NAME = {
    "001": "LeakyReLU",
    "070": "Sqrt",
    "103": "MSE_Loss",
    "104": "PointwiseConv2d",
    "105": "Scatter_add",
}


def _detect_cncc():
    """探测 cncc 编译器路径"""
    neuware_home = os.environ.get("NEUWARE_HOME", "/usr/local/neuware")
    cncc = os.path.join(neuware_home, "bin", "cncc")
    if os.path.isfile(cncc) and os.access(cncc, os.X_OK):
        return cncc
    cncc = shutil.which("cncc")
    if cncc:
        return cncc
    raise RuntimeError(
        "未找到 cncc 编译器。请设置环境变量 NEUWARE_HOME 指向 Neuware SDK 安装目录。"
    )


def _extract_bang_func_params(mlu_path):
    """从 .mlu 文件中提取 bang_func 的参数声明列表

    返回: list[str] 形如 ["torch::Tensor input", "double negative_slope"]
    """
    content = mlu_path.read_text()
    m = re.search(r"bang_func\s*\(([^)]*)\)", content)
    if not m:
        raise RuntimeError(f"未在 {mlu_path} 中找到 bang_func 定义")
    params_str = m.group(1)
    if not params_str.strip():
        return []
    params = []
    for part in params_str.split(","):
        part = part.strip()
        if part:
            params.append(part)
    return params


def compile_and_load(mlu_path):
    """编译 .mlu 文件并加载为 Python 模块

    分三步:
    1. cncc 将 .mlu 编译为 .o (BANG C 内核)
    2. 生成包装器 .cpp 并利用 torch cpp_extension 编译、链接为 .so
    3. 加载 .so 并返回模块
    """
    from torch.utils.cpp_extension import include_paths, load

    mlu_path = pathlib.Path(mlu_path).resolve()
    stem = mlu_path.stem
    obj_path = mlu_path.with_suffix(".o")

    cncc = _detect_cncc()

    # ---------- Step 1: cncc 编译 .mlu -> .o ----------
    torch_includes = include_paths()
    cncc_cmd = [
        cncc,
        str(mlu_path),
        "-o",
        str(obj_path),
        "--bang-mlu-arch=mtp_372",
        "-c",
        "-O3",
        "-std=c++17",
        "-fPIC",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ]
    for inc in torch_includes:
        cncc_cmd += ["-I", inc]

    python_include = sysconfig.get_paths().get("include", "")
    if python_include:
        cncc_cmd += ["-I", python_include]

    mlu_include = os.path.join(os.path.dirname(torch_mlu.__file__), "include")
    cncc_cmd += ["-I", mlu_include]

    print(f"  cncc: {' '.join(cncc_cmd)}")
    result = subprocess.run(cncc_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"cncc 编译失败:\n{result.stderr}\n{result.stdout}")

    # ---------- Step 2: 生成包装器 + torch cpp_extension 链接 ----------
    params = _extract_bang_func_params(mlu_path)
    param_str = ", ".join(params) if params else ""

    py_args = []
    for p in params:
        parts = p.rsplit(None, 1)
        ptype, pname = parts if len(parts) == 2 else (p, "")
        if ptype == "c10::optional<torch::Tensor>":
            py_args.append(f'py::arg("{pname}") = py::none()')
        else:
            py_args.append(f'py::arg("{pname}")')
    py_args_str = ", ".join(py_args)

    wrapper_code = f"""\
#include <torch/extension.h>
namespace py = pybind11;

// bang_func 在 .o 中定义，此处仅做声明供 pybind11 绑定
torch::Tensor bang_func({param_str});

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("bang_func", &bang_func, {py_args_str});
}}
"""
    wrapper_path = mlu_path.parent / f"{stem}_wrapper.cpp"
    wrapper_path.write_text(wrapper_code)

    try:
        neuware_home = os.environ.get("NEUWARE_HOME", "/usr/local/neuware")
        neuware_lib = os.path.join(neuware_home, "lib64")
        if not os.path.isdir(neuware_lib):
            neuware_lib = os.path.join(neuware_home, "lib")

        module = load(
            name=f"bang_{stem}",
            sources=[str(wrapper_path)],
            extra_ldflags=[
                str(obj_path),
                f"-L{neuware_lib}",
                f"-Wl,-rpath,{neuware_lib}",
                "-lcnrt",
            ],
            verbose=False,
        )
    finally:
        wrapper_path.unlink(missing_ok=True)

    if not hasattr(module, "bang_func"):
        raise RuntimeError("编译成功但模块中未找到 bang_func")
    return module


def test_operator(name, meta, device="mlu"):
    """测试单个算子的正确性和性能"""
    print(f"\n{'='*60}")
    print(f"  测试: {name}")
    print(f"  文件: {meta['file']}")
    print(f"{'='*60}")

    shape = meta["shape"]
    extra = meta.get("extra", {})
    ref_fn = meta["ref"]
    args = meta["args"]

    mlu_path = pathlib.Path(meta["file"])
    if not mlu_path.exists():
        print(f"  SKIP: {mlu_path} 不存在")
        return False

    print(f"  编译加载 {mlu_path} ...")
    try:
        module = compile_and_load(mlu_path)
    except Exception as e:
        print(f"  FAIL: 编译失败 - {e}")
        return False

    # 生成测试数据
    torch.manual_seed(42)
    if isinstance(shape, list):
        inputs_cpu = [torch.randn(*s) for s in shape]
    else:
        inputs_cpu = [torch.randn(*shape) for _ in range(len(args) - len(extra))]
    inputs_mlu = [t.to(device) for t in inputs_cpu]

    # 运行 MLU kernel（预热 + 计时）
    bang_func = module.bang_func
    extra_vals = list(extra.values())
    with torch.no_grad():
        for _ in range(3):
            bang_func(*inputs_mlu, *extra_vals)
        torch.mlu.synchronize()

        N_ITER = 100
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            result_mlu = bang_func(*inputs_mlu, *extra_vals)
        torch.mlu.synchronize()
        mlu_time_ms = (time.perf_counter() - t0) / N_ITER * 1000

    # 运行 PyTorch CPU 参考
    result_mlu_cpu = result_mlu.cpu()
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            result_ref = ref_fn(*inputs_cpu, *extra_vals)
        torch_time_ms = (time.perf_counter() - t0) / N_ITER * 1000

    # 精度对比
    if isinstance(result_ref, torch.Tensor) and result_ref.numel() > 0:
        diff = (result_mlu_cpu.float() - result_ref.float()).abs().max().item()
        atol = 1e-2
        ok = diff <= atol
        status = "PASS" if ok else "FAIL (精度超标)"
        print(f"  精度: max_diff={diff:.6f}  (atol={atol})  [{status}]")
    else:
        ok = True
        print(f"  精度: 参考输出为空，跳过对比")

    # 性能对比
    if torch_time_ms > 0:
        speedup = torch_time_ms / mlu_time_ms if mlu_time_ms > 0 else float("inf")
        print(f"  性能: MLU={mlu_time_ms:.4f}ms  CPU={torch_time_ms:.4f}ms  "
              f"speedup={speedup:.2f}x")

    return ok


def get_targets_from_config():
    """从 config 文件读取要测试的题目（三位编号）"""
    config_path = pathlib.Path("config")
    if not config_path.exists():
        return None
    targets = []
    for line in config_path.read_text().strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("#"):
            targets.append(line)
    return targets


def resolve_ops(targets):
    """将题目标识（名称或三位编号）解析为 (op_name, meta) 列表"""
    selected = []
    for t in targets:
        matched = False
        # 1) 按 config 编号映射
        mapped = NUM_TO_NAME.get(t)
        if mapped and mapped in OPS_META:
            selected.append((mapped, OPS_META[mapped]))
            matched = True
        # 2) 直接按名称匹配
        for op_name, meta in OPS_META.items():
            if t == op_name:
                if not matched:
                    selected.append((op_name, meta))
                matched = True
                break
            file_stem = pathlib.Path(meta["file"]).stem
            if t == file_stem or file_stem.startswith(t):
                if not matched:
                    selected.append((op_name, meta))
                matched = True
                break
        if not matched:
            print(f"WARNING: 未找到算子 '{t}'")
    return selected


def main():
    parser = argparse.ArgumentParser(description="MLU370 BANG C 算子测试")
    parser.add_argument("ops", nargs="*", help="要测试的算子名称或 config 编号")
    parser.add_argument("--all", action="store_true", help="测试所有算子")
    args = parser.parse_args()

    if args.ops:
        targets = args.ops
    elif args.all:
        targets = list(OPS_META.keys())
    else:
        config_targets = get_targets_from_config()
        if config_targets is None or len(config_targets) == 0:
            targets = list(OPS_META.keys())
        else:
            targets = config_targets

    selected = resolve_ops(targets)

    if not selected:
        print("没有要测试的算子。")
        sys.exit(1)

    print(f"将测试 {len(selected)} 个算子: {[s[0] for s in selected]}")

    passed = 0
    failed = 0
    for name, meta in selected:
        try:
            ok = test_operator(name, meta)
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"  结果: {passed} passed, {failed} failed, {len(selected)} total")
    print(f"{'='*60}")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
