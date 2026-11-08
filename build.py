"""vkrt（Rust）と C++ シムをビルドし、torch_vulkan/ に共有ライブラリと torch_vulkan_C 拡張を置く。

Windows: run_build.cmd（MSVC 環境を読み込む）から、PyTorch の入った python で実行する。出力は vkrt.dll と *.pyd。
Linux  : PyTorch の入った python で `python build.py`（g++ と cargo が必要）。出力は libvkrt.so と *.so。
"""

import os
import shutil
import subprocess
import sys

import torch  # noqa: F401  (cpp_extension の前提)
from torch.utils.cpp_extension import load

IS_WIN = sys.platform == "win32"
HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(HERE, "build")
PKG_DIR = os.path.join(HERE, "torch_vulkan")
VKRT_DIR = os.path.join(HERE, "vkrt")
os.makedirs(BUILD_DIR, exist_ok=True)

# 1) Rust ランタイム（cdylib）
cargo = shutil.which("cargo") or os.path.expanduser(os.path.join("~", ".cargo", "bin", "cargo.exe" if IS_WIN else "cargo"))
subprocess.run([cargo, "build", "--release"], cwd=VKRT_DIR, check=True)
REL = os.path.join(VKRT_DIR, "target", "release")
runtime_lib = "vkrt.dll" if IS_WIN else "libvkrt.so"
shutil.copyfile(os.path.join(REL, runtime_lib), os.path.join(PKG_DIR, runtime_lib))

# 2) C++ シム（Rust ランタイムにリンク）
if IS_WIN:
    os.add_dll_directory(REL)  # load() がビルド直後に import するので、vkrt.dll の場所を教える
    cflags = ["/O2", "/EHsc", "/permissive-", "/utf-8"]
    ldflags = [f"/LIBPATH:{REL}", "vkrt.dll.lib"]
else:
    cflags = ["-O2"]
    # 実行時は拡張と同じフォルダ（$ORIGIN。ninja では $$ で書き、シェルに渡す分は引用する）、ビルド直後の import 用に REL も指す
    ldflags = [f"-L{REL}", "-lvkrt", f"-Wl,-rpath,{REL}", "'-Wl,-rpath,$$ORIGIN'"]
module = load(
    name="torch_vulkan_C",
    sources=[os.path.join(HERE, "csrc", f) for f in ("shim.cpp", "ops_ew.cpp", "ops_mm.cpp", "ops_rows.cpp", "ops_conv.cpp", "ops_norm.cpp", "ops_quant.cpp")],
    build_directory=BUILD_DIR,
    extra_cflags=cflags,
    extra_include_paths=[os.path.join(HERE, "csrc")],
    extra_ldflags=ldflags,
    verbose=True,
)
src = module.__file__
dst = os.path.join(PKG_DIR, os.path.basename(src))
shutil.copyfile(src, dst)
print("built:", dst)
