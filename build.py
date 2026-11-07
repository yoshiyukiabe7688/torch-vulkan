"""vkrt（Rust）と C++ シムをビルドし、torch_vulkan/ に vkrt.dll と torch_vulkan_C*.pyd を置く。

run_build.cmd（MSVC 環境を読み込む）から、sd-scripts の venv の python で実行する。
"""

import os
import shutil
import subprocess

import torch  # noqa: F401  (cpp_extension の前提)
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(HERE, "build")
PKG_DIR = os.path.join(HERE, "torch_vulkan")
VKRT_DIR = os.path.join(HERE, "vkrt")
os.makedirs(BUILD_DIR, exist_ok=True)

# 1) Rust ランタイム（cdylib）
cargo = shutil.which("cargo") or os.path.expanduser(r"~\.cargo\bin\cargo.exe")
subprocess.run([cargo, "build", "--release"], cwd=VKRT_DIR, check=True)
REL = os.path.join(VKRT_DIR, "target", "release")
shutil.copyfile(os.path.join(REL, "vkrt.dll"), os.path.join(PKG_DIR, "vkrt.dll"))

os.add_dll_directory(REL)  # load() がビルド直後に import するので、vkrt.dll の場所を教える

# 2) C++ シム（vkrt.dll のインポートライブラリにリンク）
module = load(
    name="torch_vulkan_C",
    sources=[os.path.join(HERE, "csrc", f) for f in ("shim.cpp", "ops_ew.cpp", "ops_mm.cpp", "ops_rows.cpp")],
    build_directory=BUILD_DIR,
    extra_cflags=["/O2", "/EHsc", "/permissive-", "/utf-8"],
    extra_include_paths=[os.path.join(HERE, "csrc")],
    extra_ldflags=[f"/LIBPATH:{REL}", "vkrt.dll.lib"],
    verbose=True,
)
src = module.__file__
dst = os.path.join(PKG_DIR, os.path.basename(src))
shutil.copyfile(src, dst)
print("built:", dst)
