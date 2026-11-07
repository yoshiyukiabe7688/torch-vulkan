"""C++ シムをビルドして torch_vulkan/ に torch_vulkan_C*.pyd を置く。

MSVC の環境（VsDevCmd.bat）を読み込んだシェルから、対象の venv の python で実行する:
    build.py は run_build.cmd から呼ばれる想定。
"""

import os
import shutil
import sys

import torch
from torch.utils.cpp_extension import load

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(HERE, "build")
os.makedirs(BUILD_DIR, exist_ok=True)

module = load(
    name="torch_vulkan_C",
    sources=[os.path.join(HERE, "csrc", "shim.cpp")],
    build_directory=BUILD_DIR,
    extra_cflags=["/O2", "/EHsc", "/permissive-", "/utf-8"],
    verbose=True,
)
src = module.__file__
dst = os.path.join(HERE, "torch_vulkan", os.path.basename(src))
shutil.copyfile(src, dst)
print("built:", dst)
