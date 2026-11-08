"""dequant_int8 の性能（ms と実効 GB/s。最小トラフィック = int8 読み + scale 読み + 出力書き）。
使い方: python tools/bench_dequant_int8.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import torch

import torch_vulkan

dev = torch.device("vk")
for rows, cols in [(2048, 8192), (1280, 5120), (320, 2880), (640, 1280)]:
    g = torch.Generator().manual_seed(0)
    q = torch.randint(-127, 128, (rows, cols), generator=g, dtype=torch.int8).to(dev)
    scale = (torch.rand(rows, cols // 64, generator=g) * 0.02 + 1e-4).to(torch.bfloat16).to(dev)
    for dt in (torch.bfloat16, torch.float32, torch.float16):
        for _ in range(5):
            torch_vulkan.dequant_int8(q, scale, dt)
        torch_vulkan.synchronize()
        n = 200
        t0 = time.perf_counter()
        for _ in range(n):
            torch_vulkan.dequant_int8(q, scale, dt)
        torch_vulkan.synchronize()
        ms = (time.perf_counter() - t0) / n * 1e3
        nbytes = q.numel() + scale.numel() * 2 + q.numel() * torch.empty((), dtype=dt).element_size()
        print(f"[{rows},{cols}] -> {str(dt):14s} {ms:7.3f} ms  {nbytes / ms / 1e6:6.1f} GB/s  ({nbytes / 1e6:.1f} MB)")
