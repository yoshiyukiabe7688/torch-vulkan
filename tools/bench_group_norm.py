"""group_norm の fwd / bwd の時間（ms）と、最小トラフィック換算の実効 GB/s を測る。
使い方: python tools/bench_group_norm.py [N C H W G]
fwd の最小トラフィック = 入力読み + 出力書き（2 回分）、bwd = x と dy を読み dx を書く（3 回分）。
"""
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch_vulkan  # noqa: E402,F401

dev = torch.device("vk")
shape = tuple(int(a) for a in sys.argv[1:5]) if len(sys.argv) >= 5 else (1, 320, 64, 64)
G = int(sys.argv[5]) if len(sys.argv) >= 6 else 32
N, C = shape[0], shape[1]


def timeit(fn, iters=50, warm=5):
    for _ in range(warm):
        fn()
    torch.vk.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.vk.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


for dt in (torch.float32, torch.bfloat16):
    x = torch.randn(shape).to(dt).to(dev).requires_grad_(False)
    w = torch.randn(C).to(dt).to(dev)
    b = torch.randn(C).to(dt).to(dev)
    gy = torch.randn(shape).to(dt).to(dev)
    nbytes = x.numel() * x.element_size()
    f = lambda: torch.ops.aten.native_group_norm(x, w, b, N, C, shape[2] * shape[3], G, 1e-5)
    y, mean, rstd = f()
    g = lambda: torch.ops.aten.native_group_norm_backward(gy, x, mean, rstd, w, N, C, shape[2] * shape[3], G, [True, False, False])
    gw = lambda: torch.ops.aten.native_group_norm_backward(gy, x, mean, rstd, w, N, C, shape[2] * shape[3], G, [False, True, True])
    for name, fn, traffic in (("fwd", f, 2), ("bwd(dx)", g, 3), ("bwd(dw,db)", gw, 2)):
        ms = timeit(fn)
        print(f"{dt} {shape} G={G} {name}: {ms:.3f} ms, {traffic * nbytes / ms / 1e6:.1f} GB/s（最小トラフィック換算）")
