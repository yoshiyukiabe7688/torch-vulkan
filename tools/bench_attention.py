"""SDPA の fwd+bwd の時間とピークメモリ。使い方: bench_attention.py [B H S D dtype]
VKT_NO_FUSED_ATTN=1 を付けると従来の数式経路（bmm + softmax）。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import torch_vulkan

B, H, S, D = (int(x) for x in sys.argv[1:5]) if len(sys.argv) >= 5 else (1, 16, 1024, 128)
dtype = {"bf16": torch.bfloat16, "f32": torch.float32}[sys.argv[5] if len(sys.argv) >= 6 else "bf16"]
dev = torch.device("vk")
g = torch.Generator().manual_seed(0)
mk = lambda: (torch.randn(B, S, H, D, generator=g) * 0.5).to(dtype).to(dev)
qb, kb, vb, gb = mk(), mk(), mk(), mk()


def step():
    q, k, v = [t.transpose(1, 2).detach().requires_grad_(True) for t in (qb, kb, vb)]
    o = F.scaled_dot_product_attention(q, k, v)
    o.backward(gb.transpose(1, 2))
    return o, q.grad


for _ in range(3):
    step()
torch_vulkan.synchronize()
torch_vulkan.reset_peak_live_bytes()
base = torch_vulkan.live_bytes() if hasattr(torch_vulkan, "live_bytes") else 0
n = 20
t0 = time.perf_counter()
for _ in range(n):
    step()
torch_vulkan.synchronize()
dt = (time.perf_counter() - t0) / n
peak = torch_vulkan.peak_live_bytes() if hasattr(torch_vulkan, "peak_live_bytes") else 0
flops = 4 * B * H * S * S * D * (1 + 2.5)  # fwd 4 N^2 D、bwd 約 2.5 倍（概算）
print(f"B={B} H={H} S={S} D={D} {dtype} fused={'no' if os.environ.get('VKT_NO_FUSED_ATTN') else 'yes'}: "
      f"{dt*1e3:.2f} ms/step (fwd+bwd), {flops/dt/1e12:.2f} TFLOPS(概算), peak {peak/2**20:.0f} MiB")
