"""SDPA(実形状: 1 x 16 heads x 1024 tokens x 128)を vk で fwd+bwd し、フォールバックと時間を調べる。"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn.functional as F
import torch_vulkan
dt = torch.bfloat16 if "bf16" in sys.argv else torch.float32
dev = torch.device("vk")
S = int(os.environ.get("S", 1024))
g = torch.Generator().manual_seed(0)
SK = int(os.environ.get("SK", S))   # K/V のトークン数（クロスアテンションなら 512）
q = torch.randn(1, 16, S, 128, generator=g).to(dt).to(dev).requires_grad_(True)
k, v = [torch.randn(1, 16, SK, 128, generator=g).to(dt).to(dev).requires_grad_(True) for _ in range(2)]
mask = None
if os.environ.get("MASK"):
    mask = (torch.rand(1, 1, 1, SK, generator=g) > 0.2).to(dev)  # bool マスク [1,1,1,SK]
torch_vulkan.reset_fallback_counts()
t0 = time.perf_counter()
o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
o.float().pow(2).mean().backward()
torch_vulkan.synchronize()
print(f"dtype={dt} S={S} time={time.perf_counter()-t0:.2f}s")
print("fallback:", torch_vulkan.fallback_counts())
print("kernels:", torch_vulkan.kernel_counts())
print("stats:", {k_: round(v_, 3) for k_, v_ in torch_vulkan.stats().items()})
# CPU 参照
qc, kc, vc = [t.detach().cpu().clone().requires_grad_(True) for t in (q, k, v)]
oc = F.scaled_dot_product_attention(qc, kc, vc, attn_mask=None if mask is None else mask.cpu())
oc.float().pow(2).mean().backward()
print("out err:", (o.detach().cpu().float() - oc.detach().float()).abs().max().item(), "dq err:", (q.grad.cpu().float() - qc.grad.float()).abs().max().item(), "dq scale:", qc.grad.float().abs().max().item())
