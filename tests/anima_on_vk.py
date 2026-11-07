"""縮小 Anima（LoRA 付き）の fwd+bwd をダミー "vk" デバイスで実行し、CPU と照合する。
フォールバックした ATen 演算の一覧が、Vulkan カーネルで実装すべき演算表（実測）になる。
使い方: python anima_on_vk.py [fp32|bf16]
"""

import os
import sys
import time

sys.path.insert(0, r"C:\Users\<user>\sd-scripts-no-cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

import torch_vulkan
from library import anima_models
from networks.lora import LoRAModule

dtype = torch.bfloat16 if "bf16" in sys.argv else torch.float32
dev = torch.device("vk")

cfg = dict(
    max_img_h=64, max_img_w=64, max_frames=8, in_channels=16, out_channels=16,
    patch_spatial=2, patch_temporal=1, concat_padding_mask=True,
    model_channels=512, num_blocks=2, num_heads=4,  # head_dim=128（本物と同じ）
    crossattn_emb_channels=1024, pos_emb_cls="rope3d", pos_emb_learnable=True,
    pos_emb_interpolation="crop", min_fps=1, max_fps=30,
    use_adaln_lora=True, adaln_lora_dim=64,
    rope_h_extrapolation_ratio=4.0, rope_w_extrapolation_ratio=4.0, rope_t_extrapolation_ratio=1.0,
    rope_enable_fps_modulation=False, use_llm_adapter=False, attn_mode="torch", split_attn=False,
)


def build():
    torch.manual_seed(0)
    model = anima_models.Anima(**cfg).to(dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    loras = []
    for bname, block in model.blocks.named_modules():
        for cname, child in list(block.named_children()):
            if isinstance(child, nn.Linear) and any(k in cname for k in ("q_proj", "k_proj", "v_proj", "output_proj", "layer1", "layer2")):
                m = LoRAModule(f"{bname}_{cname}", child, 1.0, 4, 1)
                m.apply_to()
                loras.append(m)
    params = []
    g = torch.Generator().manual_seed(1)
    for m in loras:
        m.to(dtype)
        nn.init.normal_(m.lora_up.weight, std=0.02, generator=g)  # up をゼロのままにすると down の勾配が 0 になるので非ゼロにする
        for p in m.parameters():
            p.requires_grad_(True)
            params.append(p)
    return model, loras, params


def make_inputs():
    B, T, H, W = 1, 1, 32, 32
    g = torch.Generator().manual_seed(2)
    x = torch.randn(B, 16, T, H, W, generator=g).to(dtype)
    t = torch.rand(B, generator=g).to(dtype)
    ctx = torch.randn(B, 32, 1024, generator=g).to(dtype)
    pad = torch.zeros(B, 1, H, W, dtype=dtype)
    return x, t, ctx, pad


def run(device):
    model, loras, params = build()
    model.to(device)
    for m in loras:
        m.to(device)
    x, t, ctx, pad = (a.to(device) for a in make_inputs())
    model.train()
    t0 = time.perf_counter()
    out = model(x, t, ctx, padding_mask=pad)
    loss = out.float().pow(2).mean()
    loss.backward()
    dt = time.perf_counter() - t0
    grads = [p.grad.detach().float().cpu() for p in params]
    return out.detach().float().cpu(), loss.item(), grads, dt


print(f"dtype={dtype}")
out_c, loss_c, grads_c, dt_c = run("cpu")
torch_vulkan.reset_fallback_counts()
out_v, loss_v, grads_v, dt_v = run(dev)
counts = torch_vulkan.fallback_counts()

tol = 1e-4 if dtype == torch.float32 else 3e-2
out_err = ((out_c - out_v).abs().max() / out_c.abs().max()).item()  # 相対誤差（bf16 は丸めで絶対誤差が大きくなる）
grad_err = max(((a - b).abs().max() / (a.abs().max() + 1e-12)).item() for a, b in zip(grads_c, grads_v))
print(f"CPU: loss={loss_c:.6f} {dt_c:.2f}s   vk(ダミー): loss={loss_v:.6f} {dt_v:.2f}s")
print(f"出力の最大相対誤差={out_err:.3e}  LoRA 勾配の最大相対誤差={grad_err:.3e}  (許容 {tol})")
print(f"フォールバックした演算: {len(counts)} 種, 合計 {sum(counts.values())} 回")
for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {name:45s} {n}")
ok = out_err < tol and grad_err < tol
print("RESULT:", "OK" if ok else "NG")
sys.exit(0 if ok else 1)
