"""attention 単体の精度: 同じ bf16 入力に対し、fp32 の CPU 数式経路を基準に、
  fused（vk 融合）/ old（vk の bmm+softmax。VKT_NO_FUSED_ATTN=1 を子プロセスで）/ cpu_bf16 の出力・勾配の相対誤差を比べる。
使い方: attn_acc_unit.py [S] [scale]   （scale: q,k の振幅。大きいほど softmax が尖る）"""
import os
import subprocess
import sys

import torch
import torch.nn.functional as F

S = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
AMP = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
B, H, D = 1, 16, 128
mode = os.environ.get("MODE", "parent")


def inputs(seed):
    g = torch.Generator().manual_seed(seed)
    mk = lambda a: (torch.randn(B, S, H, D, generator=g) * a).to(torch.bfloat16)
    return mk(AMP), mk(AMP), mk(1.0), mk(1.0)


def run(dev, seeds, dtype=torch.bfloat16):
    res = []
    for sd in seeds:
        qb, kb, vb, gb = inputs(sd)
        q, k, v = [t.to(dev).to(dtype).transpose(1, 2).detach().requires_grad_(True) for t in (qb, kb, vb)]
        o = F.scaled_dot_product_attention(q, k, v)
        o.backward(gb.to(dev).to(dtype).transpose(1, 2))
        res.append([t.detach().float().cpu() for t in (o, q.grad, k.grad, v.grad)])
    return res


if mode == "child":
    import torch_vulkan  # noqa
    out = run(torch.device("vk"), range(4))
    torch.save(out, os.environ["OUT"])
    sys.exit(0)

seeds = range(4)
ref = run("cpu", seeds, torch.float32)
cpubf = run("cpu", seeds, torch.bfloat16)
env = dict(os.environ, MODE="child", PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for name, extra in (("fused", {}), ("old", {"VKT_NO_FUSED_ATTN": "1"})):
    env["OUT"] = f"attn_acc_{name}.pt"
    subprocess.run([sys.executable] + sys.argv[:1] + sys.argv[1:], env=dict(env, **extra), check=True)
vk = {n: torch.load(f"attn_acc_{n}.pt") for n in ("fused", "old")}
vk["cpu_bf16"] = cpubf
err = lambda a, b: ((a - b).norm() / b.norm()).item()
print(f"S={S} amp={AMP}  相対誤差(出力 / dq / dk / dv)、4 シード平均")
for n, r in vk.items():
    cols = [sum(err(r[i][j], ref[i][j]) for i in range(4)) / 4 for j in range(4)]
    print(f"{n:9s}", "  ".join(f"{c:.4f}" for c in cols))
