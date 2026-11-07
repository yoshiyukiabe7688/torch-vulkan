"""実 Anima の 1 step (fwd+bwd) を指定モードで実行し、LoRA 勾配・ブロック出力・ブロック出力勾配を保存する。
使い方: python gradcmp.py <mode> <out.pt> [--up rand] [--nblocks N]
  mode: cpu32（基準, fp32）/ cpu16（CPU bf16）/ vk（協調行列あり）/ vk_tiled（VKT_NO_COOP=1 相当）
環境変数: SD_SCRIPTS_DIR（sd-scripts のパス）、ANIMA_DIT（重み）、ANIMA_CACHE_DIR（潜在とテキスト出力のキャッシュ）、PYTHONPATH に torch-vulkan。
  --sdpa32: vk のみ、アテンションのロジットと softmax を fp32 で計算（精度確認用。標準より約 70% 遅い）
"""
import argparse
import os
import sys

sys.path.insert(0, os.environ["SD_SCRIPTS_DIR"])

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("mode")
ap.add_argument("out")
ap.add_argument("--up", default="zero")  # zero: 学習開始時と同じ（lora_up=0）/ rand: lora_up を非ゼロにして lora_down の勾配も見る
ap.add_argument("--img", default="char_00")
ap.add_argument("--seed", type=int, default=123)
ap.add_argument("--sdpa32", action="store_true")  # vk のみ: アテンションのロジットと softmax を fp32 で計算する
args = ap.parse_args()

mode = args.mode
if mode == "vk_tiled":
    os.environ["VKT_NO_COOP"] = "1"
use_vk = mode.startswith("vk")
bf16 = mode != "cpu32"
dtype = torch.bfloat16 if bf16 else torch.float32

if use_vk:
    import torch_vulkan  # noqa: F401  (デバイス登録)

dev = torch.device("vk" if use_vk else "cpu")

from library import anima_utils
from networks import lora_anima

M = os.environ["ANIMA_DIT"]  # anima-base-v1.0.safetensors のパス
D = os.environ["ANIMA_CACHE_DIR"]  # <img>_0512x0512_anima.npz と <img>_anima_te.npz があるフォルダ（sd-scripts のキャッシュ出力）

model = anima_utils.load_anima_model("cpu", M, "torch", False, "cpu", dtype, False)
model.requires_grad_(False)
torch.manual_seed(0)  # lora_down の初期値をモード間で揃える
network = lora_anima.create_network(1.0, 16, 16, None, [], model)
network.apply_to([], model, False, True)
network.requires_grad_(True)
if args.up == "rand":
    g = torch.Generator().manual_seed(7)
    for n, p in network.named_parameters():
        if "lora_up" in n:
            p.data = (torch.randn(p.shape, generator=g) * 0.02)
for p_ in network.parameters():  # bf16 で表せる値に丸めてから使う（fp32 基準とビット同一の初期値にする）
    p_.data = p_.data.bfloat16().float()
network.to(dtype)
model.to(dev)
network.to(dev)
model.train()

# 決定的な入力（実データのキャッシュ + 固定の乱数）
lat = torch.from_numpy(np.load(f"{D}/{args.img}_0512x0512_anima.npz")["latents_64x64"]).unsqueeze(0)  # [1,16,64,64] fp32
te = np.load(f"{D}/{args.img}_anima_te.npz")
prompt_embeds = torch.from_numpy(te["prompt_embeds"]).unsqueeze(0)
attn_mask = torch.from_numpy(te["attn_mask"]).unsqueeze(0)
t5_ids = torch.from_numpy(te["t5_input_ids"].astype(np.int64)).unsqueeze(0)
t5_mask = torch.from_numpy(te["t5_attn_mask"].astype(np.int64)).unsqueeze(0)
gen = torch.Generator().manual_seed(args.seed)
noise = torch.randn(lat.shape, generator=gen)
t = torch.sigmoid(torch.randn(1, generator=gen))  # sigmoid サンプリング
noisy = (1 - t.view(1, 1, 1, 1)) * lat + t.view(1, 1, 1, 1) * noise
target = noise - lat
print(f"t={t.item():.4f}", flush=True)

to = lambda x, dt=None: x.to(dev) if dt is None else x.to(dev, dtype=dt)
inp = to(noisy.unsqueeze(2), dtype)
ts = to(t, dtype)
pe = to(prompt_embeds, dtype)
pad = torch.zeros(1, 1, 64, 64, dtype=dtype, device=dev)

# ブロック出力と、その勾配を記録（二分探索用）
acts, gacts = {}, {}
def mk_hook(i):
    def hook(mod, i_, o):
        acts[i] = o.detach().float().cpu()
        if o.requires_grad:
            o.register_hook(lambda g, i=i: gacts.__setitem__(i, g.detach().float().cpu()))
    return hook
for i, b in enumerate(model.blocks):
    b.register_forward_hook(mk_hook(i))


if args.sdpa32:
    import math
    import torch.nn.functional as F
    _orig = F.scaled_dot_product_attention
    def sdpa32(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
        if q.device.type != "vk" or dropout_p != 0.0 or is_causal:
            return _orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
        sc = scale if scale is not None else 1.0 / math.sqrt(q.shape[-1])
        with torch.autocast(device_type="vk", enabled=False):
            logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * sc
            if attn_mask is not None:
                logits = logits.masked_fill(~attn_mask, float("-inf")) if attn_mask.dtype == torch.bool else logits + attn_mask.float()
            p = torch.softmax(logits, dim=-1).to(q.dtype)
            return torch.matmul(p, v)
    F.scaled_dot_product_attention = sdpa32

ctx = torch.autocast(device_type=dev.type, dtype=torch.bfloat16) if bf16 else torch.autocast(device_type="cpu", enabled=False)
with ctx:
    pred = model(inp, ts, pe, padding_mask=pad, target_input_ids=to(t5_ids), target_attention_mask=to(t5_mask), source_attention_mask=to(attn_mask))
pred = pred.squeeze(2)
loss = torch.nn.functional.mse_loss(pred.float(), to(target, torch.float32))
loss.backward()
print(f"mode={mode} loss={loss.item():.8f}", flush=True)

grads = {n: p.grad.detach().float().cpu() for n, p in network.named_parameters() if p.grad is not None}
torch.save({"loss": loss.item(), "pred": pred.detach().float().cpu(), "grads": grads, "acts": acts, "gacts": gacts}, args.out)
print(f"saved {args.out}: grads={len(grads)} acts={len(acts)} gacts={len(gacts)}")
