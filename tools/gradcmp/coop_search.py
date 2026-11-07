"""協調行列を使う mm の範囲を二分探索し、勾配の爆発を起こす呼び出しを特定する。
使い方: python coop_search.py <cpu32 の出力.pt> --img X --seed N --up zero|rand（環境変数は gradcmp.py と同じ）
"""
"""実 Anima の 1 step (fwd+bwd) を指定モードで実行し、LoRA 勾配・ブロック出力・ブロック出力勾配を保存する。
使い方: python gradcmp.py <mode> <out.pt> [--up rand] [--nblocks N]
  mode: cpu32（基準, fp32）/ cpu16（CPU bf16）/ vk（協調行列あり）/ vk_tiled（VKT_NO_COOP=1 相当）
環境: SD_SCRIPTS_DIR（sd-scripts のパス）、PYTHONPATH に torch-vulkan。
"""
import argparse
import os
import sys

sys.path.insert(0, os.environ["SD_SCRIPTS_DIR"])

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("ref")
ap.add_argument("--up", default="zero")  # zero: 学習開始時と同じ（lora_up=0）/ rand: lora_up を非ゼロにして lora_down の勾配も見る
ap.add_argument("--img", default="char_00")
ap.add_argument("--seed", type=int, default=123)
args = ap.parse_args()

mode = "vk"
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


ref = torch.load(args.ref)
ref_g0 = ref["gacts"][0].norm().item()
ctx = torch.autocast(device_type="vk", dtype=torch.bfloat16)

def step(lo, hi, kind):
    for p in network.parameters():
        p.grad = None
    acts.clear(); gacts.clear()
    torch_vulkan.reset_coop_calls()
    torch_vulkan.set_coop_range(lo, hi, kind)
    with ctx:
        pred = model(inp, ts, pe, padding_mask=pad, target_input_ids=to(t5_ids), target_attention_mask=to(t5_mask), source_attention_mask=to(attn_mask))
    loss = torch.nn.functional.mse_loss(pred.squeeze(2).float(), to(target, torch.float32))
    loss.backward()
    n = torch_vulkan.coop_calls()
    r = gacts[0].norm().item() / ref_g0   # ブロック 0 の出力勾配のノルム比（基準 = 1）
    e = (gacts[0] - ref["gacts"][0]).norm().item() / ref_g0
    return n, r, e

BIG = 2**63
n, r, e = step(0, BIG, 0); print(f"all coop      : calls={n} norm比={r:.2f} 相対誤差={e:.2f}", flush=True)
_, r, e = step(0, 0, 0);   print(f"none (tiled)  : norm比={r:.2f} 相対誤差={e:.2f}", flush=True)
for kind, name in ((1, "bmm のみ"), (2, "mm のみ")):
    _, r, e = step(0, BIG, kind); print(f"coop {name:8s}: norm比={r:.2f} 相対誤差={e:.2f}", flush=True)
THR = 10.0
# 先頭からの接頭辞 [0, hi) で爆発し始める最小の hi
lo_, hi_ = 0, n
while hi_ - lo_ > 1:
    mid = (lo_ + hi_) // 2
    _, r, e = step(0, mid, 0)
    print(f"  prefix [0,{mid}) norm比={r:.2f}", flush=True)
    if r > THR: hi_ = mid
    else: lo_ = mid
print(f"接頭辞で爆発する最小の hi = {hi_}（呼び出し #{hi_-1} までを協調行列にすると爆発）", flush=True)
