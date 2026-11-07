"""gradcmp.py の出力を基準(ref)と比較する。使い方: python compare.py ref.pt a.pt [b.pt ...]"""
import sys
import torch


def rel_l2(a, b):
    return ((a - b).norm() / (b.norm() + 1e-30)).item()


def sign_agree(a, b):
    m = b != 0
    return ((a[m] > 0) == (b[m] > 0)).float().mean().item()


ref = torch.load(sys.argv[1])
for path in sys.argv[2:]:
    r = torch.load(path)
    gu = [n for n in ref["grads"] if "lora_up" in n]
    gd = [n for n in ref["grads"] if "lora_down" in n]
    def agg(names):
        a = torch.cat([r["grads"][n].flatten() for n in names]); b = torch.cat([ref["grads"][n].flatten() for n in names])
        return rel_l2(a, b), sign_agree(a, b)
    print(f"== {path}: loss={r['loss']:.8f} (ref {ref['loss']:.8f}) pred相対誤差={rel_l2(r['pred'], ref['pred']):.3e}")
    e, s = agg(gu); print(f"   lora_up   grad 相対L2={e:.4f} 符号一致={s:.4f}")
    if gd and ref['grads'][gd[0]].abs().max() > 0:
        e, s = agg(gd); print(f"   lora_down grad 相対L2={e:.4f} 符号一致={s:.4f}")
    print("   block: act相対L2 / gact相対L2")
    for i in sorted(ref["acts"]):
        print(f"   {i:2d}: {rel_l2(r['acts'][i], ref['acts'][i]):.4f} / {rel_l2(r['gacts'][i], ref['gacts'][i]):.4f}")
