"""融合アテンション（F.scaled_dot_product_attention の vk 実装）を CPU の fp32 数式経路と照合する。
forward の出力と、q / k / v の勾配を比べる。bf16 / f16 は入力を丸めた値を fp32 に戻して CPU で計算した結果と比べる。
「融合カーネルが実行され、数式経路（bmm + softmax）へ落ちていない」ことも確認する。
"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch_vulkan  # noqa: E402

dev = torch.device("vk")
DTYPES = [torch.float32, torch.bfloat16, torch.float16]
REL = {torch.float32: 3e-5, torch.bfloat16: 2e-2, torch.float16: 4e-3}
OVR = "aten::_scaled_dot_product_fused_attention_overrideable"


@pytest.fixture(autouse=True)
def _fused_on():
    """既定（auto）は形とデバイスで経路を選ぶ。ここでは常に融合カーネルを使う設定で検証する。"""
    torch_vulkan.set_fused_attention("on")
    yield
    torch_vulkan.set_fused_attention("auto")

# (B, H, Sq, Sk, D)
SHAPES = [
    (1, 1, 32, 32, 64),     # ちょうど 1 タイル
    (1, 2, 40, 50, 64),     # 端数あり
    (2, 3, 33, 33, 40),     # SD1.5 型（D=40）
    (1, 4, 100, 77, 64),    # クロスアテンション型（Sk = 77）
    (1, 2, 64, 64, 128),    # D = 128（Anima）
    (1, 2, 17, 5, 20),      # D が 32 未満・Sk < 32
    (1, 2, 130, 130, 96),   # 複数ブロック・D が 32 の倍数
    (1, 2, 35, 37, 30),     # D が 4 の倍数でない（ベクトル読み込みを使わない経路）
    (1, 1, 48, 48, 36),     # D=36
]


def rnd(shape, dtype=torch.float32, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return ((torch.rand(shape, generator=g) * 2 - 1) * scale).to(dtype)


def rel_err(a, b):
    a = a.float()
    b = b.float()
    return ((a - b).norm() / (b.norm() + 1e-12)).item()


def reference(q, k, v, g):
    """CPU fp32 の数式経路。q / k / v は丸め済みの値を fp32 にしたもの。"""
    qf, kf, vf = [t.detach().float().clone().requires_grad_(True) for t in (q, k, v)]
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        o = F.scaled_dot_product_attention(qf, kf, vf)
    o.backward(g.float())
    return o.detach(), qf.grad, kf.grad, vf.grad


def run_vk(q, k, v, g):
    qv, kv, vv = [t.to(dev).requires_grad_(True) for t in (q, k, v)]
    o = F.scaled_dot_product_attention(qv, kv, vv)
    o.backward(g.to(dev))
    return o.detach().cpu(), qv.grad.cpu(), kv.grad.cpu(), vv.grad.cpu()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_fused_matches_cpu(shape, dtype):
    B, H, Sq, Sk, D = shape
    q = rnd((B, H, Sq, D), dtype, 1)
    k = rnd((B, H, Sk, D), dtype, 2)
    v = rnd((B, H, Sk, D), dtype, 3)
    g = rnd((B, H, Sq, D), dtype, 4)
    ro, rq, rk, rv = reference(q, k, v, g)
    torch_vulkan.reset_fallback_counts()
    o, dq, dk, dv = run_vk(q, k, v, g)
    assert o.dtype == dtype and dq.dtype == dtype and dk.dtype == dtype and dv.dtype == dtype
    for name, got, ref in (("out", o, ro), ("dq", dq, rq), ("dk", dk, rk), ("dv", dv, rv)):
        assert got.shape == ref.shape, name
        assert rel_err(got, ref) < REL[dtype], f"{name} {shape} {dtype}: {rel_err(got, ref)}"
    kc = torch_vulkan.kernel_counts()
    assert kc.get(OVR, 0) >= 1 and kc.get(OVR + "_backward", 0) >= 3, kc
    fb = torch_vulkan.fallback_counts()
    assert not any("softmax" in n or "bmm" in n for n in fb), fb


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_strided_inputs_bshd(dtype):
    """実際の呼び出しと同じ [B,S,H,D] → transpose(1,2) のビュー（非連続）。"""
    B, H, S, D = 2, 4, 70, 64
    qb = rnd((B, S, H, D), dtype, 5)
    kb = rnd((B, S, H, D), dtype, 6)
    vb = rnd((B, S, H, D), dtype, 7)
    gb = rnd((B, S, H, D), dtype, 8)
    q, k, v, g = [t.transpose(1, 2) for t in (qb, kb, vb, gb)]
    ro, rq, rk, rv = reference(q, k, v, g)
    qv, kv, vv = [t.to(dev).transpose(1, 2).detach().requires_grad_(True) for t in (qb, kb, vb)]
    gv = gb.to(dev).transpose(1, 2)
    o = F.scaled_dot_product_attention(qv, kv, vv)
    o.backward(gv)
    assert rel_err(o.detach().cpu(), ro) < REL[dtype]
    assert rel_err(qv.grad.cpu(), rq) < REL[dtype]
    assert rel_err(kv.grad.cpu(), rk) < REL[dtype]
    assert rel_err(vv.grad.cpu(), rv) < REL[dtype]
    # 出力は [B,S,H,D] 連続のビュー（後段の transpose(1,2).reshape がコピー不要）
    assert o.transpose(1, 2).is_contiguous()


def test_large_logits_are_fp32():
    """logits が大きい入力でも、bf16 に丸めずに softmax へ入る（数式経路より CPU fp32 に近い）。"""
    B, H, S, D = 1, 2, 256, 64
    q = rnd((B, H, S, D), torch.bfloat16, 11, 6.0)
    k = rnd((B, H, S, D), torch.bfloat16, 12, 6.0)
    v = rnd((B, H, S, D), torch.bfloat16, 13)
    ref = reference(q, k, v, torch.zeros_like(q))[0]
    got = F.scaled_dot_product_attention(q.to(dev), k.to(dev), v.to(dev)).cpu()
    assert rel_err(got, ref) < 6e-3  # bf16 の出力丸めのみ


def test_unsupported_falls_back_to_math():
    """マスク付き・D > 128 は従来の数式経路で動く（結果は合う）。"""
    B, H, S = 1, 2, 24
    q = rnd((B, H, S, 160), torch.float32, 21)
    k = rnd((B, H, S, 160), torch.float32, 22)
    v = rnd((B, H, S, 160), torch.float32, 23)
    ref = F.scaled_dot_product_attention(q, k, v)
    got = F.scaled_dot_product_attention(q.to(dev), k.to(dev), v.to(dev)).cpu()
    assert rel_err(got, ref) < 3e-5
    mask = torch.zeros(S, S)
    mask[:, S // 2:] = float("-inf")
    q2, k2, v2 = [t[..., :64].contiguous() for t in (q, k, v)]
    ref2 = F.scaled_dot_product_attention(q2, k2, v2, attn_mask=mask)
    got2 = F.scaled_dot_product_attention(q2.to(dev), k2.to(dev), v2.to(dev), attn_mask=mask.to(dev)).cpu()
    assert rel_err(got2, ref2) < 3e-5


def test_auto_policy():
    """auto: 小さい f32 は従来の数式経路（融合カーネルは使わない）。f32 には協調行列が無く、score 行列も小さいため。"""
    torch_vulkan.set_fused_attention("auto")
    q = rnd((1, 2, 64, 64), torch.float32, 41)
    torch_vulkan.reset_fallback_counts()
    got = F.scaled_dot_product_attention(q.to(dev), q.to(dev), q.to(dev)).cpu()
    assert torch_vulkan.kernel_counts().get(OVR, 0) == 0
    assert rel_err(got, F.scaled_dot_product_attention(q, q, q)) < 3e-5
    torch_vulkan.set_fused_attention("off")
    torch_vulkan.reset_fallback_counts()
    F.scaled_dot_product_attention(q.to(dev, torch.bfloat16), q.to(dev, torch.bfloat16), q.to(dev, torch.bfloat16))
    assert torch_vulkan.kernel_counts().get(OVR, 0) == 0


@pytest.mark.parametrize("dtype", DTYPES)
def test_unaligned_views(dtype):
    """q / k / v がストレージ内で中途半端な位置から始まる（ポインタがベクトル読み込みに揃わない）ときも正しい。"""
    B, H, S, D = 1, 2, 45, 64
    n = B * H * S * D
    flats = [rnd((n + 3,), dtype, 10 + i) for i in range(3)]
    gflat = rnd((n + 3,), dtype, 20)
    views = [f[3:].reshape(B, H, S, D) for f in flats]
    gv = gflat[3:].reshape(B, H, S, D)
    ro, rq, rk, rv = reference(*views, gv)
    q, k, v = [f.to(dev)[3:].view(B, H, S, D).detach().requires_grad_(True) for f in flats]
    o = F.scaled_dot_product_attention(q, k, v)
    o.backward(gv.to(dev))
    for name, got, ref in (("out", o, ro), ("dq", q.grad, rq), ("dk", k.grad, rk), ("dv", v.grad, rv)):
        assert rel_err(got.cpu(), ref) < REL[dtype], f"{name} {dtype}: {rel_err(got.cpu(), ref)}"
