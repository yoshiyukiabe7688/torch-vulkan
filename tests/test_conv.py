"""conv2d（forward / backward）と nearest upsample を、CPU の fp32 と照合する。
bf16 / f16 は入力を丸めた値を fp32 に戻して CPU で計算した結果と比べる（CPU の bf16 畳み込みは環境によって backward が使えないため）。
「GPU カーネルが実行され、CPU フォールバックは起きていない」ことも確認する。
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
# ノルム相対誤差の許容（出力のスケールに対する）。K の長い和でも丸めが相殺するので bf16 でも小さめに収まる。
REL = {torch.float32: 2e-5, torch.bfloat16: 2e-2, torch.float16: 4e-3}


@pytest.fixture(autouse=True)
def _reset():
    torch_vulkan.reset_fallback_counts()
    yield


def rnd(shape, dtype=torch.float32, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return ((torch.rand(shape, generator=g) * 2 - 1) * scale).to(dtype)


def rel_close(gpu, ref, dt, what=""):
    g, c = gpu.detach().cpu().float(), ref.detach().float()
    assert g.shape == c.shape, (what, g.shape, c.shape)
    err = (g - c).norm() / max(c.norm().item(), 1e-12)
    assert err < REL[dt], f"{what}: 相対誤差 {err.item():.3g}"


def no_fallback(*names):
    fb = torch_vulkan.fallback_counts()
    bad = {n: fb[n] for n in names if n in fb}
    assert not bad, f"CPU フォールバックが発生: {bad}"


# (N, C, H, W, Co, k, stride, pad)
CASES = [
    (1, 8, 16, 16, 16, 3, 1, 1),
    (1, 8, 16, 16, 16, 3, 2, 1),
    (2, 6, 10, 12, 5, 1, 1, 0),
    (2, 3, 9, 11, 4, 3, 1, 1),
    (1, 5, 13, 7, 6, 3, 2, 0),
    (1, 64, 32, 32, 64, 3, 1, 1),
    (1, 16, 8, 8, 32, 1, 1, 0),
]


@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("bias", [True, False])
def test_conv_forward_and_input_grad(dt, case, bias):
    N, C, H, W, Co, k, s, p = case
    x = rnd((N, C, H, W), dt, 1)
    w = rnd((Co, C, k, k), dt, 2, scale=0.3)
    b = rnd((Co,), dt, 3) if bias else None
    xc = x.float().clone().requires_grad_(True)
    ref = F.conv2d(xc, w.float(), b.float() if bias else None, stride=s, padding=p)
    r = rnd(ref.shape, torch.float32, 4)
    (ref * r).sum().backward()

    xd = x.to(dev).requires_grad_(True)  # 重みは凍結（LoRA 学習と同じ: 入力の勾配だけ）
    out = F.conv2d(xd, w.to(dev), b.to(dev) if bias else None, stride=s, padding=p)
    rel_close(out, ref, dt, "forward")
    (out * r.to(dt).to(dev)).sum().backward()
    rel_close(xd.grad, xc.grad, dt, "grad_input")
    no_fallback("aten::convolution_overrideable", "aten::convolution_backward_overrideable")
    assert torch_vulkan.kernel_counts().get("aten::convolution_overrideable", 0) >= 1 or case[5] == 1


@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("case", [CASES[2], CASES[6], CASES[0], CASES[3]])
def test_conv_weight_and_bias_grad(dt, case):
    # LoRA の 1x1 conv（重みの勾配が要る）と、念のため 3x3 の重み・bias の勾配
    N, C, H, W, Co, k, s, p = case
    x = rnd((N, C, H, W), dt, 5)
    w = rnd((Co, C, k, k), dt, 6, scale=0.3)
    b = rnd((Co,), dt, 7)
    xc, wc, bc = (t.float().clone().requires_grad_(True) for t in (x, w, b))
    ref = F.conv2d(xc, wc, bc, stride=s, padding=p)
    r = rnd(ref.shape, torch.float32, 8)
    (ref * r).sum().backward()

    xd, wd, bd = x.to(dev).requires_grad_(True), w.to(dev).requires_grad_(True), b.to(dev).requires_grad_(True)
    out = F.conv2d(xd, wd, bd, stride=s, padding=p)
    (out * r.to(dt).to(dev)).sum().backward()
    rel_close(xd.grad, xc.grad, dt, "grad_input")
    rel_close(wd.grad, wc.grad, dt, "grad_weight")
    rel_close(bd.grad, bc.grad, dt, "grad_bias")
    no_fallback("aten::convolution_overrideable", "aten::convolution_backward_overrideable")


def test_conv_unsupported_cases_still_match_cpu_via_fallback():
    # groups / dilation は CPU フォールバック（結果は合う）
    x, w = rnd((1, 4, 8, 8), seed=1), rnd((4, 2, 3, 3), seed=2)
    out = F.conv2d(x.to(dev), w.to(dev), None, padding=1, groups=2)
    rel_close(out, F.conv2d(x, w, None, padding=1, groups=2), torch.float32, "groups")
    out = F.conv2d(x.to(dev), rnd((4, 4, 3, 3), seed=3).to(dev), None, padding=2, dilation=2)
    rel_close(out, F.conv2d(x, rnd((4, 4, 3, 3), seed=3), None, padding=2, dilation=2), torch.float32, "dilation")


@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("shape,scale", [((1, 4, 8, 8), 2), ((2, 3, 5, 7), (2, 3)), ((1, 2, 4, 6), 1)])
def test_nearest_upsample_forward_backward(dt, shape, scale):
    x = rnd(shape, dt, 1)
    xc = x.float().clone().requires_grad_(True)
    ref = F.interpolate(xc, scale_factor=scale, mode="nearest")
    r = rnd(ref.shape, torch.float32, 2)
    (ref * r).sum().backward()
    xd = x.to(dev).requires_grad_(True)
    out = F.interpolate(xd, scale_factor=scale, mode="nearest")
    rel_close(out, ref, dt, "upsample")
    (out * r.to(dt).to(dev)).sum().backward()
    rel_close(xd.grad, xc.grad, dt, "upsample grad")
    no_fallback("aten::upsample_nearest2d", "aten::upsample_nearest2d_backward")


def test_nearest_upsample_non_integer_factor_falls_back_correctly():
    x = rnd((1, 2, 5, 5), seed=3)
    out = F.interpolate(x.to(dev), size=(8, 8), mode="nearest")
    rel_close(out, F.interpolate(x, size=(8, 8), mode="nearest"), torch.float32, "non-integer")
