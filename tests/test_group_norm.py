"""native_group_norm（forward / backward）の GPU カーネルを、float64 参照・CPU autograd と照合する。形は SD1.5 / SDXL の実形。"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch_vulkan  # noqa: E402

dev = torch.device("vk")
DTYPES = [torch.float32, torch.bfloat16, torch.float16]
TOL = {torch.float32: 5e-5, torch.bfloat16: 2.5e-2, torch.float16: 4e-3}
# (入力形状, groups)。SD1.5 の実形（G=32）と、batch 2・小さい形・groups が小さい場合
SHAPES = [
    ((1, 320, 64, 64), 32),
    ((1, 1280, 8, 8), 32),
    ((1, 2560, 16, 16), 32),
    ((2, 64, 8, 8), 4),
    ((2, 96, 5, 7), 3),
    ((2, 32, 1, 1), 32),  # グループあたり 1 要素（分散 0）
]


@pytest.fixture(autouse=True)
def _reset():
    torch_vulkan.reset_fallback_counts()
    yield


def rnd(shape, dtype=torch.float32, seed=0, scale=1.0, shift=0.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g) * scale + shift).to(dtype)


def close(out, ref64, dtype, what=""):
    o, r = out.cpu().double(), ref64.double()
    assert o.shape == r.shape, (what, o.shape, r.shape)
    scale = max(r.abs().max().item(), 1.0)
    err = (o - r).abs().max().item()
    assert err <= TOL[dtype] * scale, f"{what}: max err {err} (scale {scale})"


def no_fallback():
    fb = torch_vulkan.fallback_counts()
    bad = {n: v for n, v in fb.items() if "group_norm" in n}
    assert not bad, f"CPU フォールバックが発生: {bad}"


@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize("shape,G", SHAPES)
def test_forward_matches_float64(dt, affine, eps, shape, G):
    N, C = shape[0], shape[1]
    x = rnd(shape, dt, seed=1, scale=2.0, shift=0.5)
    w = rnd((C,), dt, seed=2, scale=0.5, shift=1.0) if affine else None
    b = rnd((C,), dt, seed=3, scale=0.5) if affine else None
    xd = x.to(dev)
    out = F.group_norm(xd, G, w.to(dev) if affine else None, b.to(dev) if affine else None, eps)
    assert out.dtype == dt
    ref = F.group_norm(x.double(), G, w.double() if affine else None, b.double() if affine else None, eps)
    close(out, ref, dt, "y")
    no_fallback()


@pytest.mark.parametrize("dt", DTYPES)
def test_mean_rstd_dtype_and_shape_match_cpu(dt):
    x = rnd((2, 64, 8, 8), dt, seed=4)
    w, b = rnd((64,), dt, seed=5), rnd((64,), dt, seed=6)
    cpu = torch.ops.aten.native_group_norm(x, w, b, 2, 64, 64, 4, 1e-5)
    gpu = torch.ops.aten.native_group_norm(x.to(dev), w.to(dev), b.to(dev), 2, 64, 64, 4, 1e-5)
    for c, g in zip(cpu, gpu):
        assert c.dtype == g.dtype and c.shape == g.shape
    close(gpu[1], cpu[1], dt, "mean")
    close(gpu[2], cpu[2].double(), dt, "rstd")
    no_fallback()


@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("shape,G", SHAPES)
def test_backward_matches_float64_autograd(dt, affine, shape, G):
    N, C = shape[0], shape[1]
    x = rnd(shape, dt, seed=1, scale=2.0, shift=0.5)
    w = rnd((C,), dt, seed=2, scale=0.5, shift=1.0)
    b = rnd((C,), dt, seed=3, scale=0.5)
    gy = rnd(shape, dt, seed=7)

    xd = x.to(dev).requires_grad_(True)
    ps = [xd]
    if affine:
        wd, bd = w.to(dev).requires_grad_(True), b.to(dev).requires_grad_(True)
        ps += [wd, bd]
    out = F.group_norm(xd, G, wd if affine else None, bd if affine else None, 1e-5)
    out.backward(gy.to(dev))

    x64 = x.double().requires_grad_(True)
    rs = [x64]
    if affine:
        w64, b64 = w.double().requires_grad_(True), b.double().requires_grad_(True)
        rs += [w64, b64]
    ref = F.group_norm(x64, G, w64 if affine else None, b64 if affine else None, 1e-5)
    ref.backward(gy.double())

    # 勾配の許容は forward より緩める（bf16 は mean / rstd が丸められて保存されるため）
    for name, p, r in zip(("dx", "dw", "db"), ps, rs):
        assert p.grad is not None and p.grad.dtype == dt
        o, rr = p.grad.cpu().double(), r.grad.double()
        scale = max(rr.abs().max().item(), 1.0)
        err = (o - rr).abs().max().item()
        mult = {torch.float32: 1.0, torch.bfloat16: 2.0, torch.float16: 2.0}[dt]
        assert err <= TOL[dt] * mult * scale * (4 if name != "dx" else 1), f"{name}: max err {err} (scale {scale})"
    no_fallback()


def test_non_contiguous_input_is_made_contiguous():
    x = rnd((2, 64, 8, 16))
    xv = x.to(dev)[:, :, :, ::2]  # 非連続
    w, b = rnd((64,), seed=2), rnd((64,), seed=3)
    out = F.group_norm(xv, 8, w.to(dev), b.to(dev), 1e-5)
    close(out, F.group_norm(x[:, :, :, ::2].double(), 8, w.double(), b.double(), 1e-5), torch.float32)
    no_fallback()


def test_weight_dtype_differs_from_input_dtype():
    x = rnd((1, 64, 8, 8), torch.bfloat16)
    w, b = rnd((64,), torch.float32, seed=2), rnd((64,), torch.float32, seed=3)
    out = F.group_norm(x.to(dev), 4, w.to(dev), b.to(dev), 1e-5)
    ref = F.group_norm(x.double(), 4, w.double(), b.double(), 1e-5)
    close(out, ref, torch.bfloat16)
    no_fallback()


def test_no_vulkan_validation_errors():
    assert torch_vulkan.validation_errors() == 0
