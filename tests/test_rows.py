"""softmax / layer_norm / sum・mean / cat の GPU カーネルを、float64 参照・CPU autograd と照合する。"""

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch_vulkan  # noqa: E402

dev = torch.device("vk")
DTYPES = [torch.float32, torch.bfloat16, torch.float16]
TOL = {torch.float32: 3e-5, torch.bfloat16: 1.2e-2, torch.float16: 2.5e-3}


@pytest.fixture(autouse=True)
def _reset():
    torch_vulkan.reset_fallback_counts()
    yield


def rnd(shape, dtype=torch.float32, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(shape, generator=g) * scale).to(dtype)


def close(out, ref64, dtype, mult=1.0):
    o, r = out.cpu().double(), ref64.double()
    assert o.shape == r.shape, (o.shape, r.shape)
    scale = max(r.abs().max().item(), 1.0)
    err = (o - r).abs().max().item()
    assert err <= TOL[dtype] * scale * mult, f"max err {err} (scale {scale})"


def no_fallback(*names):
    fb = torch_vulkan.fallback_counts()
    assert not {n: fb[n] for n in names if n in fb}, fb


# ---- softmax -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("shape,dim", [((7, 33), -1), ((3, 5, 4), 1), ((3, 5, 4), 0), ((2, 1000), 1), ((4, 5000), -1),
                                         ((5, 1), -1), ((2, 3, 4, 5), 2)])
def test_softmax_forward(dt, shape, dim):
    x = rnd(shape, dt, scale=3.0)
    out = torch.softmax(x.to(dev), dim)
    assert out.dtype == dt
    close(out, torch.softmax(x.double(), dim), dt)
    no_fallback("aten::_softmax")


def test_softmax_with_masked_entries_and_large_values():
    x = rnd((4, 40), scale=50.0)  # 大きな値でも max 引きで安定
    x[:, 5:12] = float("-inf")
    close(torch.softmax(x.to(dev), -1), torch.softmax(x.double(), -1), torch.float32)


def test_softmax_non_contiguous_input():
    x = rnd((6, 8, 9))
    xd = x.to(dev).transpose(0, 2)
    close(torch.softmax(xd, -1), torch.softmax(x.transpose(0, 2).double(), -1), torch.float32)


@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
def test_softmax_backward_matches_cpu_autograd(dt):
    x = rnd((5, 4, 37), dt, scale=2.0)
    g = rnd((5, 4, 37), dt, seed=3)
    xc = x.double().requires_grad_(True)
    torch.softmax(xc, -1).backward(g.double())
    xd = x.to(dev).requires_grad_(True)
    torch.softmax(xd, -1).backward(g.to(dev))
    close(xd.grad, xc.grad, dt, mult=2.0)
    no_fallback("aten::_softmax_backward_data")


# ---- layer_norm ------------------------------------------------------------------------------------------------------
def ref_ln(x, shape, w=None, b=None, eps=1e-5):
    return F.layer_norm(x.double(), shape, None if w is None else w.double(), None if b is None else b.double(), eps)


@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("rows,D", [(3, 7), (5, 256), (4, 1000), (2, 2048), (1, 1)])
@pytest.mark.parametrize("affine", [False, True])
def test_layer_norm_forward(dt, rows, D, affine):
    x = rnd((rows, D), dt, scale=2.0)
    w = rnd((D,), dt, 1) if affine else None
    b = rnd((D,), dt, 2) if affine else None
    out = F.layer_norm(x.to(dev), (D,), None if w is None else w.to(dev), None if b is None else b.to(dev), 1e-5)
    close(out, ref_ln(x, (D,), w, b), dt, mult=2.0)
    no_fallback("aten::native_layer_norm", "aten::native_batch_norm")


def test_layer_norm_anima_shape_without_affine_and_multi_dim_normalization():
    # Anima の AdaLN: elementwise_affine=False、入力は 5 次元 [B,T,H,W,D]
    x = rnd((1, 1, 8, 8, 128), torch.bfloat16, scale=2.0)
    close(F.layer_norm(x.to(dev), (128,), eps=1e-6), ref_ln(x, (128,), eps=1e-6), torch.bfloat16, mult=2.0)
    y = rnd((4, 3, 5), torch.float32)
    close(F.layer_norm(y.to(dev), (3, 5)), ref_ln(y, (3, 5)), torch.float32)  # 末尾 2 次元を正規化
    no_fallback("aten::native_layer_norm")


def test_layer_norm_stats_are_float32_for_half_inputs():
    x = rnd((4, 64), torch.bfloat16).to(dev)
    out, mean, rstd = torch.native_layer_norm(x, (64,), None, None, 1e-5)
    assert mean.dtype == torch.float32 and rstd.dtype == torch.float32 and out.dtype == torch.bfloat16
    assert mean.shape == (4, 1)


@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("affine", [False, True])
def test_layer_norm_backward_matches_cpu_autograd(dt, affine):
    rows, D = 6, 96
    x = rnd((rows, D), dt, scale=2.0)
    w = rnd((D,), dt, 1) if affine else None
    b = rnd((D,), dt, 2) if affine else None
    g = rnd((rows, D), dt, 3)

    xc = x.double().requires_grad_(True)
    wc = w.double().requires_grad_(True) if affine else None
    bc = b.double().requires_grad_(True) if affine else None
    F.layer_norm(xc, (D,), wc, bc, 1e-5).backward(g.double())

    xd = x.to(dev).requires_grad_(True)
    wd = w.to(dev).requires_grad_(True) if affine else None
    bd = b.to(dev).requires_grad_(True) if affine else None
    F.layer_norm(xd, (D,), wd, bd, 1e-5).backward(g.to(dev))
    close(xd.grad, xc.grad, dt, mult=4.0)
    if affine:
        close(wd.grad, wc.grad, dt, mult=4.0)
        close(bd.grad, bc.grad, dt, mult=4.0)
    no_fallback("aten::native_layer_norm_backward")


# ---- sum / mean -------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("dim", [None, -1, 0, 1, (0, 1), (1, 2), (0, 2), (0, 1, 2), (0, 3)])
@pytest.mark.parametrize("keepdim", [False, True])
@pytest.mark.parametrize("fn", ["sum", "mean"])
def test_reduce_dims(dt, dim, keepdim, fn):
    x = rnd((3, 5, 4, 6), dt)
    kwargs = {} if dim is None else {"dim": dim, "keepdim": keepdim}
    out = getattr(torch, fn)(x.to(dev), **kwargs)
    ref = getattr(torch, fn)(x.double(), **kwargs)
    assert out.dtype == dt
    close(out, ref, dt, mult=1.5)
    no_fallback("aten::" + fn)


def test_reduce_non_contiguous_and_large():
    x = rnd((40, 50, 30))
    xd = x.to(dev)
    close(xd.transpose(0, 2).sum(1), x.transpose(0, 2).double().sum(1), torch.float32, mult=3.0)
    close(xd[5:30, ::2, 3:20].mean((0, 2)), x[5:30, ::2, 3:20].double().mean((0, 2)), torch.float32, mult=3.0)
    big = rnd((1 << 22,), seed=7)
    close(big.to(dev).sum(), big.double().sum().reshape(()), torch.float32, mult=50.0)
    no_fallback("aten::sum", "aten::mean")


def test_mean_backward_and_sum_backward():
    x = rnd((4, 7))
    xd = x.to(dev).requires_grad_(True)
    (xd.mean(-1) * torch.arange(4, dtype=torch.float32, device=dev)).sum().backward()
    xc = x.clone().requires_grad_(True)
    (xc.mean(-1) * torch.arange(4, dtype=torch.float32)).sum().backward()
    assert torch.allclose(xd.grad.cpu(), xc.grad, atol=1e-6)


# ---- cat ------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("dim", [0, 1, -1])
def test_cat(dt, dim):
    parts = [rnd((3, 4, 5), dt, seed=i) for i in range(3)]
    parts[1] = rnd((3, 4, 5), dt, seed=9)
    if dim == 1:
        parts[1] = rnd((3, 2, 5), dt, seed=9)
    elif dim in (-1, 2):
        parts[1] = rnd((3, 4, 1), dt, seed=9)
    else:
        parts[1] = rnd((2, 4, 5), dt, seed=9)
    out = torch.cat([p.to(dev) for p in parts], dim)
    assert torch.equal(out.cpu(), torch.cat(parts, dim))
    no_fallback("aten::cat")


def test_cat_legacy_empty_and_noncontiguous_and_backward():
    a, b = rnd((4, 3)), rnd((4, 3), seed=1)
    out = torch.cat([a.to(dev).t(), torch.empty(0, device=dev), b.to(dev).t()], 0)
    assert torch.equal(out.cpu(), torch.cat([a.t(), b.t()], 0))
    ad = a.to(dev).requires_grad_(True)
    bd = b.to(dev).requires_grad_(True)
    (torch.cat([ad, bd], 1) * torch.arange(6, dtype=torch.float32, device=dev)).sum().backward()
    ac, bc = a.clone().requires_grad_(True), b.clone().requires_grad_(True)
    (torch.cat([ac, bc], 1) * torch.arange(6, dtype=torch.float32)).sum().backward()
    assert torch.equal(ad.grad.cpu(), ac.grad) and torch.equal(bd.grad.cpu(), bc.grad)


def test_no_vulkan_validation_errors():
    assert torch_vulkan.validation_errors() == 0
