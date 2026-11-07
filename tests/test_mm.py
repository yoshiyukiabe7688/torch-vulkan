"""行列積カーネル (mm / bmm / addmm / baddbmm) を、float64 の参照と照合する。
入力は同 dtype に丸めた値を使い、参照は丸め後の入力から float64 で計算する（accumulate 順の違いを除いて比較するため）。
"""

import os
import sys
import time

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch_vulkan  # noqa: E402

dev = torch.device("vk")
DTYPES = [torch.float32, torch.bfloat16, torch.float16]


@pytest.fixture(autouse=True)
def _reset():
    torch_vulkan.reset_fallback_counts()
    yield


def rnd(shape, dtype, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g).to(dtype)


def check(out, ref64, dtype, k):
    """out(vk) と float64 参照の比較。許容誤差は dtype と K(累積長)に依存させる。"""
    o = out.cpu().double()
    assert o.shape == ref64.shape
    scale = ref64.abs().max().item() + 1e-9
    tol = {torch.float32: 2e-5, torch.bfloat16: 1.2e-2, torch.float16: 2e-3}[dtype]
    err = (o - ref64).abs().max().item()
    assert err <= tol * max(scale, 1.0) * (1 + k ** 0.5 * 0.01), f"max err {err} scale {scale}"


def no_fallback(*names):
    fb = torch_vulkan.fallback_counts()
    assert not {n: fb[n] for n in names if n in fb}, fb


SHAPES = [(1, 1, 1), (7, 5, 3), (64, 16, 64), (65, 17, 63), (128, 129, 300), (33, 1000, 40), (5, 2048, 7)]


@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("m,k,n", SHAPES)
def test_mm_shapes(dt, m, k, n):
    a, b = rnd((m, k), dt, 1), rnd((k, n), dt, 2)
    out = torch.mm(a.to(dev), b.to(dev))
    assert out.dtype == dt and out.shape == (m, n)
    check(out, a.double() @ b.double(), dt, k)
    assert torch_vulkan.kernel_counts().get("aten::mm") == 1
    no_fallback("aten::mm")


@pytest.mark.parametrize("ta", [False, True])
@pytest.mark.parametrize("tb", [False, True])
def test_mm_transposed_operands_without_copy(ta, tb):
    m, k, n = 70, 45, 90
    a = rnd((k, m) if ta else (m, k), torch.float32, 3)
    b = rnd((n, k) if tb else (k, n), torch.float32, 4)
    ad, bd = a.to(dev), b.to(dev)
    A = ad.t() if ta else ad
    B = bd.t() if tb else bd
    ref = (a.t() if ta else a).double() @ (b.t() if tb else b).double()
    check(torch.mm(A, B), ref, torch.float32, k)
    assert "aten::clone" not in torch_vulkan.fallback_counts()
    no_fallback("aten::mm")


def test_mm_on_sliced_views_with_offset():
    a, b = rnd((20, 30), torch.float32, 5), rnd((30, 25), torch.float32, 6)
    ad, bd = a.to(dev), b.to(dev)
    out = torch.mm(ad[3:15, 4:20], bd[4:20, 2:22])
    check(out, a[3:15, 4:20].double() @ b[4:20, 2:22].double(), torch.float32, 16)


@pytest.mark.parametrize("dt", DTYPES)
def test_bmm(dt):
    a, b = rnd((5, 37, 29), dt, 1), rnd((5, 29, 41), dt, 2)
    out = torch.bmm(a.to(dev), b.to(dev))
    check(out, torch.bmm(a.double(), b.double()), dt, 29)
    out2 = torch.bmm(a.to(dev), b.to(dev).transpose(1, 2).contiguous().transpose(1, 2))  # 転置 view
    check(out2, torch.bmm(a.double(), b.double()), dt, 29)
    no_fallback("aten::bmm")


@pytest.mark.parametrize("bias_shape", [(40,), (1, 40), (30, 1), (30, 40), ()])
@pytest.mark.parametrize("dt", DTYPES)
def test_addmm_bias_broadcast_alpha_beta(bias_shape, dt):
    a, b = rnd((30, 20), dt, 1), rnd((20, 40), dt, 2)
    bias = rnd(bias_shape, dt, 3)
    for alpha, beta in [(1.0, 1.0), (0.5, 2.0), (2.0, 0.0)]:
        out = torch.addmm(bias.to(dev), a.to(dev), b.to(dev), alpha=alpha, beta=beta)
        ref = beta * bias.double() + alpha * (a.double() @ b.double()) if beta != 0 else alpha * (a.double() @ b.double())
        check(out, ref.expand(30, 40), dt, 20)
    no_fallback("aten::addmm")


def test_baddbmm():
    a, b = rnd((3, 10, 8), torch.float32, 1), rnd((3, 8, 12), torch.float32, 2)
    bias = rnd((3, 1, 12), torch.float32, 3)
    out = torch.baddbmm(bias.to(dev), a.to(dev), b.to(dev), alpha=0.5, beta=1.5)
    ref = 1.5 * bias.double() + 0.5 * torch.bmm(a.double(), b.double())
    check(out, ref.expand(3, 10, 12), torch.float32, 8)
    no_fallback("aten::baddbmm")


def test_mixed_dtype_and_empty_k_fall_back_correctly():
    a, b = rnd((4, 5), torch.float32), rnd((5, 6), torch.bfloat16)
    with pytest.raises(RuntimeError):  # PyTorch の仕様でエラー（フォールバック経由でも同じ）
        torch.mm(a.to(dev), b.to(dev))
    z = torch.mm(torch.zeros(3, 0, device=dev), torch.zeros(0, 4, device=dev))
    assert torch.equal(z.cpu(), torch.zeros(3, 4))


@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
def test_linear_forward_backward_matches_cpu(dt):
    torch.manual_seed(0)
    lin = torch.nn.Linear(48, 32).to(dt)
    x = rnd((2, 10, 48), dt, 5)
    xc = x.clone().requires_grad_(True)
    lin(xc).float().pow(2).mean().backward()
    gw, gb, gx = lin.weight.grad.clone(), lin.bias.grad.clone(), xc.grad.clone()

    lin_d = torch.nn.Linear(48, 32).to(dt)
    lin_d.load_state_dict(lin.state_dict())
    lin_d.to(dev)
    lin_d.weight.grad = None
    xd = x.to(dev).requires_grad_(True)
    torch_vulkan.reset_fallback_counts()
    lin_d(xd).float().pow(2).mean().backward()
    tol = dict(rtol=2e-4, atol=1e-5) if dt == torch.float32 else dict(rtol=3e-2, atol=3e-3)
    assert torch.allclose(lin_d.weight.grad.cpu().float(), gw.float(), **tol)
    assert torch.allclose(lin_d.bias.grad.cpu().float(), gb.float(), **tol)
    assert torch.allclose(xd.grad.cpu().float(), gx.float(), **tol)
    assert "aten::mm" not in torch_vulkan.fallback_counts()


def test_throughput_report():
    """性能の目安を出力する（合否には使わない）。3 回の平均。"""
    for dt in (torch.float32, torch.bfloat16):
        n = 2048
        a, b = torch.randn(n, n).to(dt).to(dev), torch.randn(n, n).to(dt).to(dev)
        torch.mm(a, b)
        torch_vulkan.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            c = torch.mm(a, b)
        torch_vulkan.synchronize()
        dt_s = (time.perf_counter() - t0) / 3
        print(f"\n[perf] mm {dt} {n}^3: {dt_s * 1e3:.2f} ms = {2 * n ** 3 / dt_s / 1e12:.2f} TFLOPS")


def test_no_vulkan_validation_errors():
    assert torch_vulkan.validation_errors() == 0
