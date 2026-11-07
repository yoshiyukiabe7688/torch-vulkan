"""要素演算の GPU カーネルを CPU と照合する。各ケースで「GPU カーネルが実行され、CPU フォールバックは起きていない」ことも確認する。
実行: venv\\Scripts\\python.exe -m pytest tests\\test_ops_ew.py -q -p no:cacheprovider  （cwd は torch-vulkan 直下）
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
# CPU(同 dtype) との許容誤差。bf16/f16 は丸めの 1 ulp 差を許す。
TOL = {torch.float32: (1e-5, 1e-6), torch.bfloat16: (2e-2, 1e-3), torch.float16: (4e-3, 1e-4)}


@pytest.fixture(autouse=True)
def _reset():
    torch_vulkan.reset_fallback_counts()
    yield


def rnd(shape, dtype=torch.float32, lo=-3.0, hi=3.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(shape, generator=g) * (hi - lo) + lo).to(dtype)


def close(gpu, cpu, dtype=None, scale=1.0):
    dtype = dtype or cpu.dtype
    rtol, atol = TOL[dtype]
    g, c = gpu.cpu().float(), cpu.float()
    assert g.shape == c.shape, (g.shape, c.shape)
    assert torch.allclose(g, c, rtol=rtol * scale, atol=atol * scale), f"max diff {(g - c).abs().max().item()}"


def no_fallback(*names):
    fb = torch_vulkan.fallback_counts()
    bad = {n: fb[n] for n in names if n in fb}
    assert not bad, f"CPU フォールバックが発生: {bad}"


def ran_on_gpu(name):
    assert torch_vulkan.kernel_counts().get(name, 0) >= 1, f"{name} が GPU カーネルで実行されていない: {torch_vulkan.kernel_counts()}"


# ---- 2 項演算 ---------------------------------------------------------------------------------------------
OPS = {
    "add": (torch.add, "aten::add.Tensor"),
    "sub": (torch.sub, "aten::sub.Tensor"),
    "mul": (torch.mul, "aten::mul.Tensor"),
    "div": (torch.div, "aten::div.Tensor"),
}
PAIRS = [(torch.float32, torch.float32), (torch.bfloat16, torch.bfloat16), (torch.float16, torch.float16),
         (torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32), (torch.bfloat16, torch.float16)]


@pytest.mark.parametrize("op", list(OPS))
@pytest.mark.parametrize("da,db", PAIRS)
def test_binary_same_shape_and_promotion(op, da, db):
    fn, name = OPS[op]
    a, b = rnd((5, 7), da, seed=1), rnd((5, 7), db, 0.5, 3.0, seed=2)
    ref = fn(a, b)
    out = fn(a.to(dev), b.to(dev))
    assert out.dtype == ref.dtype
    close(out, ref)
    ran_on_gpu(name)
    no_fallback(name.split(".")[0])


@pytest.mark.parametrize("op", list(OPS))
def test_binary_broadcast_and_noncontiguous(op):
    fn, name = OPS[op]
    for sa, sb in [((3, 1, 5), (1, 4, 1)), ((4, 5), (5,)), ((2, 3, 4, 5), (4, 1)), ((1,), (6, 7))]:
        a, b = rnd(sa, seed=3), rnd(sb, lo=0.5, hi=3.0, seed=4)
        close(fn(a.to(dev), b.to(dev)), fn(a, b))
    a, b = rnd((6, 8)), rnd((8, 6), lo=0.5, hi=3.0)
    close(fn(a.to(dev), b.to(dev).t()), fn(a, b.t()))  # 転置 view
    x = rnd((4, 10))
    close(fn(x.to(dev)[:, 2:7], x.to(dev)[:, 3:8]), fn(x[:, 2:7], x[:, 3:8]))  # オフセット付き slice
    ran_on_gpu(name)
    no_fallback(name.split(".")[0])


@pytest.mark.parametrize("op", list(OPS))
@pytest.mark.parametrize("dt", DTYPES)
def test_binary_with_host_scalar(op, dt):
    fn, name = OPS[op]
    x = rnd((6, 5), dt, lo=0.5, hi=3.0)
    for s in (2.5, 3):
        close(fn(x.to(dev), s), fn(x, s))
    if op in ("add", "mul"):
        close(fn(2.5, x.to(dev)), fn(2.5, x))  # スカラーが左（可換）
    ran_on_gpu(name)
    no_fallback(name.split(".")[0])


def test_alpha_and_zero_dim_vk_tensor():
    a, b = rnd((9,)), rnd((9,))
    close(torch.add(a.to(dev), b.to(dev), alpha=2.0), torch.add(a, b, alpha=2.0))
    close(torch.sub(a.to(dev), b.to(dev), alpha=0.5), torch.sub(a, b, alpha=0.5))
    s = torch.tensor(1.5)
    close(a.to(dev) * s.to(dev), a * s)  # vk 上の 0 次元


@pytest.mark.parametrize("op", ["add_", "sub_", "mul_", "div_"])
def test_inplace_on_strided_view_only_touches_the_view(op):
    x = rnd((6, 8), lo=0.5, hi=3.0)
    y = rnd((6, 4), lo=0.5, hi=3.0, seed=5)
    xd = x.to(dev)
    getattr(xd[:, 2:6], op)(y.to(dev))
    getattr(x[:, 2:6], op)(y)
    close(xd, x)
    getattr(xd, op)(2.0)
    getattr(x, op)(2.0)
    close(xd, x)
    no_fallback("aten::" + op.rstrip("_"))


# ---- 単項演算 -----------------------------------------------------------------------------------------------
UNARY = {
    "neg": (torch.neg, -3, 3), "rsqrt": (torch.rsqrt, 0.1, 5), "reciprocal": (torch.reciprocal, 0.2, 5),
    "exp": (torch.exp, -4, 4), "sqrt": (torch.sqrt, 0.0, 6), "tanh": (torch.tanh, -4, 4),
    "sigmoid": (torch.sigmoid, -6, 6), "relu": (torch.relu, -3, 3), "silu": (F.silu, -6, 6),
    "gelu": (F.gelu, -6, 6), "gelu_tanh": (lambda x: F.gelu(x, approximate="tanh"), -6, 6),
}


@pytest.mark.parametrize("name", list(UNARY))
@pytest.mark.parametrize("dt", DTYPES)
def test_unary(name, dt):
    fn, lo, hi = UNARY[name]
    x = rnd((4, 33), dt, lo, hi)
    close(fn(x.to(dev)), fn(x), scale=2.0 if name in ("exp", "gelu", "gelu_tanh") else 1.0)
    base = name.replace("_tanh", "")
    no_fallback("aten::" + base)


def test_sin_cos_with_large_arguments():
    # RoPE のように位置 x 周波数で大きな引数になる場合。範囲縮約が効いていること。
    x = torch.arange(0, 4000, dtype=torch.float32).reshape(40, 100) * 0.37
    close(torch.sin(x.to(dev)), torch.sin(x), scale=40.0)
    close(torch.cos(x.to(dev)), torch.cos(x), scale=40.0)
    no_fallback("aten::sin", "aten::cos")


@pytest.mark.parametrize("e", [2, 3, 0.5, -1, 2.5, 1.0, 0.0, -2])
def test_pow_scalar(e):
    x = rnd((5, 6), lo=0.3, hi=3.0)
    close(torch.pow(x.to(dev), e), torch.pow(x, e), scale=4.0)
    no_fallback("aten::pow")


def test_pow_negative_base_integer_exponent():
    x = rnd((50,), lo=-3.0, hi=3.0)
    for e in (2, 3, 4):
        close(torch.pow(x.to(dev), e), torch.pow(x, e), scale=4.0)


# ---- バックワード -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", DTYPES)
@pytest.mark.parametrize("act", ["silu", "gelu", "gelu_tanh"])
def test_activation_backward(act, dt):
    fn = {"silu": F.silu, "gelu": F.gelu, "gelu_tanh": lambda x: F.gelu(x, approximate="tanh")}[act]
    x = rnd((6, 17), dt, -5, 5)
    xc = x.clone().requires_grad_(True)
    xg = x.to(dev).requires_grad_(True)
    gout = rnd((6, 17), dt, seed=9)
    fn(xc).backward(gout)
    fn(xg).backward(gout.to(dev))
    close(xg.grad, xc.grad, scale=3.0)
    no_fallback("aten::silu_backward", "aten::gelu_backward")
    ran_on_gpu("aten::silu_backward" if act == "silu" else "aten::gelu_backward")


# ---- where / fill ------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dt", DTYPES)
def test_where_with_broadcast(dt):
    a, b = rnd((4, 6), dt, seed=1), rnd((1, 6), dt, seed=2)
    cond = rnd((4, 1), seed=3) > 0
    close(torch.where(cond.to(dev), a.to(dev), b.to(dev)), torch.where(cond, a, b))
    no_fallback("aten::where")


@pytest.mark.parametrize("dt", DTYPES)
def test_fill_and_zero_on_view(dt):
    x = rnd((5, 8), dt)
    xd = x.to(dev)
    xd[1:4, 2:6].fill_(1.25)
    x[1:4, 2:6].fill_(1.25)
    close(xd, x)
    xd[:, ::2].zero_()
    x[:, ::2].zero_()
    close(xd, x)
    no_fallback("aten::fill_", "aten::zero_")


# ---- 型変換つきコピー ----------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("src", [torch.float32, torch.bfloat16, torch.float16, torch.bool])
@pytest.mark.parametrize("dst", [torch.float32, torch.bfloat16, torch.float16, torch.bool])
def test_cast_between_dtypes_on_device(src, dst):
    if src == dst:
        pytest.skip("同一 dtype の .to() はコピーしない")
    x = rnd((6, 9), torch.float32, -3, 3)
    x = (x > 0.3) if src == torch.bool else x.to(src)
    xd = x.to(dev)
    out = xd.to(dst)
    assert out.device.type == "vk" and out.dtype == dst
    ref = x.to(dst)
    assert torch.equal(out.cpu().float(), ref.float()) or torch.allclose(out.cpu().float(), ref.float(), rtol=1e-2, atol=1e-3)
    ran_on_gpu("aten::copy_")


def test_contiguous_clone_and_strided_copy_stay_on_device():
    x = rnd((6, 8))
    xd = x.to(dev)
    torch_vulkan.reset_fallback_counts()
    assert torch.equal(xd.t().contiguous().cpu(), x.t().contiguous())
    assert torch.equal(xd[:, 1::2].clone().cpu(), x[:, 1::2].clone())
    dst = torch.zeros(6, 8, device=dev)
    dst[:, 1::2].copy_(xd[:, :4])
    ref = torch.zeros(6, 8)
    ref[:, 1::2].copy_(x[:, :4])
    assert torch.equal(dst.cpu(), ref)
    big = torch.zeros(3, 6, 8, device=dev)
    big.copy_(xd)  # ブロードキャスト
    assert torch.equal(big.cpu(), x.expand(3, 6, 8))
    assert torch_vulkan.fallback_counts() == {}, torch_vulkan.fallback_counts()


# ---- 結合: 学習ループ -----------------------------------------------------------------------------------------------------------
def test_mlp_training_matches_cpu_and_uses_gpu_kernels():
    def run(device):
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.SiLU(), torch.nn.Linear(32, 32), torch.nn.GELU(),
                                torch.nn.Linear(32, 4)).to(device)
        x = torch.randn(64, 16).to(device)
        y = torch.randn(64, 4).to(device)
        opt = torch.optim.SGD(m.parameters(), lr=0.05)
        ls = []
        for _ in range(15):
            opt.zero_grad()
            loss = F.mse_loss(m(x), y)
            loss.backward()
            opt.step()
            ls.append(loss.item())
        return ls

    cpu = run("cpu")
    torch_vulkan.reset_fallback_counts()
    gpu = run(dev)
    assert max(abs(a - b) for a, b in zip(cpu, gpu)) < 1e-4, (cpu, gpu)
    kc, fb = torch_vulkan.kernel_counts(), torch_vulkan.fallback_counts()
    assert kc.get("aten::silu_backward") and kc.get("aten::gelu_backward"), kc
    for n in ("aten::silu", "aten::gelu", "aten::add", "aten::mul", "aten::sub"):
        assert n not in fb, (n, fb)


def test_no_vulkan_validation_errors():
    assert torch_vulkan.validation_errors() == 0


# ---- スカラーが左にある演算（回帰: 1 - x が CPU テンソルを返して後段が CPU に混ざった） ------------------------------------------
@pytest.mark.parametrize("dt", DTYPES)
def test_scalar_on_the_left_stays_on_device_and_uses_gpu_kernels(dt):
    x = rnd((5, 7), dt, lo=0.5, hi=3.0)
    xd = x.to(dev)
    cases = {
        "1 - x": (lambda t: 1 - t), "2 / x": (lambda t: 2 / t), "0.5 + x": (lambda t: 0.5 + t), "3 * x": (lambda t: 3 * t),
        "sub alpha": (lambda t: torch.sub(2.0, t, alpha=3)), "add alpha": (lambda t: torch.add(2.0, t, alpha=2)),
    }
    for name, fn in cases.items():
        out = fn(xd)
        assert out.device.type == "vk", f"{name}: 出力が {out.device} になった"
        close(out, fn(x), dt, scale=2.0)
    no_fallback("aten::sub", "aten::add", "aten::div", "aten::mul")


def test_cpu_fallback_result_is_moved_back_to_device_when_first_arg_is_cpu_scalar():
    # atan2 は GPU カーネルが無くフォールバックする。最初の引数が CPU の 0 次元でも、結果は vk に置かれること。
    x = rnd((4, 4), lo=0.5, hi=3.0)
    out = torch.atan2(torch.tensor(2.0), x.to(dev))
    assert out.device.type == "vk"
    assert torch.allclose(out.cpu(), torch.atan2(torch.tensor(2.0), x), atol=1e-6)


# ---- オプティマイザ用: addcmul_ / addcdiv_ / lerp_ と Scalar オーバーロード ------------------------------------------------------
@pytest.mark.parametrize("dt", DTYPES)
def test_addcmul_addcdiv_lerp(dt):
    a, b, c = rnd((6, 9), dt, seed=1), rnd((6, 9), dt, seed=2), rnd((6, 9), dt, 0.5, 3.0, seed=3)
    ad = a.to(dev)
    close(torch.addcmul(ad, b.to(dev), c.to(dev), value=0.3), torch.addcmul(a, b, c, value=0.3), dt, scale=2.0)
    close(torch.addcdiv(ad, b.to(dev), c.to(dev), value=-0.7), torch.addcdiv(a, b, c, value=-0.7), dt, scale=2.0)
    close(torch.lerp(ad, b.to(dev), 0.1), torch.lerp(a, b, 0.1), dt, scale=2.0)
    x, xd = a.clone(), a.to(dev)
    xd.addcmul_(b.to(dev), c.to(dev), value=0.3); x.addcmul_(b, c, value=0.3)
    xd.addcdiv_(b.to(dev), c.to(dev), value=2.0); x.addcdiv_(b, c, value=2.0)
    xd.lerp_(b.to(dev), 0.25); x.lerp_(b, 0.25)
    close(xd, x, dt, scale=4.0)
    no_fallback("aten::addcmul", "aten::addcdiv", "aten::lerp")


@pytest.mark.parametrize("dt", DTYPES)
def test_scalar_overloads_do_not_promote_dtype(dt):
    x = rnd((5, 5), dt, lo=0.5, hi=3.0)
    xd = x.to(dev)
    for name, fn in {"mul_": lambda t: t.mul_(0.999), "add_": lambda t: t.add_(0.5, alpha=2), "sub_": lambda t: t.sub_(1.5),
                     "div_": lambda t: t.div_(3.0)}.items():
        a, b = x.clone(), xd.clone()
        fn(a); fn(b)
        assert b.dtype == dt, name
        close(b, a, dt, scale=2.0)
    for fn in (lambda t: torch.mul(t, 2.5), lambda t: torch.add(t, 1.5, alpha=2), lambda t: torch.sub(t, 0.5), lambda t: torch.div(t, 4.0)):
        out = fn(xd)
        assert out.dtype == dt
        close(out, fn(x), dt, scale=2.0)
    no_fallback("aten::mul", "aten::add", "aten::sub", "aten::div")


@pytest.mark.parametrize("dt", [torch.float32, torch.bfloat16])
def test_adamw_matches_cpu(dt):
    def run(device):
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 4)).to(dt).to(device)
        x, y = torch.randn(32, 16).to(dt).to(device), torch.randn(32, 4).to(dt).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.01, foreach=False)
        ls = []
        for _ in range(10):
            opt.zero_grad()
            loss = F.mse_loss(m(x).float(), y.float())
            loss.backward()
            opt.step()
            ls.append(loss.item())
        return ls

    cpu = run("cpu")
    torch_vulkan.reset_fallback_counts()
    gpu = run(dev)
    tol = 1e-4 if dt == torch.float32 else 5e-2
    assert max(abs(a - b) / max(abs(a), 1e-6) for a, b in zip(cpu, gpu)) < tol, (cpu, gpu)
    fb = torch_vulkan.fallback_counts()
    assert not {n for n in fb if n in ("aten::addcmul", "aten::addcdiv", "aten::lerp", "aten::mul", "aten::add", "aten::div", "aten::sqrt")}, fb


# ---- 連続メモリ(次元結合後 1 次元)の高速経路 ew1_*（1 スレッド 4 要素、整数除算なし） ------------------------------------------
@pytest.mark.parametrize("n", [1, 3, 4, 5, 1023, 1024, 1025, 4097, 100_003])
@pytest.mark.parametrize("dt", DTYPES)
def test_fast_path_matches_cpu_for_awkward_sizes(n, dt):
    # 4 要素/スレッド・1024 要素/グループの境界（端数）で、結果が CPU と一致すること
    a, b = rnd((n,), dt, seed=1), rnd((n,), dt, 0.5, 3.0, seed=2)
    close(a.to(dev) * b.to(dev), a * b, dt)
    close(a.to(dev) + 2.5, a + 2.5, dt)
    close(torch.silu(a.to(dev)) if hasattr(torch, "silu") else F.silu(a.to(dev)), F.silu(a), dt, scale=2.0)
    no_fallback("aten::mul", "aten::add", "aten::silu")


def test_fast_path_handles_scalar_broadcast_and_noncontiguous_inputs_via_general_path():
    x = rnd((64, 33))
    xd = x.to(dev)
    close(xd * xd[:1], x * x[:1])  # ブロードキャスト（2 次元に分かれる）
    close(xd.t() * 2.0, x.t() * 2.0)  # 転置
    close(xd[:, 3:30] + xd[:, 5:32], x[:, 3:30] + x[:, 5:32])  # 非連続な部分
    s = torch.tensor(1.5).to(dev)
    close(xd * s, x * 1.5)  # 0 次元（vk 上）
    no_fallback("aten::mul", "aten::add")
