"""実デバイス（9070 XT）メモリ上での転送・view・GPU カーネルの検証。pytest で実行する。
実行: venv\\Scripts\\python.exe -m pytest tests\\test_gpu_ops.py -q  （torch_vulkan を import できる cwd / PYTHONPATH で）
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch_vulkan  # noqa: E402

dev = torch.device("vk")


@pytest.fixture(autouse=True)
def _reset():
    torch_vulkan.reset_fallback_counts()
    yield


def test_runs_on_real_gpu():
    assert "Radeon" in torch_vulkan.device_name() or "AMD" in torch_vulkan.device_name()


def test_roundtrip_dtypes():
    for dt in (torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.uint8, torch.bool):
        x = (torch.rand(7, 5) * 10).to(dt)
        assert torch.equal(x.to(dev).cpu(), x), dt


def test_cpu_to_vk_with_dtype_conversion_and_broadcast():
    x = torch.arange(6, dtype=torch.float32)
    assert torch.equal(x.to(dev).to(torch.int32).cpu(), x.to(torch.int32))
    dst = torch.zeros(3, 4, device=dev)
    dst.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))  # (4,) -> (3,4) ブロードキャスト
    assert torch.equal(dst.cpu(), torch.tensor([1.0, 2.0, 3.0, 4.0]).expand(3, 4))


def test_noncontiguous_source_and_destination():
    x = torch.randn(4, 6)
    assert torch.equal(x.t().to(dev).cpu(), x.t())  # 転置した CPU テンソルを転送
    v = x.to(dev).t()  # デバイス上の転置 view をホストへ
    assert torch.equal(v.cpu(), x.t())
    d = torch.zeros(4, 6, device=dev)
    d.t().copy_(torch.arange(24, dtype=torch.float32).reshape(6, 4))  # 非連続な宛先への書き込み
    assert torch.equal(d.cpu(), torch.arange(24, dtype=torch.float32).reshape(6, 4).t())


def test_view_with_storage_offset_reads_and_writes():
    x = torch.arange(100, dtype=torch.float32)
    d = x.to(dev)
    assert torch.equal(d[10:20].cpu(), x[10:20])
    d[10:20] = -1.0
    ref = x.clone()
    ref[10:20] = -1.0
    assert torch.equal(d.cpu(), ref)


def test_device_to_device_copy_paths():
    x = torch.randn(8, 8)
    a = x.to(dev)
    assert torch.equal(a.clone().cpu(), x)  # 連続 d2d
    assert torch.equal(a.t().contiguous().cpu(), x.t().contiguous())  # 非連続 -> ホスト経由
    assert torch.equal(a.to(torch.float16).cpu(), x.to(torch.float16))  # dtype 変換


def test_empty_tensors_do_not_crash():
    e = torch.empty(0, 3, device=dev)
    assert e.cpu().shape == (0, 3)
    assert (e + e).shape == (0, 3)


def test_large_tensor_roundtrip_and_memory_is_released():
    before = torch_vulkan.live_bytes()
    x = torch.randn(64 << 20) # 256MB
    d = x.to(dev)
    assert torch_vulkan.live_bytes() - before >= x.numel() * 4
    assert torch.equal(d.cpu(), x)
    del d
    assert torch_vulkan.live_bytes() == before


def test_resize_keeps_existing_data():
    d = torch.arange(10, dtype=torch.float32).to(dev)
    d.resize_(20)
    assert torch.equal(d.cpu()[:10], torch.arange(10, dtype=torch.float32))


@pytest.mark.parametrize("n", [1, 255, 256, 257, 100_003, 4_000_000])
def test_add_runs_on_gpu_kernel_and_matches_cpu(n):
    a, b = torch.randn(n), torch.randn(n)
    out = a.to(dev) + b.to(dev)
    assert torch.equal(out.cpu(), a + b)  # 1 回の fp32 加算なので CPU と完全一致する
    assert torch_vulkan.kernel_counts().get("aten::add.Tensor") == 1
    assert "aten::add" not in torch_vulkan.fallback_counts()


def test_add_2d_and_3d_shapes():
    for shape in [(3, 5), (2, 4, 6), (1, 1, 1, 7)]:
        a, b = torch.randn(shape), torch.randn(shape)
        assert torch.equal((a.to(dev) + b.to(dev)).cpu(), a + b)


def test_add_falls_back_correctly_for_unsupported_cases():
    a, b = torch.randn(4, 5), torch.randn(5)
    assert torch.equal((a.to(dev) + b.to(dev)).cpu(), a + b)  # ブロードキャスト
    assert torch.allclose((a.to(dev) + 2.5).cpu(), a + 2.5)  # スカラー
    assert torch.equal((a.t().to(dev) + a.t().to(dev)).cpu(), a.t() + a.t())  # 同形状でも view は連続化されて GPU か fallback
    x = torch.randn(3, 3)
    assert torch.allclose(torch.add(x.to(dev), x.to(dev), alpha=2.0).cpu(), x + 2 * x)  # alpha != 1
    h = torch.randn(10).to(torch.float16)
    assert torch.equal((h.to(dev) + h.to(dev)).cpu(), h + h)  # fp16 は未対応 -> fallback
    assert torch_vulkan.fallback_counts().get("aten::add", 0) >= 4


def test_add_autograd_through_gpu_kernel():
    a = torch.randn(16, requires_grad=True)
    b = torch.randn(16, requires_grad=True)
    ad = a.detach().to(dev).requires_grad_(True)
    bd = b.detach().to(dev).requires_grad_(True)
    (ad + bd).sum().backward()
    (a + b).sum().backward()
    assert torch.equal(ad.grad.cpu(), a.grad) and torch.equal(bd.grad.cpu(), b.grad)


def test_no_vulkan_validation_errors():
    assert torch_vulkan.validation_errors() == 0  # VKRT_VALIDATION=1 のときのみ意味がある
