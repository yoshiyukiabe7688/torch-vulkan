"""重みのみ INT8 の dequant（torch_vulkan.dequant_int8）を、純 PyTorch の基準実装とビット一致で照合する。
int8 テンソルが vk 上で使えること（転送・view・slice・contiguous・Module の buffer 移動・コピー）も確認する。
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch_vulkan  # noqa: E402

dev = torch.device("vk")
OUT_DTYPES = [torch.float32, torch.bfloat16, torch.float16]


@pytest.fixture(autouse=True)
def _reset():
    torch_vulkan.reset_fallback_counts()
    yield


def make(rows, cols, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randint(-127, 128, (rows, cols), generator=g, dtype=torch.int8)
    nb = (cols + 63) // 64
    scale = (torch.rand(rows, nb, generator=g) * 0.02 + 1e-4).to(torch.bfloat16)
    return q, scale


def ref(q, scale, dt):
    return torch_vulkan.dequant_int8(q, scale, dt)  # CPU 入力は基準実装（純 PyTorch）


@pytest.mark.parametrize("dt", OUT_DTYPES)
@pytest.mark.parametrize(
    "rows,cols",
    [(1, 64), (1, 36), (3, 100), (5, 4608), (2, 63), (7, 65), (4, 1), (1, 1), (320, 2880), (1280, 5120), (2048, 8192), (640, 1280), (33, 130)],
)
def test_dequant_matches_reference_bitwise(dt, rows, cols):
    q, scale = make(rows, cols, seed=rows * 1000 + cols)
    out = torch_vulkan.dequant_int8(q.to(dev), scale.to(dev), dt)
    assert out.device.type == "vk" and out.dtype == dt and out.shape == (rows, cols)
    assert torch.equal(out.cpu(), ref(q, scale, dt)), f"{dt} {rows}x{cols}: 基準とビット一致しない"
    assert torch_vulkan.fallback_counts() == {}, torch_vulkan.fallback_counts()
    assert torch_vulkan.kernel_counts().get("torch_vulkan::dequant_int8", 0) >= 1
    assert torch_vulkan.validation_errors() == 0


def test_dequant_extreme_values_and_zero_scale():
    q = torch.tensor([[-128, -127, 0, 1, 127] * 13], dtype=torch.int8)  # 65 要素（-128 も許す）
    scale = torch.tensor([[0.5, 0.0]], dtype=torch.bfloat16)
    for dt in OUT_DTYPES:
        assert torch.equal(torch_vulkan.dequant_int8(q.to(dev), scale.to(dev), dt).cpu(), ref(q, scale, dt))


def test_dequant_accepts_noncontiguous_inputs_by_making_them_contiguous():
    q, scale = make(8, 200, 3)
    qd = q.to(dev)
    sub = qd[:, :128]  # 非連続（列スライス）
    sc = scale[:, :2]
    out = torch_vulkan.dequant_int8(sub, sc.to(dev), torch.bfloat16)
    assert torch.equal(out.cpu(), ref(q[:, :128].contiguous(), sc.contiguous(), torch.bfloat16))


def test_dequant_view_with_storage_offset():
    # 行スライス（storage offset あり、連続）。i8vec4 の読み出しが整列していなくても正しいこと
    q, scale = make(9, 64, 5)
    qd = q.to(dev)
    out = torch_vulkan.dequant_int8(qd[1:8], scale[1:8].contiguous().to(dev), torch.float32)
    assert torch.equal(out.cpu(), ref(q[1:8], scale[1:8].contiguous(), torch.float32))
    odd = torch.arange(3 * 64 + 1, dtype=torch.int64).to(torch.int8)[1:].view(3, 64)  # 1 バイトずれた先頭
    sc = (torch.rand(3, 1) + 0.1).to(torch.bfloat16)
    assert torch.equal(torch_vulkan.dequant_int8(odd.to(dev), sc.to(dev), torch.bfloat16).cpu(), ref(odd, sc, torch.bfloat16))


# ---- int8 テンソルが vk 上で使えること（転送・view・Module の buffer・ブロックスワップが使う操作） ---------------------------
def test_int8_tensor_roundtrip_and_views():
    q = torch.randint(-128, 128, (5, 100), dtype=torch.int8)
    qd = q.to(dev)
    assert qd.dtype == torch.int8 and qd.device.type == "vk"
    assert torch.equal(qd.cpu(), q)
    assert torch.equal(qd.clone().cpu(), q)
    assert torch.equal(qd.view(10, 50).cpu(), q.view(10, 50))
    assert torch.equal(qd.reshape(-1).cpu(), q.reshape(-1))
    assert torch.equal(qd[2].cpu(), q[2])
    assert torch.equal(qd[1:4, 10:50].cpu(), q[1:4, 10:50])
    assert torch.equal(qd[:, 10:50].contiguous().cpu(), q[:, 10:50])
    assert torch.equal(qd.t().contiguous().cpu(), q.t().contiguous())
    assert torch.equal(torch.empty_like(q, device=dev).copy_(q).cpu(), q)
    assert qd.to(dev) is qd


def test_int8_buffers_move_with_module_to_vk_and_back():
    m = torch.nn.Module()
    q = torch.randint(-128, 128, (4, 128), dtype=torch.int8)
    scale = (torch.rand(4, 2) + 0.1).to(torch.bfloat16)
    m.register_buffer("q", q.clone())
    m.register_buffer("scale", scale.clone())
    m.to(dev)
    assert m.q.device.type == "vk" and m.scale.device.type == "vk"
    assert torch.equal(torch_vulkan.dequant_int8(m.q, m.scale, torch.bfloat16).cpu(), ref(q, scale, torch.bfloat16))
    m.to("cpu")
    assert m.q.device.type == "cpu" and torch.equal(m.q, q) and torch.equal(m.scale, scale)


def test_int8_pinned_host_to_device_and_back_like_block_swap():
    # ブロックスワップは、ホスト側の原本（pin_memory）から非同期で GPU のバッファへ転送する
    q = torch.randint(-128, 128, (16, 256), dtype=torch.int8)
    host = q.pin_memory()
    buf = torch.empty_like(q, device=dev)
    buf.copy_(host, non_blocking=True)
    assert torch.equal(buf.cpu(), q)


def test_int8_parameter_without_grad_moves_to_vk():
    p = torch.nn.Parameter(torch.randint(-128, 128, (4, 64), dtype=torch.int8), requires_grad=False)
    pd = p.to(dev)
    assert pd.dtype == torch.int8 and pd.device.type == "vk"
    assert torch.equal(pd.cpu(), p.data)
