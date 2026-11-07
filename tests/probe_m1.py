"""M1 ビルド疎通プローブ: ダミー "vk" デバイスでの基本動作と R1〜R3 の確認。各項目は独立に成否を出す。"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch_vulkan

results = []


def check(name):
    def deco(fn):
        try:
            detail = fn()
            results.append((name, "OK", detail))
        except Exception as e:  # noqa: BLE001
            results.append((name, "NG", f"{type(e).__name__}: {str(e)[:300]}"))
            traceback.print_exc(limit=3)
        return fn

    return deco


dev = torch.device("vk")


@check("device 作成・往復コピー")
def _():
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    y = x.to(dev)
    assert y.device.type == "vk", y.device
    z = y.cpu()
    assert torch.equal(x, z)
    return f"device={y.device}"


@check("要素演算・matmul（フォールバック経由）")
def _():
    torch_vulkan.reset_fallback_counts()
    a = torch.randn(4, 5).to(dev)
    b = torch.randn(5, 3).to(dev)
    c = (a @ b + 1).relu()
    ref = ((a.cpu() @ b.cpu()) + 1).relu()
    assert torch.allclose(c.cpu(), ref, atol=1e-6)
    return f"fallbacks={torch_vulkan.fallback_counts()}"


@check("view 系（transpose/reshape/slice）がフォールバック無しで動く")
def _():
    torch_vulkan.reset_fallback_counts()
    a = torch.randn(4, 6).to(dev)
    torch_vulkan.reset_fallback_counts()
    v = a.t().reshape(2, 12)[:, 1:5].unsqueeze(0).expand(3, 2, 4)
    assert v.device.type == "vk"
    return f"fallbacks={torch_vulkan.fallback_counts()}"


@check("view への in-place 書き込みが元テンソルに反映される")
def _():
    a = torch.zeros(4, 4).to(dev)
    a[1:3, 1:3] = 5.0
    ref = torch.zeros(4, 4)
    ref[1:3, 1:3] = 5.0
    assert torch.equal(a.cpu(), ref), a.cpu()
    return "ok"


@check("autograd: Linear の fwd/bwd が CPU と一致")
def _():
    torch.manual_seed(0)
    lin = torch.nn.Linear(8, 4)
    x = torch.randn(3, 8)
    lin_v = torch.nn.Linear(8, 4).to(dev)
    lin_v.load_state_dict(lin.state_dict())
    lin(x).pow(2).sum().backward()
    lin_v(x.to(dev)).pow(2).sum().backward()
    assert torch.allclose(lin.weight.grad, lin_v.weight.grad.cpu(), atol=1e-5)
    return "grad 一致"


def _train(device, steps=20):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 4))
    x = torch.randn(64, 16)
    y = torch.randn(64, 4)
    model = model.to(device)
    x, y = x.to(device), y.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@check("M1: 小さな MLP の学習で loss が CPU と一致")
def _():
    torch_vulkan.reset_fallback_counts()
    cpu = _train("cpu")
    vk = _train(dev)
    diff = max(abs(a - b) for a, b in zip(cpu, vk))
    assert diff < 1e-5, f"最大差 {diff}"
    top = sorted(torch_vulkan.fallback_counts().items(), key=lambda kv: -kv[1])[:5]
    return f"loss {cpu[0]:.4f}->{cpu[-1]:.4f}, 最大差 {diff:.2e}, fallback 上位 {top}"


@check("R3: torch.utils.checkpoint (use_reentrant=False)")
def _():
    lin = torch.nn.Linear(8, 8).to(dev)
    x = torch.randn(2, 8, device=dev, requires_grad=True)
    out = torch.utils.checkpoint.checkpoint(lambda t: lin(t).relu(), x, use_reentrant=False)
    out.sum().backward()
    assert lin.weight.grad is not None
    return "ok"


@check("R3: torch.utils.checkpoint (use_reentrant=True)")
def _():
    lin = torch.nn.Linear(8, 8).to(dev)
    x = torch.randn(2, 8, device=dev, requires_grad=True)
    out = torch.utils.checkpoint.checkpoint(lambda t: lin(t).relu(), x, use_reentrant=True)
    out.sum().backward()
    assert lin.weight.grad is not None
    return "ok"


@check("R3: checkpoint + dropout の乱数再現（preserve_rng_state）")
def _():
    lin = torch.nn.Linear(8, 8).to(dev)
    x = torch.randn(2, 8, device=dev, requires_grad=True)
    out = torch.utils.checkpoint.checkpoint(lambda t: torch.nn.functional.dropout(lin(t), 0.5), x, use_reentrant=False)
    out.sum().backward()
    return "ok"


@check("R2: torch.autocast(device_type='vulkan', dtype=bfloat16)")
def _():
    lin = torch.nn.Linear(8, 8).to(dev)
    x = torch.randn(2, 8, device=dev)
    with torch.autocast(device_type="vk", dtype=torch.bfloat16):
        y = lin(x)
    return f"出力 dtype={y.dtype}"


@check("R2: torch.autocast(device_type='vulkan', enabled=False)")
def _():
    with torch.autocast(device_type="vk", dtype=torch.float32, enabled=False):
        y = torch.randn(2, 2, device=dev) * 2
    return f"dtype={y.dtype}"


@check("bf16 テンソルの演算（フォールバック経由）")
def _():
    a = torch.randn(4, 4).to(dev).bfloat16()
    b = (a @ a).float().cpu()
    ref = (a.cpu() @ a.cpu()).float()
    assert torch.allclose(b, ref, atol=1e-2)
    return "ok"


@check("R1: accelerate.Accelerator の device 判定")
def _():
    from accelerate import Accelerator

    acc = Accelerator()
    return f"device={acc.device}, state.device={acc.state.device}, distributed_type={acc.state.distributed_type}"


def _make_acc(mixed_precision="no"):
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState, PartialState

    AcceleratorState._reset_state(reset_partial_state=True)  # 同一プロセスで mixed_precision を変えるためのリセット
    acc = Accelerator(mixed_precision=mixed_precision)
    # accelerate は vk を知らないので cpu と判定する。sd-scripts の accelerator.device を vk にするには上書きが要る。
    PartialState().device = dev
    AcceleratorState().device = dev
    return acc


@check("R1: device を vk に上書きすると accelerator.device が vk になる")
def _():
    acc = _make_acc()
    assert acc.device.type == "vk", acc.device
    return f"acc.device={acc.device}"


@check("R1: prepare(model, optimizer) がモデルを vk へ移し、学習ステップが通る")
def _():
    acc = _make_acc()
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    model, opt = acc.prepare(model, opt)
    p = next(model.parameters())
    assert p.device.type == "vk", p.device
    x = torch.randn(8, 4, device=dev)
    loss = model(x).pow(2).mean()
    acc.backward(loss)
    acc.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return f"param device={p.device}, loss={loss.item():.4f}"


@check("R1/R2: Accelerator(mixed_precision='bf16') の autocast() が vk で通る")
def _():
    acc = _make_acc("bf16")
    lin = torch.nn.Linear(8, 8).to(dev)
    x = torch.randn(2, 8, device=dev)
    with acc.autocast():
        y = lin(x)
    return f"native_amp={acc.native_amp}, 出力 dtype={y.dtype}（bf16 になれば autocast が効いている）"


@check("safetensors の保存・読込（vulkan テンソル）")
def _():
    from safetensors.torch import save_file, load_file

    t = {"w": torch.randn(3, 3)}
    save_file({k: v.to(dev).cpu().contiguous() for k, v in t.items()}, "_tmp_probe.safetensors")
    r = load_file("_tmp_probe.safetensors")
    os.remove("_tmp_probe.safetensors")
    assert torch.equal(r["w"], t["w"])
    return "ok"


width = max(len(n) for n, _, _ in results)
print("\n==== 結果 ====")
for n, s, d in results:
    print(f"[{s}] {n.ljust(width)}  {d}")
sys.exit(0 if all(s == "OK" for _, s, _ in results) else 1)
