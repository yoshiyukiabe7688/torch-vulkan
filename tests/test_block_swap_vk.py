"""vk 上で sd-scripts の H2D 専用ブロックスワップ（LoRAStreamOffloader）が、全ブロック常駐の実行と同じ結果になることを確認する。

ブロックの重み（凍結した Linear）を CPU に置き、リング上の GPU バッファへ順次コピーして計算する。
LoRA 相当の学習可能パラメータの勾配が、全ブロック常駐の実行と一致すれば、スワップが重みを取り違えていない。
SD_SCRIPTS_DIR（sd-scripts のパス）が必要。無ければスキップする。
"""

import os
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

SD_SCRIPTS_DIR = os.environ.get("SD_SCRIPTS_DIR")
if not SD_SCRIPTS_DIR:
    pytest.skip("SD_SCRIPTS_DIR が未設定", allow_module_level=True)
sys.path.insert(0, SD_SCRIPTS_DIR)

import torch_vulkan  # noqa: E402,F401
from library.custom_offloading_utils import BlockSwapConfig, create_offloader  # noqa: E402

DEV = torch.device("vk")
D, N_BLOCKS, RANK = 64, 6, 4


class Lora(nn.Module):
    """ブロックの外（ネットワーク側）に置く学習可能パラメータ。実際の LoRA と同じ置き場にする。"""

    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(D, RANK, dtype=torch.bfloat16) * 0.1)
        self.b = nn.Parameter(torch.randn(RANK, D, dtype=torch.bfloat16) * 0.1)


class Block(nn.Module):
    def __init__(self, lora: Lora):
        super().__init__()
        self.lin1 = nn.Linear(D, 4 * D, bias=False, dtype=torch.bfloat16)
        self.lin2 = nn.Linear(4 * D, D, bias=False, dtype=torch.bfloat16)
        self.norm = nn.LayerNorm(D, dtype=torch.bfloat16)
        self._lora = [lora]  # 登録しない（block.to() に巻き込まれないように）

    def forward(self, x):
        h = self.lin2(F.gelu(self.lin1(self.norm(x))))
        lo = self._lora[0]
        return x + h + (x @ lo.a) @ lo.b


class Net(nn.Module):
    def __init__(self, swap: int, ring: int = 2):
        super().__init__()
        torch.manual_seed(0)
        self.loras = nn.ModuleList([Lora() for _ in range(N_BLOCKS)])
        self.blocks = nn.ModuleList([Block(self.loras[i]) for i in range(N_BLOCKS)])
        self.blocks.requires_grad_(False)
        self.swap = swap
        if swap:
            self.loras.to(DEV)
            cfg = BlockSwapConfig(device=DEV, supports_backward=True, h2d_only=True, ring_size=ring)
            self.offloader = create_offloader("block", self.blocks, N_BLOCKS, swap, cfg)
            self.offloader.prepare_block_devices_before_forward(self.blocks)
        else:
            self.to(DEV)
        for lo in self.loras:
            for p in lo.parameters():
                p.requires_grad_(True)

    def forward(self, x):
        for i, block in enumerate(self.blocks):
            if self.swap:
                self.offloader.wait_for_block(i)
            x = checkpoint(block, x, use_reentrant=False)
            if self.swap:
                self.offloader.submit_move_blocks_forward(self.blocks, i)
        return x


def _run(net: Net, steps: int = 2):
    out = []
    for step in range(steps):
        torch.manual_seed(100 + step)
        x = torch.randn(2, 16, D).to(torch.bfloat16).to(DEV).requires_grad_(True)
        for lo in net.loras:
            for p in lo.parameters():
                p.grad = None
        loss = net(x).float().pow(2).mean()
        loss.backward()
        grads = [p.grad.detach().float().cpu() for lo in net.loras for p in lo.parameters()]
        out.append((loss.item(), grads))
    return out


@pytest.fixture(scope="module")
def reference():
    return _run(Net(swap=0))


@pytest.mark.parametrize("swap,ring", [(2, 2), (4, 2), (3, 1), (4, 3)])
def test_block_swap_matches_resident(reference, swap, ring):
    got = _run(Net(swap=swap, ring=ring))
    for (loss_r, grads_r), (loss_g, grads_g) in zip(reference, got):
        assert loss_g == loss_r
        for gr, gg in zip(grads_r, grads_g):
            assert torch.equal(gr, gg)


def test_swapped_weights_leave_the_device():
    """スワップ対象ブロックの重みが GPU に常駐していない（確保量が減っている）こと。"""
    torch_vulkan.synchronize()
    resident = Net(swap=0)
    torch_vulkan.synchronize()
    live_resident = torch_vulkan.live_bytes()
    del resident
    torch_vulkan.empty_cache()
    swapped = Net(swap=4, ring=2)
    torch_vulkan.synchronize()
    live_swapped = torch_vulkan.live_bytes()
    block_bytes = 2 * (D * 4 * D) * 2  # lin1 + lin2 の bf16
    # 4 ブロックを CPU に置き、GPU にはリング 2 枠だけ → 2 ブロック分は減る
    assert live_resident - live_swapped >= 2 * block_bytes
    del swapped
