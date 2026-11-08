"""torch.vk が sd-scripts の --vram_budget_mb に必要なメモリ API を公開していること。"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch_vulkan  # noqa: E402


def test_device_module_exposes_memory_api():
    for name in ("memory_info", "set_memory_limit", "peak_live_bytes", "reset_peak_live_bytes"):
        assert hasattr(torch.vk, name), name
    info = torch.vk.memory_info()
    assert info["budget"] > 0 and info["heap"] >= info["budget"] > info["usage"] >= 0


def test_memory_limit_is_enforced_through_device_module():
    torch.vk.set_memory_limit(64)
    try:
        torch.vk.reset_peak_live_bytes()
        x = torch.zeros(1024, device="vk")
        assert torch.vk.peak_live_bytes() >= x.numel() * 4
        try:
            torch.empty(256 << 20, dtype=torch.uint8, device="vk")  # 上限 64MiB を超える確保
        except RuntimeError:
            pass
        else:
            raise AssertionError("上限を超える確保が通ってしまった")
    finally:
        torch.vk.set_memory_limit(0)
