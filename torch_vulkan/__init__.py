"""PrivateUse1 デバイス "vk" の Python 側。

演算ごとの Python 呼び出しは避ける方針なので、ここはデバイス登録と `torch.vk` モジュール
（is_available / get_rng_state / empty_cache など）の提供だけを行う。
M1 ビルド疎通の段階では実体はホストメモリのダミー。
"""

import importlib
import os
import sys
import types

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_extension():
    # build.py が torch_vulkan/ に vkrt.dll と torch_vulkan_C*.pyd を出力する。
    # Windows の Python は PATH を DLL 検索に使わないので、vkrt.dll のあるここを明示する。
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(_THIS_DIR)
    spec_dir = _THIS_DIR
    if spec_dir not in sys.path:
        sys.path.insert(0, spec_dir)
    return importlib.import_module("torch_vulkan_C")


_C = _load_extension()

_cpu_rng_state_cache = None


def is_available() -> bool:
    return True


def device_count() -> int:
    return 1


def current_device() -> int:
    return 0


def set_device(device) -> None:
    return None


def synchronize(device=None) -> None:
    _C.synchronize()


def device_name() -> str:
    return _C.device_name()


def live_bytes() -> int:
    return _C.live_bytes()


def peak_live_bytes() -> int:
    """確保中バイト数の最大値（reset_peak_live_bytes 以降）。学習中の VRAM ピークの目安。"""
    return _C.peak_live_bytes()


def reset_peak_live_bytes() -> None:
    _C.reset_peak_live_bytes()


def set_memory_limit(megabytes: int) -> None:
    """予約メモリの上限（MiB, 0 で無制限）。小さい VRAM の GPU を模擬する。超えると OOM（RuntimeError）になる。
    環境変数 VKT_MEM_LIMIT_MB でも指定できる（import 時に反映）。最初の確保より前に呼ぶこと。"""
    _C.set_memory_limit(int(megabytes) << 20)


def kernel_counts() -> dict:
    return dict(_C.kernel_counts())


def validation_errors() -> int:
    return _C.validation_errors()


def empty_cache() -> None:
    _C.empty_cache()  # 完全に空いたスラブをドライバへ返す


def reserved_bytes() -> int:
    return _C.reserved_bytes()


def live_blocks() -> int:
    return _C.live_blocks()


def slab_count() -> int:
    return _C.slab_count()


def manual_seed(seed: int) -> None:
    # ノイズは CPU で生成する方針なので、デバイス側の乱数は CPU の生成器に任せる。
    return None


def manual_seed_all(seed: int) -> None:
    return None


def seed() -> None:
    return None


def _is_in_bad_fork() -> bool:
    return False


def get_amp_supported_dtype():
    # torch.autocast(device_type="vk") が許可する dtype。
    return [torch.bfloat16, torch.float16]


class device:  # noqa: N801  torch.cuda.device と同じ名前・用途のコンテキストマネージャ
    def __init__(self, device=None):
        self.idx = device

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def get_rng_state(device="vk") -> torch.Tensor:
    return torch.zeros(1, dtype=torch.uint8)


def set_rng_state(new_state, device="vk") -> None:
    return None


def stats() -> dict:
    return dict(_C.stats())


def fallback_time_by_op() -> dict:
    return dict(_C.fallback_time_by_op())


def _print_stats_at_exit():
    st = stats()
    print("\n[torch_vulkan stats] " + ", ".join(f"{k}={v:.3f}" for k, v in st.items()))
    top = sorted(fallback_time_by_op().items(), key=lambda kv: -kv[1])[:8]
    print("[torch_vulkan stats] fallback 時間 上位: " + ", ".join(f"{k}={v:.2f}s" for k, v in top))
    if os.environ.get("VKRT_PROFILE"):
        rows = [r.split("\t") for r in _C.profile().splitlines()]
        total = sum(float(r[1]) for r in rows) or 1.0
        print(f"[torch_vulkan profile] GPU 時間合計 {total:.1f} ms")
        for name, ms, n in rows[:18]:
            print(f"[torch_vulkan profile] {name:34s} {float(ms):9.1f} ms {float(ms) / total * 100:5.1f}%  x{n}  ({float(ms) / int(n) * 1000:.0f} us/回)")
    print("[torch_vulkan stats] kernel 回数 上位: " + str(sorted(kernel_counts().items(), key=lambda kv: -kv[1])[:8]))


if os.environ.get("VKT_STATS") or os.environ.get("VKRT_PROFILE"):
    import atexit

    atexit.register(_print_stats_at_exit)


def coop_caps() -> int:
    """協調行列カーネルの対応状況（bit0=f16, bit1=bf16）。0 ならタイル型 GEMM のみ。"""
    return _C.coop_caps()


def set_coop_enabled(on: bool) -> None:
    """協調行列カーネルの使用を切り替える（テストでタイル型との比較に使う）。環境変数 VKT_NO_COOP=1 でも無効にできる。"""
    _C.set_coop_enabled(bool(on))


def fallback_counts() -> dict:
    return dict(_C.fallback_counts())


def reset_fallback_counts() -> None:
    _C.reset_counts()


def _register():
    # PrivateUse1 は「互換な dense デバイス」に含まれないため、module.to("vk") が Parameter を新しい
    # オブジェクトに差し替えてしまい、先に作った optimizer が古い CPU パラメータを握る（学習が静かに進まない）。
    # swap_tensors 方式にすると Parameter の同一性が保たれる。sd-scripts は optimizer を prepare()（=to()）より先に作る。
    torch.__future__.set_swap_module_params_on_conversion(True)
    # "vulkan" は PyTorch 標準（推論専用）の Vulkan デバイス型と衝突し、PrivateUse1 に届かないので "vk" を使う。
    torch.utils.rename_privateuse1_backend("vk")
    module = types.ModuleType("torch.vk")
    for name in (
        "is_available",
        "device_count",
        "current_device",
        "set_device",
        "synchronize",
        "empty_cache",
        "manual_seed",
        "manual_seed_all",
        "seed",
        "_is_in_bad_fork",
        "get_amp_supported_dtype",
        "device",
        "get_rng_state",
        "set_rng_state",
    ):
        setattr(module, name, globals()[name])
    torch._register_device_module("vk", module)
    # Tensor.vulkan() / .is_vulkan は torch.utils.rename_privateuse1_backend 側で既に生成される（二重登録はエラー）。


_register()


def set_coop_range(lo: int = 0, hi: int = 2**64 - 1, kind: int = 0) -> None:
    """デバッグ用: 協調行列の対象になる mm の通し番号 [lo, hi) だけ協調行列を使う（それ以外はタイル型）。
    kind: 0=すべて, 1=bmm のみ, 2=mm のみ。誤差の原因となる呼び出しの二分探索に使う。"""
    _C.set_coop_range(int(lo), int(hi), int(kind))


def reset_coop_calls() -> None:
    """デバッグ用: 通し番号を 0 に戻す。"""
    _C.reset_coop_calls()


def coop_calls() -> int:
    """デバッグ用: 協調行列の対象になった mm 呼び出しの通し番号（リセット後の累計）。"""
    return _C.coop_calls()


if os.environ.get("VKT_MEM_LIMIT_MB"):
    set_memory_limit(int(os.environ["VKT_MEM_LIMIT_MB"]))
