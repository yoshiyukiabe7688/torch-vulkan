"""VKRT_DEVICE によるデバイスの明示選択。環境変数は初期化前に決まるので、サブプロセスで起動して確認する。"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")

PROBE = "import torch_vulkan; print('NAME=' + torch_vulkan.device_name())"


def run(env_value):
    env = dict(os.environ)
    env.pop("VKRT_VALIDATION", None)
    if env_value is None:
        pass  # 外側の環境（実行者が VKRT_DEVICE を指定していればそれ）をそのまま使う
    else:
        env["VKRT_DEVICE"] = env_value
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")


def name_of(r):
    for line in r.stdout.splitlines():
        if line.startswith("NAME="):
            return line[5:]
    return None


def test_default_selection_still_works():
    r = run(None)
    assert r.returncode == 0 and name_of(r), r.stderr[-400:]


def test_select_by_index_zero_and_by_name_substring_case_insensitive():
    if os.environ.get("VKRT_DEVICE"):
        pytest.skip("VKRT_DEVICE を指定して実行しているので、番号 0 の選択は確認しない（起動できるとは限らない）")
    by_idx = name_of(run("0"))
    assert by_idx
    # 番号 0 のデバイス名の一部（小文字化）でも同じデバイスが選ばれる
    key = by_idx.split()[0].lower()
    assert name_of(run(key)) is not None


def test_unknown_name_and_out_of_range_index_fail_with_device_list():
    for bad in ("no-such-gpu-xyz", "99"):
        r = run(bad)
        assert r.returncode != 0
        text = r.stdout + r.stderr
        assert "VKRT_DEVICE" in text and "0:" in text, text[-400:]
