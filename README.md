# torch-vulkan（開発中）

PyTorch の PrivateUse1 デバイスとして Vulkan バックエンドを実装するための作業ディレクトリ。
sd-scripts フォークの CPU/非CUDA 化（`C:\Users\<user>\sd-scripts-no-cuda\.claude\design\no-cuda-backend.md`）の Phase 2。

## 現状（M1 ビルド疎通）
実体はまだ**ホストメモリのダミー**（Vulkan 呼び出しなし）。未実装の演算は CPU フォールバックし、演算名ごとの回数を集計する。

- デバイス名は **`vk`**。`vulkan` は PyTorch 標準（推論専用）の Vulkan デバイス型と衝突し、PrivateUse1 に届かない。
- `import torch_vulkan` で `torch.__future__.set_swap_module_params_on_conversion(True)` を有効にする。
  無いと `module.to("vk")` が Parameter を差し替え、先に作った optimizer が古いパラメータを握って学習が進まない。

## ビルドとテスト
```
run_build.cmd                                  # VsDevCmd を読み込み、sd-scripts の venv で build.py を実行
venv\Scripts\python.exe tests\probe_m1.py      # 基本動作 + R1〜R3（17 項目）
venv\Scripts\python.exe tests\anima_on_vk.py fp32|bf16   # 縮小 Anima(LoRA) の fwd+bwd を CPU と照合
```
（venv は `C:\Users\<user>\sd-scripts-no-cuda\venv`。MSVC は `/utf-8` が必須：日本語コメントの末尾バイトが次行を巻き込む）
