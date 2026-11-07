# torch-vulkan

PyTorch の **PrivateUse1 デバイス `vk`** として Vulkan バックエンドを実装し、**CUDA 無しで GPU 学習**するためのプロジェクトです。
PyTorch 標準の Vulkan は推論専用（autograd 非対応）なので、学習用に自作しました。
[sd-scripts（fork）](https://github.com/yoshiyukiabe7688/sd-scripts-no-cuda) の Anima / Anima-Turbo の LoRA 学習で使うことを目的としています。

- Rust（ash）製の Vulkan ランタイム `vkrt` + C++ シム（`TORCH_LIBRARY_IMPL(aten, PrivateUse1)` で ATen 演算を登録）+ GLSL カーネル。
- 未実装の演算は CPU にフォールバックします（回数を演算名ごとに集計）。
- 検証環境: Windows 11 / Radeon RX 9070 XT（Vulkan 1.4）/ PyTorch 2.8.0。Anima-base / Anima-Turbo の LoRA 学習（512px、batch 1、bf16）で約 1.4 秒/step。

## 実装済み
- 要素演算（ストライド・ブロードキャスト・f32/bf16/f16 混在）、活性化と backward、`where`、型変換つきコピー、オプティマイザ用の `addcmul_` / `addcdiv_` / `lerp_`。
- `mm` / `bmm` / `addmm` / `baddbmm`（タイル型 GEMM と、`VK_KHR_cooperative_matrix` による bf16/f16 版。非対応デバイスでは自動でタイル型）。
- `softmax`（+backward、`_safe_softmax`）、`layer_norm`（+backward）、`sum` / `mean`、`cat`。
- デバイスメモリは `VK_KHR_buffer_device_address` のアドレスを `data_ptr` とし、スラブ + サブアロケータで確保。dispatch はバッチ記録して必要なときだけ待つ。
- 畳み込みの GPU カーネルは未実装です（`convolution` は CPU で計算。サンプル画像生成の VAE デコード用）。そのため SDXL などの畳み込み主体のモデルは対象外です。

## 使い方
sd-scripts 側で `SD_SCRIPTS_BACKEND=vk` を設定し、`PYTHONPATH` にこのリポジトリを追加して学習スクリプトを実行します。
詳細な手順・コマンド例・制限は sd-scripts 側の `docs/anima_vulkan.md` を参照してください。

```powershell
$env:SD_SCRIPTS_BACKEND = "vk"
$env:PYTHONPATH = "<このリポジトリのパス>"
python anima_train_network.py ...   # 通常の引数。--mixed_precision=bf16 --full_bf16 など
```

主な環境変数: `VKT_NO_COOP=1`（協調行列を使わない）、`VKT_STATS=1`（時間・回数を表示）、`VKT_TRACE_FALLBACK=1`（CPU フォールバックの引数を表示）、`VKRT_PROFILE=1`（カーネル別の GPU 時間）、`VKRT_VALIDATION=1`（Vulkan 検証レイヤ）。

## ビルド
必要なもの: Visual Studio 2022（MSVC）、Rust（cargo）、PyTorch が入った Python（ヘッダと lib を使う）。シェーダを変更する場合は Vulkan SDK（glslc / spirv-val）。生成済みの SPIR-V はコミットしてあります。

```bat
set TORCH_VULKAN_PYTHON=<PyTorch が入った python.exe のパス>
run_build.cmd
```

`run_build.cmd` が MSVC 環境を読み込み、cargo で `vkrt` を、続いて C++ シムをビルドして `torch_vulkan\` に `vkrt.dll` と `torch_vulkan_C*.pyd` を置きます。
MSVC には `/utf-8` が必須です（日本語コメントの末尾バイトが次の行を巻き込むため。`build.py` で指定済み）。
シェーダの再生成: `python tools\gen_shaders.py`。

## テスト
GPU 演算は CPU / float64 と照合します（`VKRT_VALIDATION=1` で Vulkan 検証レイヤを有効にすると、使い方の違反も検出できます）。

```bat
set VKRT_VALIDATION=1
python -m pytest tests -q
```

- ランタイム単体のテスト: `cd vkrt && cargo test --release`
- 縮小 Anima の fwd+bwd を CPU と照合: `set SD_SCRIPTS_DIR=<sd-scripts のパス>` を設定して `python tests\anima_on_vk.py bf16`（`fp32` も可）

## ライセンス
MIT License（[LICENSE](LICENSE) を参照）。改変・再配布・商用利用は自由で、著作権表示とライセンス文の保持のみが条件です。
