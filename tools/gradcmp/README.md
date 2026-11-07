# gradcmp: 実 Anima の勾配を基準（CPU fp32）と比較するツール

bf16 の実行（CPU / vk タイル型 / vk 協調行列）が、fp32 の基準に対して LoRA 勾配をどれだけ再現するかを測る。
`mm` の協調行列版とタイル型の差や、SDPA の精度の影響を切り分けるために作った。

```bat
set SD_SCRIPTS_DIR=<sd-scripts のパス>
set ANIMA_DIT=<anima-base-v1.0.safetensors>
set ANIMA_CACHE_DIR=<キャッシュ済みの潜在とテキスト出力があるフォルダ>
set PYTHONPATH=<このリポジトリ>

python tools\gradcmp\gradcmp.py cpu32 ref.pt  --img char_00 --seed 123     REM 基準（fp32）
python tools\gradcmp\gradcmp.py vk    vk.pt   --img char_00 --seed 123     REM 協調行列あり
python tools\gradcmp\gradcmp.py vk_tiled t.pt --img char_00 --seed 123     REM タイル型
python tools\gradcmp\compare.py ref.pt vk.pt t.pt                          REM 勾配・ブロック出力の相対誤差
python tools\gradcmp\permod.py ref.pt vk.pt                                REM モジュール別の誤差寄与
python tools\gradcmp\coop_search.py ref.pt --img char_00 --seed 123        REM 協調行列を使う範囲の二分探索
```

注意:
- 結果は入力（画像・シード）で大きく変わる。必ず複数の入力で比べること。
- `lora_down` の初期値は `torch.manual_seed(0)` で揃えてある（揃えないと基準と無相関になる）。
- `coop_search.py` は `torch_vulkan.set_coop_range(lo, hi, kind)`（協調行列を使う mm の通し番号の範囲）を使う。
