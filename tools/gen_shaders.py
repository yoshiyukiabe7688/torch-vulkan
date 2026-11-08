"""vkrt/shaders/ew.comp.in を演算 x dtype で展開し、SPIR-V と Rust のカーネル表(kernels_gen.rs)を生成する。

使い方: venv\\Scripts\\python.exe tools\\gen_shaders.py   （glslc / spirv-val は Vulkan SDK のものを使う）
生成物（vkrt/shaders/gen/*.spv, vkrt/src/kernels_gen.rs）はコミットする（ビルド時に glslc を要求しないため）。
カーネル名: ew_<op>_<a>_<b>_<out>（2 項）/ ew_<op>_<a>_<out>（1 項・スカラー）。dtype: f32 / bf16 / f16 / u8(bool)。
"""

import os
import subprocess
import sys
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SHADERS = os.path.join(ROOT, "vkrt", "shaders")
GEN = os.path.join(SHADERS, "gen")
SDK = os.environ.get("VULKAN_SDK", r"C:\VulkanSDK\1.4.363.0")
GLSLC = os.path.join(SDK, "Bin", "glslc.exe")
SPIRV_VAL = os.path.join(SDK, "Bin", "spirv-val.exe")
PUSH_SIZE = 128  # ew.comp.in の push constant ブロックのバイト数（C++ 側の構造体と一致させる）
MM_PUSH_SIZE = 112  # mm.comp.in
# 行単位カーネル: (テンプレート, 名前の接頭辞, MODE or None, push constant バイト数)
ROW_KERNELS = [
    ("softmax.comp.in", "softmax_fwd", 0, 40),
    ("softmax.comp.in", "softmax_bwd", 1, 40),
    ("ln.comp.in", "ln_fwd", 0, 72),
    ("ln.comp.in", "ln_bwd", 1, 72),
    ("ln.comp.in", "ln_wgrad", 2, 72),
    ("reduce.comp.in", "reduce", None, 80),
    ("im2col.comp.in", "im2col", 0, 72),
    ("im2col.comp.in", "col2im", 1, 72),
    ("upsample.comp.in", "up_fwd", 0, 40),
    ("upsample.comp.in", "up_bwd", 1, 40),
    ("gn.comp.in", "gn_fwd", 0, 88),
    ("gn.comp.in", "gn_bwd", 1, 88),
    ("gn.comp.in", "gn_wgrad", 2, 88),
    ("dq8.comp.in", "dq8", 0, 40),
    ("attn.comp.in", "attn_fwd", 0, 128),
    ("attn.comp.in", "attn_dkdv", 1, 128),
    ("attn.comp.in", "attn_dq", 2, 128),
    ("attn.comp.in", "attn_delta", 3, 128),
]

KIND = {"f32": 0, "bf16": 1, "f16": 2, "u8": 3}
FLOATS = ["f32", "bf16", "f16"]


def promote(a, b):
    """PyTorch の型昇格（浮動小数のみ）: 同じならそのまま、それ以外（bf16+f16 を含む）は f32、f32 が絡めば f32。"""
    return a if a == b else "f32"


# (名前, 式, 入力) 入力: "ab" = 2 項テンソル / "as" = テンソル + スカラー s0 / "a" = 単項 / "gs" = 勾配と self / "abc" = where / "" = fill
BINARY = {
    "add": "x + S0 * y",
    "sub": "x - S0 * y",
    "mul": "x * y",
    "div": "x / y",
}
SCALAR = {
    "adds": "x + S0",
    "muls": "x * S0",
    "divs": "x / S0",
    "pows": "pow_f(x, S0)",
    "axpb": "S0 * x + S1",  # スカラー左の add / sub（alpha 付き）: S + alpha*x
    "rdivs": "S0 / x",  # スカラー左の div
}
UNARY = {
    "neg": "-x",
    "rsqrt": "inversesqrt(x)",
    "recip": "1.0 / x",
    "exp": "exp(x)",
    "sin": "sin_f(x)",
    "cos": "cos_f(x)",
    "sqrt": "sqrt(x)",
    "tanh": "tanh_f(x)",
    "sigmoid": "sigmoid_f(x)",
    "relu": "max(x, 0.0)",
    "silu": "x / (1.0 + exp(-x))",
    "gelu_erf": "0.5 * x * (1.0 + erf_f(x * 0.70710678118654752440))",
    "gelu_tanh": "0.5 * x * (1.0 + tanh_f(0.79788456080286535588 * (x + 0.044715 * x * x * x)))",
}
TERNARY = {  # x = self, y = tensor1, z = tensor2（同 dtype）。オプティマイザ用
    "addcmul": "x + S0 * y * z",
    "addcdiv": "x + S0 * y / z",
}
BACKWARD = {  # x = grad, y = self（同 dtype）
    "silu_bwd": "silu_bwd(x, y)",
    "gelu_erf_bwd": "gelu_erf_bwd(x, y)",
    "gelu_tanh_bwd": "gelu_tanh_bwd(x, y)",
    "lerp": "x + S0 * (y - x)",  # lerp(self, end, weight): x=self, y=end（同 dtype）
}


def variants():
    """(カーネル名, 式, a, b, c, out) のリスト。b / c が None なら未使用。"""
    out = []
    for op, expr in BINARY.items():
        for a, b in product(FLOATS, FLOATS):
            o = promote(a, b)
            out.append((f"ew_{op}_{a}_{b}_{o}", expr, a, b, None, o))
    for op, expr in SCALAR.items():
        for a in FLOATS:
            out.append((f"ew_{op}_{a}_{a}", expr, a, None, None, a))
    for op, expr in UNARY.items():
        for a in FLOATS:
            out.append((f"ew_{op}_{a}_{a}", expr, a, None, None, a))
    for op, expr in BACKWARD.items():
        for a in FLOATS:
            out.append((f"ew_{op}_{a}_{a}_{a}", expr, a, a, None, a))
    for op, expr in TERNARY.items():
        for a in FLOATS:
            out.append((f"ew_{op}_{a}", expr, a, a, a, a))
    for a in FLOATS:  # where(cond, self, other)
        out.append((f"ew_where_{a}_{a}_{a}", "z != 0.0 ? x : y", a, a, "u8", a))
    for o in FLOATS:  # fill（入力なし。S0 を書き込む）
        out.append((f"ew_fill_{o}", "S0", None, None, None, o))
    for a, o in product(KIND, KIND):  # 型変換つきストライドコピー
        out.append((f"ew_cast_{a}_{o}", "x", a, None, None, o))
    return out


def compile_one(name, expr, a, b, c, o, fast=False):
    defs = [f"-DOP_EXPR={expr}", f"-DO_KIND={KIND[o]}"] + (["-DFAST1D"] if fast else [])
    for tag, dt in (("A", a), ("B", b), ("C", c)):
        defs.append(f"-D{tag}_KIND={KIND[dt] if dt else 0}")
        if dt:
            defs.append(f"-DHAS_{tag}")
    src = os.path.join(SHADERS, "ew.comp.in")
    dst = os.path.join(GEN, f"{name}.spv")
    cmd = [GLSLC, "-fshader-stage=compute", "--target-env=vulkan1.3", "-O", f"-I{SHADERS}", *defs, src, "-o", dst]
    subprocess.run(cmd, check=True)
    subprocess.run([SPIRV_VAL, "--target-env", "vulkan1.3", dst], check=True)


def compile_mm(dt, coop=False):
    name = f"mm_coop_{dt}" if coop else f"mm_{dt}"
    src = os.path.join(SHADERS, "mm_coop.comp.in" if coop else "mm.comp.in")
    dst = os.path.join(GEN, f"{name}.spv")
    subprocess.run([GLSLC, "-fshader-stage=compute", "--target-env=vulkan1.3", "-O", f"-I{SHADERS}", f"-DKIND={KIND[dt]}", src, "-o", dst], check=True)
    subprocess.run([SPIRV_VAL, "--target-env", "vulkan1.3", dst], check=True)
    return name


def compile_row(template, prefix, mode, dt):
    name = f"{prefix}_{dt}"
    src = os.path.join(SHADERS, template)
    dst = os.path.join(GEN, f"{name}.spv")
    defs = [f"-DKIND={KIND[dt]}"] + ([f"-DMODE={mode}"] if mode is not None else [])
    subprocess.run([GLSLC, "-fshader-stage=compute", "--target-env=vulkan1.3", "-O", f"-I{SHADERS}", *defs, src, "-o", dst], check=True)
    subprocess.run([SPIRV_VAL, "--target-env", "vulkan1.3", dst], check=True)
    return name


def main():
    os.makedirs(GEN, exist_ok=True)
    entries = []  # (name, push_size)
    for v in variants():
        compile_one(*v)
        entries.append((v[0], PUSH_SIZE))
        fast_name = "ew1_" + v[0][3:]  # 連続メモリ(次元結合後 1 次元)の高速経路。名前の接頭辞 ew_ -> ew1_
        compile_one(fast_name, *v[1:], fast=True)
        entries.append((fast_name, PUSH_SIZE))
    for dt in FLOATS:
        entries.append((compile_mm(dt), MM_PUSH_SIZE))
    for dt in ("bf16", "f16"):  # 協調行列版（vkrt が対応を判定し、非対応デバイスでは使われない）
        entries.append((compile_mm(dt, coop=True), MM_PUSH_SIZE))
    for mode, nm in ((0, "fwd"), (1, "dkdv"), (2, "dq")):  # 融合アテンションの協調行列版（bf16 / f16 のみ）
        for dt in ("bf16", "f16"):
            entries.append((compile_row("attn_coop.comp.in", f"attn_coop_{nm}", mode, dt), 104))
    for template, prefix, mode, size in ROW_KERNELS:
        for dt in FLOATS:
            entries.append((compile_row(template, prefix, mode, dt), size))
    lines = [
        "// 自動生成: tools/gen_shaders.py。手で編集しない。",
        f"pub const EW_PUSH_SIZE: u32 = {PUSH_SIZE};",
        f"pub const MM_PUSH_SIZE: u32 = {MM_PUSH_SIZE};",
        "pub const GEN_KERNELS: &[(&str, &[u8], u32)] = &[",
    ]
    for name, size in entries:
        lines.append(f'    ("{name}", include_bytes!("../shaders/gen/{name}.spv"), {size}),')
    lines.append("];")
    with open(os.path.join(ROOT, "vkrt", "src", "kernels_gen.rs"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"generated {len(entries)} kernels")


if __name__ == "__main__":
    sys.exit(main())
