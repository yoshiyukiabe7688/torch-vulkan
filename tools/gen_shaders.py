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
PUSH_SIZE = 152  # ew.comp.in の push constant ブロックのバイト数（C++ 側の構造体と一致させる）

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
BACKWARD = {  # x = grad, y = self（同 dtype）
    "silu_bwd": "silu_bwd(x, y)",
    "gelu_erf_bwd": "gelu_erf_bwd(x, y)",
    "gelu_tanh_bwd": "gelu_tanh_bwd(x, y)",
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
    for a in FLOATS:  # where(cond, self, other)
        out.append((f"ew_where_{a}_{a}_{a}", "z != 0.0 ? x : y", a, a, "u8", a))
    for o in FLOATS:  # fill（入力なし。S0 を書き込む）
        out.append((f"ew_fill_{o}", "S0", None, None, None, o))
    for a, o in product(KIND, KIND):  # 型変換つきストライドコピー
        out.append((f"ew_cast_{a}_{o}", "x", a, None, None, o))
    return out


def compile_one(name, expr, a, b, c, o):
    defs = [f"-DOP_EXPR={expr}", f"-DO_KIND={KIND[o]}"]
    for tag, dt in (("A", a), ("B", b), ("C", c)):
        defs.append(f"-D{tag}_KIND={KIND[dt] if dt else 0}")
        if dt:
            defs.append(f"-DHAS_{tag}")
    src = os.path.join(SHADERS, "ew.comp.in")
    dst = os.path.join(GEN, f"{name}.spv")
    cmd = [GLSLC, "-fshader-stage=compute", "--target-env=vulkan1.3", "-O", *defs, src, "-o", dst]
    subprocess.run(cmd, check=True)
    subprocess.run([SPIRV_VAL, "--target-env", "vulkan1.3", dst], check=True)


def main():
    os.makedirs(GEN, exist_ok=True)
    vs = variants()
    for v in vs:
        compile_one(*v)
    lines = [
        "// 自動生成: tools/gen_shaders.py。手で編集しない。",
        f"pub const EW_PUSH_SIZE: u32 = {PUSH_SIZE};",
        "pub const GEN_KERNELS: &[(&str, &[u8], u32)] = &[",
    ]
    for name, *_ in vs:
        lines.append(f'    ("{name}", include_bytes!("../shaders/gen/{name}.spv"), {PUSH_SIZE}),')
    lines.append("];")
    with open(os.path.join(ROOT, "vkrt", "src", "kernels_gen.rs"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"generated {len(vs)} kernels")


if __name__ == "__main__":
    sys.exit(main())
