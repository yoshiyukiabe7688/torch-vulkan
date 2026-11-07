// 行単位カーネル(softmax / layer_norm / reduce)共通の型ヘルパ。KIND: 0=f32, 1=bf16, 2=f16（入出力テンソルの dtype）。
// 計算は常に fp32。mean / rstd などの補助出力は常に f32（ldF / stF）。
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(buffer_reference, scalar, buffer_reference_align = 4) buffer RF32 { float v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer RBF16 { uint16_t v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer RF16 { uint16_t v[]; };
// fp16 は 16 ビット整数で保存し、変換は GLSL 標準の pack/unpackHalf2x16 で行う（shaderFloat16 機能が不要。
// RX 550(Polaris)・GTX 1060(Pascal) は shaderFloat16 を持たない）。
float h2f(uint16_t h) { return unpackHalf2x16(uint(h)).x; }
// f32 -> f16 は最近接偶数丸め(RTE)を自前で行う。packHalf2x16 は丸めが実装依存で、AMD(RADV/Windows 版とも)では
// 切り捨て(RTZ)になり CPU と 1 ulp ずれる（実測: ランダム入力の約半数がビット不一致）。
uint16_t f2h(float f) {
    uint u = floatBitsToUint(f);
    uint sign = (u >> 16) & 0x8000u;
    uint e = (u >> 23) & 0xFFu;
    uint m = u & 0x7FFFFFu;
    if (e == 0xFFu) return uint16_t(sign | 0x7C00u | (m != 0u ? (0x200u | (m >> 13)) : 0u));  // inf / NaN
    int ne = int(e) - 127 + 15;
    if (ne >= 31) return uint16_t(sign | 0x7C00u);  // オーバーフロー -> inf
    if (ne <= 0) {                                   // 非正規化数 or 0
        if (ne < -10) return uint16_t(sign);
        m |= 0x800000u;
        uint shift = uint(14 - ne);
        uint hm = m >> shift;
        uint rem = m & ((1u << shift) - 1u);
        uint halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (hm & 1u) != 0u)) hm++;
        return uint16_t(sign | hm);
    }
    uint hm = m >> 13;
    uint rem = m & 0x1FFFu;
    uint h = sign | (uint(ne) << 10) | hm;
    if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u) != 0u)) h++;  // 繰り上がりは指数部へ伝播してよい
    return uint16_t(h);
}


float bf2f(uint16_t v) { return uintBitsToFloat(uint(v) << 16); }
uint16_t f2bf(float x) {
    if (isnan(x)) return uint16_t(0x7FC0u);
    uint u = floatBitsToUint(x);
    u += 0x7FFFu + ((u >> 16) & 1u);  // 最近接偶数丸め
    return uint16_t(u >> 16);
}

#if KIND == 0
float ldK(uint64_t base, uint i) { return RF32(base).v[i]; }
void stK(uint64_t base, uint i, float x) { RF32(base).v[i] = x; }
#elif KIND == 1
float ldK(uint64_t base, uint i) { return bf2f(RBF16(base).v[i]); }
void stK(uint64_t base, uint i, float x) { RBF16(base).v[i] = f2bf(x); }
#else
float ldK(uint64_t base, uint i) { return h2f(RF16(base).v[i]); }
void stK(uint64_t base, uint i, float x) { RF16(base).v[i] = f2h(x); }
#endif

float ldF(uint64_t base, uint i) { return RF32(base).v[i]; }
void stF(uint64_t base, uint i, float x) { RF32(base).v[i] = x; }

// 256 スレッドのワークグループ内リダクション（共有メモリのみ。サブグループ演算は使わない）。
shared float red_buf[256];

float block_sum(float v) {
    uint t = gl_LocalInvocationID.x;
    red_buf[t] = v;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1) {
        if (t < s) red_buf[t] += red_buf[t + s];
        barrier();
    }
    float r = red_buf[0];
    barrier();
    return r;
}

float block_max(float v) {
    uint t = gl_LocalInvocationID.x;
    red_buf[t] = v;
    barrier();
    for (uint s = 128u; s > 0u; s >>= 1) {
        if (t < s) red_buf[t] = max(red_buf[t], red_buf[t + s]);
        barrier();
    }
    float r = red_buf[0];
    barrier();
    return r;
}
