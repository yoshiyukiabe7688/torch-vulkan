// 行単位カーネル(softmax / layer_norm / reduce)共通の型ヘルパ。KIND: 0=f32, 1=bf16, 2=f16（入出力テンソルの dtype）。
// 計算は常に fp32。mean / rstd などの補助出力は常に f32（ldF / stF）。
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_16bit_storage : require

layout(buffer_reference, scalar, buffer_reference_align = 4) buffer RF32 { float v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer RBF16 { uint16_t v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 2) buffer RF16 { float16_t v[]; };

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
float ldK(uint64_t base, uint i) { return float(RF16(base).v[i]); }
void stK(uint64_t base, uint i, float x) { RF16(base).v[i] = float16_t(x); }
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
