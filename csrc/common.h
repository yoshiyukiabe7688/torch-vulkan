// シム全体で共有するヘルパ。実装は shim.cpp（namespace vkt）。
#pragma once

#include <ATen/ATen.h>
#include <ATen/native/CPUFallback.h>
#include <torch/library.h>

#include <optional>
#include <string>
#include <vector>

#include "vkrt.h"

namespace vkt {

constexpr c10::DeviceType kDev = c10::DeviceType::PrivateUse1;

std::string last_error();
void vk_check(int rc, const char* what);
void ensure_init();

// 統計: CPU フォールバックした演算 / GPU カーネルで実行した演算の回数
void count_fallback(const std::string& name);
void count_kernel(const std::string& name);

// 未実装の演算を CPU で実行する（boxed）。型付きカーネルからは call_fallback_fn 経由で呼ぶ。
void vulkan_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack);

inline bool is_vk(const at::Tensor& t) { return t.defined() && t.device().type() == kDev; }
inline uint64_t addr_of(const at::Tensor& t) { return reinterpret_cast<uint64_t>(t.data_ptr()); }

// デバイス上の dtype 変換つきストライドコピー（ops_ew.cpp）。非対応の組み合わせなら false。
bool ew_copy(const at::Tensor& dst, const at::Tensor& src);

// 協調行列カーネルの有効/無効切替と対応状況(bit0=f16, bit1=bf16)（ops_mm.cpp）
void set_coop_enabled(bool on);
bool coop_is_enabled();

// 融合アテンションの方針（0=使わない、1=常に使う、2=auto）。ops_attn.cpp
void set_fused_attention(int mode);
uint32_t coop_caps();
void set_coop_range(uint64_t lo, uint64_t hi, int kind);
void reset_coop_calls();
uint64_t coop_calls();

// conv2d の forward を GPU（im2col + mm）で実行する（ops_conv.cpp）。対応しない形なら nullopt。
std::optional<at::Tensor> conv_forward_gpu(const at::Tensor& input, const at::Tensor& weight, const std::optional<at::Tensor>& bias,
                                           const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                                           const std::vector<int64_t>& dilation, bool transposed, int64_t groups);

// 重みのみ INT8 の dequant（ops_quant.cpp）。q: int8 [rows, cols]、scale: bf16 [rows, ceil(cols/64)]（64 要素ブロック）。
at::Tensor dequant_int8_vk(const at::Tensor& q, const at::Tensor& scale, at::ScalarType out_dtype);

// カーネル名 -> 番号（キャッシュ付き）。無ければ例外。
int kernel_index(const std::string& name);

// 型付きカーネルから CPU フォールバックへ戻すためのショートカット
//   return VKT_FALLBACK(mul_Tensor, a, b);
#define VKT_FALLBACK(op, ...) \
  at::native::call_fallback_fn<&vkt::vulkan_fallback, at::_ops::op>::call(__VA_ARGS__)

// SymInt 引数を持つ演算用（native_layer_norm など）
#define VKT_FALLBACK_SYMINT(op, ...) \
  at::native::call_fallback_fn_symint<&vkt::vulkan_fallback, at::_ops::op>::call(__VA_ARGS__)

}  // namespace vkt
