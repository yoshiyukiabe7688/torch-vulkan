// シム全体で共有するヘルパ。実装は shim.cpp（namespace vkt）。
#pragma once

#include <ATen/ATen.h>
#include <ATen/native/CPUFallback.h>
#include <torch/library.h>

#include <string>

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
uint32_t coop_caps();

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
