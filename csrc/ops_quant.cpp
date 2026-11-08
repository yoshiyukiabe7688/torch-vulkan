// 重みのみ INT8 の dequant（vkrt/shaders/dq8.comp.in）。out = float(q) * float(scale) を out_dtype に丸める。
//   q: int8 [rows, cols] 連続、scale: bf16 [rows, ceil(cols / 64)] 連続（各行を 64 要素ずつのブロックに分ける）。
#include <ATen/Operators.h>

#include <algorithm>
#include <cstdint>
#include <string>

#include "common.h"

namespace {

using namespace vkt;

struct Dq8Push {
  uint64_t q, scale, out;
  uint32_t total, cols, nblocks, flags;
};
static_assert(sizeof(Dq8Push) == 40, "dq8.comp.in の push constant は 40 バイト");

const char* okind(at::ScalarType t) {
  switch (t) {
    case at::kFloat: return "f32";
    case at::kBFloat16: return "bf16";
    case at::kHalf: return "f16";
    default: return nullptr;
  }
}

}  // namespace

namespace vkt {

at::Tensor dequant_int8_vk(const at::Tensor& q, const at::Tensor& scale, at::ScalarType out_dtype) {
  TORCH_CHECK(is_vk(q) && is_vk(scale), "dequant_int8: q と scale は vk 上のテンソルであること");
  TORCH_CHECK(q.scalar_type() == at::kChar && q.dim() == 2 && q.is_contiguous(), "dequant_int8: q は連続な int8 の 2 次元テンソル");
  TORCH_CHECK(scale.scalar_type() == at::kBFloat16 && scale.dim() == 2 && scale.is_contiguous(),
              "dequant_int8: scale は連続な bf16 の 2 次元テンソル");
  const char* k = okind(out_dtype);
  TORCH_CHECK(k != nullptr, "dequant_int8: out_dtype は float32 / bfloat16 / float16");
  int64_t rows = q.size(0), cols = q.size(1);
  int64_t nblocks = (cols + 63) / 64;
  TORCH_CHECK(scale.size(0) == rows && scale.size(1) == nblocks, "dequant_int8: scale の形が [rows, ceil(cols/64)] と合わない");
  int64_t total = rows * cols;
  TORCH_CHECK(total < (int64_t(1) << 32), "dequant_int8: 要素数が大きすぎます");
  at::Tensor out = at::empty({rows, cols}, q.options().dtype(out_dtype));
  if (total == 0) return out;
  uint32_t flags = (cols % 64 == 0 ? 1u : 0u) | (total % 4 == 0 ? 2u : 0u);
  Dq8Push P{addr_of(q), addr_of(scale), addr_of(out), (uint32_t)total, (uint32_t)cols, (uint32_t)nblocks, flags};
  uint64_t threads = ((uint64_t)total + 3) / 4;
  uint64_t groups = (threads + 255) / 256;
  uint64_t gx = std::min<uint64_t>(groups, 65535);
  uint64_t gy = (groups + gx - 1) / gx;
  TORCH_CHECK(gy < 65536, "vk: dispatch が大きすぎます");
  std::string kernel = std::string("dq8_") + k;
  vk_check(vkrt_dispatch((uint32_t)kernel_index(kernel), &P, sizeof(P), (uint32_t)gx, (uint32_t)gy, 1), kernel.c_str());
  count_kernel("torch_vulkan::dequant_int8");
  return out;
}

}  // namespace vkt
