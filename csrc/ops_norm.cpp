// group_norm の forward / backward（カーネル本体は vkrt/shaders/gn.comp.in）。
// 入力は連続な NCHW。非連続なら先に GPU 上で連続化する。mean / rstd は入力と同じ dtype・形状 [N, G]（PyTorch CPU と同じ）。
// backward は grad_input を専用カーネルで、grad_weight / grad_bias をチャネルごとのカーネルで計算する。
#include <ATen/Operators.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

#include "common.h"

namespace {

using namespace vkt;

const char* fkind(at::ScalarType t) {
  switch (t) {
    case at::kFloat: return "f32";
    case at::kBFloat16: return "bf16";
    case at::kHalf: return "f16";
    default: return nullptr;
  }
}

at::Tensor contig(const at::Tensor& t) { return t.is_contiguous() ? t : t.contiguous(); }

struct GnPush {
  uint64_t x, y, g, mean, rstd, w, b;
  uint32_t M, HW, Cg, G, C, N;
  float eps;
  uint32_t pad;
};
static_assert(sizeof(GnPush) == 88, "gn.comp.in の push constant は 88 バイト");

void dispatch(const std::string& kernel, const GnPush& P, uint64_t groups, const char* aten_name) {
  TORCH_CHECK(groups < (uint64_t(1) << 32), "vk: dispatch が大きすぎます");
  vk_check(vkrt_dispatch((uint32_t)kernel_index(kernel), &P, sizeof(P), (uint32_t)groups, 1, 1), kernel.c_str());
  count_kernel(aten_name);
}

// 形状の検証。通れば true で P の形状フィールドを埋める。
bool gn_shape_ok(const at::Tensor& input, int64_t N, int64_t C, int64_t HW, int64_t group, GnPush& P) {
  if (!is_vk(input) || !fkind(input.scalar_type())) return false;
  if (N <= 0 || C <= 0 || HW <= 0 || group <= 0 || C % group != 0) return false;
  if (input.numel() != N * C * HW || input.numel() >= (int64_t(1) << 32)) return false;
  P.M = (uint32_t)((C / group) * HW);
  P.HW = (uint32_t)HW;
  P.Cg = (uint32_t)(C / group);
  P.G = (uint32_t)group;
  P.C = (uint32_t)C;
  P.N = (uint32_t)N;
  return true;
}

// weight / bias は入力と同じ dtype で読む。違う dtype なら GPU 上で変換する。undefined / 無しなら空のテンソル。
// 要素数が C でない・vk 上でないものは ok=false。
at::Tensor affine_param(const std::optional<at::Tensor>& p, const at::Tensor& input, int64_t C, bool& ok) {
  if (!p.has_value() || !p->defined()) return at::Tensor();
  if (!is_vk(*p) || p->numel() != C) {
    ok = false;
    return at::Tensor();
  }
  at::Tensor t = p->scalar_type() == input.scalar_type() ? *p : p->to(input.scalar_type());
  return contig(t);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> k_native_group_norm(const at::Tensor& input, const std::optional<at::Tensor>& weight,
                                                                   const std::optional<at::Tensor>& bias, c10::SymInt N_, c10::SymInt C_,
                                                                   c10::SymInt HxW_, int64_t group, double eps) {
  int64_t N = N_.expect_int(), C = C_.expect_int(), HW = HxW_.expect_int();
  GnPush P{};
  bool ok = gn_shape_ok(input, N, C, HW, group, P);
  at::Tensor w, b;
  if (ok) {
    w = affine_param(weight, input, C, ok);
    b = affine_param(bias, input, C, ok);
  }
  if (ok) {
    const char* k = fkind(input.scalar_type());
    at::Tensor x = contig(input);
    at::Tensor y = at::empty(x.sizes(), x.options());
    at::Tensor mean = at::empty({N, group}, x.options());
    at::Tensor rstd = at::empty({N, group}, x.options());
    P.x = addr_of(x);
    P.y = addr_of(y);
    P.mean = addr_of(mean);
    P.rstd = addr_of(rstd);
    P.w = w.defined() ? addr_of(w) : 0;
    P.b = b.defined() ? addr_of(b) : 0;
    P.eps = (float)eps;
    dispatch(std::string("gn_fwd_") + k, P, (uint64_t)N * group, "aten::native_group_norm");
    return {y, mean, rstd};
  }
  return VKT_FALLBACK_SYMINT(native_group_norm, input, weight, bias, N_, C_, HxW_, group, eps);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> k_native_group_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input, const at::Tensor& mean, const at::Tensor& rstd,
    const std::optional<at::Tensor>& weight, c10::SymInt N_, c10::SymInt C_, c10::SymInt HxW_, int64_t group,
    std::array<bool, 3> output_mask) {
  int64_t N = N_.expect_int(), C = C_.expect_int(), HW = HxW_.expect_int();
  GnPush P{};
  bool ok = gn_shape_ok(input, N, C, HW, group, P) && is_vk(grad_out) && grad_out.scalar_type() == input.scalar_type() &&
            grad_out.numel() == input.numel() && is_vk(mean) && is_vk(rstd) && mean.numel() == N * group && rstd.numel() == N * group &&
            fkind(mean.scalar_type()) && fkind(rstd.scalar_type());
  at::Tensor w;
  if (ok) w = affine_param(weight, input, C, ok);
  if (ok) {
    const char* k = fkind(input.scalar_type());
    at::Tensor g = contig(grad_out), x = contig(input);
    auto stat = [&](const at::Tensor& t) { return contig(t.scalar_type() == input.scalar_type() ? t : t.to(input.scalar_type())); };
    at::Tensor mu = stat(mean), rs = stat(rstd);
    P.x = addr_of(x);
    P.g = addr_of(g);
    P.mean = addr_of(mu);
    P.rstd = addr_of(rs);
    P.w = w.defined() ? addr_of(w) : 0;
    at::Tensor dx, dw, db;
    if (output_mask[0]) {
      dx = at::empty(x.sizes(), x.options());
      P.y = addr_of(dx);
      dispatch(std::string("gn_bwd_") + k, P, (uint64_t)N * group, "aten::native_group_norm_backward");
    }
    if (output_mask[1] || output_mask[2]) {
      // weight / bias の勾配（dtype は weight と同じ。weight が無ければ入力と同じ）
      at::ScalarType wdt = (weight.has_value() && weight->defined()) ? weight->scalar_type() : input.scalar_type();
      at::Tensor dw_k, db_k;
      if (output_mask[1]) dw_k = at::empty({C}, x.options());
      if (output_mask[2]) db_k = at::empty({C}, x.options());
      P.y = output_mask[1] ? addr_of(dw_k) : 0;
      P.b = output_mask[2] ? addr_of(db_k) : 0;
      dispatch(std::string("gn_wgrad_") + k, P, (uint64_t)C, "aten::native_group_norm_backward");
      if (output_mask[1]) dw = wdt == input.scalar_type() ? dw_k : dw_k.to(wdt);
      if (output_mask[2]) db = wdt == input.scalar_type() ? db_k : db_k.to(wdt);
    }
    return {dx, dw, db};
  }
  return VKT_FALLBACK_SYMINT(native_group_norm_backward, grad_out, input, mean, rstd, weight, N_, C_, HxW_, group, output_mask);
}

}  // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("native_group_norm", &k_native_group_norm);
  m.impl("native_group_norm_backward", &k_native_group_norm_backward);
}
