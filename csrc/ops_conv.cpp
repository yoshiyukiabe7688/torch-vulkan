// conv2d（forward / backward）と nearest upsample。
//   conv: im2col（vkrt/shaders/im2col.comp.in）+ 既存の mm（協調行列 / タイル型）。
//   対象: NCHW・groups=1・dilation=1・非 transposed・f32/bf16/f16（入力と重みは同 dtype）。それ以外は CPU フォールバック。
//   1x1・stride 1・padding 0 は im2col を省く（入力がそのまま cols）。
//   backward は入力の勾配（col2im は gather 式）と重みの勾配（mm）、bias の勾配（sum）。
#include <ATen/Operators.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "common.h"

namespace {

using namespace vkt;
using IntVec = std::vector<int64_t>;

const char* fkind(at::ScalarType t) {
  switch (t) {
    case at::kFloat: return "f32";
    case at::kBFloat16: return "bf16";
    case at::kHalf: return "f16";
    default: return nullptr;
  }
}

struct Im2colPush {
  uint64_t x, cols;
  uint32_t N, C, H, W, OH, OW, KH, KW, SH, SW, PH, PW, total, pad;
};
static_assert(sizeof(Im2colPush) == 72, "im2col.comp.in の push constant は 72 バイト");

struct UpPush {
  uint64_t src, dst;
  uint32_t NC, H, W, OH, OW, total;
};
static_assert(sizeof(UpPush) == 40, "upsample.comp.in の push constant は 40 バイト");

// 2 次元ディスパッチ（gx <= 65535）。カーネル側は idx = (gy * gridDim.x + gx) * 256 + local。
void dispatch_1d(const std::string& kernel, const void* push, uint32_t push_size, uint64_t total, const char* aten_name) {
  uint64_t groups = (total + 255) / 256;
  uint64_t gx = std::min<uint64_t>(groups, 65535);
  uint64_t gy = (groups + gx - 1) / gx;
  TORCH_CHECK(gy < 65536, "vk: dispatch が大きすぎます");
  vk_check(vkrt_dispatch((uint32_t)kernel_index(kernel), push, push_size, (uint32_t)gx, (uint32_t)gy, 1), kernel.c_str());
  count_kernel(aten_name);
}

struct ConvGeom {
  int64_t N, C, H, W, Co, KH, KW, SH, SW, PH, PW, OH, OW;
  bool direct;  // 1x1・stride 1・padding 0: im2col 不要
  int64_t K() const { return C * KH * KW; }
  int64_t L() const { return OH * OW; }
};

bool conv_geom(const at::Tensor& input, const at::Tensor& weight, const IntVec& stride, const IntVec& padding, const IntVec& dilation,
               bool transposed, int64_t groups, ConvGeom& g) {
  if (transposed || groups != 1) return false;
  if (!is_vk(input) || !is_vk(weight)) return false;
  if (input.dim() != 4 || weight.dim() != 4) return false;
  if (!fkind(input.scalar_type()) || input.scalar_type() != weight.scalar_type()) return false;
  if (stride.size() != 2 || padding.size() != 2 || dilation.size() != 2) return false;
  if (dilation[0] != 1 || dilation[1] != 1) return false;
  if (stride[0] < 1 || stride[1] < 1 || padding[0] < 0 || padding[1] < 0) return false;
  g.N = input.size(0); g.C = input.size(1); g.H = input.size(2); g.W = input.size(3);
  g.Co = weight.size(0); g.KH = weight.size(2); g.KW = weight.size(3);
  if (weight.size(1) != g.C) return false;
  g.SH = stride[0]; g.SW = stride[1]; g.PH = padding[0]; g.PW = padding[1];
  int64_t oh = g.H + 2 * g.PH - g.KH, ow = g.W + 2 * g.PW - g.KW;
  if (oh < 0 || ow < 0) return false;
  g.OH = oh / g.SH + 1; g.OW = ow / g.SW + 1;
  g.direct = g.KH == 1 && g.KW == 1 && g.SH == 1 && g.SW == 1 && g.PH == 0 && g.PW == 0;
  int64_t cols_elems = g.N * g.K() * g.L();
  int64_t gx_elems = g.N * g.C * g.H * g.W;
  if (cols_elems >= (int64_t(1) << 32) || gx_elems >= (int64_t(1) << 32) || g.N * g.Co * g.L() >= (int64_t(1) << 32)) return false;
  if (g.N * g.C * g.H * g.W == 0 || g.Co == 0) return false;
  return true;
}

// 入力から cols [N, K, L] を作る。direct のときは入力のビュー。
at::Tensor make_cols(const at::Tensor& input, const ConvGeom& g) {
  at::Tensor x = input.is_contiguous() ? input : input.contiguous();
  if (g.direct) return x.view({g.N, g.C, g.H * g.W});
  at::Tensor cols = at::empty({g.N, g.K(), g.L()}, x.options());
  Im2colPush P{addr_of(x), addr_of(cols), (uint32_t)g.N, (uint32_t)g.C, (uint32_t)g.H, (uint32_t)g.W, (uint32_t)g.OH, (uint32_t)g.OW,
               (uint32_t)g.KH, (uint32_t)g.KW, (uint32_t)g.SH, (uint32_t)g.SW, (uint32_t)g.PH, (uint32_t)g.PW,
               (uint32_t)(g.N * g.K() * g.L()), 0};
  dispatch_1d(std::string("im2col_") + fkind(x.scalar_type()), &P, sizeof(P), (uint64_t)P.total, "aten::convolution_overrideable");
  return cols;
}

// out[n] = A @ B[n]（A: [M, K]、B: [N, K, L]）。N==1 は mm、それ以外は bmm（A を expand）。
at::Tensor left_matmul(const at::Tensor& A, const at::Tensor& B) {
  int64_t N = B.size(0);
  if (N == 1) return at::mm(A, B.select(0, 0)).unsqueeze(0);
  return at::bmm(A.unsqueeze(0).expand({N, A.size(0), A.size(1)}), B);
}

at::Tensor conv_forward_impl(const at::Tensor& input, const at::Tensor& weight, const std::optional<at::Tensor>& bias, const ConvGeom& g) {
  at::Tensor cols = make_cols(input, g);
  at::Tensor w2 = (weight.is_contiguous() ? weight : weight.contiguous()).view({g.Co, g.K()});
  at::Tensor out = left_matmul(w2, cols).view({g.N, g.Co, g.OH, g.OW});  // [N, Co, L] -> NCHW
  if (bias.has_value() && bias->defined()) out = at::add(out, bias->view({1, g.Co, 1, 1}));
  return out;
}

IntVec to_ints(c10::SymIntArrayRef a) {
  IntVec v;
  for (const auto& s : a) v.push_back(s.guard_int(__FILE__, __LINE__));
  return v;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> conv_backward_cpu(const at::Tensor& grad_out, const at::Tensor& input, const at::Tensor& weight,
                                                                 const IntVec& stride, const IntVec& padding, const IntVec& dilation,
                                                                 bool transposed, const IntVec& output_padding, int64_t groups,
                                                                 std::array<bool, 3> mask) {
  count_fallback("aten::convolution_backward_overrideable");
  auto r = at::convolution_backward(grad_out.to(at::kCPU), input.to(at::kCPU), weight.to(at::kCPU), c10::nullopt, stride, padding, dilation,
                                    transposed, output_padding, groups, mask);
  auto mv = [&](const at::Tensor& t, const at::Tensor& like) { return t.defined() ? t.to(like.device()) : at::Tensor(); };
  return {mv(std::get<0>(r), input), mv(std::get<1>(r), weight), mv(std::get<2>(r), weight)};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> k_convolution_backward_overrideable(
    const at::Tensor& grad_output, const at::Tensor& input, const at::Tensor& weight, c10::SymIntArrayRef stride_, c10::SymIntArrayRef padding_,
    c10::SymIntArrayRef dilation_, bool transposed, c10::SymIntArrayRef output_padding_, c10::SymInt groups_, std::array<bool, 3> output_mask) {
  IntVec stride = to_ints(stride_), padding = to_ints(padding_), dilation = to_ints(dilation_), output_padding = to_ints(output_padding_);
  int64_t groups = groups_.guard_int(__FILE__, __LINE__);
  ConvGeom g;
  if (!conv_geom(input, weight, stride, padding, dilation, transposed, groups, g) || !is_vk(grad_output) ||
      grad_output.scalar_type() != input.scalar_type() || grad_output.dim() != 4 || grad_output.size(1) != g.Co ||
      grad_output.size(2) != g.OH || grad_output.size(3) != g.OW || grad_output.size(0) != g.N) {
    return conv_backward_cpu(grad_output, input, weight, stride, padding, dilation, transposed, output_padding, groups, output_mask);
  }
  at::Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
  at::Tensor G = go.view({g.N, g.Co, g.L()});
  at::Tensor w2 = (weight.is_contiguous() ? weight : weight.contiguous()).view({g.Co, g.K()});
  at::Tensor gi, gw, gb;
  if (output_mask[0]) {
    at::Tensor gcols = left_matmul(w2.t(), G);  // [N, K, L]
    if (g.direct) {
      gi = gcols.view({g.N, g.C, g.H, g.W});
    } else {
      gi = at::empty({g.N, g.C, g.H, g.W}, go.options());
      at::Tensor gc = gcols.is_contiguous() ? gcols : gcols.contiguous();
      Im2colPush P{addr_of(gi), addr_of(gc), (uint32_t)g.N, (uint32_t)g.C, (uint32_t)g.H, (uint32_t)g.W, (uint32_t)g.OH, (uint32_t)g.OW,
                   (uint32_t)g.KH, (uint32_t)g.KW, (uint32_t)g.SH, (uint32_t)g.SW, (uint32_t)g.PH, (uint32_t)g.PW,
                   (uint32_t)(g.N * g.C * g.H * g.W), 0};
      dispatch_1d(std::string("col2im_") + fkind(go.scalar_type()), &P, sizeof(P), (uint64_t)P.total, "aten::convolution_backward_overrideable");
    }
  }
  if (output_mask[1]) {
    at::Tensor cols = make_cols(input, g);  // [N, K, L]
    at::Tensor gw2;
    if (g.N == 1) {
      gw2 = at::mm(G.select(0, 0), cols.select(0, 0).t());
    } else {
      gw2 = at::bmm(G, cols.transpose(1, 2)).sum(0);
    }
    gw = gw2.reshape(weight.sizes());
  }
  if (output_mask[2]) gb = at::sum(go, IntVec{0, 2, 3});
  return {gi, gw, gb};
}

// ---- nearest upsample --------------------------------------------------------------------------------------
// 整数倍（OH % H == 0 && OW % W == 0）のみ GPU。それ以外は CPU フォールバック。
bool up_ok(const at::Tensor& t, int64_t H, int64_t W, int64_t OH, int64_t OW) {
  return is_vk(t) && fkind(t.scalar_type()) && t.dim() == 4 && H > 0 && W > 0 && OH % H == 0 && OW % W == 0 && OH >= H && OW >= W &&
         t.numel() < (int64_t(1) << 32) && t.size(0) * t.size(1) * OH * OW < (int64_t(1) << 32);
}

at::Tensor k_upsample_nearest2d(const at::Tensor& input, c10::SymIntArrayRef output_size, std::optional<double> scales_h,
                                std::optional<double> scales_w) {
  IntVec os = to_ints(output_size);
  if (os.size() == 2 && input.dim() == 4 && up_ok(input, input.size(2), input.size(3), os[0], os[1])) {
    at::Tensor x = input.is_contiguous() ? input : input.contiguous();
    int64_t NC = x.size(0) * x.size(1);
    at::Tensor out = at::empty({x.size(0), x.size(1), os[0], os[1]}, x.options());
    UpPush P{addr_of(x), addr_of(out), (uint32_t)NC, (uint32_t)x.size(2), (uint32_t)x.size(3), (uint32_t)os[0], (uint32_t)os[1],
             (uint32_t)(NC * os[0] * os[1])};
    dispatch_1d(std::string("up_fwd_") + fkind(x.scalar_type()), &P, sizeof(P), (uint64_t)P.total, "aten::upsample_nearest2d");
    return out;
  }
  return VKT_FALLBACK_SYMINT(upsample_nearest2d, input, output_size, scales_h, scales_w);
}

at::Tensor k_upsample_nearest2d_backward(const at::Tensor& grad_output, c10::SymIntArrayRef output_size, c10::SymIntArrayRef input_size,
                                         std::optional<double> scales_h, std::optional<double> scales_w) {
  IntVec os = to_ints(output_size), is = to_ints(input_size);
  if (os.size() == 2 && is.size() == 4 && grad_output.dim() == 4 && up_ok(grad_output, is[2], is[3], os[0], os[1]) &&
      grad_output.size(2) == os[0] && grad_output.size(3) == os[1]) {
    at::Tensor go = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();
    int64_t NC = is[0] * is[1];
    at::Tensor gx = at::empty({is[0], is[1], is[2], is[3]}, go.options());
    UpPush P{addr_of(go), addr_of(gx), (uint32_t)NC, (uint32_t)is[2], (uint32_t)is[3], (uint32_t)os[0], (uint32_t)os[1], (uint32_t)(NC * is[2] * is[3])};
    dispatch_1d(std::string("up_bwd_") + fkind(go.scalar_type()), &P, sizeof(P), (uint64_t)P.total, "aten::upsample_nearest2d_backward");
    return gx;
  }
  return VKT_FALLBACK_SYMINT(upsample_nearest2d_backward, grad_output, output_size, input_size, scales_h, scales_w);
}

}  // namespace

namespace vkt {

// convolution_overrideable（shim.cpp）から呼ぶ。GPU で実行できない形なら nullopt。
std::optional<at::Tensor> conv_forward_gpu(const at::Tensor& input, const at::Tensor& weight, const std::optional<at::Tensor>& bias,
                                           const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                                           const std::vector<int64_t>& dilation, bool transposed, int64_t groups) {
  ConvGeom g;
  if (!conv_geom(input, weight, stride, padding, dilation, transposed, groups, g)) return std::nullopt;
  if (bias.has_value() && bias->defined() && (!is_vk(*bias) || bias->scalar_type() != input.scalar_type() || bias->numel() != g.Co)) return std::nullopt;
  return conv_forward_impl(input, weight, bias, g);
}

}  // namespace vkt

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("convolution_backward_overrideable", &k_convolution_backward_overrideable);
  m.impl("upsample_nearest2d", &k_upsample_nearest2d);
  m.impl("upsample_nearest2d_backward", &k_upsample_nearest2d_backward);
}
