// 行単位の演算: softmax(+backward) / layer_norm(+backward) / sum・mean / cat。
// カーネル本体は vkrt/shaders/{softmax,ln,reduce}.comp.in。入力が非連続なら先に GPU 上で連続化する。
#include <ATen/Operators.h>

#include <algorithm>
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

at::Tensor contig(const at::Tensor& t) { return t.is_contiguous() ? t : t.contiguous(); }

void dispatch(const std::string& kernel, const void* push, uint32_t push_size, uint64_t gx, uint64_t gy, const char* aten_name) {
  TORCH_CHECK(gx < (uint64_t(1) << 32) && gy < (uint64_t(1) << 32), "vk: dispatch が大きすぎます");
  vk_check(vkrt_dispatch((uint32_t)kernel_index(kernel), push, push_size, (uint32_t)gx, (uint32_t)gy, 1), kernel.c_str());
  count_kernel(aten_name);
}

// ---- softmax ---------------------------------------------------------------------------------------------
struct SoftmaxPush {
  uint64_t x, y, g;
  uint32_t outer, R, inner, safe;
};
static_assert(sizeof(SoftmaxPush) == 40, "softmax.comp.in の push constant は 40 バイト");

bool softmax_dims(const at::Tensor& t, int64_t dim, uint32_t& outer, uint32_t& R, uint32_t& inner) {
  if (t.dim() == 0) {
    outer = 1; R = 1; inner = 1;
    return true;
  }
  dim = at::maybe_wrap_dim(dim, t.dim());
  int64_t o = 1, i = 1;
  for (int64_t d = 0; d < dim; ++d) o *= t.size(d);
  for (int64_t d = dim + 1; d < t.dim(); ++d) i *= t.size(d);
  int64_t r = t.size(dim);
  if (o * i >= (int64_t(1) << 32) || t.numel() >= (int64_t(1) << 32)) return false;
  outer = (uint32_t)o; R = (uint32_t)r; inner = (uint32_t)i;
  return true;
}

// softmax の共通実装。入力を dtype へ（必要なら GPU 上で）変換してから行カーネルを実行する。
std::optional<at::Tensor> try_softmax(const at::Tensor& self, int64_t dim, std::optional<at::ScalarType> out_dtype, bool safe,
                                      const char* aten_name) {
  if (!is_vk(self) || self.numel() == 0) return std::nullopt;
  at::ScalarType dt = out_dtype.value_or(self.scalar_type());
  const char* k = fkind(dt);
  if (!k || !fkind(self.scalar_type())) return std::nullopt;
  uint32_t outer, R, inner;
  if (!softmax_dims(self, dim, outer, R, inner)) return std::nullopt;
  at::Tensor x = contig(self.scalar_type() == dt ? self : self.to(dt));
  at::Tensor out = at::empty(x.sizes(), x.options());
  SoftmaxPush P{addr_of(x), addr_of(out), 0, outer, R, inner, safe ? 1u : 0u};
  dispatch(std::string("softmax_fwd_") + k, &P, sizeof(P), (uint64_t)outer * inner, 1, aten_name);
  return out;
}

at::Tensor k_softmax(const at::Tensor& self, int64_t dim, bool half_to_float) {
  // half_to_float: 半精度の入力を fp32 で計算して fp32 を返す
  std::optional<at::ScalarType> dt = half_to_float ? std::optional<at::ScalarType>(at::kFloat) : std::nullopt;
  if (auto r = try_softmax(self, dim, dt, false, "aten::_softmax")) return *r;
  return VKT_FALLBACK(_softmax, self, dim, half_to_float);
}

at::Tensor k_safe_softmax(const at::Tensor& self, int64_t dim, std::optional<at::ScalarType> dtype) {
  if (auto r = try_softmax(self, dim, dtype, true, "aten::_safe_softmax")) return *r;
  return VKT_FALLBACK(_safe_softmax, self, dim, dtype);
}

at::Tensor k_softmax_backward(const at::Tensor& g, const at::Tensor& y, int64_t dim, at::ScalarType input_dtype) {
  const char* k = (is_vk(g) && is_vk(y)) ? fkind(y.scalar_type()) : nullptr;
  uint32_t outer, R, inner;
  if (k && fkind(g.scalar_type()) && fkind(input_dtype) && g.sizes() == y.sizes() && y.numel() > 0 &&
      softmax_dims(y, dim, outer, R, inner)) {
    // 計算は y の dtype で行い、結果を input_dtype（元の入力の dtype）へ変換する
    at::Tensor gc = contig(g.scalar_type() == y.scalar_type() ? g : g.to(y.scalar_type()));
    at::Tensor yc = contig(y);
    at::Tensor dx = at::empty(yc.sizes(), yc.options());
    SoftmaxPush P{addr_of(dx), addr_of(yc), addr_of(gc), outer, R, inner, 0};
    dispatch(std::string("softmax_bwd_") + k, &P, sizeof(P), (uint64_t)outer * inner, 1, "aten::_softmax_backward_data");
    return input_dtype == dx.scalar_type() ? dx : dx.to(input_dtype);
  }
  return VKT_FALLBACK(_softmax_backward_data, g, y, dim, input_dtype);
}

// ---- layer_norm ---------------------------------------------------------------------------------------------
struct LnPush {
  uint64_t x, y, g, mean, rstd, w, b;
  uint32_t rows, D;
  float eps;
  uint32_t pad;
};
static_assert(sizeof(LnPush) == 72, "ln.comp.in の push constant は 72 バイト");

int64_t prod(c10::IntArrayRef s) {
  int64_t n = 1;
  for (auto v : s) n *= v;
  return n;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> k_native_layer_norm(const at::Tensor& input, c10::SymIntArrayRef normalized_shape_sym,
                                                                   const std::optional<at::Tensor>& weight,
                                                                   const std::optional<at::Tensor>& bias, double eps) {
  auto normalized_shape = C10_AS_INTARRAYREF_SLOW(normalized_shape_sym);
  const char* k = is_vk(input) ? fkind(input.scalar_type()) : nullptr;
  int64_t D = prod(normalized_shape);
  bool ok = k && D > 0 && input.numel() > 0 && (int64_t)normalized_shape.size() <= input.dim() &&
            input.numel() < (int64_t(1) << 32);
  auto param_ok = [&](const std::optional<at::Tensor>& p) {
    return !p.has_value() || !p->defined() || (is_vk(*p) && p->scalar_type() == input.scalar_type() && p->numel() == D);
  };
  ok = ok && param_ok(weight) && param_ok(bias);
  if (ok) {
    at::Tensor x = contig(input);
    int64_t rows = x.numel() / D;
    at::Tensor y = at::empty(x.sizes(), x.options());
    std::vector<int64_t> stat_shape(x.sizes().begin(), x.sizes().end() - normalized_shape.size());
    stat_shape.insert(stat_shape.end(), normalized_shape.size(), 1);
    at::Tensor mean = at::empty(stat_shape, x.options().dtype(at::kFloat));
    at::Tensor rstd = at::empty(stat_shape, x.options().dtype(at::kFloat));
    at::Tensor w = (weight.has_value() && weight->defined()) ? contig(*weight) : at::Tensor();
    at::Tensor b = (bias.has_value() && bias->defined()) ? contig(*bias) : at::Tensor();
    LnPush P{addr_of(x), addr_of(y), 0, addr_of(mean), addr_of(rstd), w.defined() ? addr_of(w) : 0, b.defined() ? addr_of(b) : 0,
             (uint32_t)rows, (uint32_t)D, (float)eps, 0};
    dispatch(std::string("ln_fwd_") + k, &P, sizeof(P), (uint64_t)rows, 1, "aten::native_layer_norm");
    return {y, mean, rstd};
  }
  return VKT_FALLBACK_SYMINT(native_layer_norm, input, normalized_shape_sym, weight, bias, eps);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> k_native_layer_norm_backward(
    const at::Tensor& grad_out, const at::Tensor& input, c10::SymIntArrayRef normalized_shape_sym, const at::Tensor& mean,
    const at::Tensor& rstd, const std::optional<at::Tensor>& weight, const std::optional<at::Tensor>& bias,
    std::array<bool, 3> output_mask) {
  auto normalized_shape = C10_AS_INTARRAYREF_SLOW(normalized_shape_sym);
  const char* k = (is_vk(grad_out) && is_vk(input)) ? fkind(input.scalar_type()) : nullptr;
  int64_t D = prod(normalized_shape);
  bool has_w = weight.has_value() && weight->defined();
  bool ok = k && D > 0 && input.numel() > 0 && grad_out.scalar_type() == input.scalar_type() && grad_out.sizes() == input.sizes() &&
            is_vk(mean) && is_vk(rstd) && mean.scalar_type() == at::kFloat && rstd.scalar_type() == at::kFloat &&
            input.numel() < (int64_t(1) << 32) && (!has_w || (is_vk(*weight) && weight->scalar_type() == input.scalar_type() && weight->numel() == D));
  if (ok) {
    at::Tensor g = contig(grad_out), x = contig(input), mu = contig(mean), rs = contig(rstd);
    at::Tensor w = has_w ? contig(*weight) : at::Tensor();
    int64_t rows = x.numel() / D;
    at::Tensor dx, dw, db;
    LnPush P{addr_of(x), 0, addr_of(g), addr_of(mu), addr_of(rs), has_w ? addr_of(w) : 0, 0, (uint32_t)rows, (uint32_t)D, 0.f, 0};
    if (output_mask[0]) {
      dx = at::empty(x.sizes(), x.options());
      P.y = addr_of(dx);
      dispatch(std::string("ln_bwd_") + k, &P, sizeof(P), (uint64_t)rows, 1, "aten::native_layer_norm_backward");
    }
    bool want_dw = output_mask[1] && has_w;
    bool want_db = output_mask[2] && bias.has_value() && bias->defined();
    if (want_dw || want_db) {
      if (want_dw) dw = at::empty(weight->sizes(), weight->options());
      if (want_db) db = at::empty(bias->sizes(), bias->options());
      P.y = want_dw ? addr_of(dw) : 0;
      P.b = want_db ? addr_of(db) : 0;
      dispatch(std::string("ln_wgrad_") + k, &P, sizeof(P), (uint64_t)((D + 255) / 256), 1, "aten::native_layer_norm_backward");
    }
    return {dx, dw, db};
  }
  return VKT_FALLBACK_SYMINT(native_layer_norm_backward, grad_out, input, normalized_shape_sym, mean, rstd, weight, bias, output_mask);
}

// ---- sum / mean -----------------------------------------------------------------------------------------------
struct ReducePush {
  uint64_t x, out;
  uint32_t n_out, n_red;
  uint32_t ksz[4], kst[4];
  uint32_t rsz[2], rst[2];
  float div;
  uint32_t pad;
};
static_assert(sizeof(ReducePush) == 80, "reduce.comp.in の push constant は 80 バイト");

struct RDim {
  int64_t size, stride;
};

// 隣接する次元の stride が連続なら 1 本にまとめる（集合として同じ要素を列挙するので縮約でも保持でも正しい）
void merge_dims(std::vector<RDim>& v) {
  std::vector<RDim> out;
  for (auto d : v) {
    if (d.size == 1) continue;
    if (!out.empty() && out.back().stride == d.stride * d.size) {
      out.back().size *= d.size;
      out.back().stride = d.stride;
    } else {
      out.push_back(d);
    }
  }
  v = out;
}

std::optional<at::Tensor> try_reduce(const at::Tensor& self, at::OptionalIntArrayRef dim_opt, bool keepdim,
                                     std::optional<at::ScalarType> dtype, bool mean, const char* aten_name) {
  if (!is_vk(self)) return std::nullopt;
  const char* k = fkind(self.scalar_type());
  if (!k || self.numel() == 0 || (dtype.has_value() && *dtype != self.scalar_type())) return std::nullopt;
  int64_t nd = self.dim();
  std::vector<bool> reduced(nd, false);
  if (!dim_opt.has_value() || dim_opt->empty()) {
    std::fill(reduced.begin(), reduced.end(), true);
  } else {
    for (auto d : *dim_opt) reduced[at::maybe_wrap_dim(d, nd)] = true;
  }
  std::vector<RDim> kept, red;
  IntVec out_sizes;
  int64_t n_red = 1, n_out = 1;
  for (int64_t d = 0; d < nd; ++d) {
    if (self.stride(d) < 0) return std::nullopt;
    if (reduced[d]) {
      red.push_back({self.size(d), self.stride(d)});
      n_red *= self.size(d);
      if (keepdim) out_sizes.push_back(1);
    } else {
      kept.push_back({self.size(d), self.stride(d)});
      n_out *= self.size(d);
      out_sizes.push_back(self.size(d));
    }
  }
  merge_dims(kept);
  merge_dims(red);
  if (kept.size() > 4 || red.size() > 2 || n_out >= (int64_t(1) << 32) || n_red >= (int64_t(1) << 32)) return std::nullopt;
  ReducePush P{};
  P.n_out = (uint32_t)n_out;
  P.n_red = (uint32_t)n_red;
  for (int i = 0; i < 4; ++i) P.ksz[i] = 1;
  for (int i = 0; i < 2; ++i) P.rsz[i] = 1;
  int64_t max_off = 0;
  for (size_t i = 0; i < kept.size(); ++i) {
    int j = 4 - (int)kept.size() + (int)i;
    P.ksz[j] = (uint32_t)kept[i].size;
    P.kst[j] = (uint32_t)kept[i].stride;
    max_off += (kept[i].size - 1) * kept[i].stride;
  }
  for (size_t i = 0; i < red.size(); ++i) {
    int j = 2 - (int)red.size() + (int)i;
    P.rsz[j] = (uint32_t)red[i].size;
    P.rst[j] = (uint32_t)red[i].stride;
    max_off += (red[i].size - 1) * red[i].stride;
  }
  if (max_off >= (int64_t(1) << 32)) return std::nullopt;
  at::Tensor out = at::empty(out_sizes, self.options());
  P.x = addr_of(self);
  P.out = addr_of(out);
  P.div = mean ? (float)n_red : 1.0f;
  dispatch(std::string("reduce_") + k, &P, sizeof(P), (uint64_t)n_out, 1, aten_name);
  return out;
}

at::Tensor k_sum_dim(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim, std::optional<at::ScalarType> dtype) {
  if (auto r = try_reduce(self, dim, keepdim, dtype, false, "aten::sum.dim_IntList")) return *r;
  return VKT_FALLBACK(sum_dim_IntList, self, dim, keepdim, dtype);
}
at::Tensor k_mean_dim(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim, std::optional<at::ScalarType> dtype) {
  if (auto r = try_reduce(self, dim, keepdim, dtype, true, "aten::mean.dim")) return *r;
  return VKT_FALLBACK(mean_dim, self, dim, keepdim, dtype);
}
at::Tensor k_sum(const at::Tensor& self, std::optional<at::ScalarType> dtype) {
  if (auto r = try_reduce(self, std::nullopt, false, dtype, false, "aten::sum")) return *r;
  return VKT_FALLBACK(sum, self, dtype);
}
at::Tensor k_mean(const at::Tensor& self, std::optional<at::ScalarType> dtype) {
  if (auto r = try_reduce(self, std::nullopt, false, dtype, true, "aten::mean")) return *r;
  return VKT_FALLBACK(mean, self, dtype);
}

// ---- cat -----------------------------------------------------------------------------------------------------------
// 出力を確保し、各入力を出力のスライスへ GPU 上でコピーする。
at::Tensor k_cat(const at::ITensorListRef& tensors, int64_t dim) {
  std::vector<at::Tensor> ts;
  for (const at::Tensor& t : tensors) ts.push_back(t);
  bool ok = !ts.empty();
  at::ScalarType dt = ok ? ts[0].scalar_type() : at::kFloat;
  for (auto& t : ts) {
    ok = ok && is_vk(t) && t.scalar_type() == dt && fkind(dt) != nullptr;
  }
  std::vector<at::Tensor> parts;  // 旧仕様の「形状 [0] の空テンソル」は無視する
  if (ok) {
    for (auto& t : ts) {
      if (!(t.dim() == 1 && t.size(0) == 0)) parts.push_back(t);
    }
    ok = !parts.empty();
  }
  if (ok) {
    int64_t nd = parts[0].dim();
    ok = nd > 0;
    if (ok) {
      dim = at::maybe_wrap_dim(dim, nd);
      IntVec sizes(parts[0].sizes().begin(), parts[0].sizes().end());
      sizes[dim] = 0;
      for (auto& t : parts) {
        ok = ok && t.dim() == nd;
        if (!ok) break;
        for (int64_t d = 0; d < nd; ++d) {
          if (d != dim) ok = ok && t.size(d) == parts[0].size(d);
        }
        sizes[dim] += t.size(dim);
      }
      if (ok) {
        at::Tensor out = at::empty(sizes, parts[0].options());
        int64_t off = 0;
        for (auto& t : parts) {
          if (t.size(dim) > 0) out.narrow(dim, off, t.size(dim)).copy_(t);
          off += t.size(dim);
        }
        count_kernel("aten::cat");
        return out;
      }
    }
  }
  return VKT_FALLBACK(cat, tensors, dim);
}

}  // namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_softmax", &k_softmax);
  m.impl("_safe_softmax", &k_safe_softmax);
  m.impl("_softmax_backward_data", &k_softmax_backward);
  m.impl("native_layer_norm", &k_native_layer_norm);
  m.impl("native_layer_norm_backward", &k_native_layer_norm_backward);
  m.impl("sum.dim_IntList", &k_sum_dim);
  m.impl("mean.dim", &k_mean_dim);
  m.impl("sum", &k_sum);
  m.impl("mean", &k_mean);
  m.impl("cat", &k_cat);
}
