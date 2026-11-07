// 要素演算の ATen 登録。カーネル本体は vkrt/shaders/ew.comp.in（tools/gen_shaders.py が展開）。
// 方針: 条件を満たすときだけ GPU カーネルで実行し、満たさなければ std::nullopt を返して CPU フォールバックに戻す。
//   GPU 経路の条件: テンソルが vk 上・dtype が f32/bf16/f16（bool は copy / where の条件のみ）・次元結合後 5 次元以下。
#include <ATen/Operators.h>
#include <ATen/ExpandUtils.h>

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "common.h"

namespace {

using namespace vkt;
using IntVec = std::vector<int64_t>;

// ew.comp.in の push constant ブロックと同一レイアウト（152 バイト）
struct EwPush {
  uint64_t out, a, b, c;
  uint32_t n, pad;
  uint32_t sz[5], so[5], sa[5], sb[5], sc[5];
  float s0, s1;
};
static_assert(sizeof(EwPush) == 152, "EwPush は ew.comp.in の push constant と同じ 152 バイトであること");

const char* kind_name(at::ScalarType t) {
  switch (t) {
    case at::kFloat: return "f32";
    case at::kBFloat16: return "bf16";
    case at::kHalf: return "f16";
    case at::kBool: return "u8";
    default: return nullptr;
  }
}

// 計算対象にできる浮動小数 dtype か
const char* float_kind(at::ScalarType t) {
  const char* k = kind_name(t);
  return (k && t != at::kBool) ? k : nullptr;
}

// t を out_sizes へブロードキャストしたときの stride（要素単位）。非対応なら false。
bool bcast_strides(const at::Tensor& t, const IntVec& out_sizes, IntVec& st) {
  int nd = (int)out_sizes.size();
  st.assign(nd, 0);
  for (int i = 0; i < nd; ++i) {
    int j = i - (nd - (int)t.dim());
    if (j < 0) continue;
    if (t.size(j) == out_sizes[i]) {
      if (t.stride(j) < 0) return false;
      st[i] = t.stride(j);
    } else if (t.size(j) == 1) {
      st[i] = 0;
    } else {
      return false;
    }
  }
  return (int)t.dim() <= nd;
}

IntVec contiguous_strides(const IntVec& sizes) {
  IntVec st(sizes.size());
  int64_t acc = 1;
  for (int i = (int)sizes.size() - 1; i >= 0; --i) {
    st[i] = acc;
    acc *= std::max<int64_t>(sizes[i], 1);
  }
  return st;
}

// 次元を結合（隣接次元の stride が連続なら 1 本にする）して 5 次元に収める。
bool ew_prepare(EwPush& P, const IntVec& sizes, const IntVec& out_strides, const at::Tensor* a, const at::Tensor* b,
                const at::Tensor* c) {
  const at::Tensor* ops[3] = {a, b, c};
  std::array<IntVec, 4> st;
  st[0] = out_strides;
  for (int k = 0; k < 3; ++k) {
    if (ops[k]) {
      if (!bcast_strides(*ops[k], sizes, st[k + 1])) return false;
    } else {
      st[k + 1].assign(sizes.size(), 0);
    }
  }
  int64_t numel = 1;
  for (auto s : sizes) numel *= s;
  if (numel <= 0 || numel >= (int64_t(1) << 32)) return false;

  struct Dim {
    int64_t size;
    int64_t stride[4];
  };
  std::vector<Dim> merged;
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] == 1) continue;
    Dim d{sizes[i], {st[0][i], st[1][i], st[2][i], st[3][i]}};
    if (!merged.empty()) {
      Dim& outer = merged.back();
      bool ok = true;
      for (int k = 0; k < 4; ++k) ok = ok && (outer.stride[k] == d.stride[k] * d.size);
      if (ok) {
        outer.size *= d.size;
        for (int k = 0; k < 4; ++k) outer.stride[k] = d.stride[k];
        continue;
      }
    }
    merged.push_back(d);
  }
  if (merged.size() > 5) return false;

  P = EwPush{};
  P.n = (uint32_t)numel;
  for (int i = 0; i < 5; ++i) P.sz[i] = 1;
  int base = 5 - (int)merged.size();
  for (size_t j = 0; j < merged.size(); ++j) {
    for (int k = 0; k < 4; ++k) {
      if (merged[j].stride[k] > (int64_t)UINT32_MAX) return false;
    }
    P.sz[base + j] = (uint32_t)merged[j].size;
    P.so[base + j] = (uint32_t)merged[j].stride[0];
    P.sa[base + j] = (uint32_t)merged[j].stride[1];
    P.sb[base + j] = (uint32_t)merged[j].stride[2];
    P.sc[base + j] = (uint32_t)merged[j].stride[3];
  }
  return true;
}

void ew_run(const std::string& kernel, const char* aten_name, EwPush& P, const at::Tensor& out, const at::Tensor* a,
            const at::Tensor* b, const at::Tensor* c, float s0 = 0.f, float s1 = 0.f) {
  P.out = addr_of(out);
  P.a = a ? addr_of(*a) : 0;
  P.b = b ? addr_of(*b) : 0;
  P.c = c ? addr_of(*c) : 0;
  P.s0 = s0;
  P.s1 = s1;
  uint32_t groups = (P.n + 255) / 256;
  vk_check(vkrt_dispatch((uint32_t)kernel_index(kernel), &P, sizeof(P), groups, 1, 1), kernel.c_str());
  count_kernel(aten_name);
}

bool host_scalar(const at::Tensor& t) {
  return t.defined() && t.device().is_cpu() && t.dim() == 0 &&
         (t.scalar_type() == at::kFloat || t.scalar_type() == at::kDouble || t.scalar_type() == at::kLong ||
          t.scalar_type() == at::kInt || t.scalar_type() == at::kBFloat16 || t.scalar_type() == at::kHalf ||
          t.scalar_type() == at::kBool);
}

// ---- 2 項演算 (add / sub / mul / div) ------------------------------------------------------------------
enum class Bin { Add, Sub, Mul, Div };

const char* bin_name(Bin op) {
  switch (op) {
    case Bin::Add: return "add";
    case Bin::Sub: return "sub";
    case Bin::Mul: return "mul";
    default: return "div";
  }
}

// self_inplace が非 null なら in-place（結果を *self_inplace に書く）。GPU で実行できなければ nullopt。
std::optional<at::Tensor> try_binary(Bin op, const at::Tensor& a, const at::Tensor& b, double alpha, at::Tensor* self_inplace,
                                     const char* aten_name) {
  if (is_vk(a) && is_vk(b)) {
    const char* ka = float_kind(a.scalar_type());
    const char* kb = float_kind(b.scalar_type());
    if (!ka || !kb) return std::nullopt;
    at::ScalarType rt = at::result_type(a, b);
    // カーネル表は「同じ dtype ならそのまま、違えば f32」。0 次元テンソルが絡む昇格規則の違いは CPU に任せる。
    at::ScalarType expect = (a.scalar_type() == b.scalar_type()) ? a.scalar_type() : at::kFloat;
    if (rt != expect) return std::nullopt;
    IntVec sizes;
    try {
      sizes = at::infer_size(a.sizes(), b.sizes());
    } catch (...) {
      return std::nullopt;
    }
    EwPush P;
    if (self_inplace) {
      if (rt != self_inplace->scalar_type() || IntVec(self_inplace->sizes().begin(), self_inplace->sizes().end()) != sizes) {
        return std::nullopt;
      }
      IntVec so(self_inplace->strides().begin(), self_inplace->strides().end());
      if (!ew_prepare(P, sizes, so, &a, &b, nullptr)) return std::nullopt;
      ew_run(std::string("ew_") + bin_name(op) + "_" + ka + "_" + kb + "_" + kind_name(rt), aten_name, P, *self_inplace, &a, &b,
             nullptr, (float)alpha);
      return *self_inplace;
    }
    if (!ew_prepare(P, sizes, contiguous_strides(sizes), &a, &b, nullptr)) return std::nullopt;
    at::Tensor out = at::empty(sizes, a.options().dtype(rt));
    ew_run(std::string("ew_") + bin_name(op) + "_" + ka + "_" + kb + "_" + kind_name(rt), aten_name, P, out, &a, &b, nullptr,
           (float)alpha);
    return out;
  }

  // テンソル(vk) と ホスト上の 0 次元(ラップされたスカラー)
  const at::Tensor* t = nullptr;
  const at::Tensor* s = nullptr;
  bool scalar_on_left = false;
  if (is_vk(a) && host_scalar(b)) {
    t = &a;
    s = &b;
  } else if (!self_inplace && host_scalar(a) && is_vk(b) && (op == Bin::Mul || (op == Bin::Add && alpha == 1.0))) {
    t = &b;  // 可換演算のみ左右を入れ替える
    s = &a;
    scalar_on_left = true;
  } else {
    return std::nullopt;
  }
  (void)scalar_on_left;
  const char* kt = float_kind(t->scalar_type());
  if (!kt) return std::nullopt;
  if (at::result_type(*t, *s) != t->scalar_type()) return std::nullopt;  // 型が変わる（整数テンソル + 実数スカラー等）は CPU へ
  double v = s->item<double>();
  const char* sop;
  double s0;
  switch (op) {
    case Bin::Add: sop = "adds"; s0 = alpha * v; break;
    case Bin::Sub: sop = "adds"; s0 = -alpha * v; break;
    case Bin::Mul: sop = "muls"; s0 = v; break;
    default: sop = "divs"; s0 = v; break;
  }
  IntVec sizes(t->sizes().begin(), t->sizes().end());
  if (self_inplace) {
    if (self_inplace->scalar_type() != t->scalar_type() || self_inplace->sizes() != t->sizes()) return std::nullopt;
  }
  EwPush P;
  if (self_inplace) {
    IntVec so(self_inplace->strides().begin(), self_inplace->strides().end());
    if (!ew_prepare(P, sizes, so, t, nullptr, nullptr)) return std::nullopt;
    ew_run(std::string("ew_") + sop + "_" + kt + "_" + kt, aten_name, P, *self_inplace, t, nullptr, nullptr, (float)s0);
    return *self_inplace;
  }
  if (!ew_prepare(P, sizes, contiguous_strides(sizes), t, nullptr, nullptr)) return std::nullopt;
  at::Tensor out = at::empty(sizes, t->options());
  ew_run(std::string("ew_") + sop + "_" + kt + "_" + kt, aten_name, P, out, t, nullptr, nullptr, (float)s0);
  return out;
}

#define VKT_BINARY(NAME, OP, ATEN)                                                                         \
  at::Tensor NAME##_Tensor(const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha) {              \
    if (auto r = try_binary(OP, a, b, alpha.toDouble(), nullptr, ATEN ".Tensor")) return *r;                 \
    return VKT_FALLBACK(NAME##_Tensor, a, b, alpha);                                                         \
  }                                                                                                          \
  at::Tensor& NAME##__Tensor(at::Tensor& self, const at::Tensor& b, const at::Scalar& alpha) {               \
    if (try_binary(OP, self, b, alpha.toDouble(), &self, ATEN "_.Tensor")) return self;                      \
    return VKT_FALLBACK(NAME##__Tensor, self, b, alpha);                                                     \
  }

VKT_BINARY(add, Bin::Add, "aten::add")
VKT_BINARY(sub, Bin::Sub, "aten::sub")

at::Tensor mul_Tensor(const at::Tensor& a, const at::Tensor& b) {
  if (auto r = try_binary(Bin::Mul, a, b, 1.0, nullptr, "aten::mul.Tensor")) return *r;
  return VKT_FALLBACK(mul_Tensor, a, b);
}
at::Tensor& mul__Tensor(at::Tensor& self, const at::Tensor& b) {
  if (try_binary(Bin::Mul, self, b, 1.0, &self, "aten::mul_.Tensor")) return self;
  return VKT_FALLBACK(mul__Tensor, self, b);
}
at::Tensor div_Tensor(const at::Tensor& a, const at::Tensor& b) {
  if (auto r = try_binary(Bin::Div, a, b, 1.0, nullptr, "aten::div.Tensor")) return *r;
  return VKT_FALLBACK(div_Tensor, a, b);
}
at::Tensor& div__Tensor(at::Tensor& self, const at::Tensor& b) {
  if (try_binary(Bin::Div, self, b, 1.0, &self, "aten::div_.Tensor")) return self;
  return VKT_FALLBACK(div__Tensor, self, b);
}

// ---- 単項演算 ---------------------------------------------------------------------------------------------
std::optional<at::Tensor> try_unary(const char* op, const at::Tensor& a, const char* aten_name, float s0 = 0.f) {
  if (!is_vk(a)) return std::nullopt;
  const char* k = float_kind(a.scalar_type());
  if (!k || a.numel() == 0) return std::nullopt;
  IntVec sizes(a.sizes().begin(), a.sizes().end());
  EwPush P;
  if (!ew_prepare(P, sizes, contiguous_strides(sizes), &a, nullptr, nullptr)) return std::nullopt;
  at::Tensor out = at::empty(sizes, a.options());
  ew_run(std::string("ew_") + op + "_" + k + "_" + k, aten_name, P, out, &a, nullptr, nullptr, s0);
  return out;
}

#define VKT_UNARY(NAME, KOP)                                                              \
  at::Tensor k_##NAME(const at::Tensor& a) {                                               \
    if (auto r = try_unary(KOP, a, "aten::" #NAME)) return *r;                              \
    return VKT_FALLBACK(NAME, a);                                                          \
  }

VKT_UNARY(neg, "neg")
VKT_UNARY(rsqrt, "rsqrt")
VKT_UNARY(reciprocal, "recip")
VKT_UNARY(exp, "exp")
VKT_UNARY(sin, "sin")
VKT_UNARY(cos, "cos")
VKT_UNARY(sqrt, "sqrt")
VKT_UNARY(tanh, "tanh")
VKT_UNARY(sigmoid, "sigmoid")
VKT_UNARY(relu, "relu")
VKT_UNARY(silu, "silu")

at::Tensor pow_Tensor_Scalar(const at::Tensor& a, const at::Scalar& e) {
  if (e.isFloatingPoint() || e.isIntegral(false)) {
    if (auto r = try_unary("pows", a, "aten::pow.Tensor_Scalar", (float)e.toDouble())) return *r;
  }
  return VKT_FALLBACK(pow_Tensor_Scalar, a, e);
}

at::Tensor k_gelu(const at::Tensor& a, c10::string_view approximate) {
  const char* op = approximate == "none" ? "gelu_erf" : (approximate == "tanh" ? "gelu_tanh" : nullptr);
  if (op) {
    if (auto r = try_unary(op, a, "aten::gelu")) return *r;
  }
  return VKT_FALLBACK(gelu, a, approximate);
}

// ---- バックワード (grad, self) ------------------------------------------------------------------------------
std::optional<at::Tensor> try_backward(const char* op, const at::Tensor& g, const at::Tensor& x, const char* aten_name) {
  if (!is_vk(g) || !is_vk(x) || g.scalar_type() != x.scalar_type() || g.numel() == 0) return std::nullopt;
  const char* k = float_kind(g.scalar_type());
  if (!k) return std::nullopt;
  IntVec sizes;
  try {
    sizes = at::infer_size(g.sizes(), x.sizes());
  } catch (...) {
    return std::nullopt;
  }
  EwPush P;
  if (!ew_prepare(P, sizes, contiguous_strides(sizes), &g, &x, nullptr)) return std::nullopt;
  at::Tensor out = at::empty(sizes, g.options());
  ew_run(std::string("ew_") + op + "_" + k + "_" + k + "_" + k, aten_name, P, out, &g, &x, nullptr);
  return out;
}

at::Tensor k_silu_backward(const at::Tensor& g, const at::Tensor& x) {
  if (auto r = try_backward("silu_bwd", g, x, "aten::silu_backward")) return *r;
  return VKT_FALLBACK(silu_backward, g, x);
}

at::Tensor k_gelu_backward(const at::Tensor& g, const at::Tensor& x, c10::string_view approximate) {
  const char* op = approximate == "none" ? "gelu_erf_bwd" : (approximate == "tanh" ? "gelu_tanh_bwd" : nullptr);
  if (op) {
    if (auto r = try_backward(op, g, x, "aten::gelu_backward")) return *r;
  }
  return VKT_FALLBACK(gelu_backward, g, x, approximate);
}

// ---- where / fill ---------------------------------------------------------------------------------------------
at::Tensor where_self(const at::Tensor& cond, const at::Tensor& a, const at::Tensor& b) {
  if (is_vk(cond) && is_vk(a) && is_vk(b) && cond.scalar_type() == at::kBool && a.scalar_type() == b.scalar_type() &&
      float_kind(a.scalar_type())) {
    IntVec sizes;
    bool ok = true;
    try {
      sizes = at::infer_size(at::infer_size(a.sizes(), b.sizes()), cond.sizes());
    } catch (...) {
      ok = false;
    }
    EwPush P;
    if (ok && ew_prepare(P, sizes, contiguous_strides(sizes), &a, &b, &cond)) {
      at::Tensor out = at::empty(sizes, a.options());
      const char* k = float_kind(a.scalar_type());
      ew_run(std::string("ew_where_") + k + "_" + k + "_" + k, "aten::where.self", P, out, &a, &b, &cond);
      return out;
    }
  }
  return VKT_FALLBACK(where_self, cond, a, b);
}

bool try_fill(at::Tensor& self, double v, const char* aten_name) {
  if (!is_vk(self) || self.numel() == 0) return false;
  const char* k = float_kind(self.scalar_type());
  if (!k) return false;
  IntVec sizes(self.sizes().begin(), self.sizes().end());
  IntVec so(self.strides().begin(), self.strides().end());
  EwPush P;
  if (!ew_prepare(P, sizes, so, nullptr, nullptr, nullptr)) return false;
  ew_run(std::string("ew_fill_") + k, aten_name, P, self, nullptr, nullptr, nullptr, (float)v);
  return true;
}

at::Tensor& fill__Scalar(at::Tensor& self, const at::Scalar& v) {
  if ((v.isFloatingPoint() || v.isIntegral(false)) && try_fill(self, v.toDouble(), "aten::fill_.Scalar")) return self;
  return VKT_FALLBACK(fill__Scalar, self, v);
}

at::Tensor& zero_(at::Tensor& self) {
  if (try_fill(self, 0.0, "aten::zero_")) return self;
  return VKT_FALLBACK(zero_, self);
}

}  // namespace

namespace vkt {

// デバイス上の dtype 変換つきストライドコピー（copy_ / _to_copy / clone / contiguous の実体）。
bool ew_copy(const at::Tensor& dst, const at::Tensor& src) {
  if (!is_vk(dst) || !is_vk(src) || dst.numel() == 0) return false;
  const char* kd = kind_name(dst.scalar_type());
  const char* ks = kind_name(src.scalar_type());
  if (!kd || !ks) return false;
  IntVec sizes(dst.sizes().begin(), dst.sizes().end());
  IntVec so(dst.strides().begin(), dst.strides().end());
  EwPush P;
  if (!ew_prepare(P, sizes, so, &src, nullptr, nullptr)) return false;
  ew_run(std::string("ew_cast_") + ks + "_" + kd, "aten::copy_", P, dst, &src, nullptr, nullptr);
  return true;
}

}  // namespace vkt

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("add.Tensor", &add_Tensor);
  m.impl("add_.Tensor", &add__Tensor);
  m.impl("sub.Tensor", &sub_Tensor);
  m.impl("sub_.Tensor", &sub__Tensor);
  m.impl("mul.Tensor", &mul_Tensor);
  m.impl("mul_.Tensor", &mul__Tensor);
  m.impl("div.Tensor", &div_Tensor);
  m.impl("div_.Tensor", &div__Tensor);
  m.impl("neg", &k_neg);
  m.impl("rsqrt", &k_rsqrt);
  m.impl("reciprocal", &k_reciprocal);
  m.impl("exp", &k_exp);
  m.impl("sin", &k_sin);
  m.impl("cos", &k_cos);
  m.impl("sqrt", &k_sqrt);
  m.impl("tanh", &k_tanh);
  m.impl("sigmoid", &k_sigmoid);
  m.impl("relu", &k_relu);
  m.impl("silu", &k_silu);
  m.impl("pow.Tensor_Scalar", &pow_Tensor_Scalar);
  m.impl("gelu", &k_gelu);
  m.impl("silu_backward", &k_silu_backward);
  m.impl("gelu_backward", &k_gelu_backward);
  m.impl("where.self", &where_self);
  m.impl("fill_.Scalar", &fill__Scalar);
  m.impl("zero_", &zero_);
}
