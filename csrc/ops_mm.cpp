// 行列積の ATen 登録（mm / bmm / addmm / baddbmm）。カーネル本体は vkrt/shaders/mm.comp.in。
// 転置・スライスなどの任意ストライドはコピー無しで直接読む。条件外（dtype が f32/bf16/f16 で揃っていない等）は CPU へ戻す。
#include <ATen/Operators.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>

#include "common.h"

namespace {

using namespace vkt;

// mm.comp.in の push constant ブロックと同一レイアウト（112 バイト）
struct MmPush {
  uint64_t a, b, c, bias;
  uint32_t M, N, K, batch;
  uint32_t a_sb, a_sm, a_sk;
  uint32_t b_sb, b_sk, b_sn;
  uint32_t c_sb, c_sm, c_sn;
  uint32_t bias_sb, bias_sm, bias_sn;
  float alpha, beta;
  uint32_t has_bias, pad;
};
static_assert(sizeof(MmPush) == 112, "MmPush は mm.comp.in の push constant と同じ 112 バイトであること");

std::atomic<bool>& coop_enabled() {
  static std::atomic<bool> v{std::getenv("VKT_NO_COOP") == nullptr};
  return v;
}

uint32_t coop_caps_cached() {
  static uint32_t caps = [] {
    ensure_init();
    return vkrt_coop_caps();
  }();
  return caps;
}

const char* mm_kind(at::ScalarType t) {
  switch (t) {
    case at::kFloat: return "f32";
    case at::kBFloat16: return "bf16";
    case at::kHalf: return "f16";
    default: return nullptr;
  }
}

bool fits_u32(int64_t v) { return v >= 0 && v < (int64_t(1) << 32); }

// A(M,K), B(K,N) を [batch] 付きで解釈して実行する。a / b は 2 次元（mm）または 3 次元（bmm）。bias は任意（broadcast 可）。
std::optional<at::Tensor> try_gemm(const at::Tensor& a, const at::Tensor& b, const at::Tensor* bias, double alpha, double beta,
                                   const char* aten_name) {
  if (!is_vk(a) || !is_vk(b) || a.scalar_type() != b.scalar_type() || a.dim() != b.dim()) return std::nullopt;
  const char* kind = mm_kind(a.scalar_type());
  if (!kind) return std::nullopt;
  bool batched = a.dim() == 3;
  if (!batched && a.dim() != 2) return std::nullopt;
  int64_t batch = batched ? a.size(0) : 1;
  int64_t M = a.size(batched ? 1 : 0), K = a.size(batched ? 2 : 1);
  int64_t N = b.size(batched ? 2 : 1);
  if ((batched && b.size(0) != batch) || b.size(batched ? 1 : 0) != K) return std::nullopt;
  if (M == 0 || N == 0 || K == 0 || batch == 0) return std::nullopt;
  auto s = [&](const at::Tensor& t, int d) { return t.stride(d); };
  int off = batched ? 1 : 0;
  int64_t a_sb = batched ? s(a, 0) : 0, a_sm = s(a, off), a_sk = s(a, off + 1);
  int64_t b_sb = batched ? s(b, 0) : 0, b_sk = s(b, off), b_sn = s(b, off + 1);
  int64_t c_sb = M * N, c_sm = N, c_sn = 1;

  MmPush P{};
  bool use_bias = bias != nullptr && beta != 0.0;
  if (use_bias) {
    if (!is_vk(*bias) || bias->scalar_type() != a.scalar_type()) return std::nullopt;
    // bias を (batch, M, N) へブロードキャストした stride
    std::vector<int64_t> full = batched ? std::vector<int64_t>{batch, M, N} : std::vector<int64_t>{M, N};
    if ((int)bias->dim() > (int)full.size()) return std::nullopt;
    std::vector<int64_t> st(full.size(), 0);
    for (size_t i = 0; i < full.size(); ++i) {
      int j = (int)i - ((int)full.size() - (int)bias->dim());
      if (j < 0) continue;
      if (bias->size(j) == full[i]) st[i] = bias->stride(j);
      else if (bias->size(j) == 1) st[i] = 0;
      else return std::nullopt;
    }
    P.bias = addr_of(*bias);
    P.bias_sb = batched ? (uint32_t)st[0] : 0;
    P.bias_sm = (uint32_t)st[batched ? 1 : 0];
    P.bias_sn = (uint32_t)st[batched ? 2 : 1];
    for (auto v : st) if (!fits_u32(v)) return std::nullopt;
  }
  // 32bit オフセットに収まること（最大到達オフセットで判定）
  auto max_off = [&](int64_t sb, int64_t sm, int64_t sk, int64_t nb, int64_t nm, int64_t nk) {
    return (nb - 1) * sb + (nm - 1) * sm + (nk - 1) * sk;
  };
  if (a_sb < 0 || a_sm < 0 || a_sk < 0 || b_sb < 0 || b_sk < 0 || b_sn < 0) return std::nullopt;
  if (!fits_u32(max_off(a_sb, a_sm, a_sk, batch, M, K)) || !fits_u32(max_off(b_sb, b_sk, b_sn, batch, K, N)) ||
      !fits_u32(batch * M * N)) {
    return std::nullopt;
  }

  at::Tensor out = batched ? at::empty({batch, M, N}, a.options()) : at::empty({M, N}, a.options());
  P.a = addr_of(a);
  P.b = addr_of(b);
  P.c = addr_of(out);
  P.M = (uint32_t)M;
  P.N = (uint32_t)N;
  P.K = (uint32_t)K;
  P.batch = (uint32_t)batch;
  P.a_sb = (uint32_t)a_sb; P.a_sm = (uint32_t)a_sm; P.a_sk = (uint32_t)a_sk;
  P.b_sb = (uint32_t)b_sb; P.b_sk = (uint32_t)b_sk; P.b_sn = (uint32_t)b_sn;
  P.c_sb = (uint32_t)c_sb; P.c_sm = (uint32_t)c_sm; P.c_sn = (uint32_t)c_sn;
  P.alpha = (float)alpha;
  P.beta = (float)beta;
  P.has_bias = use_bias ? 1 : 0;

  uint32_t gx = (uint32_t)((N + 63) / 64), gy = (uint32_t)((M + 63) / 64), gz = (uint32_t)batch;
  // bf16 / f16 で、デバイスが協調行列に対応し、十分大きい行列なら協調行列版を使う（それ以外は従来のタイル型）。
  // 協調行列版のタイルは 64x64 / 128 スレッド。これは mm.comp.in(256 スレッド)とは別のカーネルなので、グリッドは同じ。
  bool use_coop = false;
  if (coop_enabled().load() && M >= 32 && N >= 32 && K >= 32) {
    uint32_t caps = coop_caps_cached();
    use_coop = (a.scalar_type() == at::kBFloat16 && (caps & 2)) || (a.scalar_type() == at::kHalf && (caps & 1));
  }
  if (use_coop) {
    // 128bit ベクトルロードが使える条件: 16 バイト整列、バッチ stride が 8 要素の倍数、連続軸でない stride も 8 の倍数、端数なし
    auto aligned16 = [](const at::Tensor& t) { return addr_of(t) % 16 == 0; };
    bool a_vec = aligned16(a) && a_sb % 8 == 0 && ((a_sk == 1 && a_sm % 8 == 0 && K % 8 == 0) || (a_sm == 1 && a_sk % 8 == 0 && M % 8 == 0));
    bool b_vec = aligned16(b) && b_sb % 8 == 0 && ((b_sn == 1 && b_sk % 8 == 0 && N % 8 == 0) || (b_sk == 1 && b_sn % 8 == 0 && K % 8 == 0));
    P.pad = (a_vec ? 1u : 0u) | (b_vec ? 2u : 0u);
  }
  std::string kname = std::string(use_coop ? "mm_coop_" : "mm_") + kind;
  vk_check(vkrt_dispatch((uint32_t)kernel_index(kname), &P, sizeof(P), gx, gy, gz), "dispatch mm");
  // デバッグ: VKT_VERIFY_MM=1 なら、協調行列の結果を同じ入力のタイル型の結果と照合し、乖離した呼び出しを報告する
  static const bool verify = std::getenv("VKT_VERIFY_MM") != nullptr;
  if (verify && use_coop) {
    at::Tensor ref = batched ? at::empty({batch, M, N}, a.options()) : at::empty({M, N}, a.options());
    MmPush Q = P;
    Q.c = addr_of(ref);
    Q.pad = 0;
    vk_check(vkrt_dispatch((uint32_t)kernel_index(std::string("mm_") + kind), &Q, sizeof(Q), gx, gy, gz), "dispatch mm(verify)");
    at::Tensor d = (out.to(at::kFloat) - ref.to(at::kFloat)).abs();
    double dmax = d.max().item<double>();
    double rmax = ref.to(at::kFloat).abs().max().item<double>();
    // 要素ごとの相対差（参照が小さい要素は分母に行の最大値を使う）
    // 要素ごとの相対差。最大値の 1e-4 倍未満の要素は分母を床で押さえる。2 ulp(bf16 で約 1.6%)を超える要素の割合を rel に使う。
    at::Tensor rf = ref.to(at::kFloat);
    at::Tensor denom = rf.abs().clamp_min(rmax * 1e-4);
    double rel = (d / denom > 0.016).to(at::kFloat).mean().item<double>();
    static int n_calls = 0, n_bad = 0;
    n_calls++;
    if (rel > 1e-3) {
      n_bad++;
      std::cerr << "[vkt VERIFY-MM BAD] rel=" << rel << " dmax=" << dmax << " rmax=" << rmax << " M=" << M << " N=" << N << " K=" << K
                << " batch=" << batch << " a_strides=(" << a_sb << "," << a_sm << "," << a_sk << ") b_strides=(" << b_sb << "," << b_sk << ","
                << b_sn << ") vec_flags=" << P.pad << " dtype=" << kind << " (" << n_bad << "/" << n_calls << ")" << std::endl;
    }
  }
  count_kernel(aten_name);
  return out;
}

at::Tensor k_mm(const at::Tensor& a, const at::Tensor& b) {
  if (auto r = try_gemm(a, b, nullptr, 1.0, 0.0, "aten::mm")) return *r;
  return VKT_FALLBACK(mm, a, b);
}

at::Tensor k_bmm(const at::Tensor& a, const at::Tensor& b) {
  if (auto r = try_gemm(a, b, nullptr, 1.0, 0.0, "aten::bmm")) return *r;
  return VKT_FALLBACK(bmm, a, b);
}

at::Tensor k_addmm(const at::Tensor& self, const at::Tensor& a, const at::Tensor& b, const at::Scalar& beta, const at::Scalar& alpha) {
  if (a.dim() == 2) {
    if (auto r = try_gemm(a, b, &self, alpha.toDouble(), beta.toDouble(), "aten::addmm")) return *r;
  }
  return VKT_FALLBACK(addmm, self, a, b, beta, alpha);
}

at::Tensor k_baddbmm(const at::Tensor& self, const at::Tensor& a, const at::Tensor& b, const at::Scalar& beta, const at::Scalar& alpha) {
  if (a.dim() == 3) {
    if (auto r = try_gemm(a, b, &self, alpha.toDouble(), beta.toDouble(), "aten::baddbmm")) return *r;
  }
  return VKT_FALLBACK(baddbmm, self, a, b, beta, alpha);
}

}  // namespace

namespace vkt {
void set_coop_enabled(bool on) { coop_enabled().store(on); }
uint32_t coop_caps() { return coop_caps_cached(); }
}  // namespace vkt

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("mm", &k_mm);
  m.impl("bmm", &k_bmm);
  m.impl("addmm", &k_addmm);
  m.impl("baddbmm", &k_baddbmm);
}
