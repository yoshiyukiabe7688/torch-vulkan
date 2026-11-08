// 融合アテンション（flash attention 型）: F.scaled_dot_product_attention を vk で受ける。
// カーネル本体は vkrt/shaders/attn.comp.in（MODE 0 fwd / 1 dK,dV / 2 dQ / 3 delta）。
// PyTorch 側の経路: `_fused_sdp_choice` のスタブ（PrivateUse1）が overrideable を返すと、
// `_scaled_dot_product_fused_attention_overrideable`（forward）と同 `_backward` が呼ばれる。autograd は PyTorch 本体に任せる。
// 対応: 4 次元 [B,H,S,D]、D <= 128、f32 / bf16 / f16、マスク・dropout・causal・GQA 無し。それ以外は従来の数式経路（bmm + softmax）。
// logits（QK^T）は fp32 のまま保持され、N x N の行列はメモリに作られない。
#include <ATen/Operators.h>
#include <ATen/SDPBackend.h>
#include <ATen/native/transformers/attention.h>

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>

#include "common.h"

namespace {

using namespace vkt;

struct AttnPush {
  uint64_t a[8];
  uint32_t qs[3];  // q の (batch, head, seq) ストライド
  uint32_t ks[3];  // k と v 共通
  uint32_t ds[3];  // dO
  uint32_t B, H, Sq, Sk, D;
  float scale;
  uint32_t pad;
};
static_assert(sizeof(AttnPush) == 128, "attn.comp.in の push constant は 128 バイト");

struct CoopPush {
  uint64_t a[8];
  uint32_t B, H, Sq, Sk, D, Sqp, Skp, Dp;
  float scale;
  uint32_t pad;
};
static_assert(sizeof(CoopPush) == 104, "attn_coop.comp.in の push constant は 104 バイト");

const char* kind_name(at::ScalarType t) {
  switch (t) {
    case at::kFloat: return "f32";
    case at::kBFloat16: return "bf16";
    case at::kHalf: return "f16";
    default: return nullptr;
  }
}

// VKT_FUSED_ATTN: "0" = 使わない、"1" = 常に使う、未設定 = auto（協調行列が使えるとき、または score 行列が大きいとき）。
// VKT_NO_FUSED_ATTN=1 は "0" と同じ。
int policy_from_env() {
  const char* n = std::getenv("VKT_NO_FUSED_ATTN");
  if (n && n[0] && n[0] != '0') return 0;
  const char* e = std::getenv("VKT_FUSED_ATTN");
  if (!e || !e[0]) return 2;
  return e[0] == '0' ? 0 : 1;
}
std::atomic<int>& policy_ref() {
  static std::atomic<int> p{policy_from_env()};
  return p;
}
int fused_policy() { return policy_ref().load(); }

// auto のときに使うか。協調行列版は従来経路より速い。スカラー版（協調行列の無いデバイス）は従来経路より遅いが、
// score 行列（B*H*Sq*Sk の fp32 換算）が大きいとメモリを大きく節約できるので、その場合だけ使う。
bool fused_wanted(const at::Tensor& q, const at::Tensor& k) {
  int p = fused_policy();
  if (p == 0) return false;
  if (p == 1) return true;
  if (coop_is_enabled() && ((q.scalar_type() == at::kBFloat16 && (coop_caps() & 2)) || (q.scalar_type() == at::kHalf && (coop_caps() & 1)))) return true;
  double score_bytes = 4.0 * (double)q.size(0) * q.size(1) * q.size(2) * k.size(2);
  return score_bytes >= 256.0 * 1024 * 1024;
}

// 対応する形・条件か。選択スタブと forward の両方で同じ判定を使う。
bool attn_supported(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const std::optional<at::Tensor>& mask, double dropout_p,
                    bool is_causal, bool enable_gqa) {
  if (!fused_wanted(q, k)) return false;
  if (mask.has_value() && mask->defined()) return false;
  if (dropout_p != 0.0 || is_causal || enable_gqa) return false;
  if (!is_vk(q) || !is_vk(k) || !is_vk(v)) return false;
  if (q.dim() != 4 || k.dim() != 4 || v.dim() != 4) return false;
  if (!kind_name(q.scalar_type()) || !kind_name(k.scalar_type()) || !kind_name(v.scalar_type())) return false;
  int64_t D = q.size(3);
  if (D < 1 || D > 128 || k.size(3) != D || v.size(3) != D) return false;
  if (q.size(0) != k.size(0) || q.size(0) != v.size(0) || q.size(1) != k.size(1) || q.size(1) != v.size(1)) return false;
  if (k.size(2) != v.size(2) || k.size(2) < 1 || q.size(2) < 1 || q.size(0) < 1 || q.size(1) < 1) return false;
  // 32 ビットのオフセットに収まること
  if (q.numel() >= (int64_t(1) << 31) || k.numel() >= (int64_t(1) << 31)) return false;
  return true;
}

int64_t fused_sdp_choice_vk(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const std::optional<at::Tensor>& mask, double dropout_p,
                            bool is_causal, std::optional<double>, bool enable_gqa) {
  return attn_supported(q, k, v, mask, dropout_p, is_causal, enable_gqa) ? static_cast<int64_t>(at::SDPBackend::overrideable)
                                                                          : static_cast<int64_t>(at::SDPBackend::math);
}

// 協調行列版を使えるか（bf16 / f16 で、デバイスが対応し、無効化されていない）
bool use_coop_attn(at::ScalarType t) {
  if (!coop_is_enabled()) return false;
  uint32_t caps = coop_caps();
  return (t == at::kBFloat16 && (caps & 2)) || (t == at::kHalf && (caps & 1));
}

int64_t round_up(int64_t v, int64_t m) { return (v + m - 1) / m * m; }

// [B,H,S,D]（任意のストライド）を、0 埋めした連続の [B,H,Spad,Dpad] へ詰め直す（協調行列カーネルの入力）
at::Tensor pad_bhsd(const at::Tensor& t, int64_t Spad, int64_t Dpad) {
  at::Tensor z = at::zeros({t.size(0), t.size(1), Spad, Dpad}, t.options());
  z.slice(2, 0, t.size(2)).slice(3, 0, t.size(3)).copy_(t);
  return z;
}

void dispatch_coop(const char* name, at::ScalarType dt, const CoopPush& P, uint64_t gx, uint64_t gy, const char* aten_name) {
  std::string kn = std::string(name) + "_" + kind_name(dt);
  vk_check(vkrt_dispatch((uint32_t)kernel_index(kn), &P, sizeof(P), (uint32_t)gx, (uint32_t)gy, 1), kn.c_str());
  count_kernel(aten_name);
}

void dispatch(const char* name, const std::string& dt, const AttnPush& P, uint64_t gx, uint64_t gy, const char* aten_name) {
  std::string kn = std::string(name) + "_" + dt;
  vk_check(vkrt_dispatch((uint32_t)kernel_index(kn), &P, sizeof(P), (uint32_t)gx, (uint32_t)gy, 1), kn.c_str());
  count_kernel(aten_name);
  // スカラー版は遅い（協調行列の無いデバイス）。1 回の dispatch が長い（B*H*Sq*Sk*D が大きい）ときは直後に同期して、
  // 溜まったバッチが長時間になるのを防ぐ（amdgpu は約 10 秒の submit をハングと判定して ERROR_DEVICE_LOST にする）。
  double work = (double)P.B * P.H * P.Sq * P.Sk * P.D;
  if (work >= 2.0e9) vk_check(vkrt_synchronize(), "attention sync");
}

uint32_t u32(int64_t v) {
  TORCH_CHECK(v >= 0 && v < (int64_t(1) << 32), "vk: attention のストライドが大きすぎます");
  return (uint32_t)v;
}

// d 方向が連続で、k と v のストライドが等しい形に揃える
struct Prepared {
  at::Tensor q, k, v;
};
Prepared prep_qkv(const at::Tensor& q0, const at::Tensor& k0, const at::Tensor& v0) {
  at::ScalarType dt = q0.scalar_type();
  at::Tensor q = q0.scalar_type() == dt ? q0 : q0.to(dt);
  at::Tensor k = k0.scalar_type() == dt ? k0 : k0.to(dt);
  at::Tensor v = v0.scalar_type() == dt ? v0 : v0.to(dt);
  if (q.stride(3) != 1) q = q.contiguous();
  if (k.stride(3) != 1) k = k.contiguous();
  if (v.stride(3) != 1) v = v.contiguous();
  if (k.strides() != v.strides()) {
    k = k.contiguous();
    v = v.contiguous();
  }
  return {q, k, v};
}

// [B,S,H,D] 連続で確保して [B,H,S,D] に permute したビューを返す（標準の SDPA の出力と同じ向き）。
at::Tensor alloc_bshd(int64_t B, int64_t H, int64_t S, int64_t D, const at::TensorOptions& opt) {
  return at::empty({B, S, H, D}, opt).permute({0, 2, 1, 3});
}

using FwdRet = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, c10::SymInt, c10::SymInt, at::Tensor, at::Tensor, at::Tensor>;

FwdRet k_sdpa_fwd(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value, const std::optional<at::Tensor>& attn_bias,
                  double dropout_p, bool is_causal, bool /*return_debug_mask*/, std::optional<double> scale) {
  TORCH_CHECK(attn_supported(query, key, value, attn_bias, dropout_p, is_causal, false), "vk: この形の attention は融合カーネルが未対応です");
  ensure_init();
  auto [q, k, v] = prep_qkv(query, key, value);
  int64_t B = q.size(0), H = q.size(1), Sq = q.size(2), Sk = k.size(2), D = q.size(3);
  at::Tensor out = alloc_bshd(B, H, Sq, D, q.options());
  at::Tensor lse = at::empty({B, H, Sq}, q.options().dtype(at::kFloat));
  if (use_coop_attn(q.scalar_type())) {
    int64_t Dp = round_up(D, 16), Sqp = round_up(Sq, 64), Skp = round_up(Sk, 64);
    at::Tensor qp = pad_bhsd(q, Sqp, Dp), kp = pad_bhsd(k, Skp, Dp), vp = pad_bhsd(v, Skp, Dp);
    CoopPush C{};
    // 逆伝播の delta を P と整合させるため、丸め前の fp32 の O も作って（philox_offset の枠で）持ち回す
    at::Tensor o32 = at::empty({B, Sq, H, D}, q.options().dtype(at::kFloat));
    C.a[0] = addr_of(qp), C.a[1] = addr_of(kp), C.a[2] = addr_of(vp), C.a[4] = addr_of(lse), C.a[6] = addr_of(out), C.a[7] = addr_of(o32);
    C.B = (uint32_t)B, C.H = (uint32_t)H, C.Sq = (uint32_t)Sq, C.Sk = (uint32_t)Sk, C.D = (uint32_t)D;
    C.Sqp = (uint32_t)Sqp, C.Skp = (uint32_t)Skp, C.Dp = (uint32_t)Dp;
    C.scale = (float)scale.value_or(1.0 / std::sqrt((double)D));
    dispatch_coop("attn_coop_fwd", q.scalar_type(), C, Sqp / 64, B * H, "aten::_scaled_dot_product_fused_attention_overrideable");
    at::Tensor el = at::empty({0}, at::TensorOptions().dtype(at::kLong));
    return FwdRet{out, lse, el, el, c10::SymInt(Sq), c10::SymInt(Sk), el, o32, at::empty({0}, at::TensorOptions().dtype(q.scalar_type()))};
  }
  at::Tensor o32;  // q が f32 のときは不要（out がそのまま fp32）
  if (q.scalar_type() != at::kFloat) o32 = at::empty({B, Sq, H, D}, q.options().dtype(at::kFloat));
  AttnPush P{};
  P.a[0] = addr_of(q);
  P.a[1] = addr_of(k);
  P.a[2] = addr_of(v);
  P.a[3] = addr_of(out);
  P.a[4] = addr_of(lse);
  if (o32.defined()) P.a[5] = addr_of(o32);
  P.qs[0] = u32(q.stride(0)), P.qs[1] = u32(q.stride(1)), P.qs[2] = u32(q.stride(2));
  P.ks[0] = u32(k.stride(0)), P.ks[1] = u32(k.stride(1)), P.ks[2] = u32(k.stride(2));
  P.B = (uint32_t)B, P.H = (uint32_t)H, P.Sq = (uint32_t)Sq, P.Sk = (uint32_t)Sk, P.D = (uint32_t)D;
  P.scale = (float)scale.value_or(1.0 / std::sqrt((double)D));
  dispatch("attn_fwd", kind_name(q.scalar_type()), P, (Sq + 31) / 32, B * H, "aten::_scaled_dot_product_fused_attention_overrideable");
  at::Tensor empty_l = at::empty({0}, at::TensorOptions().dtype(at::kLong));
  return FwdRet{out, lse, empty_l, empty_l, c10::SymInt(Sq), c10::SymInt(Sk), empty_l, o32.defined() ? o32 : empty_l, at::empty({0}, at::TensorOptions().dtype(q.scalar_type()))};
}

using BwdRet = std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor>;

BwdRet k_sdpa_bwd(const at::Tensor& grad_out, const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
                  const at::Tensor& attn_bias, std::array<bool, 4> mask, const at::Tensor& out, const at::Tensor& lse, const at::Tensor&,
                  const at::Tensor&, c10::SymInt, c10::SymInt, double dropout_p, bool is_causal, const at::Tensor&, const at::Tensor& o32_saved,
                  std::optional<double> scale) {
  TORCH_CHECK(attn_supported(query, key, value, attn_bias.defined() ? std::optional<at::Tensor>(attn_bias) : std::nullopt, dropout_p, is_causal, false) &&
                  is_vk(grad_out) && is_vk(out) && is_vk(lse),
              "vk: この形の attention の backward は融合カーネルが未対応です");
  ensure_init();
  auto [q, k, v] = prep_qkv(query, key, value);
  int64_t B = q.size(0), H = q.size(1), Sq = q.size(2), Sk = k.size(2), D = q.size(3);
  at::ScalarType dt = q.scalar_type();
  at::Tensor dO = grad_out.scalar_type() == dt ? grad_out : grad_out.to(dt);
  if (dO.stride(3) != 1) dO = dO.contiguous();
  at::Tensor o_bshd = (out.scalar_type() == dt ? out : out.to(dt)).permute({0, 2, 1, 3}).contiguous();  // [B,Sq,H,D] 連続
  at::Tensor lse_c = lse.contiguous();
  at::Tensor delta = at::empty({B, H, Sq}, q.options().dtype(at::kFloat));
  const std::string dn = kind_name(dt);

  AttnPush P{};
  P.a[0] = addr_of(q);
  P.a[1] = addr_of(k);
  P.a[2] = addr_of(v);
  P.a[3] = addr_of(dO);
  P.a[4] = addr_of(lse_c);
  P.a[5] = addr_of(delta);
  P.qs[0] = u32(q.stride(0)), P.qs[1] = u32(q.stride(1)), P.qs[2] = u32(q.stride(2));
  P.ks[0] = u32(k.stride(0)), P.ks[1] = u32(k.stride(1)), P.ks[2] = u32(k.stride(2));
  P.ds[0] = u32(dO.stride(0)), P.ds[1] = u32(dO.stride(1)), P.ds[2] = u32(dO.stride(2));
  P.B = (uint32_t)B, P.H = (uint32_t)H, P.Sq = (uint32_t)Sq, P.Sk = (uint32_t)Sk, P.D = (uint32_t)D;
  P.scale = (float)scale.value_or(1.0 / std::sqrt((double)D));
  const char* an = "aten::_scaled_dot_product_fused_attention_overrideable_backward";

  // delta = sum_d out * dO
  AttnPush PD = P;
  if (dt != at::kFloat && o32_saved.defined() && o32_saved.scalar_type() == at::kFloat && is_vk(o32_saved) && o32_saved.numel() == B * Sq * H * D) {
    PD.a[0] = addr_of(o32_saved);  // 丸め前の fp32 の O
    PD.pad = 1;
  } else {
    PD.a[0] = addr_of(o_bshd);
  }
  dispatch("attn_delta", dn, PD, (B * H * Sq + 255) / 256, 1, an);

  at::Tensor dq, dk, dv;
  if (use_coop_attn(dt)) {
    int64_t Dp = round_up(D, 16), Sqp = round_up(Sq, 64), Skp = round_up(Sk, 64);
    at::Tensor qp = pad_bhsd(q, Sqp, Dp), kp = pad_bhsd(k, Skp, Dp), vp = pad_bhsd(v, Skp, Dp), dop = pad_bhsd(dO, Sqp, Dp);
    CoopPush C{};
    C.a[0] = addr_of(qp), C.a[1] = addr_of(kp), C.a[2] = addr_of(vp), C.a[3] = addr_of(dop), C.a[4] = addr_of(lse_c), C.a[5] = addr_of(delta);
    C.B = (uint32_t)B, C.H = (uint32_t)H, C.Sq = (uint32_t)Sq, C.Sk = (uint32_t)Sk, C.D = (uint32_t)D;
    C.Sqp = (uint32_t)Sqp, C.Skp = (uint32_t)Skp, C.Dp = (uint32_t)Dp;
    C.scale = P.scale;
    if (mask[1] || mask[2]) {
      at::Tensor dk_ = alloc_bshd(B, H, Sk, D, q.options());
      at::Tensor dv_ = alloc_bshd(B, H, Sk, D, q.options());
      CoopPush C1 = C;
      C1.a[6] = addr_of(dk_), C1.a[7] = addr_of(dv_);
      dispatch_coop("attn_coop_dkdv", dt, C1, Skp / 64, B * H, an);
      dk = dk_;
      dv = dv_;
    }
    if (mask[0]) {
      at::Tensor dq_ = alloc_bshd(B, H, Sq, D, q.options());
      CoopPush C2 = C;
      C2.a[6] = addr_of(dq_);
      dispatch_coop("attn_coop_dq", dt, C2, Sqp / 64, B * H, an);
      dq = dq_;
    }
    return BwdRet{dq, dk, dv, at::Tensor()};
  }
  if (mask[1] || mask[2]) {
    at::Tensor dk_ = alloc_bshd(B, H, Sk, D, q.options());
    at::Tensor dv_ = alloc_bshd(B, H, Sk, D, q.options());
    AttnPush P1 = P;
    P1.a[6] = addr_of(dk_);
    P1.a[7] = addr_of(dv_);
    dispatch("attn_dkdv", dn, P1, (Sk + 31) / 32, B * H, an);
    dk = dk_;
    dv = dv_;
  }
  if (mask[0]) {
    at::Tensor dq_ = alloc_bshd(B, H, Sq, D, q.options());
    AttnPush P2 = P;
    P2.a[6] = addr_of(dq_);
    dispatch("attn_dq", dn, P2, (Sq + 31) / 32, B * H, an);
    dq = dq_;
  }
  return BwdRet{dq, dk, dv, at::Tensor()};
}

}  // namespace

namespace at::native {
REGISTER_PRIVATEUSE1_DISPATCH(_fused_sdp_choice_stub, &fused_sdp_choice_vk);
}

namespace vkt {
void set_fused_attention(int mode) { policy_ref().store(mode); }
}  // namespace vkt

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_scaled_dot_product_fused_attention_overrideable", &k_sdpa_fwd);
  m.impl("_scaled_dot_product_fused_attention_overrideable_backward", &k_sdpa_bwd);
}
