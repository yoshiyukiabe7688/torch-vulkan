// PrivateUse1 バックエンド "vk" の C++ シム。デバイスメモリと演算は Rust 製ランタイム vkrt（C ABI）が担う。
// - 実装済みの演算は GPU カーネルで実行し、未実装の演算は CPU フォールバックする（回数を演算名ごとに集計）。
// - 演算の登録は C++ から行う（Python 経由にすると 1 step に数千演算ある学習でオーバーヘッドが大きい）。
// - data_ptr は vkrt が返すデバイスアドレス（VK_KHR_buffer_device_address）。ホストからは参照できない。

#include <ATen/EmptyTensor.h>
#include <ATen/autocast_mode.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/unfold_native.h>
#include <ATen/ops/view_native.h>
#include <c10/core/Allocator.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Exception.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <string>

#include <unordered_map>

#include "common.h"

namespace vkt {

// ---- vkrt 呼び出しの補助 -------------------------------------------------------
std::string last_error() {
  char buf[1024];
  vkrt_last_error(buf, sizeof(buf));
  return buf;
}

void vk_check(int rc, const char* what) {
  TORCH_CHECK(rc == 0, "vkrt ", what, " failed: ", last_error());
}

void ensure_init() {
  static std::once_flag once;
  std::call_once(once, [] { vk_check(vkrt_init(), "init"); });
}

// ---- 統計（フォールバック / GPU カーネルの実行回数） -------------------------------------
std::mutex g_stats_mutex;
std::map<std::string, int64_t> g_fallback_counts;
std::map<std::string, int64_t> g_kernel_counts;
std::atomic<uint64_t> g_fallback_ns{0};
std::map<std::string, uint64_t> g_fallback_ns_by_op;

void count_fallback(const std::string& name) {
  std::lock_guard<std::mutex> lock(g_stats_mutex);
  g_fallback_counts[name]++;
}

void count_kernel(const std::string& name) {
  std::lock_guard<std::mutex> lock(g_stats_mutex);
  g_kernel_counts[name]++;
}

// ---- CPU フォールバック ------------------------------------------------------------------------
void vulkan_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  auto t_start = std::chrono::steady_clock::now();
  struct Done {
    std::chrono::steady_clock::time_point t;
    const c10::OperatorHandle& op;
    ~Done() {
      uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t).count();
      g_fallback_ns += ns;
      std::lock_guard<std::mutex> lock(g_stats_mutex);
      g_fallback_ns_by_op[op.schema().operator_name().name] += ns;
    }
  } done{t_start, op};
  // デバッグ用: VKT_TRACE_FALLBACK=1 でフォールバックした演算のスキーマ(オーバーロード込み)を stderr に出す
  static const bool trace = std::getenv("VKT_TRACE_FALLBACK") != nullptr;
  if (trace) {
    std::cerr << "[vkt fallback] " << op.schema() << std::endl;
    // vk と CPU(0 次元でない)のテンソルが混在していれば警告する。CUDA ならエラーになる状況で、CPU 結果が返って後段が壊れる原因になる。
    auto args = torch::jit::last(*stack, op.schema().arguments().size());
    bool has_vk = false, has_cpu = false;
    std::string desc;
    for (const auto& a : args) {
      if (a.isTensor() && a.toTensor().defined()) {
        const auto& t = a.toTensor();
        if (is_vk(t)) has_vk = true;
        else if (t.dim() > 0) has_cpu = true;
        desc += std::string(is_vk(t) ? " vk" : " cpu") + c10::str(t.sizes()) + c10::str(t.scalar_type());
      }
    }
    std::cerr << "[vkt fallback-args]" << desc << std::endl;
    if (has_vk && has_cpu) std::cerr << "[vkt MIXED-DEVICE] " << op.schema().operator_name() << ":" << desc << std::endl;
  }
  count_fallback(op.schema().operator_name().name);

  // cpu_fallback は「最初のテンソル引数」のデバイスへ結果を戻す。最初が CPU の 0 次元スカラー(例: 1 - x の 1)で
  // 他に vk のテンソルがあると、結果が CPU に置かれて後段が静かに CPU に混ざる。その場合は結果を vk へ移す。
  bool first_is_cpu = false, seen_first = false, has_vk = false;
  {
    auto args = torch::jit::last(*stack, op.schema().arguments().size());
    for (const auto& a : args) {
      if (a.isTensor() && a.toTensor().defined()) {
        if (!seen_first) {
          seen_first = true;
          first_is_cpu = !is_vk(a.toTensor());
        }
        has_vk = has_vk || is_vk(a.toTensor());
      } else if (a.isTensorList()) {
        for (const at::Tensor& t : a.toTensorList()) has_vk = has_vk || is_vk(t);
      }
    }
  }
  at::native::cpu_fallback(op, stack);
  if (first_is_cpu && has_vk) {
    size_t n_ret = op.schema().returns().size();
    auto rets = torch::jit::last(*stack, n_ret);
    for (size_t i = 0; i < n_ret; ++i) {
      auto& v = (*stack)[stack->size() - n_ret + i];
      if (v.isTensor() && v.toTensor().defined() && !is_vk(v.toTensor())) {
        v = v.toTensor().to(c10::Device(kDev, 0));
      }
    }
  }
}

int kernel_index(const std::string& name) {
  static std::mutex m;
  static std::unordered_map<std::string, int> cache;
  std::lock_guard<std::mutex> lock(m);
  auto it = cache.find(name);
  if (it != cache.end()) return it->second;
  ensure_init();
  int k = vkrt_kernel_index(name.c_str());
  TORCH_CHECK(k >= 0, "vkrt: カーネル '", name, "' が見つかりません");
  cache.emplace(name, k);
  return k;
}

}  // namespace vkt

using namespace vkt;

namespace {

// ---- アロケータ ----------------------------------------------------------------------
// 現状は 1 テンソル = 1 回の vkAllocateMemory。割り当て数の上限（約 4096）対策のサブアロケータは後続。
struct VulkanAllocator final : public c10::Allocator {
  static void deleter(void* p) {
    if (p) vkrt_free(reinterpret_cast<uint64_t>(p));
  }
  c10::DataPtr allocate(size_t n) override {
    ensure_init();
    if (n == 0) return {nullptr, nullptr, &deleter, c10::Device(kDev, 0)};
    uint64_t addr = vkrt_alloc(n);
    TORCH_CHECK(addr != 0, "vk: デバイスメモリの確保に失敗 (", n, " bytes): ", last_error());
    void* p = reinterpret_cast<void*>(addr);
    return {p, p, &deleter, c10::Device(kDev, 0)};
  }
  c10::DeleterFnPtr raw_deleter() const override { return &deleter; }
  void copy_data(void* dest, const void* src, std::size_t count) const override {
    vk_check(vkrt_copy_d2d(reinterpret_cast<uint64_t>(dest), reinterpret_cast<uint64_t>(src), count), "copy_data");
  }
};
VulkanAllocator g_allocator;

// ---- デバイスガード（単一デバイス） --------------------------------------------------
struct VulkanGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  c10::DeviceType type() const override { return kDev; }
  c10::Device exchangeDevice(c10::Device) const override { return c10::Device(kDev, 0); }
  c10::Device getDevice() const override { return c10::Device(kDev, 0); }
  void setDevice(c10::Device) const override {}
  void uncheckedSetDevice(c10::Device) const noexcept override {}
  c10::Stream getStream(c10::Device d) const override { return c10::Stream(c10::Stream::DEFAULT, d); }
  c10::Stream exchangeStream(c10::Stream s) const override { return s; }
  bool queryStream(const c10::Stream&) const override { return true; }
  void synchronizeStream(const c10::Stream&) const override { vkrt_synchronize(); }
  c10::DeviceIndex deviceCount() const noexcept override { return 1; }
};
C10_REGISTER_GUARD_IMPL(PrivateUse1, VulkanGuardImpl);

// ---- フック ----------------------------------------------------------------------------
struct VulkanHooks final : public at::PrivateUse1HooksInterface {
  bool isBuilt() const override { return true; }
  bool isAvailable() const override { return true; }
  bool hasPrimaryContext(c10::DeviceIndex) const override { return true; }
  at::Device getDeviceFromPtr(void*) const override { return c10::Device(kDev, 0); }
  bool isPinnedPtr(const void*) const override { return false; }
  // resize_ などでストレージを拡張するときに呼ばれる。
  void resizePrivateUse1Bytes(const c10::Storage& storage, size_t newsize) const override {
    auto* impl = storage.unsafeGetStorageImpl();
    size_t old = impl->nbytes();
    if (newsize == old) return;
    c10::DataPtr fresh = g_allocator.allocate(newsize);
    size_t keep = std::min(old, newsize);
    if (keep > 0) {
      vk_check(vkrt_copy_d2d(reinterpret_cast<uint64_t>(fresh.get()), reinterpret_cast<uint64_t>(impl->data()), keep), "resize copy");
    }
    impl->set_data_ptr_noswap(std::move(fresh));
    impl->set_nbytes(newsize);
  }
};
VulkanHooks g_hooks;
struct HooksRegistrar {
  HooksRegistrar() { at::RegisterPrivateUse1HooksInterface(&g_hooks); }
};
HooksRegistrar g_hooks_registrar;

// ---- テンソルの転送 -----------------------------------------------------------------------
// data_ptr() から最終要素までの要素数（stride は非負を仮定）
int64_t span_elems(const at::Tensor& t) {
  if (t.numel() == 0) return 0;
  int64_t n = 1;
  for (int64_t i = 0; i < t.dim(); ++i) {
    TORCH_CHECK(t.stride(i) >= 0, "vk: 負の stride は未対応");
    n += (t.size(i) - 1) * t.stride(i);
  }
  return n;
}

// ホスト tensor src を、デバイス tensor dst へコピーする（dtype 変換・ブロードキャスト・非連続に対応）。
void copy_cpu_to_vk(const at::Tensor& dst, const at::Tensor& src) {
  if (dst.numel() == 0) return;
  at::Tensor s = src.to(dst.scalar_type()).expand(dst.sizes()).contiguous();
  if (dst.is_contiguous()) {
    vk_check(vkrt_upload(addr_of(dst), s.data_ptr(), dst.nbytes()), "upload");
    return;
  }
  // 非連続の宛先: span を読み出し、ホスト側で view に書き込み、span を書き戻す（read-modify-write）。
  int64_t elems = span_elems(dst);
  size_t bytes = elems * dst.element_size();
  at::Tensor host = at::empty({(int64_t)bytes}, at::TensorOptions().dtype(at::kByte));
  vk_check(vkrt_download(host.data_ptr(), addr_of(dst), bytes), "download(span)");
  at::from_blob(host.data_ptr(), dst.sizes(), dst.strides(), dst.options().device(at::kCPU)).copy_(s);
  vk_check(vkrt_upload(addr_of(dst), host.data_ptr(), bytes), "upload(span)");
}

// デバイス tensor src を、ホスト tensor dst へコピーする。
void copy_vk_to_cpu(const at::Tensor& dst, const at::Tensor& src) {
  if (src.numel() == 0) return;
  int64_t elems = span_elems(src);
  size_t bytes = elems * src.element_size();
  at::Tensor host = at::empty({(int64_t)bytes}, at::TensorOptions().dtype(at::kByte));
  vk_check(vkrt_download(host.data_ptr(), addr_of(src), bytes), "download");
  dst.copy_(at::from_blob(host.data_ptr(), src.sizes(), src.strides(), src.options().device(at::kCPU)));
}

void copy_impl(const at::Tensor& dst, const at::Tensor& src) {
  if (!is_vk(src) && !is_vk(dst)) {
    dst.copy_(src);
  } else if (!is_vk(src)) {
    copy_cpu_to_vk(dst, src);
  } else if (!is_vk(dst)) {
    copy_vk_to_cpu(dst, src);
  } else if (src.scalar_type() == dst.scalar_type() && src.sizes() == dst.sizes() && src.is_contiguous() && dst.is_contiguous()) {
    if (src.numel() > 0 && addr_of(src) != addr_of(dst)) {
      vk_check(vkrt_copy_d2d(addr_of(dst), addr_of(src), dst.nbytes()), "d2d");
    }
  } else if (ew_copy(dst, src.sizes() == dst.sizes() ? src : src.expand(dst.sizes()))) {
    // GPU 上で dtype 変換・ストライド・ブロードキャスト付きコピー
  } else {
    // GPU カーネルが対応しない組み合わせ（整数型など）は、いったんホストを経由する。
    at::Tensor tmp = at::empty(src.sizes(), src.options().device(at::kCPU));
    copy_vk_to_cpu(tmp, src);
    copy_cpu_to_vk(dst, tmp);
  }
}

at::Tensor _copy_from(const at::Tensor& self, const at::Tensor& dst, bool /*non_blocking*/) {
  copy_impl(dst, self);  // self = src
  return dst;
}

at::Tensor _copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
  if (dst.sizes() != self.sizes()) {
    dst.resize_(self.sizes());
  }
  copy_impl(dst, self);
  return dst;
}

// ---- ATen カーネル（メモリ・view） --------------------------------------------------------------
at::Tensor empty_memory_format(
    c10::IntArrayRef size,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout>,
    std::optional<at::Device>,
    std::optional<bool>,
    std::optional<at::MemoryFormat> memory_format) {
  return at::detail::empty_generic(size, &g_allocator, c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
                                   c10::dtype_or_default(dtype), memory_format);
}

at::Tensor empty_strided(
    c10::IntArrayRef size,
    c10::IntArrayRef stride,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout>,
    std::optional<at::Device>,
    std::optional<bool>) {
  return at::detail::empty_strided_generic(size, stride, &g_allocator, c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
                                           c10::dtype_or_default(dtype));
}

// view 系はストレージを共有する必要があり CPU フォールバックでは代替できないので、バックエンド側で登録する。
at::Tensor wrapper_view(const at::Tensor& self, c10::SymIntArrayRef size) {
  return at::native::view(self, C10_AS_INTARRAYREF_SLOW(size));
}

at::Tensor wrapper_as_strided(
    const at::Tensor& self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride, std::optional<c10::SymInt> storage_offset) {
  return at::native::as_strided_tensorimpl(
      self, C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride),
      storage_offset.has_value() ? std::optional<int64_t>(storage_offset->guard_int(__FILE__, __LINE__)) : std::nullopt);
}

at::Tensor wrapper_reshape_alias(const at::Tensor& self, c10::SymIntArrayRef size, c10::SymIntArrayRef stride) {
  return at::native::_reshape_alias(self, C10_AS_INTARRAYREF_SLOW(size), C10_AS_INTARRAYREF_SLOW(stride));
}

at::Tensor wrapper_unfold(const at::Tensor& self, int64_t dimension, int64_t size, int64_t step) {
  return at::native::unfold(self, dimension, size, step);
}

// at::native::resize_ は CPU 版の resize_bytes_cpu を使い、デバイスアドレスにホストの memcpy をしてしまうので自前で実装する。
const at::Tensor& wrapper_resize_(
    const at::Tensor& self, c10::SymIntArrayRef size_sym, std::optional<at::MemoryFormat> memory_format) {
  auto size = C10_AS_INTARRAYREF_SLOW(size_sym);
  auto* impl = self.unsafeGetTensorImpl();
  int64_t numel = 1;
  for (auto s : size) numel *= s;
  if (numel > 0) {
    size_t needed = (impl->storage_offset() + numel) * self.element_size();
    if (needed > self.storage().nbytes()) {
      g_hooks.resizePrivateUse1Bytes(self.storage(), needed);  // 拡張時のみ。縮小ではストレージを縮めない
    }
  }
  impl->set_sizes_contiguous(size);
  if (memory_format.has_value() && *memory_format != at::MemoryFormat::Contiguous && *memory_format != at::MemoryFormat::Preserve) {
    impl->empty_tensor_restride(*memory_format);
  }
  return self;
}

// 畳み込みは未知のバックエンドでは「上書き用」演算(convolution_overrideable)に行き着き、CPU にもその実装が無いので、
// 通常のフォールバックでは動かない。GPU カーネルは未実装（LoRA 学習の演算には含まれない。サンプル画像生成の VAE デコードだけが使う）
// なので、入力を CPU へ移して通常の at::convolution で計算し、結果を戻す。forward のみ（VAE は学習しない）。
at::Tensor k_convolution_overrideable(const at::Tensor& input, const at::Tensor& weight, const std::optional<at::Tensor>& bias,
                                      c10::SymIntArrayRef stride, c10::SymIntArrayRef padding, c10::SymIntArrayRef dilation, bool transposed,
                                      c10::SymIntArrayRef output_padding, c10::SymInt groups) {
  count_fallback("aten::convolution_overrideable");
  std::optional<at::Tensor> b = std::nullopt;
  if (bias.has_value() && bias->defined()) b = bias->to(at::kCPU);
  at::Tensor out = at::convolution_symint(input.to(at::kCPU), weight.to(at::kCPU), b, stride, padding, dilation, transposed, output_padding, groups);
  return out.to(input.device());
}

}  // namespace

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&vulkan_fallback>());
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", &empty_memory_format);
  m.impl("empty_strided", &empty_strided);
  m.impl("_copy_from", &_copy_from);
  m.impl("_copy_from_and_resize", &_copy_from_and_resize);
  m.impl("view", &wrapper_view);
  m.impl("as_strided", &wrapper_as_strided);
  m.impl("_reshape_alias", &wrapper_reshape_alias);
  m.impl("unfold", &wrapper_unfold);
  m.impl("resize_", &wrapper_resize_);
  m.impl("convolution_overrideable", &k_convolution_overrideable);
}

// ---- autocast（torch.autocast(device_type="vk")） -------------------------------------------------------
// 既定は素通し（fallthrough）。CUDA と同じ方針で、行列積系は低精度、softmax / norm / 損失は fp32 に揃える。
TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
  KERNEL_PRIVATEUSEONE(linear, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(mm, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(bmm, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(addmm, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(baddbmm, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(matmul, lower_precision_fp)
  KERNEL_PRIVATEUSEONE(layer_norm, fp32)
  KERNEL_PRIVATEUSEONE(softmax, int, fp32)
  KERNEL_PRIVATEUSEONE(mse_loss, fp32)
}

// ---- Python バインディング ----------------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", [] { ensure_init(); });
  m.def("device_name", [] {
    ensure_init();
    char buf[256];
    vkrt_device_name(buf, sizeof(buf));
    return std::string(buf);
  });
  m.def("max_alloc", [] { ensure_init(); return vkrt_max_alloc(); });
  m.def("live_bytes", [] { ensure_init(); return vkrt_live_bytes(); });
  m.def("synchronize", [] { ensure_init(); vk_check(vkrt_synchronize(), "synchronize"); });
  m.def("empty_cache", [] { ensure_init(); return vkrt_empty_cache(); });
  m.def("reserved_bytes", [] { ensure_init(); return vkrt_reserved_bytes(); });
  m.def("live_blocks", [] { ensure_init(); return vkrt_live_blocks(); });
  m.def("slab_count", [] { ensure_init(); return vkrt_slab_count(); });
  m.def("validation_errors", [] { return vkrt_validation_errors(); });
  m.def("fallback_counts", [] {
    std::lock_guard<std::mutex> lock(g_stats_mutex);
    return g_fallback_counts;
  });
  m.def("kernel_counts", [] {
    std::lock_guard<std::mutex> lock(g_stats_mutex);
    return g_kernel_counts;
  });
  m.def("stats", [] {
    ensure_init();
    std::map<std::string, double> r;
    r["dispatch_s"] = vkrt_stat(0) * 1e-9;
    r["dispatch_n"] = (double)vkrt_stat(1);
    r["transfer_s"] = vkrt_stat(2) * 1e-9;
    r["transfer_n"] = (double)vkrt_stat(3);
    r["pipeline_create_s"] = vkrt_stat(4) * 1e-9;
    r["upload_MB"] = vkrt_stat(5) / 1048576.0;
    r["download_MB"] = vkrt_stat(6) / 1048576.0;
    r["gpu_wait_s"] = vkrt_stat(7) * 1e-9;  // バッチの完了待ち(= GPU が実際に忙しかった時間の目安)
    r["fallback_s"] = g_fallback_ns.load() * 1e-9;
    return r;
  });
  m.def("fallback_time_by_op", [] {
    std::lock_guard<std::mutex> lock(g_stats_mutex);
    std::map<std::string, double> r;
    for (auto& kv : g_fallback_ns_by_op) r[kv.first] = kv.second * 1e-9;
    return r;
  });
  m.def("profile", [] {
    ensure_init();
    std::string buf(1 << 16, '\0');
    size_t n = vkrt_profile_dump(buf.data(), buf.size());
    buf.resize(std::min(n, buf.size() - 1));
    return buf;
  });
  m.def("coop_caps", [] { return vkt::coop_caps(); });
  m.def("set_coop_enabled", [](bool on) { vkt::set_coop_enabled(on); });
  m.def("reset_counts", [] {
    std::lock_guard<std::mutex> lock(g_stats_mutex);
    g_fallback_counts.clear();
    g_kernel_counts.clear();
  });
}
