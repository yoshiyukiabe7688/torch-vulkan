// M1 ビルド疎通用の最小 PrivateUse1 バックエンド（"vulkan"）。
// 現段階の実体はホストメモリ（Vulkan 呼び出しはまだ無い）。目的は次の確認:
//   (1) libtorch 2.8 のヘッダ + MSVC でビルドできること
//   (2) accelerate / autocast / torch.utils.checkpoint が独自デバイスと組み合わせられること（設計書 R1〜R3）
// 未実装の演算は CPU フォールバックし、演算名ごとの回数を集計する。

#include <ATen/EmptyTensor.h>
#include <ATen/autocast_mode.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/unfold_native.h>
#include <ATen/ops/view_native.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/Allocator.h>
#include <c10/core/impl/alloc_cpu.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/util/Exception.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <atomic>
#include <cstring>
#include <map>
#include <mutex>
#include <string>

namespace {

constexpr c10::DeviceType kDev = c10::DeviceType::PrivateUse1;

// ---- フォールバック統計 -------------------------------------------------
std::mutex g_stats_mutex;
std::map<std::string, int64_t> g_fallback_counts;

// ---- アロケータ（ホストメモリ。後で Vulkan バッファに置き換える） -------------
struct VulkanAllocator final : public c10::Allocator {
  static void deleter(void* p) { c10::free_cpu(p); }
  c10::DataPtr allocate(size_t n) override {
    void* p = n ? c10::alloc_cpu(n) : nullptr;
    return {p, p, &deleter, c10::Device(kDev, 0)};
  }
  c10::DeleterFnPtr raw_deleter() const override { return &deleter; }
  void copy_data(void* dest, const void* src, std::size_t count) const override { std::memcpy(dest, src, count); }
};
VulkanAllocator g_allocator;

// ---- デバイスガード（単一デバイス） --------------------------------------
struct VulkanGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  c10::DeviceType type() const override { return kDev; }
  c10::Device exchangeDevice(c10::Device) const override { return c10::Device(kDev, 0); }
  c10::Device getDevice() const override { return c10::Device(kDev, 0); }
  void setDevice(c10::Device) const override {}
  void uncheckedSetDevice(c10::Device) const noexcept override {}
  c10::Stream getStream(c10::Device d) const override { return c10::Stream(c10::Stream::DEFAULT, d); }
  c10::Stream exchangeStream(c10::Stream s) const override { return s; }
  bool queryStream(const c10::Stream&) const override { return true; }
  void synchronizeStream(const c10::Stream&) const override {}
  c10::DeviceIndex deviceCount() const noexcept override { return 1; }
};
C10_REGISTER_GUARD_IMPL(PrivateUse1, VulkanGuardImpl);

// ---- フック ---------------------------------------------------------------
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
    std::memcpy(fresh.get(), impl->data(), std::min(old, newsize));
    impl->set_data_ptr_noswap(std::move(fresh));
    impl->set_nbytes(newsize);
  }
};
VulkanHooks g_hooks;
struct HooksRegistrar {
  HooksRegistrar() { at::RegisterPrivateUse1HooksInterface(&g_hooks); }
};
HooksRegistrar g_hooks_registrar;

// ---- ATen カーネル ----------------------------------------------------------
// ホストメモリ上にあるテンソルを CPU テンソルとして見る（コピーなし）。
at::Tensor as_cpu_view(const at::Tensor& t) {
  if (t.device().type() != kDev) return t;
  return at::from_blob(t.data_ptr(), t.sizes(), t.strides(), t.options().device(at::kCPU));
}

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

at::Tensor _copy_from(const at::Tensor& self, const at::Tensor& dst, bool /*non_blocking*/) {
  // self = src。どちらかが CPU でも、ホストメモリ同士なので CPU の copy_ で足りる。
  as_cpu_view(dst).copy_(as_cpu_view(self));
  return dst;
}

at::Tensor _copy_from_and_resize(const at::Tensor& self, const at::Tensor& dst) {
  if (dst.sizes() != self.sizes()) {
    dst.resize_(self.sizes());
  }
  as_cpu_view(dst).copy_(as_cpu_view(self));
  return dst;
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

const at::Tensor& wrapper_resize_(
    const at::Tensor& self, c10::SymIntArrayRef size, std::optional<at::MemoryFormat> memory_format) {
  return at::native::resize_(self, C10_AS_INTARRAYREF_SLOW(size), memory_format);
}

void vulkan_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  {
    std::lock_guard<std::mutex> lock(g_stats_mutex);
    g_fallback_counts[op.schema().operator_name().name]++;
  }
  at::native::cpu_fallback(op, stack);
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
}

// ---- autocast（torch.autocast(device_type="vk")） -----------------------------
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

// ---- Python バインディング --------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fallback_counts", []() {
    std::lock_guard<std::mutex> lock(g_stats_mutex);
    return g_fallback_counts;
  });
  m.def("reset_fallback_counts", []() {
    std::lock_guard<std::mutex> lock(g_stats_mutex);
    g_fallback_counts.clear();
  });
}
