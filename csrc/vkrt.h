// vkrt（Rust 製 Vulkan ランタイム）の C ABI。実装は vkrt/src/lib.rs。
#pragma once
#include <cstddef>
#include <cstdint>

extern "C" {
int vkrt_init();
size_t vkrt_last_error(char* buf, size_t cap);
size_t vkrt_device_name(char* buf, size_t cap);
uint64_t vkrt_max_alloc();
uint64_t vkrt_live_bytes();
uint64_t vkrt_alloc(uint64_t size);
int vkrt_free(uint64_t addr);
int vkrt_upload(uint64_t dst, const void* src, uint64_t n);
int vkrt_download(void* dst, uint64_t src, uint64_t n);
int vkrt_copy_d2d(uint64_t dst, uint64_t src, uint64_t n);
int vkrt_kernel_index(const char* name);
int vkrt_dispatch(uint32_t kernel, const void* push, uint32_t push_len, uint32_t gx, uint32_t gy, uint32_t gz);
int vkrt_synchronize();
size_t vkrt_empty_cache();
uint64_t vkrt_reserved_bytes();
size_t vkrt_live_blocks();
size_t vkrt_slab_count();
size_t vkrt_validation_errors();
uint64_t vkrt_stat(uint32_t i);
size_t vkrt_profile_dump(char* buf, size_t cap);
uint32_t vkrt_coop_caps();
}
