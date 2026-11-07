//! vkrt: torch-vulkan 用の Vulkan ランタイム（C ABI の cdylib）。
//!
//! 方針:
//! - デバイスメモリは `VK_KHR_buffer_device_address` のアドレスをそのまま PyTorch の `data_ptr` にする。
//!   ビュー（ポインタ + オフセット）がそのまま成立し、シェーダは `buffer_reference` で直接読み書きする。
//! - 1 回の割り当ては `maxMemoryAllocationSize`（このPCでは 2GB）以下。大きい一時テンソルは設計で避ける。
//! - サブグループサイズは決め打ちしない（IREE の失敗の教訓）。カーネルは subgroup 演算に依存させない。
//! - 現段階は同期実行（1 操作ごとに submit + fence 待ち）。コマンドのバッチ化は後続の最適化で行う。

use ash::{Device, Entry, Instance, ext::debug_utils, vk};
use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicUsize, Ordering},
    ffi::{CStr, c_char, c_void},
    io::Cursor,
    sync::{Mutex, OnceLock},
};

/// 組み込みカーネル: (名前, SPIR-V, push constant のバイト数)
mod kernels_gen;

/// 手書きカーネル: (名前, SPIR-V, push constant のバイト数)
const BASE_KERNELS: &[(&str, &[u8], u32)] = &[("add_f32", include_bytes!("../shaders/add_f32.spv"), 32)];

/// 全カーネル（手書き + tools/gen_shaders.py の生成物）。番号 = この並びの添字。
fn kernel_table() -> &'static [(&'static str, &'static [u8], u32)] {
    static T: OnceLock<Vec<(&'static str, &'static [u8], u32)>> = OnceLock::new();
    T.get_or_init(|| BASE_KERNELS.iter().chain(kernels_gen::GEN_KERNELS.iter()).copied().collect())
}

const STAGING_BYTES: u64 = 16 << 20;
const PUSH_MAX: u32 = 256; // maxPushConstantsSize（AMD gfx1201 実測 256。起動時に実機の値と照合する）
const VALIDATION: &CStr = c"VK_LAYER_KHRONOS_validation";

const ALIGN: u64 = 512;
const SLAB_BYTES: u64 = 256 << 20;
/// これ以上の要求は専用バッファにする（解放時にドライバへ返す）。
const DEDICATED_MIN: u64 = 64 << 20;

/// デバイスメモリ 1 枚（1 つの VkBuffer）。内部をサブアロケートする。
struct Slab {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: u64,
    dedicated: bool,
    used: u64,
    /// 空き領域: オフセット -> 長さ（隣接は結合済み）
    free: BTreeMap<u64, u64>,
}

/// 利用者に渡したブロック。
struct Block {
    /// 丸めた後のサイズ（スラブ内で実際に占有する量）
    size: u64,
    /// 利用者が要求したサイズ。範囲検査はこちらで行う（パディングへのアクセスも検出する）。
    req: u64,
}

struct Kernel {
    /// 初回の dispatch で作る（118 本以上を起動時に全部作らない）
    pipeline: Option<vk::Pipeline>,
    push_size: u32,
}

struct Staging {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    size: u64,
}

struct Runtime {
    _entry: Entry,
    instance: Instance,
    device: Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    layout: vk::PipelineLayout,
    kernels: Vec<Kernel>,
    slabs: BTreeMap<u64, Slab>,
    blocks: BTreeMap<u64, Block>,
    staging: Staging,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    mem_device_local: u32,
    max_alloc: u64,
    device_name: String,
    live_bytes: u64,
}

// vk のハンドルは u64 / ポインタ。Mutex 経由でしか触らないので Send にしてよい。
unsafe impl Send for Runtime {}

static VALIDATION_ERRORS: AtomicUsize = AtomicUsize::new(0);
static RT: OnceLock<Mutex<Runtime>> = OnceLock::new();
static LAST_ERROR: Mutex<String> = Mutex::new(String::new());

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user: *mut c_void,
) -> vk::Bool32 {
    let msg = unsafe { CStr::from_ptr((*data).p_message) }.to_string_lossy();
    if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        VALIDATION_ERRORS.fetch_add(1, Ordering::Relaxed);
    }
    eprintln!("[vkrt validation {severity:?}] {msg}");
    vk::FALSE
}

type R<T> = Result<T, String>;

fn vkerr<E: std::fmt::Debug>(what: &str) -> impl FnOnce(E) -> String + '_ {
    move |e| format!("{what}: {e:?}")
}

fn find_memory_type(props: &vk::PhysicalDeviceMemoryProperties, bits: u32, want: vk::MemoryPropertyFlags) -> Option<u32> {
    (0..props.memory_type_count)
        .find(|&i| bits & (1 << i) != 0 && props.memory_types[i as usize].property_flags.contains(want))
}

impl Runtime {
    fn new() -> R<Runtime> {
        unsafe {
            let entry = Entry::load().map_err(vkerr("Vulkan ローダの読み込み"))?;
            let layers = entry.enumerate_instance_layer_properties().map_err(vkerr("レイヤ列挙"))?;
            let want_validation = std::env::var_os("VKRT_VALIDATION").is_some()
                && layers.iter().any(|l| l.layer_name_as_c_str().is_ok_and(|n| n == VALIDATION));
            let layer_ptrs: Vec<*const c_char> = if want_validation { vec![VALIDATION.as_ptr()] } else { vec![] };
            let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
            let ext_ptrs: Vec<*const c_char> = if want_validation { vec![debug_utils::NAME.as_ptr()] } else { vec![] };
            let instance = entry
                .create_instance(
                    &vk::InstanceCreateInfo::default()
                        .application_info(&app)
                        .enabled_layer_names(&layer_ptrs)
                        .enabled_extension_names(&ext_ptrs),
                    None,
                )
                .map_err(vkerr("インスタンス作成"))?;
            if want_validation {
                let loader = debug_utils::Instance::new(&entry, &instance);
                loader
                    .create_debug_utils_messenger(
                        &vk::DebugUtilsMessengerCreateInfoEXT::default()
                            .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR)
                            .message_type(
                                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                            )
                            .pfn_user_callback(Some(debug_callback)),
                        None,
                    )
                    .map_err(vkerr("デバッグメッセンジャー"))?;
            }

            let pds = instance.enumerate_physical_devices().map_err(vkerr("物理デバイス列挙"))?;
            let pd = pds
                .iter()
                .copied()
                .find(|&p| instance.get_physical_device_properties(p).device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
                .or(pds.first().copied())
                .ok_or("Vulkan デバイスが見つかりません")?;
            let props = instance.get_physical_device_properties(pd);
            let device_name = props.device_name_as_c_str().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();

            if props.limits.max_push_constants_size < PUSH_MAX {
                return Err(format!(
                    "maxPushConstantsSize={} が必要量 {PUSH_MAX} に足りません",
                    props.limits.max_push_constants_size
                ));
            }
            let mut m3 = vk::PhysicalDeviceMaintenance3Properties::default();
            let mut p2 = vk::PhysicalDeviceProperties2::default().push_next(&mut m3);
            instance.get_physical_device_properties2(pd, &mut p2);
            let max_alloc = m3.max_memory_allocation_size;

            let qfamily = instance
                .get_physical_device_queue_family_properties(pd)
                .iter()
                .position(|q| q.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .ok_or("compute キューがありません")? as u32;
            let prio = [1.0f32];
            let qinfo = [vk::DeviceQueueCreateInfo::default().queue_family_index(qfamily).queue_priorities(&prio)];
            let mut f11 = vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
                .buffer_device_address(true)
                .shader_float16(true)
                .shader_int8(true)
                .storage_buffer8_bit_access(true);
            let mut f2 = vk::PhysicalDeviceFeatures2::default()
                .features(vk::PhysicalDeviceFeatures::default().shader_int64(true).shader_int16(true))
                .push_next(&mut f11)
                .push_next(&mut f12);
            let device = instance
                .create_device(pd, &vk::DeviceCreateInfo::default().queue_create_infos(&qinfo).push_next(&mut f2), None)
                .map_err(vkerr("デバイス作成"))?;
            let queue = device.get_device_queue(qfamily, 0);

            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(qfamily)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .map_err(vkerr("コマンドプール"))?;
            let cmd = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .map_err(vkerr("コマンドバッファ"))?[0];
            let fence = device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(vkerr("フェンス"))?;

            let mem_props = instance.get_physical_device_memory_properties(pd);
            // デバイスローカルのメモリタイプ（バッファの memory_type_bits との照合は割り当て時に行う）
            let mem_device_local = (0..mem_props.memory_type_count)
                .find(|&i| mem_props.memory_types[i as usize].property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL))
                .ok_or("DEVICE_LOCAL メモリがありません")?;

            // ステージングバッファ（ホスト可視）
            let sbuf = device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(STAGING_BYTES)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST),
                    None,
                )
                .map_err(vkerr("ステージングバッファ"))?;
            let sreq = device.get_buffer_memory_requirements(sbuf);
            let smem_type = find_memory_type(
                &mem_props,
                sreq.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .ok_or("ホスト可視メモリがありません")?;
            let smem = device
                .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(sreq.size).memory_type_index(smem_type), None)
                .map_err(vkerr("ステージングメモリ"))?;
            device.bind_buffer_memory(sbuf, smem, 0).map_err(vkerr("ステージング bind"))?;
            let sptr = device.map_memory(smem, 0, STAGING_BYTES, vk::MemoryMapFlags::empty()).map_err(vkerr("map"))? as *mut u8;

            // パイプライン（descriptor 無し・push constant のみ。バッファは push constant 内のアドレスで渡す）
            let ranges = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(PUSH_MAX)];
            let layout = device
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges), None)
                .map_err(vkerr("パイプラインレイアウト"))?;
            let kernels = kernel_table().iter().map(|(_, _, push_size)| Kernel { pipeline: None, push_size: *push_size }).collect();

            Ok(Runtime {
                _entry: entry,
                instance,
                device,
                queue,
                pool,
                cmd,
                fence,
                layout,
                kernels,
                slabs: BTreeMap::new(),
                blocks: BTreeMap::new(),
                staging: Staging { buffer: sbuf, memory: smem, ptr: sptr, size: STAGING_BYTES },
                mem_props,
                mem_device_local,
                max_alloc,
                device_name,
                live_bytes: 0,
            })
        }
    }

    /// 1 回分のコマンドを記録して submit し、完了を待つ。
    fn submit_sync(&self, record: impl FnOnce(&Device, vk::CommandBuffer)) -> R<()> {
        unsafe {
            let d = &self.device;
            d.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty()).map_err(vkerr("reset"))?;
            d.begin_command_buffer(
                self.cmd,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(vkerr("begin"))?;
            record(d, self.cmd);
            d.end_command_buffer(self.cmd).map_err(vkerr("end"))?;
            let cmds = [self.cmd];
            d.queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], self.fence).map_err(vkerr("submit"))?;
            d.wait_for_fences(&[self.fence], true, u64::MAX).map_err(vkerr("wait"))?;
            d.reset_fences(&[self.fence]).map_err(vkerr("reset fence"))?;
        }
        Ok(())
    }

    /// 新しいスラブ（1 つの VkBuffer + VkDeviceMemory）を作る。
    fn new_slab(&mut self, size: u64, dedicated: bool) -> R<u64> {
        unsafe {
            let d = &self.device;
            let usage = vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS;
            let buffer = d
                .create_buffer(&vk::BufferCreateInfo::default().size(size).usage(usage), None)
                .map_err(vkerr("create_buffer"))?;
            let req = d.get_buffer_memory_requirements(buffer);
            let ty = find_memory_type(&self.mem_props, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .unwrap_or(self.mem_device_local);
            let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let memory = d
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(ty).push_next(&mut flags),
                    None,
                )
                .map_err(|e| {
                    d.destroy_buffer(buffer, None);
                    format!("allocate_memory({} bytes): {e:?}", req.size)
                })?;
            d.bind_buffer_memory(buffer, memory, 0).map_err(vkerr("bind_buffer_memory"))?;
            let base = d.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));
            let mut free = BTreeMap::new();
            free.insert(0, size);
            self.slabs.insert(base, Slab { buffer, memory, size, dedicated, used: 0, free });
            Ok(base)
        }
    }

    /// 既存スラブの空きから best-fit で切り出す。
    fn carve(&mut self, rounded: u64) -> Option<u64> {
        for (base, slab) in self.slabs.iter_mut() {
            if slab.dedicated || slab.size - slab.used < rounded {
                continue;
            }
            let best = slab.free.iter().filter(|&(_, &len)| len >= rounded).min_by_key(|&(_, &len)| len).map(|(&o, &l)| (o, l));
            if let Some((off, len)) = best {
                slab.free.remove(&off);
                if len > rounded {
                    slab.free.insert(off + rounded, len - rounded);
                }
                slab.used += rounded;
                return Some(base + off);
            }
        }
        None
    }

    fn alloc(&mut self, size: u64) -> R<u64> {
        if size == 0 {
            return Err("サイズ 0 の割り当て".into());
        }
        let rounded = size.div_ceil(ALIGN) * ALIGN;
        if rounded > self.max_alloc {
            return Err(format!("割り当てサイズ {size} が maxMemoryAllocationSize {} を超えています", self.max_alloc));
        }
        let addr = if rounded >= DEDICATED_MIN {
            let base = self.new_slab_with_retry(rounded, true)?;
            let slab = self.slabs.get_mut(&base).unwrap();
            slab.free.clear();
            slab.used = rounded;
            base
        } else if let Some(a) = self.carve(rounded) {
            a
        } else {
            self.new_slab_with_retry(SLAB_BYTES, false)?;
            self.carve(rounded).ok_or("スラブ作成直後の割り当てに失敗")?
        };
        self.blocks.insert(addr, Block { size: rounded, req: size });
        self.live_bytes += rounded;
        Ok(addr)
    }

    /// OOM のときは空きスラブを返却して 1 回だけ再試行する。
    fn new_slab_with_retry(&mut self, size: u64, dedicated: bool) -> R<u64> {
        match self.new_slab(size, dedicated) {
            Ok(b) => Ok(b),
            Err(first) => {
                if self.empty_cache() == 0 {
                    return Err(first);
                }
                let live = self.live_bytes;
                self.new_slab(size, dedicated).map_err(|e| format!("{e}（空きスラブ返却後も失敗。確保中 {live} bytes）"))
            }
        }
    }

    fn free(&mut self, addr: u64) -> R<()> {
        let blk = self.blocks.remove(&addr).ok_or_else(|| format!("未知のアドレス {addr:#x} の free"))?;
        let (&base, _) = self.slabs.range(..=addr).next_back().ok_or("スラブが見つかりません")?;
        let slab = self.slabs.get_mut(&base).unwrap();
        slab.used -= blk.size;
        self.live_bytes -= blk.size;
        if slab.dedicated {
            let slab = self.slabs.remove(&base).unwrap();
            unsafe {
                self.device.destroy_buffer(slab.buffer, None);
                self.device.free_memory(slab.memory, None);
            }
            return Ok(());
        }
        // 空きリストへ戻し、前後の空きと結合する
        let mut off = addr - base;
        let mut len = blk.size;
        if let Some((&po, &pl)) = slab.free.range(..off).next_back() {
            if po + pl == off {
                slab.free.remove(&po);
                off = po;
                len += pl;
            }
        }
        if let Some((&no, &nl)) = slab.free.range(off + len..).next() {
            if no == off + len {
                slab.free.remove(&no);
                len += nl;
            }
        }
        slab.free.insert(off, len);
        Ok(())
    }

    /// 完全に空いたスラブをドライバへ返す。返したスラブ数を返す。
    fn empty_cache(&mut self) -> usize {
        let empty: Vec<u64> = self.slabs.iter().filter(|(_, s)| s.used == 0).map(|(&b, _)| b).collect();
        for b in &empty {
            let slab = self.slabs.remove(b).unwrap();
            unsafe {
                self.device.destroy_buffer(slab.buffer, None);
                self.device.free_memory(slab.memory, None);
            }
        }
        empty.len()
    }

    fn reserved_bytes(&self) -> u64 {
        self.slabs.values().map(|s| s.size).sum()
    }

    /// アドレス範囲 [addr, addr+len) が 1 つのブロックに収まることを確認し、(buffer, buffer 内オフセット) を返す。
    fn resolve(&self, addr: u64, len: u64) -> R<(vk::Buffer, u64)> {
        let (bbase, blk) = self.blocks.range(..=addr).next_back().ok_or_else(|| format!("未知のアドレス {addr:#x}"))?;
        if addr + len > bbase + blk.req {
            return Err(format!("範囲外アクセス: addr={addr:#x} len={len} (ブロック {bbase:#x} 要求サイズ {})", blk.req));
        }
        let (sbase, slab) = self.slabs.range(..=addr).next_back().ok_or("スラブが見つかりません")?;
        Ok((slab.buffer, addr - sbase))
    }

    fn upload(&self, dst: u64, src: *const u8, n: u64) -> R<()> {
        let (buf, off) = self.resolve(dst, n)?;
        let mut done = 0u64;
        while done < n {
            let chunk = (n - done).min(self.staging.size);
            unsafe { std::ptr::copy_nonoverlapping(src.add(done as usize), self.staging.ptr, chunk as usize) };
            let (sb, o) = (self.staging.buffer, off + done);
            self.submit_sync(|d, c| unsafe {
                d.cmd_copy_buffer(c, sb, buf, &[vk::BufferCopy::default().src_offset(0).dst_offset(o).size(chunk)]);
            })?;
            done += chunk;
        }
        Ok(())
    }

    fn download(&self, dst: *mut u8, src: u64, n: u64) -> R<()> {
        let (buf, off) = self.resolve(src, n)?;
        let mut done = 0u64;
        while done < n {
            let chunk = (n - done).min(self.staging.size);
            let (sb, o) = (self.staging.buffer, off + done);
            self.submit_sync(|d, c| unsafe {
                d.cmd_copy_buffer(c, buf, sb, &[vk::BufferCopy::default().src_offset(o).dst_offset(0).size(chunk)]);
            })?;
            unsafe { std::ptr::copy_nonoverlapping(self.staging.ptr, dst.add(done as usize), chunk as usize) };
            done += chunk;
        }
        Ok(())
    }

    fn copy_d2d(&self, dst: u64, src: u64, n: u64) -> R<()> {
        let (db, doff) = self.resolve(dst, n)?;
        let (sb, soff) = self.resolve(src, n)?;
        self.submit_sync(|d, c| unsafe {
            d.cmd_copy_buffer(c, sb, db, &[vk::BufferCopy::default().src_offset(soff).dst_offset(doff).size(n)]);
        })
    }

    fn ensure_pipeline(&mut self, kernel: usize) -> R<vk::Pipeline> {
        if let Some(p) = self.kernels[kernel].pipeline {
            return Ok(p);
        }
        let (name, spv, _) = kernel_table()[kernel];
        unsafe {
            let words = ash::util::read_spv(&mut Cursor::new(spv)).map_err(vkerr("SPIR-V 読み込み"))?;
            let module = self
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)
                .map_err(vkerr(name))?;
            let stage =
                vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            let r = self.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default().stage(stage).layout(self.layout)],
                None,
            );
            self.device.destroy_shader_module(module, None);
            let pipelines = r.map_err(|(_, e)| format!("パイプライン {name}: {e:?}"))?;
            self.kernels[kernel].pipeline = Some(pipelines[0]);
            Ok(pipelines[0])
        }
    }

    fn dispatch(&mut self, kernel: u32, push: &[u8], gx: u32, gy: u32, gz: u32) -> R<()> {
        let idx = kernel as usize;
        let push_size = self.kernels.get(idx).ok_or("不明なカーネル")?.push_size;
        if push.len() as u32 != push_size || push.len() as u32 > PUSH_MAX {
            return Err(format!("push constant のサイズ不一致: {} (期待 {})", push.len(), push_size));
        }
        let pipeline = self.ensure_pipeline(idx)?;
        self.submit_sync(|d, c| unsafe {
            d.cmd_bind_pipeline(c, vk::PipelineBindPoint::COMPUTE, pipeline);
            d.cmd_push_constants(c, self.layout, vk::ShaderStageFlags::COMPUTE, 0, push);
            d.cmd_dispatch(c, gx, gy, gz);
        })
    }
}

fn with_rt<T>(f: impl FnOnce(&mut Runtime) -> R<T>) -> R<T> {
    let rt = RT.get().ok_or("vkrt_init が呼ばれていません")?;
    let mut g = rt.lock().map_err(|_| "ランタイムのロックが壊れています".to_string())?;
    f(&mut g)
}

fn report<T: Default>(r: R<T>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => {
            *LAST_ERROR.lock().unwrap() = e;
            T::default()
        }
    }
}

// ---------------------------------------------------------------------------
// C ABI。失敗時は 0 / 負値を返し、`vkrt_last_error` で理由を取れる。
// ---------------------------------------------------------------------------

/// 初期化（冪等）。成功 0、失敗 -1。
#[unsafe(no_mangle)]
pub extern "C" fn vkrt_init() -> i32 {
    if RT.get().is_some() {
        return 0;
    }
    match Runtime::new() {
        Ok(rt) => {
            let _ = RT.set(Mutex::new(rt));
            0
        }
        Err(e) => {
            *LAST_ERROR.lock().unwrap() = e;
            -1
        }
    }
}

/// 直近のエラーメッセージを buf（cap バイト、NUL 終端）へコピーし、元の長さを返す。
///
/// # Safety
/// `buf` は `cap` バイト以上の書き込み可能な領域を指すこと。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkrt_last_error(buf: *mut c_char, cap: usize) -> usize {
    let s = LAST_ERROR.lock().unwrap().clone();
    if !buf.is_null() && cap > 0 {
        let n = s.len().min(cap - 1);
        unsafe {
            std::ptr::copy_nonoverlapping(s.as_ptr(), buf as *mut u8, n);
            *buf.add(n) = 0;
        }
    }
    s.len()
}

/// 使用デバイス名を buf へ書く。
///
/// # Safety
/// `buf` は `cap` バイト以上の書き込み可能な領域を指すこと。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkrt_device_name(buf: *mut c_char, cap: usize) -> usize {
    let s = report(with_rt(|rt| Ok(rt.device_name.clone())));
    if !buf.is_null() && cap > 0 {
        let n = s.len().min(cap - 1);
        unsafe {
            std::ptr::copy_nonoverlapping(s.as_ptr(), buf as *mut u8, n);
            *buf.add(n) = 0;
        }
    }
    s.len()
}

/// 1 回の割り当ての上限（バイト）。
#[unsafe(no_mangle)]
pub extern "C" fn vkrt_max_alloc() -> u64 {
    report(with_rt(|rt| Ok(rt.max_alloc)))
}

/// 現在確保中の合計バイト数。
#[unsafe(no_mangle)]
pub extern "C" fn vkrt_live_bytes() -> u64 {
    report(with_rt(|rt| Ok(rt.live_bytes)))
}

/// デバイスメモリを確保し、デバイスアドレス（= PyTorch の data_ptr）を返す。失敗時 0。
#[unsafe(no_mangle)]
pub extern "C" fn vkrt_alloc(size: u64) -> u64 {
    report(with_rt(|rt| rt.alloc(size)))
}

/// `vkrt_alloc` で得たアドレスを解放する。成功 0、失敗 -1。
#[unsafe(no_mangle)]
pub extern "C" fn vkrt_free(addr: u64) -> i32 {
    match with_rt(|rt| rt.free(addr)) {
        Ok(()) => 0,
        Err(e) => {
            *LAST_ERROR.lock().unwrap() = e;
            -1
        }
    }
}

fn status(r: R<()>) -> i32 {
    match r {
        Ok(()) => 0,
        Err(e) => {
            *LAST_ERROR.lock().unwrap() = e;
            -1
        }
    }
}

/// ホスト → デバイス。
///
/// # Safety
/// `src` は `n` バイト読めること。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkrt_upload(dst: u64, src: *const c_void, n: u64) -> i32 {
    status(with_rt(|rt| rt.upload(dst, src as *const u8, n)))
}

/// デバイス → ホスト。
///
/// # Safety
/// `dst` は `n` バイト書けること。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkrt_download(dst: *mut c_void, src: u64, n: u64) -> i32 {
    status(with_rt(|rt| rt.download(dst as *mut u8, src, n)))
}

/// デバイス → デバイス（連続領域）。
#[unsafe(no_mangle)]
pub extern "C" fn vkrt_copy_d2d(dst: u64, src: u64, n: u64) -> i32 {
    status(with_rt(|rt| rt.copy_d2d(dst, src, n)))
}

/// 組み込みカーネルの番号を名前から引く。見つからなければ -1。
///
/// # Safety
/// `name` は NUL 終端文字列。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkrt_kernel_index(name: *const c_char) -> i32 {
    let name = unsafe { CStr::from_ptr(name) }.to_string_lossy();
    kernel_table().iter().position(|(n, _, _)| *n == name).map(|i| i as i32).unwrap_or(-1)
}

/// カーネルを dispatch する。push constant には各バッファのデバイスアドレスとパラメータを詰める。
///
/// # Safety
/// `push` は `push_len` バイト読めること。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkrt_dispatch(kernel: u32, push: *const c_void, push_len: u32, gx: u32, gy: u32, gz: u32) -> i32 {
    let bytes = unsafe { std::slice::from_raw_parts(push as *const u8, push_len as usize) };
    status(with_rt(|rt| rt.dispatch(kernel, bytes, gx, gy, gz)))
}

/// 完了待ち（現状は各操作が同期なので何もしない。バッチ化後に意味を持つ）。
#[unsafe(no_mangle)]
pub extern "C" fn vkrt_synchronize() -> i32 {
    status(with_rt(|rt| rt.submit_sync(|_, _| {})))
}

/// 検証レイヤが報告したエラー数（`VKRT_VALIDATION` 設定時のみ増える）。
#[unsafe(no_mangle)]
pub extern "C" fn vkrt_validation_errors() -> usize {
    VALIDATION_ERRORS.load(Ordering::Relaxed)
}

/// 完全に空いたスラブをドライバへ返す（torch.vk.empty_cache 相当）。返したスラブ数。
#[unsafe(no_mangle)]
pub extern "C" fn vkrt_empty_cache() -> usize {
    report(with_rt(|rt| Ok(rt.empty_cache())))
}

/// ドライバから確保済みのバイト数（スラブ合計。確保中 + キャッシュ）。
#[unsafe(no_mangle)]
pub extern "C" fn vkrt_reserved_bytes() -> u64 {
    report(with_rt(|rt| Ok(rt.reserved_bytes())))
}

/// 利用者に渡して未解放のブロック数。
#[unsafe(no_mangle)]
pub extern "C" fn vkrt_live_blocks() -> usize {
    report(with_rt(|rt| Ok(rt.blocks.len())))
}

/// スラブ数（= ドライバへの vkAllocateMemory の保持数）。
#[unsafe(no_mangle)]
pub extern "C" fn vkrt_slab_count() -> usize {
    report(with_rt(|rt| Ok(rt.slabs.len())))
}
