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
const SLAB_BYTES: u64 = 512 << 20;
/// 1 バッチに積む最大コマンド数（超えたら submit する。待たない）。
/// 1 回の submit が長時間走ると、ドライバのハング検出（Linux amdgpu: 約 10 秒、Windows TDR: 約 2 秒）で
/// デバイスが失われる。遅い GPU（RX 550 など）で 4096 個を積むと超えるので、小さめにする。VKRT_MAX_BATCH で上書きできる。
const DEFAULT_MAX_BATCH: u32 = 64;
fn max_batch() -> u32 {
    static V: OnceLock<u32> = OnceLock::new();
    *V.get_or_init(|| std::env::var("VKRT_MAX_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_MAX_BATCH))
}
/// これ以上の要求は専用バッファにする（解放時にドライバへ返す）。
const DEDICATED_MIN: u64 = 256 << 20;

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
    /// HOST_COHERENT でなければ、読み出しの前に invalidate が要る
    coherent: bool,
}

struct Runtime {
    _entry: Entry,
    instance: Instance,
    device: Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    /// 同期転送(upload/download)用
    cmd: vk::CommandBuffer,
    fence: vk::Fence,
    /// バッチ実行用(2 本を交互に使う): 記録中のものを submit したら、もう 1 本に記録しながら GPU が先のバッチを実行する
    bcmd: [vk::CommandBuffer; 2],
    bfence: [vk::Fence; 2],
    bflight: [bool; 2],
    bcur: usize,
    recording: bool,
    pending: u32,
    last_pipeline: Option<vk::Pipeline>,
    /// 協調行列カーネルが使えるか（f16 / bf16）。bit0=f16, bit1=bf16
    coop_caps: u32,
    layout: vk::PipelineLayout,
    kernels: Vec<Kernel>,
    slabs: BTreeMap<u64, Slab>,
    blocks: BTreeMap<u64, Block>,
    staging: Staging,
    /// GPU -> ホストの読み出し専用（HOST_CACHED 優先）。書き込み結合メモリを CPU から読むと約 0.2GB/s まで落ちる（RX 550 で実測）
    readback: Staging,
    mem_props: vk::PhysicalDeviceMemoryProperties,
    mem_device_local: u32,
    max_alloc: u64,
    device_name: String,
    live_bytes: u64,
}

// vk のハンドルは u64 / ポインタ。Mutex 経由でしか触らないので Send にしてよい。
unsafe impl Send for Runtime {}

static VALIDATION_ERRORS: AtomicUsize = AtomicUsize::new(0);
/// 統計(ns / 回数)。0=dispatch 時間, 1=dispatch 回数, 2=転送(upload/download/d2d)時間, 3=転送回数, 4=パイプライン作成時間
static STATS: [AtomicUsize; 8] = [const { AtomicUsize::new(0) }; 8];
fn stat_add(i: usize, v: u64) {
    STATS[i].fetch_add(v as usize, Ordering::Relaxed);
}
/// カーネル別の GPU 実行時間(ns, 回数)。VKRT_PROFILE 設定時のみ有効（1 dispatch ごとに完了を待つので遅くなる）。
static PROFILE: Mutex<BTreeMap<&'static str, (u64, u64)>> = Mutex::new(BTreeMap::new());
static PROFILE_ON: OnceLock<bool> = OnceLock::new();
fn profile_on() -> bool {
    *PROFILE_ON.get_or_init(|| std::env::var_os("VKRT_PROFILE").is_some())
}
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

/// VK_KHR_shader_bfloat16 の機能構造体（ash 0.38 には無いので自前で定義する）。
#[repr(C)]
#[derive(Clone, Copy)]
struct Bf16Features {
    s_type: vk::StructureType,
    p_next: *mut c_void,
    shader_bfloat16_type: vk::Bool32,
    shader_bfloat16_dot_product: vk::Bool32,
    shader_bfloat16_cooperative_matrix: vk::Bool32,
}
impl Default for Bf16Features {
    fn default() -> Self {
        Bf16Features {
            s_type: vk::StructureType::from_raw(1000141000), // VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR
            p_next: std::ptr::null_mut(),
            shader_bfloat16_type: 0,
            shader_bfloat16_dot_product: 0,
            shader_bfloat16_cooperative_matrix: 0,
        }
    }
}
unsafe impl vk::TaggedStructure for Bf16Features {
    const STRUCTURE_TYPE: vk::StructureType = vk::StructureType::from_raw(1000141000);
}
unsafe impl vk::ExtendsPhysicalDeviceFeatures2 for Bf16Features {}
unsafe impl vk::ExtendsDeviceCreateInfo for Bf16Features {}

/// 協調行列カーネルが使うサブグループサイズ（mm_coop.comp.in は 32 を前提に workgroup=128 → 4 サブグループ）
const COOP_SUBGROUP: u32 = 32;

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
            // 協調行列(+bf16)の対応を判定する。サブグループサイズを 32 に固定できることも必須（固定できないなら使わない）。
            let dev_exts = instance.enumerate_device_extension_properties(pd).map_err(vkerr("デバイス拡張列挙"))?;
            let has_ext = |n: &CStr| dev_exts.iter().any(|e| e.extension_name_as_c_str().is_ok_and(|x| x == n));
            let ext_coop = c"VK_KHR_cooperative_matrix";
            let ext_bf16 = c"VK_KHR_shader_bfloat16";
            let mut q_coop = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
            let mut q_bf = Bf16Features::default();
            let mut q13 = vk::PhysicalDeviceVulkan13Features::default();
            let mut q12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut q11 = vk::PhysicalDeviceVulkan11Features::default();
            let mut base_feats = vk::PhysicalDeviceFeatures::default();
            {
                let mut q2 = vk::PhysicalDeviceFeatures2::default().push_next(&mut q13).push_next(&mut q12).push_next(&mut q11);
                if has_ext(ext_coop) {
                    q2 = q2.push_next(&mut q_coop);
                }
                if has_ext(ext_bf16) {
                    q2 = q2.push_next(&mut q_bf);
                }
                instance.get_physical_device_features2(pd, &mut q2);
                base_feats = q2.features;
            }
            // 必須機能。足りなければ、黙って誤動作せず理由を返す。
            let required = [
                (q12.buffer_device_address, "bufferDeviceAddress"),
                (q12.shader_int8, "shaderInt8"),
                (q12.storage_buffer8_bit_access, "storageBuffer8BitAccess"),
                (q11.storage_buffer16_bit_access, "storageBuffer16BitAccess"),
                (base_feats.shader_int64, "shaderInt64"),
                (base_feats.shader_int16, "shaderInt16"),
            ];
            let missing: Vec<&str> = required.iter().filter(|(v, _)| *v == 0).map(|(_, n)| *n).collect();
            if !missing.is_empty() {
                return Err(format!("{device_name} は vkrt の必須機能に対応していません: {missing:?}"));
            }
            // fp16 演算は任意（RX 550 / GTX 1060 には無い）。fp16 のデータ型は pack/unpackHalf2x16 で扱うので不要。
            // VKRT_MASK_FP16 / VKRT_MASK_COOP は、古い GPU の挙動を新しい GPU 上で模擬するためのスイッチ。
            let fp16_ok = q12.shader_float16 != 0 && std::env::var_os("VKRT_MASK_FP16").is_none();
            let mask_coop = std::env::var_os("VKRT_MASK_COOP").is_some();
            let mut sg = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
            {
                let mut pp = vk::PhysicalDeviceProperties2::default().push_next(&mut sg);
                instance.get_physical_device_properties2(pd, &mut pp);
            }
            let coop_ok = !mask_coop
                && fp16_ok  // 協調行列カーネル(bf16 版も)は Float16 capability を宣言する
                && has_ext(ext_coop)
                && q_coop.cooperative_matrix != 0
                && q13.subgroup_size_control != 0
                && q12.vulkan_memory_model != 0
                && q13.compute_full_subgroups != 0
                && sg.min_subgroup_size <= COOP_SUBGROUP
                && sg.max_subgroup_size >= COOP_SUBGROUP
                && sg.required_subgroup_size_stages.contains(vk::ShaderStageFlags::COMPUTE);
            let bf16_ok = coop_ok && has_ext(ext_bf16) && q_bf.shader_bfloat16_type != 0 && q_bf.shader_bfloat16_cooperative_matrix != 0;
                        let coop_caps: u32 = if coop_ok { 1 | if bf16_ok { 2 } else { 0 } } else { 0 };
            let mut dev_ext_ptrs: Vec<*const c_char> = vec![];
            if coop_ok {
                dev_ext_ptrs.push(ext_coop.as_ptr());
            }
            if bf16_ok {
                dev_ext_ptrs.push(ext_bf16.as_ptr());
            }
            let mut e_coop = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default().cooperative_matrix(true);
            let mut e_bf = Bf16Features::default();
            e_bf.shader_bfloat16_type = 1;
            e_bf.shader_bfloat16_cooperative_matrix = 1;
            let mut e13 = vk::PhysicalDeviceVulkan13Features::default().subgroup_size_control(true).compute_full_subgroups(true);

            let mut f11 = vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
            let mut f12 = vk::PhysicalDeviceVulkan12Features::default()
                .buffer_device_address(true)
                .shader_float16(coop_ok)  // 協調行列カーネルだけが使う
                .shader_int8(true)
                .storage_buffer8_bit_access(true)
                .vulkan_memory_model(coop_ok); // 協調行列シェーダ(GL_KHR_memory_scope_semantics)が宣言する
            let mut f2 = vk::PhysicalDeviceFeatures2::default()
                .features(vk::PhysicalDeviceFeatures::default().shader_int64(true).shader_int16(true))
                .push_next(&mut f11)
                .push_next(&mut f12);
            if coop_ok {
                f2 = f2.push_next(&mut e13).push_next(&mut e_coop);
            }
            if bf16_ok {
                f2 = f2.push_next(&mut e_bf);
            }
            let device = instance
                .create_device(pd, &vk::DeviceCreateInfo::default().queue_create_infos(&qinfo).enabled_extension_names(&dev_ext_ptrs).push_next(&mut f2), None)
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

            // 読み出し専用ステージング。CPU から読むので HOST_CACHED を優先する（無ければアップロード用と同じ種類）。
            let rbuf = device
                .create_buffer(
                    &vk::BufferCreateInfo::default().size(STAGING_BYTES).usage(vk::BufferUsageFlags::TRANSFER_DST),
                    None,
                )
                .map_err(vkerr("読み出しバッファ"))?;
            let rreq = device.get_buffer_memory_requirements(rbuf);
            let hv = vk::MemoryPropertyFlags::HOST_VISIBLE;
            let (rtype, rcoherent) = [
                hv | vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_CACHED,
                hv | vk::MemoryPropertyFlags::HOST_CACHED,
                hv | vk::MemoryPropertyFlags::HOST_COHERENT,
            ]
            .iter()
            .find_map(|&f| find_memory_type(&mem_props, rreq.memory_type_bits, f).map(|t| (t, f.contains(vk::MemoryPropertyFlags::HOST_COHERENT))))
            .ok_or("読み出し用のホスト可視メモリがありません")?;
            let rmem = device
                .allocate_memory(&vk::MemoryAllocateInfo::default().allocation_size(rreq.size).memory_type_index(rtype), None)
                .map_err(vkerr("読み出しメモリ"))?;
            device.bind_buffer_memory(rbuf, rmem, 0).map_err(vkerr("読み出し bind"))?;
            let rptr = device.map_memory(rmem, 0, STAGING_BYTES, vk::MemoryMapFlags::empty()).map_err(vkerr("map"))? as *mut u8;

            // パイプライン（descriptor 無し・push constant のみ。バッファは push constant 内のアドレスで渡す）
            let ranges = [vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(PUSH_MAX)];
            let layout = device
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&ranges), None)
                .map_err(vkerr("パイプラインレイアウト"))?;
            let kernels = kernel_table().iter().map(|(_, _, push_size)| Kernel { pipeline: None, push_size: *push_size }).collect();

            let bcmds = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default().command_pool(pool).level(vk::CommandBufferLevel::PRIMARY).command_buffer_count(2),
                )
                .map_err(vkerr("バッチ用コマンドバッファ"))?;
            let bf0 = device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(vkerr("フェンス"))?;
            let bf1 = device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(vkerr("フェンス"))?;

            Ok(Runtime {
                _entry: entry,
                instance,
                device,
                queue,
                pool,
                cmd,
                fence,
                bcmd: [bcmds[0], bcmds[1]],
                bfence: [bf0, bf1],
                bflight: [false, false],
                bcur: 0,
                recording: false,
                pending: 0,
                last_pipeline: None,
                coop_caps,
                layout,
                kernels,
                slabs: BTreeMap::new(),
                blocks: BTreeMap::new(),
                staging: Staging { buffer: sbuf, memory: smem, ptr: sptr, size: STAGING_BYTES, coherent: true },
                readback: Staging { buffer: rbuf, memory: rmem, ptr: rptr, size: STAGING_BYTES, coherent: rcoherent },
                mem_props,
                mem_device_local,
                max_alloc,
                device_name,
                live_bytes: 0,
            })
        }
    }

    /// 1 回分のコマンドを記録して submit し、完了を待つ（upload / download 用）。先に積んだバッチを全部完了させる。
    fn submit_sync(&mut self, record: impl FnOnce(&Device, vk::CommandBuffer)) -> R<()> {
        self.flush(true)?;
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

    /// バッチ i の完了を待つ（飛行中でなければ何もしない）。
    fn wait_batch(&mut self, i: usize) -> R<()> {
        if self.bflight[i] {
            let t0 = std::time::Instant::now();
            unsafe {
                self.device.wait_for_fences(&[self.bfence[i]], true, u64::MAX).map_err(vkerr("バッチ待ち"))?;
                self.device.reset_fences(&[self.bfence[i]]).map_err(vkerr("reset fence"))?;
            }
            self.bflight[i] = false;
            stat_add(7, t0.elapsed().as_nanos() as u64);
        }
        Ok(())
    }

    /// 記録中のバッチを submit する。wait が true なら、飛行中の全バッチの完了まで待つ。
    fn flush(&mut self, wait: bool) -> R<()> {
        if self.recording {
            let i = self.bcur;
            unsafe {
                self.device.end_command_buffer(self.bcmd[i]).map_err(vkerr("end"))?;
                let cmds = [self.bcmd[i]];
                self.device
                    .queue_submit(self.queue, &[vk::SubmitInfo::default().command_buffers(&cmds)], self.bfence[i])
                    .map_err(vkerr("submit"))?;
            }
            self.bflight[i] = true;
            self.recording = false;
            self.bcur ^= 1;
        }
        if wait {
            self.wait_batch(0)?;
            self.wait_batch(1)?;
        }
        Ok(())
    }

    /// コマンドを現在のバッチへ記録する（submit も待ちもしない）。コマンド間には全メモリバリアを入れる。
    fn record(&mut self, f: impl FnOnce(&Device, vk::CommandBuffer)) -> R<()> {
        if !self.recording {
            let i = self.bcur;
            self.wait_batch(i)?; // 2 つ前のバッチがまだ飛行中なら、再利用のために待つ
            unsafe {
                self.device.reset_command_buffer(self.bcmd[i], vk::CommandBufferResetFlags::empty()).map_err(vkerr("reset"))?;
                self.device
                    .begin_command_buffer(self.bcmd[i], &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT))
                    .map_err(vkerr("begin"))?;
            }
            self.recording = true;
            self.pending = 0;
            self.last_pipeline = None;
            // 先に submit したバッチの結果を読めるよう、バッチの先頭にもバリアを置く
            self.barrier(self.bcmd[i]);
        }
        let cmd = self.bcmd[self.bcur];
        f(&self.device, cmd);
        self.barrier(cmd);
        self.pending += 1;
        if self.pending >= max_batch() {
            self.flush(false)?;
        }
        Ok(())
    }

    fn barrier(&self, cmd: vk::CommandBuffer) {
        let mb = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE,
            )];
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &mb,
                &[],
                &[],
            );
        }
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
            self.flush(true)?; // 実行中のコマンドが参照しているかもしれないので、破棄前に完了を待つ
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
        let _ = self.flush(true); // スラブ破棄の前に GPU 完了を待つ
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

    fn upload(&mut self, dst: u64, src: *const u8, n: u64) -> R<()> {
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

    fn download(&mut self, dst: *mut u8, src: u64, n: u64) -> R<()> {
        let (buf, off) = self.resolve(src, n)?;
        let mut done = 0u64;
        while done < n {
            let chunk = (n - done).min(self.readback.size);
            let (sb, o) = (self.readback.buffer, off + done);
            self.submit_sync(|d, c| unsafe {
                d.cmd_copy_buffer(c, buf, sb, &[vk::BufferCopy::default().src_offset(o).dst_offset(0).size(chunk)]);
            })?;
            unsafe {
                if !self.readback.coherent {
                    let range = vk::MappedMemoryRange::default().memory(self.readback.memory).offset(0).size(vk::WHOLE_SIZE);
                    self.device.invalidate_mapped_memory_ranges(&[range]).map_err(vkerr("invalidate"))?;
                }
                std::ptr::copy_nonoverlapping(self.readback.ptr, dst.add(done as usize), chunk as usize);
            }
            done += chunk;
        }
        Ok(())
    }

    fn copy_d2d(&mut self, dst: u64, src: u64, n: u64) -> R<()> {
        let (db, doff) = self.resolve(dst, n)?;
        let (sb, soff) = self.resolve(src, n)?;
        self.record(|d, c| unsafe {
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
            let is_coop = name.starts_with("mm_coop");
            if is_coop {
                let need = if name.ends_with("_bf16") { 2 } else { 1 };
                if self.coop_caps & need == 0 {
                    self.device.destroy_shader_module(module, None);
                    return Err(format!("カーネル {name} はこのデバイスで使えません（協調行列/サブグループサイズ固定が非対応）"));
                }
            }
            let mut req = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default().required_subgroup_size(COOP_SUBGROUP);
            let mut stage =
                vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(module).name(c"main");
            if is_coop {
                stage = stage.flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS).push_next(&mut req);
            }
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
        let t_pipe = std::time::Instant::now();
        let pipeline = self.ensure_pipeline(idx)?;
        stat_add(4, t_pipe.elapsed().as_nanos() as u64);
        let t0 = std::time::Instant::now();
        let layout = self.layout;
        let need_bind = self.last_pipeline != Some(pipeline) || !self.recording;
        let r = self.record(|d, c| unsafe {
            if need_bind {
                d.cmd_bind_pipeline(c, vk::PipelineBindPoint::COMPUTE, pipeline);
            }
            d.cmd_push_constants(c, layout, vk::ShaderStageFlags::COMPUTE, 0, push);
            d.cmd_dispatch(c, gx, gy, gz);
        });
        self.last_pipeline = Some(pipeline);
        if profile_on() {
            let tp = std::time::Instant::now();
            self.flush(true)?;
            let ns = tp.elapsed().as_nanos() as u64;
            let name = kernel_table()[idx].0;
            let mut m = PROFILE.lock().unwrap();
            let e = m.entry(name).or_insert((0, 0));
            e.0 += ns;
            e.1 += 1;
        }
        stat_add(0, t0.elapsed().as_nanos() as u64);
        stat_add(1, 1);
        r
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
    let t0 = std::time::Instant::now();
    let r = status(with_rt(|rt| rt.upload(dst, src as *const u8, n)));
    stat_add(2, t0.elapsed().as_nanos() as u64);
    stat_add(3, 1);
    stat_add(5, n);
    r
}

/// デバイス → ホスト。
///
/// # Safety
/// `dst` は `n` バイト書けること。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkrt_download(dst: *mut c_void, src: u64, n: u64) -> i32 {
    let t0 = std::time::Instant::now();
    let r = status(with_rt(|rt| rt.download(dst as *mut u8, src, n)));
    stat_add(2, t0.elapsed().as_nanos() as u64);
    stat_add(3, 1);
    stat_add(6, n);
    r
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
    status(with_rt(|rt| rt.flush(true)))
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

/// 統計値の取得（番号は STATS の定義参照）。5=アップロード総バイト, 6=ダウンロード総バイト。
#[unsafe(no_mangle)]
pub extern "C" fn vkrt_stat(i: u32) -> u64 {
    STATS.get(i as usize).map(|a| a.load(Ordering::Relaxed) as u64).unwrap_or(0)
}

/// プロファイル結果（"名前\t合計ms\t回数\n" を合計時間の降順で）を buf へ書く。
///
/// # Safety
/// `buf` は `cap` バイト以上の書き込み可能な領域を指すこと。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vkrt_profile_dump(buf: *mut c_char, cap: usize) -> usize {
    let m = PROFILE.lock().unwrap();
    let mut v: Vec<_> = m.iter().collect();
    v.sort_by(|a, b| b.1.0.cmp(&a.1.0));
    let mut out = String::new();
    for (name, (ns, n)) in v {
        out.push_str(&format!("{name}\t{:.3}\t{n}\n", *ns as f64 * 1e-6));
    }
    if !buf.is_null() && cap > 0 {
        let k = out.len().min(cap - 1);
        unsafe {
            std::ptr::copy_nonoverlapping(out.as_ptr(), buf as *mut u8, k);
            *buf.add(k) = 0;
        }
    }
    out.len()
}

/// 使える協調行列カーネルのビットマスク: bit0=f16, bit1=bf16（サブグループサイズ 32 固定が必要）。
#[unsafe(no_mangle)]
pub extern "C" fn vkrt_coop_caps() -> u32 {
    report(with_rt(|rt| Ok(rt.coop_caps)))
}
