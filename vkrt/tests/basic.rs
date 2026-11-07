//! vkrt の基本動作: 確保・往復コピー・d2d コピー・add_f32 カーネル・範囲外アクセスの検出。
use std::ffi::c_void;
use vkrt::*;

fn init() {
    assert_eq!(vkrt_init(), 0, "init 失敗");
}

#[test]
fn upload_download_roundtrip_and_d2d() {
    init();
    let n = 1000usize;
    let src: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let a = vkrt_alloc((n * 4) as u64);
    let b = vkrt_alloc((n * 4) as u64);
    assert!(a != 0 && b != 0);
    unsafe {
        assert_eq!(vkrt_upload(a, src.as_ptr() as *const c_void, (n * 4) as u64), 0);
        assert_eq!(vkrt_copy_d2d(b, a, (n * 4) as u64), 0);
        let mut out = vec![0f32; n];
        assert_eq!(vkrt_download(out.as_mut_ptr() as *mut c_void, b, (n * 4) as u64), 0);
        assert_eq!(out, src);
        // オフセット付き（view 相当）: b+400 から 100 要素
        let mut part = vec![0f32; 100];
        assert_eq!(vkrt_download(part.as_mut_ptr() as *mut c_void, b + 400, 400), 0);
        assert_eq!(part, src[100..200]);
    }
    assert_eq!(vkrt_free(a), 0);
    assert_eq!(vkrt_free(b), 0);
}

#[test]
fn large_transfer_exceeds_staging_size() {
    init();
    let n = (40usize << 20) / 4; // 40MB > ステージング 16MB
    let src: Vec<f32> = (0..n).map(|i| (i % 1000) as f32).collect();
    let a = vkrt_alloc((n * 4) as u64);
    unsafe {
        assert_eq!(vkrt_upload(a, src.as_ptr() as *const c_void, (n * 4) as u64), 0);
        let mut out = vec![0f32; n];
        assert_eq!(vkrt_download(out.as_mut_ptr() as *mut c_void, a, (n * 4) as u64), 0);
        assert!(out == src);
    }
    assert_eq!(vkrt_free(a), 0);
}

#[test]
fn add_f32_kernel_matches_cpu() {
    init();
    let n = 100_003usize; // 256 の倍数でない
    let x: Vec<f32> = (0..n).map(|i| (i as f32).sin()).collect();
    let y: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).cos()).collect();
    let (da, db, dc) = (vkrt_alloc((n * 4) as u64), vkrt_alloc((n * 4) as u64), vkrt_alloc((n * 4) as u64));
    unsafe {
        vkrt_upload(da, x.as_ptr() as *const c_void, (n * 4) as u64);
        vkrt_upload(db, y.as_ptr() as *const c_void, (n * 4) as u64);
        let k = vkrt_kernel_index(c"add_f32".as_ptr());
        assert!(k >= 0);
        let push: [u64; 4] = [da, db, dc, n as u64];
        let groups = n.div_ceil(256) as u32;
        assert_eq!(vkrt_dispatch(k as u32, push.as_ptr() as *const c_void, 32, groups, 1, 1), 0);
        let mut out = vec![0f32; n];
        vkrt_download(out.as_mut_ptr() as *mut c_void, dc, (n * 4) as u64);
        for i in 0..n {
            assert_eq!(out[i], x[i] + y[i], "i={i}");
        }
    }
    vkrt_free(da);
    vkrt_free(db);
    vkrt_free(dc);
}

#[test]
fn out_of_range_and_unknown_addresses_are_rejected() {
    init();
    let a = vkrt_alloc(64);
    let mut buf = [0u8; 128];
    unsafe {
        assert_eq!(vkrt_download(buf.as_mut_ptr() as *mut c_void, a, 128), -1, "範囲外は拒否される");
        assert_eq!(vkrt_download(buf.as_mut_ptr() as *mut c_void, 0x1234, 4), -1, "未知アドレスは拒否される");
    }
    assert_eq!(vkrt_free(0x1234), -1);
    assert_eq!(vkrt_free(a), 0);
    assert_eq!(vkrt_free(a), -1, "二重 free は拒否される");
}

#[test]
fn allocation_above_device_limit_fails() {
    init();
    let limit = vkrt_max_alloc();
    assert!(limit > 0);
    assert_eq!(vkrt_alloc(limit + 1), 0);
}

/// 検証レイヤ有効時（VKRT_VALIDATION=1）に、Vulkan の使い方違反が 0 件であること。他のテストと並行実行されるので、
/// 最後に走らせるのではなく、ここでも一通りの操作をしてから確認する。
#[test]
fn no_validation_errors() {
    init();
    let a = vkrt_alloc(4096);
    let b = vkrt_alloc(4096);
    let data = [1u8; 4096];
    unsafe {
        vkrt_upload(a, data.as_ptr() as *const c_void, 4096);
        vkrt_copy_d2d(b, a, 4096);
        let k = vkrt_kernel_index(c"add_f32".as_ptr()) as u32;
        let push: [u64; 4] = [a, b, b, 1024];
        vkrt_dispatch(k, push.as_ptr() as *const c_void, 32, 4, 1, 1);
    }
    vkrt_synchronize();
    vkrt_free(a);
    vkrt_free(b);
    assert_eq!(vkrt_validation_errors(), 0);
}
