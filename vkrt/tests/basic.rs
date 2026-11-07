//! vkrt の基本動作: 確保・往復コピー・d2d コピー・add_f32 カーネル・範囲外アクセスの検出。
use std::ffi::c_void;
use vkrt::*;

static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// ランタイムはプロセス全体で 1 つ（統計もグローバル）なので、テストを直列化する。
fn init() -> std::sync::MutexGuard<'static, ()> {
    let g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(vkrt_init(), 0, "init 失敗");
    g
}

#[test]
fn upload_download_roundtrip_and_d2d() {
    let _g = init();
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
    let _g = init();
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
    let _g = init();
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
    let _g = init();
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
    let _g = init();
    let limit = vkrt_max_alloc();
    assert!(limit > 0);
    assert_eq!(vkrt_alloc(limit + 1), 0);
}

/// 検証レイヤ有効時（VKRT_VALIDATION=1）に、Vulkan の使い方違反が 0 件であること。他のテストと並行実行されるので、
/// 最後に走らせるのではなく、ここでも一通りの操作をしてから確認する。
#[test]
fn no_validation_errors() {
    let _g = init();
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

#[test]
fn many_small_allocations_do_not_hit_driver_allocation_limit() {
    let _g = init();
    // ドライバの割り当て上限（約 4096）を大きく超える数の小ブロックを同時に保持できること
    let slabs_before = vkrt_slab_count();
    let addrs: Vec<u64> = (0..20_000).map(|i| vkrt_alloc(1000 + (i % 7) * 100)).collect();
    assert!(addrs.iter().all(|&a| a != 0));
    assert!(vkrt_slab_count() - slabs_before < 20, "スラブ数が多すぎる: {}", vkrt_slab_count());
    // アドレスの重複が無いこと
    let mut sorted = addrs.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), addrs.len());
    for a in addrs {
        assert_eq!(vkrt_free(a), 0);
    }
}

#[test]
fn freed_neighbours_are_coalesced_and_data_is_isolated() {
    let _g = init();
    let a = vkrt_alloc(4096);
    let b = vkrt_alloc(4096);
    let c = vkrt_alloc(4096);
    let (da, dc) = ([0xAAu8; 4096], [0xCCu8; 4096]);
    unsafe {
        vkrt_upload(a, da.as_ptr() as *const c_void, 4096);
        vkrt_upload(c, dc.as_ptr() as *const c_void, 4096);
    }
    // a と c の間の b を解放して、より大きなブロックを確保しても a, c が壊れない
    vkrt_free(b);
    let big = vkrt_alloc(8192);
    let bigdata = [0x55u8; 8192];
    unsafe {
        vkrt_upload(big, bigdata.as_ptr() as *const c_void, 8192);
        let (mut ra, mut rc) = ([0u8; 4096], [0u8; 4096]);
        vkrt_download(ra.as_mut_ptr() as *mut c_void, a, 4096);
        vkrt_download(rc.as_mut_ptr() as *mut c_void, c, 4096);
        assert!(ra == da && rc == dc, "隣接ブロックが破壊された");
    }
    vkrt_free(a);
    vkrt_free(c);
    vkrt_free(big);
    // 返却後は隣接空きが結合されているので、元の合計サイズの連続領域を取れる
    let all = vkrt_alloc(3 * 4096 + 8192);
    assert!(all != 0);
    vkrt_free(all);
}

#[test]
fn empty_cache_returns_idle_slabs_and_large_blocks_are_dedicated() {
    let _g = init();
    let big = vkrt_alloc(100 << 20); // 専用スラブ
    assert!(big != 0);
    let reserved = vkrt_reserved_bytes();
    assert!(reserved >= 100 << 20);
    vkrt_free(big); // 専用スラブは解放時にドライバへ返る
    assert!(vkrt_reserved_bytes() < reserved);
    let small = vkrt_alloc(1024);
    vkrt_free(small);
    assert!(vkrt_empty_cache() >= 1, "空きスラブが返却されない");
}
