//! VK_KHR_cooperative_matrix の対応形状・型の組み合わせを列挙する。
use ash::{Entry, khr, vk};

fn main() {
    unsafe {
        let entry = Entry::load().unwrap();
        let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);
        let instance = entry.create_instance(&vk::InstanceCreateInfo::default().application_info(&app), None).unwrap();
        let pds = instance.enumerate_physical_devices().unwrap();
        let pd = *pds
            .iter()
            .find(|&&p| instance.get_physical_device_properties(p).device_type == vk::PhysicalDeviceType::DISCRETE_GPU)
            .unwrap_or(&pds[0]);
        let loader = khr::cooperative_matrix::Instance::new(&entry, &instance);
        let props = loader.get_physical_device_cooperative_matrix_properties(pd).unwrap();
        println!("count = {}", props.len());
        for p in props {
            println!(
                "M{} N{} K{}  A={:?} B={:?} C={:?} R={:?} scope={:?} sat={}",
                p.m_size, p.n_size, p.k_size, p.a_type, p.b_type, p.c_type, p.result_type, p.scope, p.saturating_accumulation
            );
        }
    }
}
