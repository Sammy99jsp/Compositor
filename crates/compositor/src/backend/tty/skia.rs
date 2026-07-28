use std::sync::Arc;

use anyhow::Context;
use skia_safe;
use smithay::reexports::ash::{self, vk::Handle};

impl super::vulkan::Device {
    pub fn skia_context(self: Arc<Self>) -> anyhow::Result<skia_safe::gpu::DirectContext> {
        let device = self;
        let instance = device.instance.handle().as_raw() as _;
        let physical_device = device.physical.as_raw() as _;
        let queue = (device.queue.as_raw() as _, device.queue_family as usize);
        let raw_device = device.handle().as_raw() as _;

        let get_proc = move |of: skia_safe::gpu::vk::GetProcOf| unsafe {
            let instance = &device.instance;
            let entry = &instance.entry;
            match of {
                skia_safe::gpu::vk::GetProcOf::Instance(inst, name) => {
                    entry.get_instance_proc_addr(ash::vk::Instance::from_raw(inst as _), name)
                }
                skia_safe::gpu::vk::GetProcOf::Device(dev, name) => {
                    instance.get_device_proc_addr(ash::vk::Device::from_raw(dev as _), name)
                }
            }
            .map(|f| f as _)
            .unwrap_or(std::ptr::null())
        };

        // SAFETY: the Vulkan handles will outlive this backend context
        let ctx = unsafe {
            skia_safe::gpu::vk::BackendContext::new_builder(
                instance,
                physical_device,
                raw_device,
                queue,
                &get_proc,
                None,
            )
            .build()
        };

        skia_safe::gpu::direct_contexts::make_vulkan(&ctx, None)
            .context("could not create skia vulkan direct context")
    }
}
