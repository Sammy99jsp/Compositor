use std::{
    collections::HashSet,
    os::{fd::AsRawFd, unix::prelude::OwnedFd},
    sync::Arc,
    task::Poll,
};

use smithay::reexports::{
    ash::{self, vk},
    drm::buffer::Buffer,
    gbm,
};
use thiserror::Error;

use crate::{
    backend::tty::{
        Backend, BackendBuffer, Card, DrmState, drmx,
        utils::Guard,
        vulkan::utils::{CommandPool, Device, ExportableSemaphore, ImportableFence, Instance},
    },
    colors::COLORS,
};

pub mod utils;

pub struct Vulkan {
    device: Arc<Device>,
    pool: Arc<CommandPool>,
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        // Ensure device has finished all of the queues before trying to deallocate/destroy anything.
        unsafe {
            let _ = self.device.device_wait_idle();
        }
    }
}

pub struct VulkanBuffer {
    device: Arc<Device>,
    memory: vk::DeviceMemory,
    image: vk::Image,
    image_view: vk::ImageView,
    fence: ImportableFence,
    semaphore: ExportableSemaphore,
    pool: Arc<CommandPool>,
    cmd: vk::CommandBuffer,
    cmd_fence: vk::Fence,
    size: (u32, u32),
}

#[derive(Debug, Error)]
pub enum VulkanError {
    #[error(transparent)]
    Loading(#[from] ash::LoadingError),

    #[error(transparent)]
    Vulkan(#[from] ash::vk::Result),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("no supported DMA memory type can be imported into Vulkan")]
    UnsupportedDmaMemory,

    #[error(transparent)]
    Fd(#[from] gbm::InvalidFdError),

    #[error("{0}")]
    Other(String),
}

impl From<String> for VulkanError {
    fn from(value: String) -> Self {
        Self::Other(value)
    }
}

impl Backend for Vulkan {
    type Error = VulkanError;

    fn new(card: &Card) -> Result<Self, Self::Error> {
        let instance = Instance::load()?;
        let device = Arc::new(instance.device_for(card)?);

        Ok(Self {
            pool: Arc::new(CommandPool::new(device.clone())?),
            device,
        })
    }

    fn modifiers(&self, format: drmx::Format) -> HashSet<u64> {
        let device = &self.device;

        // Get the number of available format modifiers.
        let format_modifier_len = {
            let mut modifiers = vk::DrmFormatModifierPropertiesListEXT::default();
            let mut fmt_props2 = vk::FormatProperties2::default().push_next(&mut modifiers);
            unsafe {
                device.instance.get_physical_device_format_properties2(
                    device.physical,
                    format.into(),
                    &mut fmt_props2,
                )
            };

            modifiers.drm_format_modifier_count as usize
        };
        // Then, load all the format modifiers into a list.
        let modifier_properties = {
            let mut mod_props =
                vec![vk::DrmFormatModifierPropertiesEXT::default(); format_modifier_len];
            let mut modifier_list = vk::DrmFormatModifierPropertiesListEXT::default()
                .drm_format_modifier_properties(&mut mod_props);
            let mut fmt_props2 = vk::FormatProperties2::default().push_next(&mut modifier_list);
            unsafe {
                device.instance.get_physical_device_format_properties2(
                    device.physical,
                    format.into(),
                    &mut fmt_props2,
                )
            };

            mod_props
        };
        // Only select the modifiers for formats with COLOR
        modifier_properties
            .into_iter()
            .filter(|a| {
                a.drm_format_modifier_tiling_features
                    .contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT)
            })
            .map(|a| a.drm_format_modifier)
            .collect::<HashSet<_>>()
    }

    type Buffer = VulkanBuffer;
    fn new_buffer(
        &self,
        drm: &DrmState,
        bo: &gbm::BufferObject<()>,
    ) -> Result<Self::Buffer, Self::Error> {
        let cmd = {
            let [cmd] = unsafe {
                self.device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_buffer_count(1)
                            .command_pool(**self.pool)
                            .level(vk::CommandBufferLevel::PRIMARY),
                    )?
                    .try_into()
                    .expect("one allocated command buffer")
            };

            Guard::new(cmd, |cmd| {
                unsafe { self.device.free_command_buffers(**self.pool, &[cmd]) };
            })
        };

        // Create vk::Image
        let image = {
            let plane_layouts = (0..bo.plane_count())
                .map(|i| i as i32)
                .map(|i| {
                    vk::SubresourceLayout::default()
                        .offset(bo.offset(i) as _)
                        .row_pitch(bo.stride_for_plane(i) as _)
                })
                .collect::<Vec<_>>();

            let mut mod_info = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
                .drm_format_modifier(bo.modifier().into())
                .plane_layouts(&plane_layouts);

            let mut ext_info = vk::ExternalMemoryImageCreateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);

            let (width, height) = bo.size();
            let info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(drm.format.into())
                .extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED)
                .push_next(&mut ext_info)
                .push_next(&mut mod_info);

            let img = unsafe { self.device.create_image(&info, None)? };

            Guard::new(img, |img| unsafe { self.device.destroy_image(img, None) })
        };

        // Use the external memory via the fd provided by GBM
        let (imported_fd, memory) = {
            // (A) Create a DMA file descriptor (owned by Vulkan)
            let fd = bo.fd()?;
            let fd_props = {
                let mut props = vk::MemoryFdPropertiesKHR::default();
                let ext_mem_device = ash::khr::external_memory_fd::Device::new(
                    &self.device.instance,
                    self.device.as_ref(),
                );
                unsafe {
                    ext_mem_device.get_memory_fd_properties(
                        vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                        fd.as_raw_fd(),
                        &mut props,
                    )?;
                };

                props
            };

            let reqs = unsafe { self.device.get_image_memory_requirements(*image) };

            // Get first mem_type compatible with Image and fd.
            let compatible_mem_types = reqs.memory_type_bits & fd_props.memory_type_bits;
            if compatible_mem_types == 0 {
                return Err(VulkanError::UnsupportedDmaMemory);
            }
            let mem_type_index = compatible_mem_types.trailing_zeros();

            // (A) Vulkan will take ownership over our fd.
            let mut import = vk::ImportMemoryFdInfoKHR::default()
                .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
                .fd(fd.as_raw_fd());

            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(*image);

            let info = vk::MemoryAllocateInfo::default()
                .allocation_size(reqs.size)
                .memory_type_index(mem_type_index)
                .push_next(&mut import)
                .push_next(&mut dedicated);

            let memory = unsafe { self.device.allocate_memory(&info, None)? };

            (
                fd,
                Guard::new(memory, |memory| unsafe {
                    self.device.free_memory(memory, None)
                }),
            )
        };

        unsafe { self.device.bind_image_memory(*image, *memory, 0)? };

        // (A) Okay, now everything is good, we can forget the file descriptor here, since Vulkan will close it.
        std::mem::forget(imported_fd);

        let image_view = {
            let info = vk::ImageViewCreateInfo::default()
                .components(vk::ComponentMapping::default())
                .format(drm.format.into())
                .image(*image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_array_layer(0)
                        .layer_count(1)
                        .base_mip_level(0)
                        .level_count(1),
                );

            let image_view = unsafe { self.device.create_image_view(&info, None)? };

            Guard::new(image_view, |view| unsafe {
                self.device.destroy_image_view(view, None)
            })
        };

        let cmd_fence = unsafe {
            Guard::new(
                self.device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )?,
                |fence| self.device.destroy_fence(fence, None),
            )
        };

        Ok(VulkanBuffer {
            device: self.device.clone(),
            memory: memory.finish(),
            image: image.finish(),
            image_view: image_view.finish(),
            semaphore: ExportableSemaphore::new(self.device.clone())?,
            fence: ImportableFence::new(self.device.clone())?,
            pool: self.pool.clone(),
            cmd: cmd.finish(),
            cmd_fence: cmd_fence.finish(),
            size: {
                let (w, h) = drm.mode.size();
                (w as u32, h as u32)
            },
        })
    }

    fn render(
        &mut self,
        buffer: &mut Self::Buffer,
        frame: usize,
    ) -> Result<std::task::Poll<()>, Self::Error> {
        let fence_signal = unsafe {
            self.device.get_fence_status(*buffer.fence)?
                && self.device.get_fence_status(buffer.cmd_fence)?
        };

        if !fence_signal {
            return Ok(Poll::Pending);
        }

        unsafe {
            self.device
                .reset_fences(&[*buffer.fence, buffer.cmd_fence])?;
            self.device
                .reset_command_buffer(buffer.cmd, vk::CommandBufferResetFlags::default())?;
        }

        {
            unsafe {
                self.device.begin_command_buffer(
                    buffer.cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )?
            }

            let barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::TOP_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(buffer.image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );

            // UNDEFINED -> GENERAL layout
            unsafe {
                self.device.cmd_pipeline_barrier2(
                    buffer.cmd,
                    &vk::DependencyInfo::default()
                        .image_memory_barriers(core::slice::from_ref(&barrier)),
                )
            };

            // Fill with some color.
            {
                let color = COLORS[frame % COLORS.len()];
                let attachment = vk::RenderingAttachmentInfo::default()
                    .image_layout(vk::ImageLayout::GENERAL)
                    .image_view(buffer.image_view)
                    // RED
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue { float32: color },
                    });

                let (width, height) = buffer.size;
                let render = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: width as _,
                            height: height as _,
                        },
                    })
                    .layer_count(1)
                    .color_attachments(core::slice::from_ref(&attachment));

                unsafe {
                    self.device.cmd_begin_rendering(buffer.cmd, &render);
                }
                unsafe { self.device.cmd_end_rendering(buffer.cmd) };
            }

            unsafe { self.device.end_command_buffer(buffer.cmd)? }
        }

        unsafe {
            self.device.queue_submit(
                self.device.queue,
                &[vk::SubmitInfo::default()
                    .command_buffers(&[buffer.cmd])
                    .signal_semaphores(buffer.semaphore.as_slice())],
                buffer.cmd_fence,
            )?;
        }

        Ok(Poll::Ready(()))
    }
}

impl Drop for VulkanBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.cmd_fence, None);
            self.device.free_command_buffers(**self.pool, &[self.cmd]);
            self.device.destroy_image_view(self.image_view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

impl BackendBuffer for VulkanBuffer {
    type Backend = Vulkan;

    fn import_sync(&mut self, fd: OwnedFd) -> Result<(), <Self::Backend as Backend>::Error> {
        self.fence.import(fd)?;
        Ok(())
    }

    fn export_sync(&mut self) -> Result<Option<OwnedFd>, <Self::Backend as Backend>::Error> {
        self.semaphore.sync_fd().map_err(VulkanError::Vulkan)
    }
}
