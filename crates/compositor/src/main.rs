pub mod backend;
pub mod colors;
pub mod config;

use std::{
    collections::{HashSet, VecDeque},
    os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd, RawFd},
    sync::Arc,
    task::Poll,
    time::Duration,
};

// use skia_safe::prelude::NativeAccess;

use anyhow::Context as _;
use smithay::reexports::{
    ash::{self, vk},
    calloop,
    drm::{self, Device, buffer::Buffer, control::Device as _},
    gbm,
};

use crate::{
    backend::tty::{self, utils::Guard},
    colors::COLORS,
};

fn main() -> anyhow::Result<()> {
    pretty_env_logger::init();

    let card = tty::Card::find_primary()?;

    log::trace!("DRM Driver: {:?}", card.get_driver());
    let resources = card.resource_handles()?;

    let card = Arc::new(gbm::Device::new(card)?);

    let instance = tty::vulkan::Instance::load()?;
    let device = Arc::new(instance.device_for(&card)?);

    // Pick a suitable CRTC and mode for the active connector.
    let connector = resources
        .connectors()
        .iter()
        .filter_map(|&con| card.get_connector(con, true).ok())
        // TODO: Make this slightly more configurable...
        .find(|con| con.state() == drm::control::connector::State::Connected)
        .ok_or(anyhow::anyhow!("Not connected to any display!"))?;

    let crtc = {
        let compatible_crtcs = connector
            .encoders()
            .iter()
            .filter_map(|&enc| card.get_encoder(enc).ok())
            .flat_map(|enc| resources.filter_crtcs(enc.possible_crtcs()))
            .collect::<Vec<_>>();

        let crtc_before = connector
            .current_encoder()
            .and_then(|enc| card.get_encoder(enc).ok())
            .and_then(|encoder| encoder.crtc());
        compatible_crtcs
            .iter()
            .find(|&crtc| crtc_before.as_ref().is_some_and(|old| old == crtc))
            .or(compatible_crtcs.first())
            .copied()
            .ok_or(anyhow::anyhow!("Cannot find a suitable CRTC!"))?
    };

    let mode = {
        let modes = connector.modes();
        modes
            .iter()
            .find(|&mode| {
                mode.mode_type()
                    .contains(drm::control::ModeTypeFlags::PREFERRED)
            })
            .or(modes.first())
            .copied()
            .ok_or(anyhow::anyhow!(
                "Cannot find suitable mode for connector {}",
                connector.interface().as_str()
            ))?
    };

    let planes = card
        .plane_handles()?
        .iter()
        .filter_map(|&p| card.get_plane(p).ok())
        .filter(|p| resources.filter_crtcs(p.possible_crtcs()).contains(&crtc))
        .collect::<Vec<_>>();

    let (plane, plane_props) = planes
        .iter()
        .filter_map(|p| card.get_properties(p.handle()).ok().map(|info| (p, info)))
        .filter_map(|(p, info)| {
            let ty = info.iter().find_map(|(k, v)| {
                if let Ok(info) = card.get_property(*k)
                    && info.name() == c"type"
                {
                    use drm::control::PlaneType as Type;

                    const PRIMARY: u64 = Type::Primary as u64;
                    const CURSOR: u64 = Type::Cursor as u64;
                    const OVERLAY: u64 = Type::Overlay as u64;
                    return match *v {
                        PRIMARY => Some(Type::Primary),
                        CURSOR => Some(Type::Cursor),
                        OVERLAY => Some(Type::Overlay),
                        _ => None,
                    };
                }

                None
            });

            ty.map(move |ty| (p.handle(), ty, info))
        })
        .find_map(|(p, ty, props)| (ty == drm::control::PlaneType::Primary).then_some((p, props)))
        .ok_or(anyhow::anyhow!("cannot find suitable primary plane"))?;

    let format = tty::utils::Format(gbm::Format::Xrgb8888);

    // Format modifiers...
    let usable_modifiers = {
        // DRM
        let drm_modifiers = drm_modifiers(&card, format.fourcc(), &plane_props)?;

        // Vulkan
        let vulkan_modifiers = vulkan_modifiers(&device, format.vk());

        // Intersect the DRM- and Vulkan-provided format modifiers
        let usable_modifiers = vulkan_modifiers
            .intersection(&drm_modifiers)
            .copied()
            .collect::<Vec<_>>();

        if usable_modifiers.is_empty() {
            log::warn!("no common modifier between DRM plane and Vulkan; falling back to linear");
            vec![u64::from(gbm::Modifier::Linear)]
        } else {
            usable_modifiers
        }
    };

    let usable_modifiers = usable_modifiers
        .iter()
        .copied()
        .map(gbm::Modifier::from)
        .collect::<Vec<_>>();

    // let mut skia = device.clone().skia_context()?;

    let mut output = BufferedOutput::new(
        card.clone(),
        device,
        crtc,
        plane,
        mode,
        connector.handle(),
        format,
        &usable_modifiers,
    )?;

    let mut event_loop = calloop::EventLoop::try_new()?;

    event_loop.handle().insert_source(
        tty::DrmEventNotifier::new(card.clone()),
        |event, _, output: &mut BufferedOutput| {
            if let drm::control::Event::PageFlip(page_flip_event) = event {
                log::trace!(
                    "Flip event! {:?} @ {:?}",
                    page_flip_event.frame,
                    page_flip_event.duration
                );

                if let Some(pending_req) = output.queue.pop_front()
                    && let Err(err) = output.flip(pending_req)
                {
                    println!("Error during page flip: {err:?}");
                }
            }
        },
    )?;

    let signal = event_loop.get_signal();

    event_loop
        .handle()
        .insert_source(
            calloop::timer::Timer::from_duration(Duration::from_secs(10)),
            move |_, _, _| {
                signal.stop();
                calloop::timer::TimeoutAction::Drop
            },
        )
        .map_err(|a| a.error)?;

    let Poll::Ready(req) = output.try_render()? else {
        panic!("First buffer is still being scanned! Should not be the case!")
    };
    output.flip(req)?;

    let now = std::time::Instant::now();
    event_loop.run(
        Some(Duration::from_millis(5)),
        &mut output,
        |output| match output.try_render().expect("No error within render loop") {
            Poll::Ready(req) => output.queue.push_back(req),
            Poll::Pending => (),
        },
    )?;

    let fps = output.frame as f64 / now.elapsed().as_secs_f64();
    println!("Average FPS: {fps:.3}");

    Ok(())
}

pub struct ScanoutRequest {
    pub frame_i: usize,
}

pub struct BufferedOutput {
    // DRM
    card: Arc<gbm::Device<tty::Card>>,
    cache: tty::atomic_req::AtomicRequestCache,
    connector: drm::control::connector::Handle,
    crtc: drm::control::crtc::Handle,
    plane: drm::control::plane::Handle,
    mode: drm::control::Mode,

    previous_state: drm::control::crtc::Info,

    // Vulkan
    frame: usize,
    buffers: [ScanoutBuffer; 3],
    pool: tty::vulkan::CommandPool,
    device: Arc<tty::vulkan::Device>,
    queue: VecDeque<ScanoutRequest>,
}

impl Drop for BufferedOutput {
    fn drop(&mut self) {
        // Reset the state of the CRTC
        {
            let prev = &self.previous_state;
            let _ = self.card.set_crtc(
                self.crtc,
                prev.framebuffer(),
                prev.position(),
                core::slice::from_ref(&self.connector),
                prev.mode(),
            );
        }

        unsafe {
            let _ = self.device.device_wait_idle();
        }
    }
}

impl BufferedOutput {
    pub fn new(
        card: Arc<gbm::Device<tty::Card>>,
        device: Arc<tty::vulkan::Device>,
        crtc: drm::control::crtc::Handle,
        plane: drm::control::plane::Handle,
        mode: drm::control::Mode,
        connector: drm::control::connector::Handle,
        format: tty::utils::Format,
        modifiers: &[gbm::Modifier],
    ) -> anyhow::Result<Self> {
        // Render =:render_sem:=> Scanout
        // Create a Vulkan semaphore that we can export to DRM via a SYNC_FD.

        let command_pool = {
            let info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(device.queue_family)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

            unsafe {
                Guard::new(device.create_command_pool(&info, None)?, |pool| {
                    device.destroy_command_pool(pool, None)
                })
            }
        };

        let (cmd, cmd_frames) = {
            let info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(1)
                .command_pool(*command_pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let cmd = unsafe {
                let [cmd] = device.allocate_command_buffers(&info)?.try_into().unwrap();

                Guard::new(cmd, |cmd| {
                    device.free_command_buffers(*command_pool, &[cmd])
                })
            };

            let info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(3)
                .command_pool(*command_pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let cmd_frames = unsafe {
                let cmds @ [_, _, _] = device.allocate_command_buffers(&info)?.try_into().unwrap();

                Guard::new(cmds, |cmds| {
                    device.free_command_buffers(*command_pool, cmds.as_slice())
                })
            };

            (cmd, cmd_frames)
        };

        let buffers = {
            let [cmd1, cmd2, cmd3] = cmd_frames.finish();
            let create_scan = |cmd| {
                ScanoutBuffer::new(
                    device.clone(),
                    card.clone(),
                    format,
                    mode,
                    modifiers,
                    *command_pool,
                    cmd, // &mut skia,
                )
            };

            [create_scan(cmd1)?, create_scan(cmd2)?, create_scan(cmd3)?]
        };

        anyhow::ensure!(
            buffers[0].bo.modifier() == buffers[1].bo.modifier(),
            "Both buffers should have the same format modifiers!"
        );

        // Clear the first buffer to avoid displaying garbage.
        {
            let clear_fence = unsafe {
                Guard::new(
                    device.create_fence(&vk::FenceCreateInfo::default(), None)?,
                    |fence| {
                        let _ = device.wait_for_fences(
                            core::slice::from_ref(&fence),
                            true,
                            1_000_000_000,
                        );
                        device.destroy_fence(fence, None);
                    },
                )
            };

            unsafe {
                device.begin_command_buffer(
                    *cmd,
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
                .image(buffers[2].image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );

            // UNDEFINED -> GENERAL layout
            unsafe {
                device.cmd_pipeline_barrier2(
                    *cmd,
                    &vk::DependencyInfo::default()
                        .image_memory_barriers(core::slice::from_ref(&barrier)),
                )
            };

            // Fill with BLACK.
            {
                let attachment = vk::RenderingAttachmentInfo::default()
                    .image_layout(vk::ImageLayout::GENERAL)
                    .image_view(buffers[2].image_view)
                    // RED
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    });

                let (width, height) = mode.size();
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
                    device.cmd_begin_rendering(*cmd, &render);
                }
                unsafe { device.cmd_end_rendering(*cmd) };
            }

            unsafe { device.end_command_buffer(*cmd)? }

            unsafe {
                device.queue_submit(
                    device.queue,
                    &[vk::SubmitInfo::default().command_buffers(core::slice::from_ref(&cmd))],
                    *clear_fence,
                )?
            };
        }

        let previous_state = card.get_crtc(crtc)?;

        drop(cmd);

        let mut output = Self {
            cache: tty::atomic_req::AtomicRequestCache::new(card.clone()),
            frame: 0,
            connector,
            crtc,
            plane,
            mode,
            previous_state,
            buffers,
            card,
            pool: tty::vulkan::CommandPool::new(device.clone(), command_pool.finish()),
            device,
            queue: VecDeque::with_capacity(3),
        };

        output.modeset()?;

        Ok(output)
    }

    pub fn modeset(&mut self) -> anyhow::Result<()> {
        use tty::atomic_req::*;

        let fb = self.buffers[self.frame].fb;
        let (w, h) = self.mode.size();

        // This blocks to prevent a race condition with the cleanup of the `MODE_ID` blob.
        // It's only done once anyway...
        self.cache
            .request()
            .set(self.connector, CRTC_ID, Some(self.crtc))?
            .set(self.crtc, ACTIVE, true)?
            .set(self.crtc, MODE_ID, Some(&self.mode))?
            .set(self.plane, CRTC_ID, Some(self.crtc))?
            .set(self.plane, FB_ID, Some(fb))?
            .set(self.plane, SRC_X, Fixed16::ZERO)?
            .set(self.plane, SRC_Y, Fixed16::ZERO)?
            .set(self.plane, SRC_W, Fixed16::integer(w))?
            .set(self.plane, SRC_H, Fixed16::integer(h))?
            .set(self.plane, CRTC_X, 0)?
            .set(self.plane, CRTC_Y, 0)?
            .set(self.plane, CRTC_W, w as u32)?
            .set(self.plane, CRTC_H, h as u32)?
            .commit(AtomicCommitFlags::ALLOW_MODESET)?;

        Ok(())
    }

    fn try_render(&mut self) -> anyhow::Result<Poll<ScanoutRequest>> {
        let (frame_i, buffer, color) = {
            let frame_i = self.frame % 3;
            let color = COLORS[self.frame % COLORS.len()];
            (frame_i, &mut self.buffers[frame_i], color)
        };

        let fence_signal = unsafe {
            self.device.get_fence_status(*buffer.fence)?
                && self.device.get_fence_status(buffer.cmd_fence)?
        };

        if !fence_signal {
            return Ok(Poll::Pending);
        }

        log::trace!("Frame #{} in buffer #{}", self.frame, frame_i);
        self.frame += 1;

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
                let attachment = vk::RenderingAttachmentInfo::default()
                    .image_layout(vk::ImageLayout::GENERAL)
                    .image_view(buffer.image_view)
                    // RED
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue { float32: color },
                    });

                let (width, height) = self.mode.size();
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

        // let surface = &mut self.buffers[back].surface;
        // let canvas = surface.canvas();
        // let paint = skia_safe::Paint::new(skia_safe::Color4f::from(skia_safe::Color::BLACK), None);
        // canvas
        //     .clear(skia_safe::Color::WHITE) //
        //     .draw_text_align(
        //         self.frame.to_string(),
        //         ( 0.0, 100.0),
        //         &skia_safe::Font::default(),
        //         &paint,
        //         skia_safe::utils::text_utils::Align::Left,
        //     );

        // {
        //     let semaphores = self.render_semaphore.skia();
        //     let flush_info = {
        //         let mut info = skia_safe::gpu::FlushInfo::default();
        //         unsafe { info.set_signal_semaphores(semaphores) };
        //         info
        //     };

        //     let mut ctx = surface.direct_context().unwrap();
        //     let submitted_semaphores = ctx.flush_surface_with_access(
        //         surface,
        //         skia_safe::surface::BackendSurfaceAccess::NoAccess,
        //         &flush_info,
        //     );
        //     assert_eq!(
        //         submitted_semaphores,
        //         skia_safe::gpu::ganesh::SemaphoresSubmitted::Yes
        //     );

        //     let can_wait_on_semaphore = ctx.submit(Some(skia_safe::gpu::SyncCpu::No));
        //     assert!(can_wait_on_semaphore);
        // }

        Ok(Poll::Ready(ScanoutRequest { frame_i }))
    }

    fn flip(&mut self, req: ScanoutRequest) -> anyhow::Result<()> {
        use tty::atomic_req::*;

        let ScanoutRequest { frame_i } = req;
        let buffer = &mut self.buffers[frame_i];
        let mut out_fd: RawFd = -1;

        self.cache
            .request()
            .set(self.plane, FB_ID, Some(buffer.fb))?
            .set(self.plane, IN_FENCE_FD, buffer.semaphore.sync_fd()?)?
            .set(self.crtc, OUT_FENCE_PTR, &raw mut out_fd)?
            .commit(AtomicCommitFlags::PAGE_FLIP_EVENT | AtomicCommitFlags::NONBLOCK)
            .context(format!("Submitting page flip for page #{}", self.frame))?;

        if out_fd == -1 {
            panic!("OUT_FENCE_PTR is invalid!");
        }

        let out_fd = unsafe { OwnedFd::from_raw_fd(out_fd) };
        buffer.fence.import(out_fd)?;

        Ok(())
    }

    // fn wait_for_flip(&self) -> std::io::Result<()> {
    //     loop {
    //         let mut events = match self.card.receive_events() {
    //             Ok(events) => events,
    //             Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
    //             Err(err) => return Err(err),
    //         };

    //         if events.any(|ev| matches!(ev, drm::control::Event::PageFlip(_))) {
    //             return Ok(());
    //         }
    //     }
    // }
}

pub struct ScanoutBuffer {
    device: Arc<tty::vulkan::Device>,
    card: Arc<gbm::Device<tty::Card>>,
    memory: vk::DeviceMemory,
    image: vk::Image,
    image_view: vk::ImageView,
    fb: drm::control::framebuffer::Handle,
    bo: gbm::BufferObject<()>,
    fence: tty::vulkan::ImportableFence,
    semaphore: tty::vulkan::ExportableSemaphore,
    pool: vk::CommandPool,
    cmd: vk::CommandBuffer, // surface: skia_safe::Surface,
    cmd_fence: vk::Fence,
}

impl Drop for ScanoutBuffer {
    fn drop(&mut self) {
        let _ = self.card.destroy_framebuffer(self.fb);
        unsafe {
            self.device.destroy_fence(self.cmd_fence, None);
            self.device.free_command_buffers(self.pool, &[self.cmd]);
            self.device.destroy_image_view(self.image_view, None);
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

impl ScanoutBuffer {
    pub fn new(
        device: Arc<tty::vulkan::Device>,
        card: Arc<gbm::Device<tty::Card>>,
        format: tty::utils::Format,
        mode: drm::control::Mode,
        usable_modifiers: &[gbm::Modifier],
        pool: vk::CommandPool,
        cmd: vk::CommandBuffer, // skia_context: &mut skia_safe::gpu::DirectContext,
    ) -> anyhow::Result<ScanoutBuffer> {
        // Create a BufferObject with
        let (width, height) = mode.size();

        let cmd = Guard::new(cmd, |cmd| {
            unsafe { device.free_command_buffers(pool, &[cmd]) };
        });

        let bo = card.create_buffer_object_with_modifiers2::<()>(
            width as _,
            height as _,
            format.fourcc(),
            usable_modifiers.iter().copied(),
            gbm::BufferObjectFlags::SCANOUT | gbm::BufferObjectFlags::RENDERING,
        )?;

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
                .format(format.vk())
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

            let img = unsafe { device.create_image(&info, None)? };

            Guard::new(img, |img| unsafe { device.destroy_image(img, None) })
        };

        // Use the external memory via the fd provided by GBM
        let (imported_fd, memory) = {
            // (A) Create a DMA file descriptor (owned by Vulkan)
            let fd = bo.fd()?;
            let fd_props = {
                let mut props = vk::MemoryFdPropertiesKHR::default();
                let ext_mem_device =
                    ash::khr::external_memory_fd::Device::new(&device.instance, device.as_ref());
                unsafe {
                    ext_mem_device.get_memory_fd_properties(
                        vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                        fd.as_raw_fd(),
                        &mut props,
                    )?;
                };

                props
            };

            let reqs = unsafe { device.get_image_memory_requirements(*image) };

            // Get first mem_type compatible with Image and fd.
            let compatible_mem_types = reqs.memory_type_bits & fd_props.memory_type_bits;
            if compatible_mem_types == 0 {
                anyhow::bail!("no supported DMA memory type can be imported into Vulkan")
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

            let memory = unsafe { device.allocate_memory(&info, None)? };

            (
                fd,
                Guard::new(memory, |memory| unsafe { device.free_memory(memory, None) }),
            )
        };

        unsafe { device.bind_image_memory(*image, *memory, 0)? };

        // (A) Okay, now everything is good, we can forget the file descriptor here, since Vulkan will close it.
        std::mem::forget(imported_fd);

        let image_view = {
            let info = vk::ImageViewCreateInfo::default()
                .components(vk::ComponentMapping::default())
                .format(format.vk())
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

            let image_view = unsafe { device.create_image_view(&info, None)? };

            Guard::new(image_view, |view| unsafe {
                device.destroy_image_view(view, None)
            })
        };

        let framebuffer = card.add_planar_framebuffer(&bo, drm::control::FbCmd2Flags::MODIFIERS)?;

        let cmd_fence = unsafe {
            Guard::new(
                device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )?,
                |fence| device.destroy_fence(fence, None),
            )
        };
        // let surface = {
        //     let image_info = unsafe {
        //         skia_safe::gpu::vk::ImageInfo::new(
        //             image.as_raw() as _,
        //             Default::default(),
        //             skia_safe::gpu::vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT,
        //             skia_safe::gpu::vk::ImageLayout::UNDEFINED,
        //             std::mem::transmute::<vk::Format, skia_safe::gpu::vk::Format>(format.vk()),
        //             1,
        //             device.queue_family,
        //             None,
        //             None,
        //             None,
        //         )
        //     };

        //     let render_target = skia_safe::gpu::backend_render_targets::make_vk(
        //         (width as i32, height as i32),
        //         &image_info,
        //     );

        //     skia_safe::gpu::surfaces::wrap_backend_render_target(
        //         skia_context,
        //         &render_target,
        //         skia_safe::gpu::SurfaceOrigin::TopLeft,
        //         format.skia(),
        //         None,
        //         None,
        //     )
        //     .context("wrap surface")?
        // };

        Ok(ScanoutBuffer {
            fence: tty::vulkan::ImportableFence::new(device.clone())?,
            semaphore: tty::vulkan::ExportableSemaphore::new(device.clone())?,
            memory: memory.finish(),
            image: image.finish(),
            image_view: image_view.finish(),
            fb: framebuffer,
            bo,
            card,
            pool,
            cmd: cmd.finish(),
            cmd_fence: cmd_fence.finish(),
            device,
        })
    }
}
fn drm_modifiers(
    card: &gbm::Device<tty::Card>,
    fourcc: gbm::Format,
    plane_props: &drm::control::PropertyValueSet,
) -> Result<HashSet<u64>, anyhow::Error> {
    let in_formats = plane_props
        .iter()
        .find_map(|(k, v)| (card.get_property(*k).ok()?.name() == c"IN_FORMATS").then_some(*v))
        .and_then(|blob| card.get_property_blob(blob).ok())
        .ok_or(anyhow::anyhow!("cannot get supported formats"))?;
    let in_formats = tty::format_modifier::drm_format_modifier_blob::read(&in_formats)?;
    Ok(HashSet::<u64>::from_iter(
        in_formats
            .modifiers_for(fourcc)
            .into_iter()
            .filter(|&m| m != u64::from(gbm::Modifier::Invalid)),
    ))
}

fn vulkan_modifiers(device: &tty::vulkan::Device, format: vk::Format) -> HashSet<u64> {
    // Get the number of available format modifiers.
    let format_modifier_len = {
        let mut modifiers = vk::DrmFormatModifierPropertiesListEXT::default();
        let mut fmt_props2 = vk::FormatProperties2::default().push_next(&mut modifiers);
        unsafe {
            device.instance.get_physical_device_format_properties2(
                device.physical,
                format,
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
                format,
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
