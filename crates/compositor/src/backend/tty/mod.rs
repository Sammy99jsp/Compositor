use std::{
    collections::{HashSet, VecDeque},
    os::fd::{FromRawFd as _, OwnedFd, RawFd},
    sync::Arc,
    task::Poll,
};

use smithay::reexports::{
    drm::{self, control::Device as _},
    gbm,
};

use crate::backend::tty::utils::Guard;

pub mod drmx;
pub mod utils;
pub mod vulkan;

pub type Card = gbm::Device<drmx::Card>;

pub trait Backend {
    type Error: std::error::Error;

    fn new(card: &Card) -> Result<Self, Self::Error>
    where
        Self: Sized;

    fn modifiers(&self, format: drmx::Format) -> HashSet<u64>;

    type Buffer: BackendBuffer<Backend = Self>;
    fn new_buffer(
        &self,
        drm: &DrmState,
        bo: &gbm::BufferObject<()>,
    ) -> Result<Self::Buffer, Self::Error>;

    fn render(&mut self, buffer: &mut Self::Buffer, frame: usize) -> Result<Poll<()>, Self::Error>;
}

pub trait BackendBuffer {
    type Backend: Backend;

    /// Import a sync file from DRM, which signals when scanout completes (and the backend can re-use this buffer).
    fn import_sync(&mut self, fd: OwnedFd) -> Result<(), <Self::Backend as Backend>::Error>;

    /// Export a sync file for DRM, which is signalled when the rendering completes (and DRM can scanout from this buffer)
    fn export_sync(&mut self) -> Result<Option<OwnedFd>, <Self::Backend as Backend>::Error>;
}

const LEN: usize = 3;

#[derive(thiserror::Error)]
pub enum ScanoutError<B: Backend> {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    DrmFormat(#[from] drmx::format_modifier::DrmFormatParseError),

    #[error("cannot get supported formats")]
    CannotGetDrmFormats,

    #[error(transparent)]
    Backend(B::Error),

    #[error(transparent)]
    Modeset(#[from] ModesetError),
}

impl<B: Backend> std::fmt::Debug for ScanoutError<B>
where
    B::Error: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(arg0) => f.debug_tuple("Io").field(arg0).finish(),
            Self::DrmFormat(arg0) => f.debug_tuple("DrmFormat").field(arg0).finish(),
            Self::CannotGetDrmFormats => write!(f, "CannotGetDrmFormats"),
            Self::Backend(arg0) => f.debug_tuple("Backend").field(arg0).finish(),
            Self::Modeset(arg0) => f.debug_tuple("Modeset").field(arg0).finish(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ModesetError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    AtomicReq(#[from] drmx::atomic_req::AtomicReqError),
}

pub struct BufferedOutput<B: Backend> {
    pub frame: usize,
    queue: VecDeque<ScanoutRequest>,
    pub drm: DrmState,
    pub backend: B,
    pub buffers: [ScanoutBuffer<B>; LEN],
}

impl<B: Backend> BufferedOutput<B> {
    pub fn new(
        card: Arc<Card>,
        crtc: drm::control::crtc::Handle,
        plane: drm::control::plane::Handle,
        mode: drm::control::Mode,
        connector: drm::control::connector::Handle,
        format: drmx::Format,
    ) -> Result<Self, ScanoutError<B>> {
        let backend = B::new(&card).map_err(ScanoutError::Backend)?;

        // Format modifiers...
        let modifiers = {
            // DRM
            let drm_modifiers =
                drm_modifiers::<B>(&card, format.fourcc(), &card.get_properties(plane)?)?;

            // Provided by backend...
            let vulkan_modifiers = backend.modifiers(format);

            // Intersect the DRM- and Vulkan-provided format modifiers
            let usable_modifiers = vulkan_modifiers
                .intersection(&drm_modifiers)
                .copied()
                .collect::<Vec<_>>();

            if usable_modifiers.is_empty() {
                log::warn!(
                    "no common modifier between DRM plane and Vulkan; falling back to linear"
                );
                vec![u64::from(gbm::Modifier::Linear)]
            } else {
                usable_modifiers
            }
            .into_iter()
            .map(gbm::Modifier::from)
            .collect::<Vec<_>>()
        };

        let drm = DrmState {
            previous_state: card.get_crtc(crtc)?,
            cache: drmx::atomic_req::AtomicRequestCache::new(card.clone()),
            card,
            crtc,
            plane,
            mode,
            connector,
            format,
            modifiers,
        };

        let [b1, b2, b3] = core::array::from_fn(|_| ScanoutBuffer::<B>::new(&drm, &backend));
        let buffers = [b1?, b2?, b3?];

        let mut output = Self {
            drm,
            backend,
            buffers,
            frame: 0,
            queue: VecDeque::with_capacity(3),
        };

        output.modeset()?;

        Ok(output)
    }

    pub fn modeset(&mut self) -> Result<(), ModesetError> {
        use drmx::atomic_req::*;

        let fb = self.buffers[self.frame].framebuffer;
        let drm = &mut self.drm;
        let (w, h) = drm.mode.size();

        // This blocks to prevent a race condition with the cleanup of the `MODE_ID` blob.
        // It's only done once anyway...
        drm.cache
            .request()
            .set(drm.connector, CRTC_ID, Some(drm.crtc))?
            .set(drm.crtc, ACTIVE, true)?
            .set(drm.crtc, MODE_ID, Some(&drm.mode))?
            .set(drm.plane, CRTC_ID, Some(drm.crtc))?
            .set(drm.plane, FB_ID, Some(fb))?
            .set(drm.plane, SRC_X, Fixed16::ZERO)?
            .set(drm.plane, SRC_Y, Fixed16::ZERO)?
            .set(drm.plane, SRC_W, Fixed16::integer(w))?
            .set(drm.plane, SRC_H, Fixed16::integer(h))?
            .set(drm.plane, CRTC_X, 0)?
            .set(drm.plane, CRTC_Y, 0)?
            .set(drm.plane, CRTC_W, w as u32)?
            .set(drm.plane, CRTC_H, h as u32)?
            .commit(AtomicCommitFlags::ALLOW_MODESET)?;

        Ok(())
    }

    pub fn render(&mut self) -> Result<Poll<()>, B::Error> {
        let frame_i = self.frame % 3;
        let buffer = &mut self.buffers[frame_i];

        let poll = self.backend.render(&mut buffer.backend, self.frame)?;
        if let Poll::Ready(()) = poll {
            self.frame += 1;
            self.queue.push_back(ScanoutRequest { frame_i });
        }

        Ok(poll)
    }

    pub fn flip(&mut self) -> Result<Poll<()>, ScanoutError<B>> {
        use drmx::atomic_req::*;

        let Some(ScanoutRequest { frame_i }) = self.queue.pop_front() else {
            return Ok(Poll::Pending);
        };

        let buffer = &mut self.buffers[frame_i];
        let drm = &mut self.drm;
        let mut out_fd: RawFd = -1;

        drm.cache
            .request()
            .set(drm.plane, FB_ID, Some(buffer.framebuffer))
            .map_err(ModesetError::from)?
            .set(
                drm.plane,
                IN_FENCE_FD,
                buffer
                    .backend
                    .export_sync()
                    .map_err(ScanoutError::Backend)?,
            )
            .map_err(ModesetError::from)?
            .set(drm.crtc, OUT_FENCE_PTR, &raw mut out_fd)
            .map_err(ModesetError::from)?
            .commit(AtomicCommitFlags::PAGE_FLIP_EVENT | AtomicCommitFlags::NONBLOCK)?;

        if out_fd == -1 {
            panic!("OUT_FENCE_PTR is invalid!");
        }

        let out_fd = unsafe { OwnedFd::from_raw_fd(out_fd) };
        buffer
            .backend
            .import_sync(out_fd)
            .map_err(ScanoutError::Backend)?;

        Ok(Poll::Ready(()))
    }
}

pub struct ScanoutRequest {
    pub frame_i: usize,
}

pub struct DrmState {
    pub card: Arc<Card>,
    pub cache: drmx::atomic_req::AtomicRequestCache,
    pub crtc: drm::control::crtc::Handle,
    pub plane: drm::control::plane::Handle,
    pub mode: drm::control::Mode,
    pub connector: drm::control::connector::Handle,
    pub format: drmx::Format,
    pub modifiers: Vec<gbm::Modifier>,
    pub previous_state: drm::control::crtc::Info,
}

impl Drop for DrmState {
    fn drop(&mut self) {
        // Reset the state of the CRTC
        let prev = &self.previous_state;
        let _ = self.card.set_crtc(
            self.crtc,
            prev.framebuffer(),
            prev.position(),
            core::slice::from_ref(&self.connector),
            prev.mode(),
        );
    }
}

pub struct ScanoutBuffer<B: Backend> {
    card: Arc<Card>,
    framebuffer: drm::control::framebuffer::Handle,

    #[allow(unused)] // Here for lifetime purpose...
    buffer_object: gbm::BufferObject<()>,

    pub backend: B::Buffer,
}

impl<B: Backend> Drop for ScanoutBuffer<B> {
    fn drop(&mut self) {
        let _ = self.card.destroy_framebuffer(self.framebuffer);
    }
}

impl<B: Backend> ScanoutBuffer<B> {
    pub fn new(drm: &DrmState, backend: &B) -> Result<Self, ScanoutError<B>> {
        let (width, height) = drm.mode.size();

        let buffer_object = drm.card.create_buffer_object_with_modifiers2::<()>(
            width as _,
            height as _,
            drm.format.fourcc(),
            drm.modifiers.iter().copied(),
            gbm::BufferObjectFlags::SCANOUT | gbm::BufferObjectFlags::RENDERING,
        )?;

        let framebuffer = Guard::new(
            drm.card
                .add_planar_framebuffer(&buffer_object, drm::control::FbCmd2Flags::MODIFIERS)?,
            |fb| {
                let _ = drm.card.destroy_framebuffer(fb);
            },
        );

        Ok(Self {
            backend: backend
                .new_buffer(drm, &buffer_object)
                .map_err(ScanoutError::Backend)?,
            framebuffer: framebuffer.finish(),
            buffer_object,
            card: drm.card.clone(),
        })
    }
}

/// Get the format modifiers supported for this plane...
fn drm_modifiers<B: Backend>(
    card: &Card,
    fourcc: gbm::Format,
    plane_props: &drm::control::PropertyValueSet,
) -> Result<HashSet<u64>, ScanoutError<B>> {
    let in_formats = plane_props
        .iter()
        .find_map(|(k, v)| (card.get_property(*k).ok()?.name() == c"IN_FORMATS").then_some(*v))
        .and_then(|blob| card.get_property_blob(blob).ok())
        .ok_or(ScanoutError::CannotGetDrmFormats)?;
    let in_formats = drmx::format_modifier::drm_format_modifier_blob::read(&in_formats)?;
    Ok(HashSet::<u64>::from_iter(
        in_formats
            .modifiers_for(fourcc)
            .into_iter()
            .filter(|&m| m != u64::from(gbm::Modifier::Invalid)),
    ))
}
