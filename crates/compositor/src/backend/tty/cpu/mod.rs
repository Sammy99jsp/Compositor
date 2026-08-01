use std::{
    collections::HashSet,
    os::{
        fd::{AsFd, AsRawFd},
        unix::prelude::OwnedFd,
    },
    sync::Arc,
    task::Poll,
};

use smithay::reexports::{
    drm::{self, control::Device},
    gbm, rustix,
};

use crate::backend::tty::{Backend, BackendBuffer, drmx::Format};

use super::Card;

pub struct Cpu {
    card: Arc<Card>,
}

pub struct CpuBuffer {
    bo: gbm::BufferObject<()>,
    format: Format,
    fence: Option<ImportedSyncFile>,
}

const TIMEOUT_NSEC: i64 = 500_000; // 0.5ms

impl Backend for Cpu {
    const NAME: &str = "CPU";
    const BUFFER_FLAGS: gbm::BufferObjectFlags = const {
        gbm::BufferObjectFlags::SCANOUT
            .union(gbm::BufferObjectFlags::LINEAR)
            .union(gbm::BufferObjectFlags::WRITE)
    };
    type Error = std::io::Error;

    fn new(card: &Arc<Card>) -> Result<Self, Self::Error>
    where
        Self: Sized,
    {
        Ok(Cpu { card: card.clone() })
    }

    fn modifiers(
        &self,
        _: super::drmx::Format,
        _: &HashSet<u64>,
    ) -> std::collections::HashSet<u64> {
        HashSet::new()
    }

    type Buffer = CpuBuffer;
    fn new_buffer(
        &self,
        drm: &super::DrmState,
        bo: gbm::BufferObject<()>,
    ) -> Result<Self::Buffer, Self::Error> {
        Ok(CpuBuffer {
            bo,
            fence: None,
            format: drm.format,
        })
    }

    fn render(&mut self, buffer: &mut Self::Buffer, frame: usize) -> Result<Poll<()>, Self::Error> {
        // Wait on the imported sync file.
        if let Some(sync) = &buffer.fence {
            const TIMEOUT: i32 = rustix::io::Errno::TIME.raw_os_error();
            match self
                .card
                .syncobj_wait(core::slice::from_ref(sync), TIMEOUT_NSEC, true, true)
            {
                Ok(_) => (),
                Err(err) if let Some(TIMEOUT) = err.raw_os_error() => return Ok(Poll::Pending),
                Err(err) => return Err(err),
            }
        }

        let (width, height) = (buffer.bo.width(), buffer.bo.height());
        let format = buffer.format;

        buffer.bo.map_mut(0, 0, width, height, |pxs| {
            let info = skia_safe::ImageInfo::new(
                (width as _, height as _),
                format.skia(),
                skia_safe::AlphaType::Premul,
                None,
            );

            let stride = pxs.stride();
            let mut surface =
                skia_safe::surfaces::wrap_pixels(&info, pxs.buffer_mut(), stride as usize, None)
                    .expect("Valid skia surface");

            let canvas = surface.canvas();

            canvas.clear(skia_safe::Color4f {
                r: 1.0,
                g: 0.0,
                b: 1.0,
                a: 1.0,
            });

            let f = skia_safe::FontMgr::new();
            let mut a = f.match_family("Arvo");
            let fface = a
                .match_style(skia_safe::FontStyle::bold())
                .expect("Find font");
            let font = &skia_safe::Font::new(fface, 128.0);
            let paint = skia_safe::Paint::new(
                skia_safe::Color4f {
                    r: 0.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                },
                None,
            );
            canvas.draw_text_align(
                format!("{frame}"),
                (100.0, 100.0),
                font,
                &paint,
                skia_safe::utils::text_utils::Align::Left,
            );

            Poll::Ready(())
        })
    }
}

impl BackendBuffer for CpuBuffer {
    type Backend = Cpu;

    fn import_sync(
        &mut self,
        cpu: &mut Cpu,
        fd: OwnedFd,
    ) -> Result<(), <Self::Backend as Backend>::Error> {
        self.fence.replace(import_sync_file(cpu.card.clone(), fd)?);
        Ok(())
    }

    fn export_sync(
        &mut self,
        _: &mut Cpu,
    ) -> Result<Option<OwnedFd>, <Self::Backend as Backend>::Error> {
        Ok(None)
    }
}

pub struct ImportedSyncFile {
    card: Arc<Card>,
    sync: drm::control::syncobj::Handle,

    #[expect(unused)] // Closed when dropped
    fd: OwnedFd,
}

impl std::ops::Deref for ImportedSyncFile {
    type Target = drm::control::syncobj::Handle;

    fn deref(&self) -> &Self::Target {
        &self.sync
    }
}

impl Drop for ImportedSyncFile {
    fn drop(&mut self) {
        let _ = self.card.destroy_syncobj(self.sync);
    }
}

// Stolen shamelessly from Smithay's workaround whilst drm-rs sorts this out:
// https://github.com/caedenhq/smithay/blob/642e53e9eb1260c223c1cc821d52df4bb65a7da8/src/wayland/drm_syncobj/sync_point.rs#L201
fn import_sync_file(card: Arc<Card>, fd: OwnedFd) -> std::io::Result<ImportedSyncFile> {
    use rustix::ioctl::{Updater, ioctl, opcode::read_write};

    let sync = card.create_syncobj(false)?;

    const DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE: rustix::ioctl::Opcode =
        read_write::<drm_ffi::drm_syncobj_handle>(drm_ffi::DRM_IOCTL_BASE, 0xC2);

    let mut args = drm_ffi::drm_syncobj_handle {
        handle: sync.into(),
        flags: drm_ffi::DRM_SYNCOBJ_FD_TO_HANDLE_FLAGS_IMPORT_SYNC_FILE,
        fd: fd.as_raw_fd(),
        pad: 0,
        point: 0,
    };

    let result = unsafe {
        ioctl(
            card.as_fd(),
            Updater::<DRM_IOCTL_SYNCOBJ_FD_TO_HANDLE, _>::new(&mut args),
        )
    };

    if let Err(err) = result {
        let _ = card.destroy_syncobj(sync);
        return Err(err.into());
    }

    Ok(ImportedSyncFile { card, sync, fd })
}
