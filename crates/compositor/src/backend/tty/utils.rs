use std::ops::Deref;

use smithay::reexports::{ash::vk, gbm};

pub struct Guard<T: Copy, F: FnOnce(T)>(Option<(T, F)>);

impl<T: Copy, F: FnOnce(T)> Guard<T, F> {
    pub const fn new(value: T, cleanup: F) -> Self {
        Self(Some((value, cleanup)))
    }

    pub fn finish(mut self) -> T {
        self.0.take().unwrap().0
    }
}

impl<T: Copy, F: FnOnce(T)> Deref for Guard<T, F> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0.as_ref().unwrap().0
    }
}

impl<T: Copy, F: FnOnce(T)> Drop for Guard<T, F> {
    fn drop(&mut self) {
        if let Some((t, f)) = self.0.take() {
            f(t)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Format(pub gbm::Format);

impl Format {
    pub fn vk(&self) -> vk::Format {
        use gbm::Format as G;
        use vk::Format as F;
        match self.0 {
            G::Xrgb8888 => F::B8G8R8A8_UNORM,
            G::Argb8888 => F::B8G8R8A8_UNORM,
            G::Xbgr8888 => F::R8G8B8A8_UNORM,
            G::Abgr8888 => F::R8G8B8A8_UNORM,

            // HDR
            G::Xbgr2101010 => F::A2B10G10R10_UNORM_PACK32,
            G::Abgr2101010 => F::A2B10G10R10_UNORM_PACK32,

            G::Rgb565 => F::R5G6B5_UNORM_PACK16,

            G::Abgr16161616f => F::R16G16B16A16_SFLOAT,

            _ => panic!("unsupported color format"),
        }
    }

    pub fn fourcc(&self) -> gbm::Format {
        self.0
    }

    pub fn skia(&self) -> skia_safe::ColorType {
        use gbm::Format as G;
        use skia_safe::ColorType as S;

        match self.0 {
            G::Xrgb8888 => S::RGB888x,
            G::Argb8888 => S::RGBA8888,
            G::Xbgr8888 => S::BGRA8888,
            G::Abgr8888 => S::BGRA8888,

            // HDR
            G::Xbgr2101010 => S::BGR101010x,
            G::Abgr2101010 => S::BGRA1010102,
            G::Abgr16161616f => S::RGBAF16,

            G::Rgb565 => S::RGB565,

            _ => panic!("unsupported color format"),
        }
    }
}
