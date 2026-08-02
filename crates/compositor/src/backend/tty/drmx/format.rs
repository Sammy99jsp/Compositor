use smithay::reexports::gbm;

#[derive(Debug, Clone, Copy)]
pub struct Format(pub gbm::Format);

impl Format {
    pub fn fourcc(&self) -> gbm::Format {
        self.0
    }

    pub fn skia(&self) -> skia_safe::ColorType {
        use gbm::Format as G;
        use skia_safe::ColorType as S;

        match self.0 {
            G::Xrgb8888 => S::BGRA8888,
            G::Argb8888 => S::BGRA8888,
            G::Xbgr8888 => S::RGB888x,
            G::Abgr8888 => S::RGBA8888,

            // HDR
            G::Xbgr2101010 => S::BGR101010x,
            G::Abgr2101010 => S::BGRA1010102,
            G::Abgr16161616f => S::RGBAF16,

            G::Rgb565 => S::RGB565,

            _ => panic!("unsupported color format"),
        }
    }
}
