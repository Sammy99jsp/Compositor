#![allow(non_camel_case_types)]

use anyhow::Context;
use smithay::reexports::drm;

#[derive(Debug)]
pub struct drm_format_modifier_blob {
    pub version: u32,
    pub flags: u32,
    pub formats: Vec<format>,
    pub modifiers: Vec<drm_format_modifier>,
}

type format = drm::buffer::DrmFourcc;

#[repr(C)]
#[derive(Debug)]
pub struct drm_format_modifier {
    pub formats: u64,
    pub offset: u32,
    _p: u32,
    pub modifier: u64,
}

#[repr(C)]
#[derive(Debug)]
struct drm_format_modifier_blob_header {
    version: u32,
    flags: u32,
    count_formats: u32,
    formats_offset: u32,
    count_modifiers: u32,
    modifiers_offset: u32,
}

impl drm_format_modifier_blob {
    pub fn read(blob: &[u8]) -> anyhow::Result<Self> {
        // SAFETY: the header's contents are all integer types, so are valid for any bit pattern...
        let drm_format_modifier_blob_header {
            version,
            flags,
            count_formats,
            formats_offset,
            count_modifiers,
            modifiers_offset,
        } = unsafe {
            extract_array(
                blob,
                1,
                0,
                "valid header not present in the blob provided by drm",
            )?
            .pop()
            .unwrap()
        };

        // Check that this is v1 to not break under future versions
        anyhow::ensure!(
            version == 1,
            "expected v1 of `drm_format_modifier_blob`, got {version}"
        );

        // SAFETY: u32-s are valid for any bit pattern.
        let formats = unsafe {
            extract_array(
                blob,
                count_formats,
                formats_offset,
                "formats slice out of bounds",
            )
        }?;

        // SAFETY: drm_format_modifier-s are composed of only integer types, so this is okay...
        let modifiers = unsafe {
            extract_array(
                blob,
                count_modifiers,
                modifiers_offset,
                "modifiers slice out of bounds",
            )
        }?;

        Ok(Self {
            version,
            flags,
            formats,
            modifiers,
        })
    }

    pub fn modifiers_for(&self, fourcc: drm::buffer::DrmFourcc) -> Vec<u64> {
        let Some(idx) = self.formats.iter().position(|&f| f == fourcc) else {
            return Vec::new();
        };

        let idx = idx as u64;

        self.modifiers
            .iter()
            .filter(|m| {
                let offset = m.offset as u64;

                // Check if our format index is contained within this bitset.
                let in_range = idx >= offset && idx < offset + 64;

                // Check if the corresponding bit for this format is set.
                let is_enabled = (m.formats >> (idx - offset)) & 1 != 0;

                in_range && is_enabled
            })
            .map(|m| m.modifier)
            .collect()
    }
}

/// # Safety
/// `<T>` must be valid for any bit pattern (e.g. a simple integer type).
unsafe fn extract_array<T>(
    blob: &[u8],
    count: u32,
    offset: u32,
    msg: &'static str,
) -> Result<Vec<T>, anyhow::Error> {
    let src = blob
        .get(
            (offset as usize)
                ..((offset as usize).checked_add(
                    size_of::<T>()
                        .checked_mul(count as usize)
                        .context("overflow in pointer arithmetic")?,
                ))
                .context("overflow in pointer arithmetic")?,
        )
        .context(msg)?;

    let mut dest = Vec::new();
    dest.reserve_exact(count as _);
    unsafe {
        // SAFETY: We have reserved exactly the correct amount of memory and checked that the
        //         source has enough bytes...
        core::ptr::copy_nonoverlapping(src.as_ptr(), dest.as_mut_ptr() as *mut u8, src.len());

        // SAFETY: We have set capacity to exactly `count`
        dest.set_len(count as _);
    }
    Ok(dest)
}
