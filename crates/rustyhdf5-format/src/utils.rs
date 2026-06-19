//! Common utility functions for parsing HDF5 binary format structures.

use crate::error::FormatError;

/// Check that `data` has at least `offset + needed` bytes available.
pub fn ensure_len(data: &[u8], offset: usize, needed: usize) -> Result<(), FormatError> {
    match offset.checked_add(needed) {
        Some(end) if end <= data.len() => Ok(()),
        _ => Err(FormatError::UnexpectedEof {
            expected: offset.saturating_add(needed),
            available: data.len(),
        }),
    }
}

/// Read a little-endian unsigned integer of `size` bytes (1, 2, 4, or 8) from `data` at `pos`.
pub fn read_offset(data: &[u8], pos: usize, size: u8) -> Result<u64, FormatError> {
    let s = size as usize;
    ensure_len(data, pos, s)?;
    let slice = &data[pos..pos + s];
    Ok(match size {
        1 => slice[0] as u64,
        2 => u16::from_le_bytes([slice[0], slice[1]]) as u64,
        4 => u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]) as u64,
        8 => u64::from_le_bytes([
            slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
        ]),
        _ => return Err(FormatError::InvalidOffsetSize(size)),
    })
}

/// Check if a 64-bit address value is the "undefined address" for a given offset size.
/// Undefined addresses are all-ones values: 0xFFFF (2-byte), 0xFFFF_FFFF (4-byte),
/// 0xFFFF_FFFF_FFFF_FFFF (8-byte).
pub fn is_undefined_offset(val: u64, offset_size: u8) -> bool {
    match offset_size {
        2 => val == 0xFFFF,
        4 => val == 0xFFFF_FFFF,
        8 => val == 0xFFFF_FFFF_FFFF_FFFF,
        _ => false,
    }
}

/// Check if `size` bytes starting at `pos` in `data` are all 0xFF (undefined address).
pub fn is_undefined_bytes(data: &[u8], pos: usize, size: u8) -> bool {
    let s = size as usize;
    if pos + s > data.len() {
        return false;
    }
    data[pos..pos + s].iter().all(|&b| b == 0xFF)
}