use crate::tensor_desc::TensorDesc;
use crate::{drop_using_function, try_unsafe, util::Result};
use crate::{Layout, Precision};
use openvino_sys::{
    self, dimensions_t, ie_blob_buffer__bindgen_ty_1, ie_blob_buffer_t, ie_blob_byte_size,
    ie_blob_free, ie_blob_get_buffer, ie_blob_get_dims, ie_blob_get_layout, ie_blob_get_precision,
    ie_blob_make_memory, ie_blob_size, ie_blob_t,
};
use std::convert::TryFrom;

/// See [Blob](https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1Blob.html).
pub struct Blob {
    pub(crate) instance: *mut ie_blob_t,
}
drop_using_function!(Blob, ie_blob_free);

impl Blob {
    /// Create a new [Blob] by copying data in to the OpenVINO-allocated memory.
    pub fn new(description: TensorDesc, data: &[u8]) -> Result<Self> {
        let mut blob = Self::allocate(description)?;
        let blob_len = blob.byte_len()?;
        assert_eq!(
            blob_len,
            data.len(),
            "The data to initialize ({} bytes) must be the same as the blob size ({} bytes).",
            data.len(),
            blob_len
        );

        // Copy the incoming data into the buffer.
        let buffer = blob.buffer_mut()?;
        buffer.copy_from_slice(data);

        Ok(blob)
    }

    /// Allocate space in OpenVINO for an empty [Blob].
    pub fn allocate(description: TensorDesc) -> Result<Self> {
        let mut instance = std::ptr::null_mut();
        try_unsafe!(ie_blob_make_memory(
            &description.instance as *const _,
            &mut instance as *mut *mut _
        ))?;
        Ok(Self { instance })
    }

    /// Return the tensor description of this [Blob].
    pub fn tensor_desc(&self) -> Result<TensorDesc> {
        let blob = self.instance as *const ie_blob_t;

        let mut layout = Layout::ANY;
        try_unsafe!(ie_blob_get_layout(blob, &mut layout as *mut _))?;

        let mut dimensions = dimensions_t {
            ranks: 0,
            dims: [0; 8usize],
        };
        try_unsafe!(ie_blob_get_dims(blob, &mut dimensions as *mut _))?;

        let mut precision = Precision::UNSPECIFIED;
        try_unsafe!(ie_blob_get_precision(blob, &mut precision as *mut _))?;

        Ok(TensorDesc::new(layout, &dimensions.dims, precision))
    }

    /// Get the number of elements contained in the [Blob].
    pub fn len(&mut self) -> Result<usize> {
        let mut size = 0;
        try_unsafe!(ie_blob_size(self.instance, &mut size as *mut _))?;
        Ok(usize::try_from(size).unwrap())
    }

    /// Get the size of the current [Blob] in bytes.
    pub fn byte_len(&mut self) -> Result<usize> {
        let mut size = 0;
        try_unsafe!(ie_blob_byte_size(self.instance, &mut size as *mut _))?;
        Ok(usize::try_from(size).unwrap())
    }

    /// Retrieve the [Blob]'s data as an immutable slice of bytes.
    pub fn buffer(&mut self) -> Result<&[u8]> {
        let mut buffer = Blob::empty_buffer();
        try_unsafe!(ie_blob_get_buffer(self.instance, &mut buffer as *mut _))?;
        let size = self.byte_len()?;
        let slice = unsafe {
            std::slice::from_raw_parts(buffer.__bindgen_anon_1.buffer as *const u8, size)
        };
        Ok(slice)
    }

    /// Retrieve the [Blob]'s data as a mutable slice of bytes.
    pub fn buffer_mut(&mut self) -> Result<&mut [u8]> {
        let mut buffer = Blob::empty_buffer();
        try_unsafe!(ie_blob_get_buffer(self.instance, &mut buffer as *mut _))?;
        let size = self.byte_len()?;
        let slice = unsafe {
            std::slice::from_raw_parts_mut(buffer.__bindgen_anon_1.buffer as *mut u8, size)
        };
        Ok(slice)
    }

    /// Retrieve the [Blob]'s data as a mutable slice of type `T`. This is `unsafe`, since the
    /// values of `T` may not have been properly initialized; however, this functionality
    /// is provided as an equivalent of what C/C++ users of OpenVINO currently do to access [Blob]s
    /// with, e.g., floating point values: `results.buffer_mut_as_type::<f32>()`.
    pub unsafe fn buffer_mut_as_type<T>(&mut self) -> Result<&mut [T]> {
        let mut buffer = Blob::empty_buffer();
        try_unsafe!(ie_blob_get_buffer(self.instance, &mut buffer as *mut _))?;
        // This is very unsafe, but very convenient: by allowing users to specify T, they can
        // retrieve the buffer in whatever shape they prefer. But we must ensure that they cannot
        // read too many bytes, so we manually calculate the resulting slice `size`.
        let size = self.byte_len()? / std::mem::size_of::<T>();
        let slice = unsafe {
            std::slice::from_raw_parts_mut(buffer.__bindgen_anon_1.buffer as *mut T, size)
        };
        Ok(slice)
    }

    /// Construct a Blob from its associated pointer.
    pub(crate) unsafe fn from_raw_pointer(instance: *mut ie_blob_t) -> Self {
        Self { instance }
    }

    fn empty_buffer() -> ie_blob_buffer_t {
        ie_blob_buffer_t {
            __bindgen_anon_1: ie_blob_buffer__bindgen_ty_1 {
                buffer: std::ptr::null_mut(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn invalid_blob_size() {
        let desc = TensorDesc::new(Layout::NHWC, &[1, 2, 2, 2], Precision::U8);
        // Blob should be 1x2x2x2 = 8 bytes but we pass in 7 bytes:
        let _ = Blob::new(desc, &[0; 7]).unwrap();
    }

    #[test]
    fn buffer_conversion() {
        const LEN: usize = 200 * 100;
        let desc = TensorDesc::new(Layout::HW, &[200, 100], Precision::U16);

        // Provide a u8 slice to create a u16 blob (twice as many items).
        let mut blob = Blob::new(desc, &[0; LEN * 2]).unwrap();

        assert_eq!(blob.len().unwrap(), LEN);
        assert_eq!(
            blob.byte_len().unwrap(),
            LEN * 2,
            "we should have twice as many bytes (u16 = u8 * 2)"
        );
        assert_eq!(
            blob.buffer().unwrap().len(),
            LEN * 2,
            "we should have twice as many items (u16 = u8 * 2)"
        );
        assert_eq!(
            unsafe { blob.buffer_mut_as_type::<f32>() }.unwrap().len(),
            LEN / 2,
            "we should have half as many items (u16 = f32 / 2)"
        );
    }
}
