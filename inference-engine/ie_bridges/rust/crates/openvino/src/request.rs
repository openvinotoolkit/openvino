use crate::blob::Blob;
use crate::{cstr, drop_using_function, try_unsafe, util::Result};
use openvino_sys::{
    ie_infer_request_free, ie_infer_request_get_blob, ie_infer_request_infer,
    ie_infer_request_set_batch, ie_infer_request_set_blob, ie_infer_request_t,
};

/// See [InferRequest](https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1InferRequest.html).
pub struct InferRequest {
    pub(crate) instance: *mut ie_infer_request_t,
}
drop_using_function!(InferRequest, ie_infer_request_free);

impl InferRequest {
    /// Set the batch size of the inference requests.
    pub fn set_batch_size(&mut self, size: usize) -> Result<()> {
        try_unsafe!(ie_infer_request_set_batch(self.instance, size as u64))
    }

    /// Assign a [Blob] to the input (i.e. `name`) on the network.
    pub fn set_blob(&mut self, name: &str, blob: Blob) -> Result<()> {
        try_unsafe!(ie_infer_request_set_blob(
            self.instance,
            cstr!(name),
            blob.instance
        ))
    }

    /// Retrieve a [Blob] from the output (i.e. `name`) on the network.
    pub fn get_blob(&mut self, name: &str) -> Result<Blob> {
        let mut instance = std::ptr::null_mut();
        try_unsafe!(ie_infer_request_get_blob(
            self.instance,
            cstr!(name),
            &mut instance as *mut *mut _
        ))?;
        Ok(unsafe { Blob::from_raw_pointer(instance) })
    }

    /// Execute the inference request.
    pub fn infer(&mut self) -> Result<()> {
        try_unsafe!(ie_infer_request_infer(self.instance))
    }
}
