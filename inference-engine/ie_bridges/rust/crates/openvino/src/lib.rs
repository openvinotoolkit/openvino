mod blob;
mod core;
mod error;
mod network;
mod request;
mod tensor_desc;
mod util;

pub use crate::core::Core;
pub use blob::Blob;
pub use error::InferenceError;
pub use network::{CNNNetwork, ExecutableNetwork};
// Re-publish some OpenVINO enums with a conventional Rust naming (see
// `crates/openvino-sys/build.rs`).
pub use openvino_sys::{
    layout_e as Layout, precision_e as Precision, resize_alg_e as ResizeAlgorithm,
};
pub use request::InferRequest;
pub use tensor_desc::TensorDesc;

/// Emit the version string of the OpenVINO C API backing this implementation.
pub fn version() -> String {
    use std::ffi::CStr;
    let mut ie_version = unsafe { openvino_sys::ie_c_api_version() };
    let str_version = unsafe { CStr::from_ptr(ie_version.api_version) }
        .to_string_lossy()
        .into_owned();
    unsafe { openvino_sys::ie_version_free(&mut ie_version as *mut openvino_sys::ie_version_t) };
    str_version
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_version() {
        assert!(version().starts_with("2.1"),)
    }
}
