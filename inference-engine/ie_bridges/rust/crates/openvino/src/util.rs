//! A collection of utility types and macros for use inside this crate.
use crate::InferenceError;

/// This alias makes the implementation slightly less verbose.
pub(crate) type Result<T> = std::result::Result<T, InferenceError>;

/// Convert a Rust string into a string to pass across the C boundary.
#[macro_export]
macro_rules! cstr {
    ($str: expr) => {
        std::ffi::CString::new($str)
            .expect("a valid C string")
            .into_raw()
    };
}

/// Convert an unsafe call to openvino-sys into an [InferenceError].
#[macro_export]
macro_rules! try_unsafe {
    ($e: expr) => {
        crate::InferenceError::from(unsafe { $e })
    };
}

/// Drop one of the Rust wrapper structures using the provided free function. This relies on all
/// Rust wrapper functions having an `instance` field pointing to their OpenVINO C structure.
#[macro_export]
macro_rules! drop_using_function {
    ($ty: ty, $free_fn: expr) => {
        impl Drop for $ty {
            fn drop(&mut self) {
                unsafe { $free_fn(&mut self.instance as *mut *mut _) }
                debug_assert!(self.instance.is_null())
            }
        }
    };
}
