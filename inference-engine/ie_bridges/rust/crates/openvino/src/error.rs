use thiserror::Error;

/// See [IEStatusCode](https://docs.openvinotoolkit.org/latest/ie_c_api/ie__c__api_8h.html#a391683b1e8e26df8b58d7033edd9ee83).
/// TODO Replace this in bindgen with [newtype_enum](https://docs.rs/bindgen/0.54.1/bindgen/struct.Builder.html#method.newtype_enum)
/// or [rustified_enum](https://docs.rs/bindgen/0.54.1/bindgen/struct.Builder.html#method.rustified_enum).
#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("general error")]
    GeneralError,
    #[error("not implemented")]
    NotImplemented,
    #[error("network not loaded")]
    NetworkNotLoaded,
    #[error("parameter mismatch")]
    ParameterMismatch,
    #[error("not found")]
    NotFound,
    #[error("out of bounds")]
    OutOfBounds,
    #[error("unexpected")]
    Unexpected,
    #[error("request busy")]
    RequestBusy,
    #[error("result not ready")]
    ResultNotReady,
    #[error("not allocated")]
    NotAllocated,
    #[error("infer not started")]
    InferNotStarted,
    #[error("network not ready")]
    NetworkNotReady,
    #[error("undefined error code: {0}")]
    Undefined(i32),
}

impl InferenceError {
    pub fn from(error_code: i32) -> Result<(), InferenceError> {
        use InferenceError::*;
        match error_code {
            openvino_sys::IEStatusCode_OK => Ok(()),
            openvino_sys::IEStatusCode_GENERAL_ERROR => Err(GeneralError),
            openvino_sys::IEStatusCode_NOT_IMPLEMENTED => Err(NotImplemented),
            openvino_sys::IEStatusCode_NETWORK_NOT_LOADED => Err(NetworkNotLoaded),
            openvino_sys::IEStatusCode_PARAMETER_MISMATCH => Err(ParameterMismatch),
            openvino_sys::IEStatusCode_NOT_FOUND => Err(NotFound),
            openvino_sys::IEStatusCode_OUT_OF_BOUNDS => Err(OutOfBounds),
            openvino_sys::IEStatusCode_UNEXPECTED => Err(Unexpected),
            openvino_sys::IEStatusCode_REQUEST_BUSY => Err(RequestBusy),
            openvino_sys::IEStatusCode_RESULT_NOT_READY => Err(ResultNotReady),
            openvino_sys::IEStatusCode_NOT_ALLOCATED => Err(NotAllocated),
            openvino_sys::IEStatusCode_INFER_NOT_STARTED => Err(InferNotStarted),
            openvino_sys::IEStatusCode_NETWORK_NOT_READ => Err(NetworkNotReady),
            _ => Err(Undefined(error_code)),
        }
    }
}
