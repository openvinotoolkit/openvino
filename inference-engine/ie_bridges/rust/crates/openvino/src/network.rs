//! Contains the network representations in OpenVINO:
//!  - [CNNNetwork] is the OpenVINO represenation of a neural network
//!  - [ExecutableNetwork] is the compiled representation of a [CNNNetwork] for a device.

use crate::request::InferRequest;
use crate::{cstr, drop_using_function, try_unsafe, util::Result};
use crate::{Layout, Precision, ResizeAlgorithm};
use openvino_sys::{
    ie_exec_network_create_infer_request, ie_exec_network_free, ie_executable_network_t,
    ie_network_free, ie_network_get_input_name, ie_network_get_output_name, ie_network_name_free,
    ie_network_set_input_layout, ie_network_set_input_precision,
    ie_network_set_input_resize_algorithm, ie_network_set_output_precision, ie_network_t,
};
use std::ffi::CStr;

pub struct CNNNetwork {
    pub(crate) instance: *mut ie_network_t,
}
drop_using_function!(CNNNetwork, ie_network_free);

impl CNNNetwork {
    /// Retrieve the name identifying the input tensor at `index`.
    pub fn get_input_name(&self, index: usize) -> Result<String> {
        let mut cname = std::ptr::null_mut();
        try_unsafe!(ie_network_get_input_name(
            self.instance,
            index as u64,
            &mut cname as *mut *mut _
        ))?;
        let name = unsafe { CStr::from_ptr(cname) }
            .to_string_lossy()
            .into_owned();
        unsafe { ie_network_name_free(&mut cname as *mut *mut _) };
        debug_assert!(cname.is_null());
        Ok(name)
    }

    /// Retrieve the name identifying the output tensor at `index`.
    pub fn get_output_name(&self, index: usize) -> Result<String> {
        let mut cname = std::ptr::null_mut();
        try_unsafe!(ie_network_get_output_name(
            self.instance,
            index as u64,
            &mut cname as *mut *mut _
        ))?;
        let name = unsafe { CStr::from_ptr(cname) }
            .to_string_lossy()
            .into_owned();
        unsafe { ie_network_name_free(&mut cname as *mut *mut _) };
        debug_assert!(cname.is_null());
        Ok(name)
    }

    /// Configure a resize algorithm for the input tensor at `input_name`.
    pub fn set_input_resize_algorithm(
        &mut self,
        input_name: &str,
        algorithm: ResizeAlgorithm,
    ) -> Result<()> {
        try_unsafe!(ie_network_set_input_resize_algorithm(
            self.instance,
            cstr!(input_name),
            algorithm
        ))
    }

    /// Configure a layout for the input tensor at `input_name`.
    pub fn set_input_layout(&mut self, input_name: &str, layout: Layout) -> Result<()> {
        try_unsafe!(ie_network_set_input_layout(
            self.instance,
            cstr!(input_name),
            layout
        ))
    }

    /// Configure the precision for the input tensor at `input_name`.
    pub fn set_input_precision(&mut self, input_name: &str, precision: Precision) -> Result<()> {
        try_unsafe!(ie_network_set_input_precision(
            self.instance,
            cstr!(input_name),
            precision
        ))
    }

    /// Configure the precision for the output tensor at `output_name`.
    pub fn set_output_precision(&mut self, output_name: &str, precision: Precision) -> Result<()> {
        try_unsafe!(ie_network_set_output_precision(
            self.instance,
            cstr!(output_name),
            precision
        ))
    }
}

pub struct ExecutableNetwork {
    pub(crate) instance: *mut ie_executable_network_t,
}
drop_using_function!(ExecutableNetwork, ie_exec_network_free);

impl ExecutableNetwork {
    /// Create an [InferRequest].
    pub fn create_infer_request(&mut self) -> Result<InferRequest> {
        let mut instance = std::ptr::null_mut();
        try_unsafe!(ie_exec_network_create_infer_request(
            self.instance,
            &mut instance as *mut *mut _
        ))?;
        Ok(InferRequest { instance })
    }
}
