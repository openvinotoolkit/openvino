//! Define the interface between Rust and OpenVINO's C [API](https://docs.openvinotoolkit.org/latest/usergroup16.html).
//! See the [binding] module for how this library calls into OpenVINO and [../build.rs] for how the
//! OpenVINO libraries are linked in.

use crate::blob::Blob;
use crate::network::{CNNNetwork, ExecutableNetwork};
use crate::tensor_desc::TensorDesc;
use crate::{cstr, drop_using_function, try_unsafe, util::Result};
use crate::{Layout, Precision};
use openvino_sys::{
    self, ie_config_t, ie_core_create, ie_core_free, ie_core_load_network, ie_core_read_network,
    ie_core_read_network_from_memory, ie_core_t,
};

/// See [Core](https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1Core.html).
pub struct Core {
    instance: *mut ie_core_t,
}
drop_using_function!(Core, ie_core_free);

impl Core {
    /// Construct a new OpenVINO [Core]--this is the primary entrypoint for constructing and using
    /// inference networks.
    pub fn new(xml_config_file: Option<&str>) -> Result<Core> {
        let file = match xml_config_file {
            None => format!("{}/plugins.xml", openvino_sys::LIBRARY_PATH),
            Some(f) => f.to_string(),
        };
        let mut instance = std::ptr::null_mut();
        try_unsafe!(ie_core_create(cstr!(file), &mut instance as *mut *mut _))?;
        Ok(Core { instance })
    }

    /// Read a [CNNNetwork] from a pair of files: `model_path` points to an XML file containing the
    /// OpenVINO network IR and `weights_path` points to the binary weights file.
    pub fn read_network_from_file(
        &mut self,
        model_path: &str,
        weights_path: &str,
    ) -> Result<CNNNetwork> {
        let mut instance = std::ptr::null_mut();
        try_unsafe!(ie_core_read_network(
            self.instance,
            cstr!(model_path),
            cstr!(weights_path),
            &mut instance as *mut *mut _,
        ))?;
        Ok(CNNNetwork { instance })
    }

    /// Read a [CNNNetwork] from a pair of byte slices: `model_content` contains the XML data
    /// describing the OpenVINO network IR and `weights_content` contains the binary weights.
    pub fn read_network_from_buffer(
        &mut self,
        model_content: &[u8],
        weights_content: &[u8],
    ) -> Result<CNNNetwork> {
        let mut instance = std::ptr::null_mut();
        let weights_desc =
            TensorDesc::new(Layout::ANY, &[weights_content.len() as u64], Precision::U8);
        let weights_blob = Blob::new(weights_desc, weights_content)?;
        try_unsafe!(ie_core_read_network_from_memory(
            self.instance,
            model_content as *const _ as *const u8,
            model_content.len() as u64,
            weights_blob.instance,
            &mut instance as *mut *mut _,
        ))?;
        Ok(CNNNetwork { instance })
    }

    /// Instantiate a [CNNNetwork] as an [ExecutableNetwork] on the specified `device`.
    pub fn load_network(
        &mut self,
        network: &CNNNetwork,
        device: &str,
    ) -> Result<ExecutableNetwork> {
        let mut instance = std::ptr::null_mut();
        // Because `ie_core_load_network` does not allow a null pointer for the configuration, we
        // construct an empty configuration struct to pass. At some point, it could be good to allow
        // users to pass a map to this function that gets converted to an `ie_config_t` (TODO).
        let empty_config = ie_config_t {
            name: std::ptr::null(),
            value: std::ptr::null(),
            next: std::ptr::null_mut(),
        };
        try_unsafe!(ie_core_load_network(
            self.instance,
            network.instance,
            cstr!(device),
            &empty_config as *const _,
            &mut instance as *mut *mut _
        ))?;
        Ok(ExecutableNetwork { instance })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_core() {
        let _ = Core::new(None).unwrap();
    }
}
