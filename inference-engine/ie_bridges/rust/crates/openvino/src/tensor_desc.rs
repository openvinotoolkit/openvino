use crate::{Layout, Precision};
use openvino_sys::{dimensions_t, tensor_desc_t};

/// See [TensorDesc](https://docs.openvinotoolkit.org/latest/classInferenceEngine_1_1TensorDesc.html).
pub struct TensorDesc {
    pub(crate) instance: tensor_desc_t,
}

impl TensorDesc {
    /// Construct a new [TensorDesc] from its C API components.
    pub fn new(layout: Layout, dimensions: &[u64], precision: Precision) -> Self {
        // Setup dimensions.
        assert!(dimensions.len() < 8);
        let mut dims = [0; 8];
        dims[..dimensions.len()].copy_from_slice(dimensions);

        // Create the description structure.
        Self {
            instance: tensor_desc_t {
                layout,
                dims: dimensions_t {
                    ranks: dimensions.len() as u64,
                    dims,
                },
                precision,
            },
        }
    }

    /// Get the number of elements described by this [TensorDesc].
    pub fn len(&self) -> usize {
        self.instance.dims.dims[..self.instance.dims.ranks as usize]
            .iter()
            .fold(1, |a, &b| a * b as usize)
    }
}
