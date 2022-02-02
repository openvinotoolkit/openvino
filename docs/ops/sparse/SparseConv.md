# SparseConv {#openvino_docs_ops_sparse_SparseConv}

**Versioned name**: *SparseConv*

**Category**: *Sparse*

**Short description**: Performs convolution with dense kernel over sparse input.

**Detailed description**: This is SparseConv layer introduced in [Open3D](https://github.com/isl-org/Open3D) project and described in [paper](https://openreview.net/pdf?id=B1lDoJSYDH). SparseConv takes input defined as a pair of features tensor and input positions tensor.

**Attributes**: no attributes

**Inputs**:

*   **1**: `features` Features tensor with shape `NxIC` where `N` is a number of features and `IC` is a number of input channels **Required.**

*   **2**: `inp_pos` Positions tensor which determines locations of features in the spase. Shape is `NxZ` where `N` is a number of features and `Z` equals to dimensionality. **Required.**

*   **3**: `out_pos` Positions tensor in which convolution is computed and then result is returned. Shape is `MxZ` where `M` is a number of output position and `Z` equals to dimensionality. **Required.**

*   **4**: `kernel` Dense weights similar to regular Convolution layer. Shape is `DxHxWxICxOC` (3D convolution). **Required.**

*   **5**: `offset` Vector of `Z` offset values. Usually, filled by zeros for odd kernel size and by `-0.5` for even kernel size.  **Required.**

**Outputs**:

*   **1**: Tensor of shape `MxOC` where `M` is a number of output positions and `OC` is a number of output channels.

```
@inproceedings{
    Ummenhofer2020Lagrangian,
    title={Lagrangian Fluid Simulation with Continuous Convolutions},
    author={Benjamin Ummenhofer and Lukas Prantl and Nils Thuerey and Vladlen Koltun},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=B1lDoJSYDH}
}
```
