# Nvidia backend support

## General information

The Nvidia backend for oneDNN can be exposed to the user via the
`dnnl::engine::kind::gpu` engine kind. Currently, for the case when user's
system has both Intel and Nvidia GPUs, `DNNL_GPU_VENDOR=NVIDIA` flag is used in
CMake, since the devices are clustered based on the device vendor ID and index
pattern can not be used to distinguish between Intel GPU and Nvidia GPU.
However, Intel is working on restructuring the engine creation, so that it would
be possible to choose engine kind and vendor kind at runtime. Also, it is
possible to create oneDNN engines using `sycl::device` objects corresponding to
Nvidia GPUs. The stream in Nvidia backend for oneDNN defines an out-of-order
SYCL queue by default. Similar to the existing oneDNN API, user can specify an
in-order queue when creating a stream if needed.

## Build command

```bash
export CC=/path/to/dpcpp/install/bin/clang
export CXX=/path/to/dpcpp/install/bin/clang++
mkdir build
cd build
cmake -DDNNL_CPU_RUNTIME=DPCPP -DDNNL_GPU_RUNTIME=DPCPP \
      -DDNNL_GPU_VENDOR=NVIDIA -G Ninja \
      -DOPENCLROOT=/path/to/the/root/folder/of/libOpenCL.so ..
```

## Memory

Currently, only the buffer-based oneDNN API is supported for Nvidia backend.

## Suported Data Types

The following table documents the supported data types.

| Data Type | Computation Mode            |
|-----------|-----------------------------|
| f32       | Training, Inference         |
| f16       | Inference                   |
| s8        | Inference (when applicable) |

## Supported Primitives and Implementation Limitations

cuDNN functions are not necessarily the same as oneDNN primitives due to lack of
standard API for DNN. For each primitive the cuDNN equivalent function is added
to the Nvidia backend for oneDNN. However, the added backend cannot provide all
functionalities supported by oneDNN primitives. The detailed limitations of each
cuDNN primitive are explained as follow.

### Batch normalization

The closest equivalent to oneDNN batch normalization can be
`cudnnBatchNormalizationForward` and `cudnnBatchNormalizationBackward`
operations. However, there are some difference between cuDNN and oneDNN batch
normalization.

#### Forward direction

* When `global_stats` flag is set for batch normalization, the mean and variance
  are input only parameters. However, cuDNN does not have the option to accept
  the mean and variance as inputs in the forward training operation. Therefore,
  `cudnnBatchNormalizationForwardInference` is used to match the oneDNN feature.
  Although inference is not supported without `global_stats` flags set.
* The cuDNN precision is different from that of oneDNN for Batch Normalization.
  (e.g `fp:0.0170898 dt:0.0170907 diff:8.27014e-07 rdiff:4.83922e-05`)
* The forward training with no flags accepts mean and variance as an output.
  However, in cuDNN the mean and variance are running mean and variance
  respectably so they are both input and output variable. Therefore, they are
  required to have a sensible value (cannot be NaN). Since oneDNN will not set
  value for the mean and variance when no flag is passed, the NaN can be
  propagated as a result. To avoid NaN propagation, `cudaMemset` function is
  used to initialize the mean and variance with zero.
* cuDNN always requires the values for scale and shift. When shift and scale are
  not defined in oneDNN, `cudaMemset` is used to initialize scale to 1 and shift
  to 0.
* For performance reason in the backward pass, cuDNN requires the mean and
  inverse variance to be saved in the forward pass. Therefore, when Nvidia
  backend is used for batch normalization, the workspace must be provided to
  save the mean and inverse variance.
* When `dnnl_fuse_norm_relu` flag is set for batch normalization, the
  `cudnnActivationForward` operation is called immediately after the batch
  normalization, since cuDNN does not have a fused batch normalization with
  `RELU`. The implementation for element-wise post operations is the same.
* When `dnnl_fuse_norm_relu` is used, the intermediate output of batch
  normalization, which is used as an input to the activation function, is saved
  in the workspace as well. This is required to compute the backward pass for
  `dnnl_fuse_norm_relu` flag.
* Forward pass supports f32, f16 and s8 data types. Although blocking is not
  supported for s8.

#### Backward direction

* cuDNN uses `alpha` and `beta` parameters to blend the `dy`, `shift` and
  `scale`. Since oneDNN does not have this feature, the `alpha` and `beta`
  values in the backward direction are set to 1 and 0 respectively to avoid
  blending.
* Nvidia backend for backward direction requires the workspace as an input
  containing the mean and inverse variance computed in the forward pass.
* The Nvidia backend for oneDNN does not support the backward direction for
  batch normalization when the flag is set to `global_stats`. This is due to the
  fact that oneDNN will skip the
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=$d_{y} -= \left ( \frac{\beta + \left ( \frac{src-mean}{\sqrt{\delta ^{2} + \epsilon }} \right )}{NHW} \right )$" >
</p>
  since the mean and variance are constant, however, cuDNN does not have an
  option to skip this operation.
* When `dnnl_fuse_norm_relu` flag is set, Nvidia backend requires the
  intermediate result of the batch normalization saved in the forward pass. This
  is used to compute the backward direction of the activation function used for
  `RELU`.

### Binary

The `cudnnOpTensor` is equivalent of oneDNN binary primitives.

* Only scales attribute is supported. Post-op attribute is not supported.
* Blocking is only supported for `int8` and only in the C dimension with either
  4 or 32 block size (same as other cuDNN primitives).

### Concat

The concat operation uses the reorder primitive to concatenate tensors over the
chosen dimension, so the same limitation as reorder applies here.

### Convolution

The `cudnnConvolutionForward`, `cudnnConvolutionBackward` and
`cudnnConvolutionBackwardFilter` is used to compute forward, backward by data or
backward by weights for a convolution operation.

* Blocking is only supported for `int8` and only in the C dimension with block
  size of 4. Input and output tensors must have the same data type.
* For int8 (s8s8s8) with post-ops the operations are performed as s8s8f32 (due
  to cuDNN limitations) then reordered to `s8` at the end which impacts
  performance.
* Direct convolution is not supported, so implicit GEMM is used in those cases.
* "Left" padding must be greater or equal to "right" padding, and the requested
  spatial output should match the output formula for two "left" padding used.
* Eltwise post-op limitations are the same as our eltwise limitation as post-ops
  are not fused.
* cuDNN requires padding tensors to 4 dimensions, so 1D convolutions are
  supported but are performed as 2D.

The following table shows the convolution status for the oneDNN Nvidia backend:

#### Forward direction
| Weights Format | Winograd Supported | Supported Input Format | Supported Output Format | Supported Data Type | Limitations                                                                                                                                                                             |
|----------------|--------------------|------------------------|-------------------------|---------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2D NCHW        | YES                | NCHW, NHWC             | NCHW, NHWC              | f32, f16            | The Winograd algorithm has limitations: <br> * Filter size must be 3x3 or 5x5. <br> * Dilation must be zero for all dimensions. <br> * Horizontal and vertical filter stride must be 1. |
| 2D NHWC        | NO                 | NHWC                   | NHWC                    | f32, f16, int8      | * Dilation must be zero in all dimensions. <br> * Output feature maps must be multiple of 4 for `int8` type.                                                                            |
| 3D NCHW        | NO                 | NCHW, NHWC             | NCHW, NHWC              | f32, f16            |                                                                                                                                                                                         |

#### Backward direction
| Weights Format | Winograd Supported | Supported Input Format | Supported Output Format | Supported Data Type | Limitations                                                                                                                                                                                                                                  |
|----------------|--------------------|------------------------|-------------------------|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2D NCHW        | YES                | NCHW, NHWC             | NCHW                    | f32, f16            | 1. Dilation must be zero for all dimensions. <br> 2. The Winograd algorithm has limitations: <br> * Filter size must be 3x3 or 5x5. <br> * Dilation must be zero for all dimensions. <br> * Horizontal and vertical filter stride must be 1. |
| 2D NHWC        | NO                 | NHWC                   | NHWC                    | f32, f16            |                                                                                                                                                                                                                                              |
| 3D NCHW        | NO                 | NCHW, NHWC             | NCHW                    | f32, f16            |                                                                                                                                                                                                                                              |
### Deconvolution

Deconvolution primitive is implemented through the convolution with swapped
input abd output channels.

* Currently, there is a bug, likely in this code, which causes crashes in
  memory_tracking for 3D backward_weights with bias when backward_weights
  without bias was also a part of the run. Cache interrogation is suspected due
  to cache-free runs are successful. Switched off in benchdnn until further
  investigation and the fix.

### Eltwise

The `cudnnActivationForward` and `cudnnActivationBackward` is the equivalent of
eltwise forward and eltwise backward in oneDNN respectively. There are some
limitations when using Nvidia backend for eltwise primitive:

* cuDNN only supports the following operations - `RELU`, `ELU`, `TANH`,
  `LOGISTIC` and `BRELU`.
* `RELU` is only supported with alpha = 0.
* cuDNN expects `x`, `y` and `dy` as inputs to the backward pass, hence, only
  `RELU` and `BRELU` operations are supported in the backward pass.
  TODO: add `ELU_DST`, `TANH_DST` and `LOGISTIC_DST` support which require `dy`.
* Forward pass supports `f32`, `f16` and `s8` data types. Although blocking is
  not supported for `s8`.
* Backward pass supports `f32` and `f16` data types.

### Inner product

The inner product primitives is an implementation of matrix multiplication plus
bias activation. There are two implementation of inner product in cuDNN backend.

#### Using GEMM

The default backend for inner product is the gemm backend using `cublasGemmEx`
for forward, backward data, and backward weight and `cudnnReduceTensor` for
backward bias. A function called `gemm_consitency_check()`, `dense_check()` is
used to see if the gemm backend can be used for inner product. `reorder_check()`
is used when reorder is required. If none of the above condition are met, it
falls back to the convolution backend. `cudnnActivationForward` operation is
used for eltwise operation and `cudnnAddTensor` is used for bias operation. The
`beta` parameter in gemm is used for the sum scale and `alpha` parameter is used
for the output scale.

#### Using convolution

For the forward direction, this operation can be implemented by using
`cudnnConvolutionBiasActivation` by converting the inner product to `1x1`
convolution. For the backward direction the inner product operation will be
equivalent of `cudnnConvolutionBackwardData`, `cudnnConvolutionBackwardWeights`
and `cudnnConvolutionBackwardBias` when applied. This implementation of inner
product has the following restrictions and performance implications:

* The only blocked layouts are those that are supported in cuDNN - namely that
  the blocking is done on the C dimension, the block size is 4, and only for
  `int8` inference. The additional requirement is that both the input and filter
  must be blocked.
* The `ReLU` and sum are supported as a fused post-op, for other post-op a
  separate call to eltwise primitive is performed. So the limitation for the
  eltwise primitive is applied here.
* Only `mask = 0` case is supported for output scale.
* The restrictions for the convolution primitive are applied here for input and
  filter format. When required, the filter is internally reordered to match the
  convolution restriction.
* For `int8` cuDNN requires both input and output feature maps to be a multiple
  of 4.

### LRN

The local response normalization primitive in the Nvidia backend is implemented
with the `cudnnLRNForward` and `cudnnLRNBackward` functions for forward and
backward propagation respectively.

* `WITHIN` algorithm is not supported.
* There is a difference in the LRN algorithm used in oneDNN and cuDNN which
  causes a mismatch when the local size is even.
* cuDNN supports NCHW tensor formats for all valid dimensions. However, it does
  not support the NHWC tensor format for above 5 dimensions.

### Matrix Multiplication

The matrix multiplication primitive in the Nvidia backend is implemented with
`cublasGemmEx` and `cublasGemmStridedBatchedEx` functions.

* Zero points support is not provided by cuBLAS and, hence, not supported by the
  Nvidia backend.
* Post-ops and output scale limitations are same as for Inner Product.

### Pooling

The pooling primitive in the Nvidia backend is implemented with the
`cudnnPoolingForward` and `cudnnPoolingBackward` functions for forward and
backward propagation respectively.

* cuDNN only allows the use of symmetric padding, i.e. padding at the beginning
  of a dimension must be the same as the padding at the end of that dimension.
  oneDNN doesn't have this limitation. Therefore,

    - Configurations where padding in the beginning is larger than padding at
      the end are supported and work as expected.
    - For configurations where padding at the end is larger than padding in the
      beginning of any dimension, the primitive returns `status::unimplemented`.

* For backward propagation cuDNN requires the parameters `x`, `y`, `dx` and
  `dy`, while oneDNN requires only `dx`, `dy` and workspace when the `MAX`
  algorithm is used. Hence, the workspace is used to store the `x` and `y`
  parameters in the forward pass for the Nvidia backend. Therefore, the
  workspace is always required when the Nvidia backend is used (except for the
  forward inference).

### Reorder

The `cudnnTransform` function is the equivalent of oneDNN reorder function.
However, there are some limitations when using SYCL_API-DNN reorder on Nvidia
GPU:

* Per dimension scaling is not supported (a single alpha and beta value is
  accepted by the transform tensor function).
* Blocking is only permitted for the channel dimension in cuDNN. This primitive
  currently supports block size of 4.
* Blocking is only supported when channel dimension is a multiple of the block
  size and the datatype is `int8`.

### Resampling

The `cudnnSpatialTfSamplerForward` and `cudnnSpatialTfSamplerBackward` are used
to implement the resampling primitive.

The Nvidia's spatial sampling is based on
[Spacial Transformer Network](https://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf)
where all the data locations are normalized between `-1 <= (xi, yi) <= 1`.

* cuDNN backend requires a grid of coordinates that can be sample-up/down based
  on `theta`. The grid is generated by `cudnnSpatialTfGridGeneratorForward`.
* The `theta` is a `MB * 2 * 3` matrix scaling factor for each coordinate and is
  used to generate the grid.
* The grid value must be normalized in range [-1 , 1]. cuDNN clamps the out of
  bounds coordinate to zero. Therefore, it is needed to manually clamp the out
  of bound coordinate to edges in order to avoid incorrect result.
* 3D spatial sampling is not supported in cuDNN.
* `Nearest neighbour` algorithm is not supported in cuDNN.
* Since cuDNN computation is different from that of oneDNN, the error threshold
  is smaller than other oneDNN implementation, so reduced testing accuracy for
  `fp32` and `fp16` data types are required.
* The backward pass requires an output parameter for `d_grid` which cannot be
  `nullptr`. However, since the grid coordinates are not a tunable parameter in
  oneDNN, a dummy memory for `d_grid` is created and is deleted when the
  destructor of the primitive is called.

### Softmax/LogSoftmax

The `cudnnSoftmaxForward` and `cudnnSoftmaxBackward` are used to implement the
softmax primitive. For logsoftmax primitive the same functions will be used and
the algorithm selection in cuDNN for the above mentioned functions will be
changed to `CUDNN_SOFTMAX_LOG`.

* The softmax axis is supported for only the channel dimension, (i.e., axis=1).
* There is a bug in cuDNN softmax for 5D tensor with format `NHWC`. When the
  channel size is greater than 1, it only applies softmax for a single channel
  and leave the others untouched.

### Sum

The sum operation uses the reorder primitive to sum tensors, so the same
limitation as reorder applies here.

### Other primitives

Rest primitives not listed above are not supported by Nvidia backend. This is
likely due to either missed functionality in cuDNN or cuBLAS, or lack of
priority in supporting of such functionality.
