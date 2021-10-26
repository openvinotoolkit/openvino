Supported Devices {#openvino_docs_IE_DG_supported_plugins_Supported_Devices}
==================

The Inference Engine can infer models in different formats with various input and output formats. This section provides supported and optimal configurations per device.

> **NOTE**: With OpenVINO™ 2020.4 release, Intel® Movidius™ Neural Compute Stick is no longer supported.

The Inference Engine provides unique capabilities to infer deep learning models on the following device types with corresponding plugins:

| Plugin                                   | Device types                                                                                                                                                |
|------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
|[GPU plugin](GPU.md)            |Intel&reg; Processor Graphics, including Intel&reg; HD Graphics and Intel&reg; Iris&reg; Graphics                                                            |
|[CPU plugin](CPU.md)              |Intel&reg; Xeon&reg; with Intel® Advanced Vector Extensions 2 (Intel® AVX2), Intel® Advanced Vector Extensions 512 (Intel® AVX-512), and AVX512_BF16, Intel&reg; Core&trade; Processors with Intel&reg; AVX2, Intel&reg; Atom&reg; Processors with Intel® Streaming SIMD Extensions (Intel® SSE) |
|[VPU plugins](VPU.md) (available in the Intel® Distribution of OpenVINO™ toolkit)            |Intel® Neural Compute Stick 2 powered by the Intel® Movidius™ Myriad™ X, Intel® Vision Accelerator Design with Intel® Movidius™ VPUs                                                                                           |
|[GNA plugin](GNA.md) (available in the Intel® Distribution of OpenVINO™ toolkit)              |Intel&reg; Speech Enabling Developer Kit, Amazon Alexa* Premium Far-Field Developer Kit, Intel&reg; Pentium&reg; Silver J5005 Processor, Intel&reg; Pentium&reg; Silver N5000 Processor, Intel&reg; Celeron&reg; J4005 Processor, Intel&reg; Celeron&reg; J4105 Processor, Intel&reg; Celeron&reg; Processor N4100, Intel&reg; Celeron&reg; Processor N4000, Intel&reg; Core&trade; i3-8121U Processor, Intel&reg; Core&trade; i7-1065G7 Processor, Intel&reg; Core&trade; i7-1060G7 Processor, Intel&reg; Core&trade; i5-1035G4 Processor, Intel&reg; Core&trade; i5-1035G7 Processor, Intel&reg; Core&trade; i5-1035G1 Processor, Intel&reg; Core&trade; i5-1030G7 Processor, Intel&reg; Core&trade; i5-1030G4 Processor, Intel&reg; Core&trade; i3-1005G1 Processor, Intel&reg; Core&trade; i3-1000G1 Processor, Intel&reg; Core&trade; i3-1000G4 Processor|
|[Multi-Device plugin](MULTI.md) |Multi-Device plugin enables simultaneous inference of the same network on several Intel&reg; devices in parallel    |   
|[Auto-Device plugin](AUTO.md) |Auto-Device plugin enables selecting Intel&reg; device for inference automatically |   
|[Heterogeneous plugin](HETERO.md) |Heterogeneous plugin enables automatic inference splitting between several Intel&reg; devices (for example if a device doesn't [support certain layers](#supported-layers)).                                                           |

Devices similar to the ones we have used for benchmarking can be accessed using [Intel® DevCloud for the Edge](https://devcloud.intel.com/edge/), a remote development environment with access to Intel® hardware and the latest versions of the Intel® Distribution of the OpenVINO™ Toolkit. [Learn more](https://devcloud.intel.com/edge/get_started/devcloud/) or [Register here](https://inteliot.force.com/DevcloudForEdge/s/).

## Supported Configurations

The Inference Engine can inference models in different formats with various input and output formats.
This page shows supported and optimal configurations for each plugin.

### Terminology

| Acronym/Term      | Description                                   |
| :-----------------| :---------------------------------------------|
|   DL              | Deep Learning                                 |
|   FP32 format     | Single-precision floating-point format        |
|   BF16 format     | Brain floating-point format                   |
|   FP16 format     | Half-precision floating-point format          |
|   I16 format      | 2-byte signed integer format                  |
|   I8 format       | 1-byte signed integer format                  |
|   U16 format      | 2-byte unsigned integer format                |
|   U8 format       | 1-byte unsigned integer format                |

NHWC, NCHW, and NCDHW refer to the representation of batches of images.
* NHWC and NCHW refer to image data layout.
* NCDHW refers to image sequence data layout.

Abbreviations in the support tables are as follows:
* N: Number of images in a batch
* D: Depth. Depend on model it could be spatial or time dimension
* H: Number of pixels in the vertical dimension
* W: Number of pixels in the horizontal dimension
* C: Number of channels

CHW, NC, C  - Tensor memory layout.
For example, the CHW value at index (c,h,w) is physically located at index (c\*H+h)\*W+w, for others by analogy.

### Supported Model Formats

|Plugin        |FP32                    |FP16                    |I8                      |
|:-------------|:----------------------:|:----------------------:|:----------------------:|
|CPU plugin    |Supported and preferred |Supported               |Supported               |
|GPU plugin    |Supported               |Supported and preferred |Supported\*             |
|VPU plugins   |Not supported           |Supported               |Not supported           |
|GNA plugin    |Supported               |Supported               |Not supported           |
<br>\* - currently, only limited set of topologies might benefit from enabling I8 model on GPU<br>
For [Multi-Device](MULTI.md) and [Heterogeneous](HETERO.md) execution
the supported models formats depends on the actual underlying devices. _Generally, FP16 is preferable as it is most ubiquitous and performant_.

### Supported Input Precision

|Plugin        |FP32      |FP16           |U8             |U16            |I8            |I16            |
|:-------------|:--------:|:-------------:|:-------------:|:-------------:|:------------:|:-------------:|
|CPU plugin    |Supported |Not supported  |Supported      |Supported      |Not supported |Supported      |
|GPU plugin    |Supported |Supported\*    |Supported\*    |Supported\*    |Not supported |Supported\*    |
|VPU plugins   |Supported |Supported      |Supported      |Not supported  |Not supported |Not supported  |
|GNA plugin    |Supported |Not supported  |Supported      |Not supported  |Supported     |Supported      |

<br>\* - Supported via `SetBlob` only, `GetBlob` returns FP32<br>
For [Multi-Device](MULTI.md) and [Heterogeneous](HETERO.md) execution
the supported input precision  depends on the actual underlying devices. _Generally, U8 is preferable as it is most ubiquitous_.

### Supported Output Precision

|Plugin        |FP32      |FP16          |
|:-------------|:--------:|:------------:|
|CPU plugin    |Supported |Not supported |
|GPU plugin    |Supported |Supported     |
|VPU plugins   |Supported |Supported     |
|GNA plugin    |Supported |Not supported |
For [Multi-Device](MULTI.md) and [Heterogeneous](HETERO.md) execution
the supported output precision  depends on the actual underlying devices. _Generally, FP32 is preferable as it is most ubiquitous_.

### Supported Input Layout

|Plugin        |NCDHW         |NCHW          |NHWC          |NC            |
|:-------------|:------------:|:------------:|:------------:|:------------:|
|CPU plugin    |Supported     |Supported     |Supported     |Supported     |
|GPU plugin    |Supported     |Supported     |Supported     |Supported     |
|VPU plugins   |Supported     |Supported     |Supported     |Supported     |
|GNA plugin    |Not supported |Supported     |Supported     |Supported     |

### Supported Output Layout

|Number of dimensions|5    |4    |3    |2    |1    |
|:-------------------|:---:|:---:|:---:|:---:|:---:|
|Layout              |NCDHW|NCHW |CHW  |NC   |C    |

For setting relevant configuration, refer to the
[Integrate with Customer Application New Request API](../Integrate_with_customer_application_new_API.md) topic
(step 3 "Configure input and output").

### Supported Layers
The following layers are supported by the plugins and by [Shape Inference feature](../ShapeInference.md):

| Layers                         | GPU           | CPU           | VPU           | GNA           | ShapeInfer    |
|:-------------------------------|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Abs                            | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| Acos                           | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| Acosh                          | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| Activation-Clamp               | Supported     |Supported\*\*\*| Supported     | Supported     | Supported     |
| Activation-ELU                 | Supported     |Supported\*\*\*| Supported     | Not Supported | Supported     |
| Activation-Exp                 | Supported     |Supported\*\*\*| Supported     | Supported     | Supported     |
| Activation-Leaky ReLU          | Supported     |Supported\*\*\*| Supported     | Supported     | Supported     |
| Activation-Not                 | Supported     |Supported\*\*\*| Supported     | Not Supported | Supported     |
| Activation-PReLU               | Supported     |Supported\*\*\*| Supported     | Not Supported | Supported     |
| Activation-ReLU                | Supported     |Supported\*\*\*| Supported     | Supported     | Supported     |
| Activation-ReLU6               | Supported     |Supported\*\*\*| Supported     | Not Supported | Supported     |
| Activation-Sigmoid/Logistic    | Supported     |Supported\*\*\*| Supported     | Supported     | Supported     |
| Activation-TanH                | Supported     |Supported\*\*\*| Supported     | Supported     | Supported     |
| ArgMax                         | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| Asin                           | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| Asinh                          | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| Atan                           | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| Atanh                          | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| BatchNormalization             | Supported     | Supported     | Supported     | Not Supported | Supported     |
| BinaryConvolution              | Supported     | Supported     | Not Supported | Not Supported | Supported     |
| Broadcast                      | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| Ceil                           | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| Concat                         | Supported     |Supported\*\*\*| Supported     | Supported     | Supported     |
| Const                          | Supported     | Supported     | Supported     | Supported     | Not Supported |
| Convolution-Dilated            | Supported     | Supported     | Supported     | Not Supported | Supported     |
| Convolution-Dilated 3D         | Supported     | Supported     | Not Supported | Not Supported | Not Supported |
| Convolution-Grouped            | Supported     | Supported     | Supported     | Not Supported | Supported     |
| Convolution-Grouped 3D         | Supported     | Supported     | Not Supported | Not Supported | Not Supported |
| Convolution-Ordinary           | Supported     | Supported     | Supported     | Supported\*   | Supported     |
| Convolution-Ordinary 3D        | Supported     | Supported     | Not Supported | Not Supported | Not Supported |
| Cos                            | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| Cosh                           | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| Crop                           | Supported     | Supported     | Supported     | Supported     | Supported     |
| CTCGreedyDecoder               | Supported\*\* | Supported\*\* | Supported\*   | Not Supported | Supported     |
| Deconvolution                  | Supported     | Supported     | Supported     | Not Supported | Supported     |
| Deconvolution 3D               | Supported     | Supported     | Not Supported | Not Supported | Not Supported |
| DeformableConvolution          | Supported     | Supported     | Not Supported | Not Supported | Supported     |
| DepthToSpace                   | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| DetectionOutput                | Supported     | Supported\*\* | Supported\*   | Not Supported | Supported     |
| Eltwise-And                    | Supported     |Supported\*\*\*| Supported     | Not Supported | Supported     |
| Eltwise-Add                    | Supported     |Supported\*\*\*| Supported     | Not Supported | Supported     |
| Eltwise-Div                    | Supported     |Supported\*\*\*| Supported     | Not Supported | Supported     |
| Eltwise-Equal                  | Supported     |Supported\*\*\*| Supported     | Not Supported | Supported     |
| Eltwise-FloorMod               | Supported     |Supported\*\*\*| Supported     | Not Supported | Supported     |
| Eltwise-Greater                | Supported     |Supported\*\*\*| Supported     | Not Supported | Supported     |
| Eltwise-GreaterEqual           | Supported     |Supported\*\*\*| Supported     | Not Supported | Supported     |
| Eltwise-Less                   | Supported     |Supported\*\*\*| Supported     | Not Supported | Supported     |
| Eltwise-LessEqual              | Supported     |Supported\*\*\*| Supported     | Not Supported | Supported     |
| Eltwise-LogicalAnd             | Supported     |Supported\*\*\*| Supported     | Not Supported | Supported     |
| Eltwise-LogicalOr              | Supported     |Supported\*\*\*| Supported     | Not Supported | Supported     |
| Eltwise-LogicalXor             | Supported     |Supported\*\*\*| Supported     | Not Supported | Supported     |
| Eltwise-Max                    | Supported     |Supported\*\*\*| Supported     | Not Supported | Supported     |
| Eltwise-Min                    | Supported     |Supported\*\*\*| Supported     | Not Supported | Supported     |
| Eltwise-Mul                    | Supported     |Supported\*\*\*| Supported     | Supported     | Supported     |
| Eltwise-NotEqual               | Supported     |Supported\*\*\*| Supported     | Not Supported | Supported     |
| Eltwise-Pow                    | Supported     |Supported\*\*\*| Supported     | Not Supported | Supported     |
| Eltwise-Prod                   | Supported     |Supported\*\*\*| Supported     | Supported     | Supported     |
| Eltwise-SquaredDiff            | Supported     |Supported\*\*\*| Supported     | Not Supported | Supported     |
| Eltwise-Sub                    | Supported     |Supported\*\*\*| Supported     | Supported     | Supported     |
| Eltwise-Sum                    | Supported     |Supported\*\*\*| Supported     | Supported     | Supported     |
| Erf                            | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| Exp                            | Supported     | Supported     | Supported     | Supported     | Supported     |
| FakeQuantize                   | Not Supported | Supported     | Not Supported | Not Supported | Supported     |
| Fill                           | Not Supported | Supported\*\* | Not Supported | Not Supported | Supported     |
| Flatten                        | Supported     | Supported     | Supported     | Not Supported | Supported     |
| Floor                          | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| FullyConnected (Inner Product) | Supported     |Supported\*\*\*| Supported     | Supported     | Supported     |
| Gather                         | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| GatherTree                     | Not Supported | Supported\*\* | Not Supported | Not Supported | Supported     |
| Gemm                           | Supported     | Supported     | Supported     | Not Supported | Supported     |
| GRN                            | Supported\*\* | Supported\*\* | Supported     | Not Supported | Supported     |
| HardSigmoid                    | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| Interp                         | Supported\*\* | Supported\*\* | Supported     | Not Supported | Supported\*   |
| Log                            | Supported     | Supported\*\* | Supported     | Supported     | Supported     |
| LRN (Norm)                     | Supported     | Supported     | Supported     | Not Supported | Supported     |
| LSTMCell                       | Supported     | Supported     | Supported     | Supported     | Not Supported |
| GRUCell                        | Supported     | Supported     | Not Supported | Not Supported | Not Supported |
| RNNCell                        | Supported     | Supported     | Not Supported | Not Supported | Not Supported |
| LSTMSequence                   | Supported     | Supported     | Supported     | Not Supported | Not Supported |
| GRUSequence                    | Supported     | Supported     | Not Supported | Not Supported | Not Supported |
| RNNSequence                    | Supported     | Supported     | Not Supported | Not Supported | Not Supported |
| LogSoftmax                     | Supported     | Supported\*\* | Not Supported | Not Supported | Not Supported |
| Memory                         | Not Supported | Supported     | Not Supported | Supported     | Supported     |
| MVN                            | Supported     | Supported\*\* | Supported\*   | Not Supported | Supported     |
| Neg                            | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| NonMaxSuppression              | Not Supported | Supported\*\* | Supported     | Not Supported | Supported     |
| Normalize                      | Supported     | Supported\*\* | Supported\*   | Not Supported | Supported     |
| OneHot                         | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| Pad                            | Supported     | Supported\*\* | Supported\*   | Not Supported | Supported     |
| Permute                        | Supported     | Supported     | Supported     | Supported\*   | Supported     |
| Pooling(AVG,MAX)               | Supported     | Supported     | Supported     | Supported     | Supported     |
| Pooling(AVG,MAX) 3D            | Supported     | Supported     | Not Supported | Not Supported | Not Supported |
| Power                          | Supported     | Supported\*\* | Supported     | Supported\*   | Supported     |
| PowerFile                      | Not Supported | Supported\*\* | Not Supported | Not Supported | Not Supported |
| PriorBox                       | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| PriorBoxClustered              | Supported\*\* | Supported\*\* | Supported     | Not Supported | Supported     |
| Proposal                       | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| PSROIPooling                   | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| Range                          | Not Supported | Supported\*\* | Not Supported | Not Supported | Supported     |
| Reciprocal                     | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| ReduceAnd                      | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| ReduceL1                       | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| ReduceL2                       | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| ReduceLogSum                   | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| ReduceLogSumExp                | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| ReduceMax                      | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| ReduceMean                     | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| ReduceMin                      | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| ReduceOr                       | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| ReduceProd                     | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| ReduceSum                      | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| ReduceSumSquare                | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| RegionYolo                     | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| ReorgYolo                      | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| Resample                       | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| Reshape                        | Supported     |Supported\*\*\*| Supported     | Supported     | Supported\*   |
| ReverseSequence                | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| RNN                            | Not Supported | Supported     | Supported     | Not Supported | Not Supported |
| ROIPooling                     | Supported\*   | Supported     | Supported     | Not Supported | Supported     |
| ScaleShift                     | Supported     |Supported\*\*\*| Supported\*   | Supported     | Supported     |
| ScatterUpdate                  | Not Supported | Supported\*\* | Supported     | Not Supported | Supported     |
| Select                         | Supported     | Supported     | Supported     | Not Supported | Supported     |
| Selu                           | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| ShuffleChannels                | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| Sign                           | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| Sin                            | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| Sinh                           | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| SimplerNMS                     | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| Slice                          | Supported     |Supported\*\*\*| Supported     | Supported     | Supported     |
| SoftMax                        | Supported     |Supported\*\*\*| Supported     | Not Supported | Supported     |
| Softplus                       | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| Softsign                       | Supported     | Supported\*\* | Not Supported | Supported     | Supported     |
| SpaceToDepth                   | Not Supported | Supported\*\* | Not Supported | Not Supported | Supported     |
| SpatialTransformer             | Not Supported | Supported\*\* | Not Supported | Not Supported | Supported     |
| Split                          | Supported     |Supported\*\*\*| Supported     | Supported     | Supported     |
| Squeeze                        | Supported     | Supported\*\* | Supported     | Supported     | Supported     |
| StridedSlice                   | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| Tan                            | Supported     | Supported\*\* | Not Supported | Not Supported | Supported     |
| TensorIterator                 | Not Supported | Supported     | Supported     | Supported     | Not Supported |
| Tile                           | Supported\*\* |Supported\*\*\*| Supported     | Not Supported | Supported     |
| TopK                           | Supported     | Supported\*\* | Supported     | Not Supported | Supported     |
| Unpooling                      | Supported     | Not Supported | Not Supported | Not Supported | Not Supported |
| Unsqueeze                      | Supported     | Supported\*\* | Supported     | Supported     | Supported     |
| Upsampling                     | Supported     | Not Supported | Not Supported | Not Supported | Not Supported |

\*- support is limited to the specific parameters. Refer to "Known Layers Limitation" section for the device [from the list of supported](Supported_Devices.md).

\*\*- support is implemented via [Extensibility mechanism](../Extensibility_DG/Intro.md).

\*\*\*- supports NCDHW layout.
