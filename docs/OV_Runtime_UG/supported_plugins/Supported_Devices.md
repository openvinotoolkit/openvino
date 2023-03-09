Supported Devices {#openvino_docs_OV_UG_supported_plugins_Supported_Devices}
==================

The OpenVINO Runtime can infer models in different formats with various input and output formats. This section provides supported and optimal configurations per device. In OpenVINO™ documentation, "device" refers to an Intel® processors used for inference, which can be a supported CPU, GPU, or GNA (Gaussian neural accelerator coprocessor), or a combination of those devices.

> **NOTE**: With OpenVINO™ 2023.0 release, support has been cancelled for all VPU accelerators based on Intel® Movidius™.

The OpenVINO Runtime provides unique capabilities to infer deep learning models on the following device types with corresponding plugins:

| Plugin                                   | Device types                                                                                                                                                |
|------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
|[GPU plugin](GPU.md)            |Intel&reg; Processor Graphics, including Intel&reg; HD Graphics and Intel&reg; Iris&reg; Graphics                                                            |
|[CPU plugin](CPU.md)              |Intel&reg; Xeon&reg; with Intel® Advanced Vector Extensions 2 (Intel® AVX2), Intel® Advanced Vector Extensions 512 (Intel® AVX-512), and AVX512_BF16, Intel&reg; Core&trade; Processors with Intel&reg; AVX2, Intel&reg; Atom&reg; Processors with Intel® Streaming SIMD Extensions (Intel® SSE) |
|[GNA plugin](GNA.md) (available in the Intel® Distribution of OpenVINO™ toolkit)              |Intel&reg; Speech Enabling Developer Kit, Amazon Alexa* Premium Far-Field Developer Kit, Intel&reg; Pentium&reg; Silver J5005 Processor, Intel&reg; Pentium&reg; Silver N5000 Processor, Intel&reg; Celeron&reg; J4005 Processor, Intel&reg; Celeron&reg; J4105 Processor, Intel&reg; Celeron&reg; Processor N4100, Intel&reg; Celeron&reg; Processor N4000, Intel&reg; Core&trade; i3-8121U Processor, Intel&reg; Core&trade; i7-1065G7 Processor, Intel&reg; Core&trade; i7-1060G7 Processor, Intel&reg; Core&trade; i5-1035G4 Processor, Intel&reg; Core&trade; i5-1035G7 Processor, Intel&reg; Core&trade; i5-1035G1 Processor, Intel&reg; Core&trade; i5-1030G7 Processor, Intel&reg; Core&trade; i5-1030G4 Processor, Intel&reg; Core&trade; i3-1005G1 Processor, Intel&reg; Core&trade; i3-1000G1 Processor, Intel&reg; Core&trade; i3-1000G4 Processor|
|[Arm® CPU plugin](ARM_CPU.md) (unavailable in the Intel® Distribution of OpenVINO™ toolkit) |Raspberry Pi™ 4 Model B, Apple® Mac mini with M1 chip, NVIDIA® Jetson Nano™, Android™ devices    |
|[Multi-Device execution](../multi_device.md) |Multi-Device execution enables simultaneous inference of the same model on several devices in parallel    |
|[Auto-Device plugin](../auto_device_selection.md) |Auto-Device plugin enables selecting Intel&reg; device for inference automatically |
|[Heterogeneous plugin](../hetero_execution.md) |Heterogeneous execution enables automatic inference splitting between several devices (for example if a device doesn't [support certain operation](#supported-layers)).                                                           |

> **NOTE**: ARM® CPU plugin is a community-level add-on to OpenVINO™. Intel® welcomes community participation in the OpenVINO™ ecosystem, technical questions and code contributions on community forums. However, this component has not undergone full release validation or qualification from Intel®, hence no official support is offered. 

Devices similar to the ones we have used for benchmarking can be accessed using [Intel® DevCloud for the Edge](https://devcloud.intel.com/edge/), a remote development environment with access to Intel® hardware and the latest versions of the Intel® Distribution of the OpenVINO™ Toolkit. [Learn more](https://devcloud.intel.com/edge/get_started/devcloud/) or [Register here](https://inteliot.force.com/DevcloudForEdge/s/).

## Supported Configurations

The OpenVINO Runtime can inference models in different formats with various input and output formats.
This page shows supported and optimal configurations for each plugin.

### Terminology

| Acronym/Term      | Description                                   |
| :-----------------| :---------------------------------------------|
|   FP32 format     | Single-precision floating-point format        |
|   BF16 format     | Brain floating-point format                   |
|   FP16 format     | Half-precision floating-point format          |
|   I16 format      | 2-byte signed integer format                  |
|   I8 format       | 1-byte signed integer format                  |
|   U16 format      | 2-byte unsigned integer format                |
|   U8 format       | 1-byte unsigned integer format                |

NHWC, NCHW, and NCDHW refer to the data ordering in batches of images:
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

|Plugin             |FP32                    |FP16                    |I8                      |
|:------------------|:----------------------:|:----------------------:|:----------------------:|
|CPU plugin         |Supported and preferred |Supported               |Supported               |
|GPU plugin         |Supported               |Supported and preferred |Supported               |
|GNA plugin         |Supported               |Supported               |Not supported           |
|Arm® CPU plugin    |Supported and preferred |Supported               |Supported (partially)   |

For [Multi-Device](../multi_device.md) and [Heterogeneous](../hetero_execution.md) executions
the supported models formats depends on the actual underlying devices. _Generally, FP16 is preferable as it is most ubiquitous and performant_.

### Supported Input Precision

|Plugin             |FP32      |FP16           |U8             |U16            |I8            |I16            |
|:------------------|:--------:|:-------------:|:-------------:|:-------------:|:------------:|:-------------:|
|CPU plugin         |Supported |Supported      |Supported      |Supported      |Supported     |Supported      |
|GPU plugin         |Supported |Supported\*    |Supported\*    |Supported\*    |Not supported |Supported\*    |
|GNA plugin         |Supported |Not supported  |Supported      |Not supported  |Supported     |Supported      |
|Arm® CPU plugin    |Supported |Supported      |Supported      |Supported      |Not supported |Not supported  |

<br>\* - Supported via `SetBlob` only, `GetBlob` returns FP32<br>
For [Multi-Device](../multi_device.md) and [Heterogeneous](../hetero_execution.md) executions
the supported input precision  depends on the actual underlying devices. _Generally, U8 is preferable as it is most ubiquitous_.

### Supported Output Precision

|Plugin             |FP32      |FP16          |
|:------------------|:--------:|:------------:|
|CPU plugin         |Supported |Supported     |
|GPU plugin         |Supported |Supported     |
|GNA plugin         |Supported |Not supported |
|Arm® CPU plugin    |Supported |Supported     |

For [Multi-Device](../multi_device.md) and [Heterogeneous](../hetero_execution.md) executions
the supported output precision  depends on the actual underlying devices. _Generally, FP32 is preferable as it is most ubiquitous_.

### Supported Input Layout

|Plugin             |NCDHW         |NCHW          |NHWC          |NC            |
|:------------------|:------------:|:------------:|:------------:|:------------:|
|CPU plugin         |Supported     |Supported     |Supported     |Supported     |
|GPU plugin         |Supported     |Supported     |Supported     |Supported     |
|GNA plugin         |Not supported |Supported     |Supported     |Supported     |
|Arm® CPU plugin    |Not supported |Supported     |Supported     |Supported     |

### Supported Output Layout

|Number of dimensions|5    |4    |3    |2    |1    |
|:-------------------|:---:|:---:|:---:|:---:|:---:|
|Layout              |NCDHW|NCHW |CHW  |NC   |C    |

For setting relevant configuration, refer to the
[Integrate with Customer Application](../integrate_with_your_application.md) topic
(step 3 "Configure input and output").

### <a name="supported-layers"></a> Supported Layers
The following layers are supported by the plugins:

@sphinxdirective
==============================  ==============  ===============  ==============  ==================
Layers                          GPU             CPU              GNA             Arm® CPU    
==============================  ==============  ===============  ==============  ==================  
Abs                             Supported       Supported\*\*    Not Supported   Supported       
Acos                            Supported       Supported\*\*    Not Supported   Supported\*\*\*\*
Acosh                           Supported       Supported\*\*    Not Supported   Supported\*\*\*\*
Activation-Clamp                Supported       Supported\*\*\*  Supported       Supported       
Activation-ELU                  Supported       Supported\*\*\*  Not Supported   Supported       
Activation-Exp                  Supported       Supported\*\*\*  Supported       Supported       
Activation-Leaky ReLU           Supported       Supported\*\*\*  Supported       Not Supported   
Activation-Not                  Supported       Supported\*\*\*  Not Supported   Not Supported   
Activation-PReLU                Supported       Supported\*\*\*  Not Supported   Supported       
Activation-ReLU                 Supported       Supported\*\*\*  Supported       Supported       
Activation-ReLU6                Supported       Supported\*\*\*  Not Supported   Not Supported   
Activation-Sigmoid/Logistic     Supported       Supported\*\*\*  Supported       Supported       
Activation-TanH                 Supported       Supported\*\*\*  Supported       Supported       
ArgMax                          Supported       Supported\*\*    Not Supported   Not Supported   
Asin                            Supported       Supported\*\*    Not Supported   Supported\*\*\*\*
Asinh                           Supported       Supported\*\*    Not Supported   Supported\*\*\*\*
Atan                            Supported       Supported\*\*    Not Supported   Supported\*\*\*\*
Atanh                           Supported       Supported\*\*    Not Supported   Supported\*\*\*\*
BatchNormalization              Supported       Supported        Not Supported   Supported       
BinaryConvolution               Supported       Supported        Not Supported   Not Supported   
Broadcast                       Supported       Supported\*\*    Not Supported   Supported       
Ceil                            Supported       Supported\*\*    Not Supported   Supported       
Concat                          Supported       Supported\*\*\*  Supported       Supported       
Const                           Supported       Supported        Supported       Supported       
Convolution-Dilated             Supported       Supported        Not Supported   Supported       
Convolution-Dilated 3D          Supported       Supported        Not Supported   Not Supported   
Convolution-Grouped             Supported       Supported        Not Supported   Supported       
Convolution-Grouped 3D          Supported       Supported        Not Supported   Not Supported   
Convolution-Ordinary            Supported       Supported        Supported\*     Supported       
Convolution-Ordinary 3D         Supported       Supported        Not Supported   Not Supported   
Cos                             Supported       Supported\*\*    Not Supported   Supported\*\*\*\*
Cosh                            Supported       Supported\*\*    Not Supported   Supported\*\*\*\*
Crop                            Supported       Supported        Supported       Not Supported   
CTCGreedyDecoder                Supported\*\*   Supported\*\*    Not Supported   Supported\*\*\*\*
Deconvolution                   Supported       Supported        Not Supported   Not Supported   
Deconvolution 3D                Supported       Supported        Not Supported   Not Supported   
DeformableConvolution           Supported       Supported        Not Supported   Not Supported   
DepthToSpace                    Supported       Supported\*\*    Not Supported   Supported\*     
DetectionOutput                 Supported       Supported\*\*    Not Supported   Supported\*\*\*\*
Eltwise-And                     Supported       Supported\*\*\*  Not Supported   Supported       
Eltwise-Add                     Supported       Supported\*\*\*  Not Supported   Supported       
Eltwise-Div                     Supported       Supported\*\*\*  Not Supported   Supported       
Eltwise-Equal                   Supported       Supported\*\*\*  Not Supported   Supported\*     
Eltwise-FloorMod                Supported       Supported\*\*\*  Not Supported   Supported\*\*\*\*
Eltwise-Greater                 Supported       Supported\*\*\*  Not Supported   Supported       
Eltwise-GreaterEqual            Supported       Supported\*\*\*  Not Supported   Supported       
Eltwise-Less                    Supported       Supported\*\*\*  Not Supported   Supported\*     
Eltwise-LessEqual               Supported       Supported\*\*\*  Not Supported   Supported\*     
Eltwise-LogicalAnd              Supported       Supported\*\*\*  Not Supported   Supported       
Eltwise-LogicalOr               Supported       Supported\*\*\*  Not Supported   Supported       
Eltwise-LogicalXor              Supported       Supported\*\*\*  Not Supported   Supported       
Eltwise-Max                     Supported       Supported\*\*\*  Not Supported   Supported       
Eltwise-Min                     Supported       Supported\*\*\*  Not Supported   Supported       
Eltwise-Mul                     Supported       Supported\*\*\*  Supported       Supported       
Eltwise-NotEqual                Supported       Supported\*\*\*  Not Supported   Supported\*     
Eltwise-Pow                     Supported       Supported\*\*\*  Not Supported   Supported       
Eltwise-Prod                    Supported       Supported\*\*\*  Supported       Not Supported   
Eltwise-SquaredDiff             Supported       Supported\*\*\*  Not Supported   Supported       
Eltwise-Sub                     Supported       Supported\*\*\*  Supported       Supported       
Eltwise-Sum                     Supported       Supported\*\*\*  Supported       Supported\*\*\*\*
Erf                             Supported       Supported\*\*    Not Supported   Supported\*\*\*\*
Exp                             Supported       Supported        Supported       Supported       
FakeQuantize                    Not Supported   Supported        Not Supported   Supported\*     
Fill                            Not Supported   Supported\*\*    Not Supported   Not Supported   
Flatten                         Supported       Supported        Not Supported   Not Supported   
Floor                           Supported       Supported\*\*    Not Supported   Supported       
FullyConnected (Inner Product)  Supported       Supported\*\*\*  Supported       Supported       
Gather                          Supported       Supported\*\*    Not Supported   Supported\*     
GatherTree                      Not Supported   Supported\*\*    Not Supported   Supported\*\*\*\*
Gemm                            Supported       Supported        Not Supported   Not Supported   
GRN                             Supported\*\*   Supported\*\*    Not Supported   Supported       
HardSigmoid                     Supported       Supported\*\*    Not Supported   Supported\*\*\*\*
Interp                          Supported\*\*   Supported\*\*    Not Supported   Supported\*     
Log                             Supported       Supported\*\*    Supported       Supported       
LRN (Norm)                      Supported       Supported        Not Supported   Supported\*     
LSTMCell                        Supported       Supported        Supported       Supported       
GRUCell                         Supported       Supported        Supported       Supported       
RNNCell                         Supported       Supported        Not Supported   Supported       
LSTMSequence                    Supported       Supported        Supported       Supported\*\*\*\*
GRUSequence                     Supported       Supported        Supported       Supported\*\*\*\*
RNNSequence                     Supported       Supported        Not Supported   Supported\*\*\*\*
LogSoftmax                      Supported       Supported\*\*    Not Supported   Supported       
Memory                          Not Supported   Supported        Supported       Not Supported   
MVN                             Supported       Supported\*\*    Not Supported   Supported\*     
Neg                             Supported       Supported\*\*    Not Supported   Supported       
NonMaxSuppression               Not Supported   Supported\*\*    Not Supported   Supported\*\*\*\*
Normalize                       Supported       Supported\*\*    Not Supported   Supported\*     
OneHot                          Supported       Supported\*\*    Not Supported   Supported\*\*\*\*
Pad                             Supported       Supported\*\*    Not Supported   Supported\*     
Permute                         Supported       Supported        Supported\*     Not Supported   
Pooling(AVG,MAX)                Supported       Supported        Supported       Supported       
Pooling(AVG,MAX) 3D             Supported       Supported        Not Supported   Supported\*     
Power                           Supported       Supported\*\*    Supported\*     Supported       
PowerFile                       Not Supported   Supported\*\*    Not Supported   Not Supported   
PriorBox                        Supported       Supported\*\*    Not Supported   Supported       
PriorBoxClustered               Supported\*\*   Supported\*\*    Not Supported   Supported       
Proposal                        Supported       Supported\*\*    Not Supported   Supported\*\*\*\*
PSROIPooling                    Supported       Supported\*\*    Not Supported   Supported\*\*\*\*
Range                           Not Supported   Supported\*\*    Not Supported   Not Supported   
Reciprocal                      Supported       Supported\*\*    Not Supported   Not Supported   
ReduceAnd                       Supported       Supported\*\*    Not Supported   Supported\*\*\*\*
ReduceL1                        Supported       Supported\*\*    Not Supported   Supported       
ReduceL2                        Supported       Supported\*\*    Not Supported   Supported       
ReduceLogSum                    Supported       Supported\*\*    Not Supported   Supported       
ReduceLogSumExp                 Supported       Supported\*\*    Not Supported   Not Supported   
ReduceMax                       Supported       Supported\*\*    Not Supported   Supported       
ReduceMean                      Supported       Supported\*\*    Not Supported   Supported       
ReduceMin                       Supported       Supported\*\*    Not Supported   Supported       
ReduceOr                        Supported       Supported\*\*    Not Supported   Supported\*\*\*\*
ReduceProd                      Supported       Supported\*\*    Not Supported   Supported       
ReduceSum                       Supported       Supported\*\*    Not Supported   Supported       
ReduceSumSquare                 Supported       Supported\*\*    Not Supported   Not Supported   
RegionYolo                      Supported       Supported\*\*    Not Supported   Supported\*\*\*\*
ReorgYolo                       Supported       Supported\*\*    Not Supported   Supported       
Resample                        Supported       Supported\*\*    Not Supported   Not Supported   
Reshape                         Supported       Supported\*\*\*  Supported       Supported       
ReverseSequence                 Supported       Supported\*\*    Not Supported   Supported\*\*\*\*
RNN                             Not Supported   Supported        Not Supported   Supported       
ROIPooling                      Supported\*     Supported        Not Supported   Supported\*\*\*\*
ScaleShift                      Supported       Supported\*\*\*  Supported       Not Supported   
ScatterUpdate                   Not Supported   Supported\*\*    Not Supported   Not Supported   
Select                          Supported       Supported        Not Supported   Supported       
Selu                            Supported       Supported\*\*    Not Supported   Supported\*\*\*\*
ShuffleChannels                 Supported       Supported\*\*    Not Supported   Supported       
Sign                            Supported       Supported\*\*    Not Supported   Supported       
Sin                             Supported       Supported\*\*    Not Supported   Supported       
Sinh                            Supported       Supported\*\*    Not Supported   Supported\*\*\*\*
SimplerNMS                      Supported       Supported\*\*    Not Supported   Not Supported   
Slice                           Supported       Supported\*\*\*  Supported       Not Supported   
SoftMax                         Supported       Supported\*\*\*  Not Supported   Supported       
Softplus                        Supported       Supported\*\*    Not Supported   Supported       
Softsign                        Supported       Supported\*\*    Supported       Not Supported   
SpaceToDepth                    Not Supported   Supported\*\*    Not Supported   Supported\*     
SpatialTransformer              Not Supported   Supported\*\*    Not Supported   Not Supported   
Split                           Supported       Supported\*\*\*  Supported       Supported       
Squeeze                         Supported       Supported\*\*    Supported       Supported       
StridedSlice                    Supported       Supported\*\*    Not Supported   Supported\*     
Tan                             Supported       Supported\*\*    Not Supported   Supported\*\*\*\*
TensorIterator                  Not Supported   Supported        Supported       Supported       
Tile                            Supported\*\*   Supported\*\*\*  Not Supported   Supported       
TopK                            Supported       Supported\*\*    Not Supported   Supported\*\*\*\*
Unpooling                       Supported       Not Supported    Not Supported   Not Supported   
Unsqueeze                       Supported       Supported\*\*    Supported       Supported       
Upsampling                      Supported       Not Supported    Not Supported   Not Supported   
==============================  ==============  ===============  ==============  ==================
@endsphinxdirective


\*- support is limited to the specific parameters. Refer to "Known Layers Limitation" section for the device [from the list of supported](Supported_Devices.md).

\*\*- support is implemented via [Extensibility mechanism](../../Extensibility_UG/Intro.md).

\*\*\*- supports NCDHW layout.

\*\*\*\*- support is implemented via runtime reference.
