# Device Plugin Support {#openvino_docs_IE_DG_Device_Plugins}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_IE_DG_InferenceEngine_QueryAPI
   openvino_docs_IE_DG_supported_plugins_CPU
   openvino_docs_IE_DG_supported_plugins_GPU
   openvino_docs_IE_DG_supported_plugins_VPU
   openvino_docs_IE_DG_supported_plugins_GNA
   openvino_docs_IE_DG_supported_plugins_AUTO
   openvino_docs_IE_DG_supported_plugins_HETERO
   openvino_docs_IE_DG_supported_plugins_MULTI
         
@endsphinxdirective

Inference Engine uses a plugin architecture. Inference Engine plugin is a software component that contains complete implementation for inference on a certain Intel® hardware device: CPU, GPU, VPU, GNA, etc. Each plugin implements the unified API and provides additional hardware-specific APIs.

The Inference Engine provides capabilities to infer deep learning models on the following device types with corresponding plugins:

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

