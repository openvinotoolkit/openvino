# Working with devices {#openvino_docs_OV_UG_Working_with_devices}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_query_api
   openvino_docs_OV_UG_supported_plugins_CPU
   openvino_docs_OV_UG_supported_plugins_GPU
   openvino_docs_IE_DG_supported_plugins_VPU
   openvino_docs_OV_UG_supported_plugins_GNA

@endsphinxdirective

The OpenVINO Runtime provides capabilities to infer deep learning models on the following device types with corresponding plugins:

| Plugin                                   | Device types                                                                                                                                                |
|------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
|[CPU plugin](CPU.md)              |Intel&reg; Xeon&reg; with Intel® Advanced Vector Extensions 2 (Intel® AVX2), Intel® Advanced Vector Extensions 512 (Intel® AVX-512), and AVX512_BF16, Intel&reg; Core&trade; Processors with Intel&reg; AVX2, Intel&reg; Atom&reg; Processors with Intel® Streaming SIMD Extensions (Intel® SSE) |
|[GPU plugin](GPU.md)            |Intel® Graphics, including Intel® HD Graphics, Intel® UHD Graphics, Intel® Iris® Graphics, Intel® Xe Graphics, Intel® Xe MAX Graphics |
|[VPU plugins](VPU.md)            |Intel® Neural Compute Stick 2 powered by the Intel® Movidius™ Myriad™ X, Intel® Vision Accelerator Design with Intel® Movidius™ VPUs                                                                                           |
|[GNA plugin](GNA.md)              |Intel&reg; Speech Enabling Developer Kit, Amazon Alexa* Premium Far-Field Developer Kit, Intel&reg; Pentium&reg; Silver J5005 Processor, Intel&reg; Pentium&reg; Silver N5000 Processor, Intel&reg; Celeron&reg; J4005 Processor, Intel&reg; Celeron&reg; J4105 Processor, Intel&reg; Celeron&reg; Processor N4100, Intel&reg; Celeron&reg; Processor N4000, Intel&reg; Core&trade; i3-8121U Processor, Intel&reg; Core&trade; i7-1065G7 Processor, Intel&reg; Core&trade; i7-1060G7 Processor, Intel&reg; Core&trade; i5-1035G4 Processor, Intel&reg; Core&trade; i5-1035G7 Processor, Intel&reg; Core&trade; i5-1035G1 Processor, Intel&reg; Core&trade; i5-1030G7 Processor, Intel&reg; Core&trade; i5-1030G4 Processor, Intel&reg; Core&trade; i3-1005G1 Processor, Intel&reg; Core&trade; i3-1000G1 Processor, Intel&reg; Core&trade; i3-1000G4 Processor|

OpenVINO runtime also has several execution capabilities which work on top of other devices:

| Capability                                   | Description                                                                                                                                                |
|------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
|[Multi-Device execution](../multi_device.md) |Multi-Device enables simultaneous inference of the same model on several devices in parallel    |
|[Auto-Device selection](../auto_device_selection.md) |Auto-Device selection enables selecting Intel&reg; device for inference automatically |
|[Heterogeneous execution](../hetero_execution.md) |Heterogeneous execution enables automatic inference splitting between several devices (for example if a device doesn't [support certain operation](#supported-layers))|
|[Automatic Batching](../automatic_batching.md) | Auto-Batching plugin enables the batching (on top of the specified device)  that is completely transparent to the application |

Devices similar to the ones we have used for benchmarking can be accessed using [Intel® DevCloud for the Edge](https://devcloud.intel.com/edge/), a remote development environment with access to Intel® hardware and the latest versions of the Intel® Distribution of the OpenVINO™ Toolkit. [Learn more](https://devcloud.intel.com/edge/get_started/devcloud/) or [Register here](https://inteliot.force.com/DevcloudForEdge/s/).


## Features support matrix
The table below demonstrates support of key features by OpenVINO device plugins.

| Capability | CPU | [GPU](./GPU.md) | GNA | VPU |
| ---------- | --- | --- | --- | --- |
| [Heterogeneous execution](../hetero_execution.md)| Yes | Yes | ? | ? |
| [Multi-device execution](../multi_device.md) | Yes | Yes | ? | ? |
| [Automatic batching](../automatic_batching.md) | No | Yes | ? | ? |
| [Multi-stream execution](@ref openvino_docs_optimization_guide_dldt_optimization_guide) | Yes | Yes | ? | ? |
| [Models caching](../Model_caching_overview.md) | Yes | Partial | ? | ? |
| [Dynamic shapes](../ov_dynamic_shapes.md) | Yes | Partial | ? | ? |
| Import/Export | Yes | No | ? | ? |
| [Preprocessing acceleration](../preprocessing_overview.md) | Yes | Yes | ? | ? |
| [Stateful models](../stateful_models_intro.md) | Yes | No | ? | ? |
| [Extensibility](@ref openvino_docs_Extensibility_UG_Intro) | Yes | Yes | ? | ? |

For more details on plugin specific feature limitation see corresponding plugin pages.
