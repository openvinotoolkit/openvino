# Inference Device Support {#openvino_docs_OV_UG_Working_with_devices}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_query_api
   openvino_docs_OV_UG_supported_plugins_CPU
   openvino_docs_OV_UG_supported_plugins_GPU
   openvino_docs_OV_UG_supported_plugins_VPU
   openvino_docs_OV_UG_supported_plugins_GNA
   openvino_docs_OV_UG_supported_plugins_ARM_CPU

@endsphinxdirective

OpenVINO™ Runtime can infer deep learning models using the following device types:

* [CPU](CPU.md)    
* [GPU](GPU.md)    
* [VPUs](VPU.md)   
* [GNA](GNA.md)   
* [Arm® CPU](ARM_CPU.md)     

For a more detailed list of hardware, see [Supported Devices](./Supported_Devices.md)

Devices similar to the ones used for benchmarking can be accessed, using [Intel® DevCloud for the Edge](https://devcloud.intel.com/edge/), a remote development environment with access to Intel® hardware and the latest versions of the Intel® Distribution of the OpenVINO™ Toolkit. [Learn more](https://devcloud.intel.com/edge/get_started/devcloud/) or [Register here](https://inteliot.force.com/DevcloudForEdge/s/).


@anchor features_support_matrix
## Feature Support Matrix
The table below demonstrates support of key features by OpenVINO device plugins.

| Capability | [CPU](CPU.md) | [GPU](GPU.md) | [GNA](GNA.md) |[Arm® CPU](ARM_CPU.md) |
| ---------- | --- | --- | --- | --- |
| [Heterogeneous execution](../hetero_execution.md)| Yes | Yes | No | Yes |
| [Multi-device execution](../multi_device.md) | Yes | Yes | Partial | Yes |
| [Automatic batching](../automatic_batching.md) | No | Yes | No | No |
| [Multi-stream execution](../../optimization_guide/dldt_deployment_optimization_tput.md) | Yes | Yes | No | Yes |
| [Models caching](../Model_caching_overview.md) | Yes | Partial | Yes | No |
| [Dynamic shapes](../ov_dynamic_shapes.md) | Yes | Partial | No | No |
| [Import/Export](../../../tools/compile_tool/README.md) | Yes | No | Yes | No |
| [Preprocessing acceleration](../preprocessing_overview.md) | Yes | Yes | No | Partial |
| [Stateful models](../network_state_intro.md) | Yes | No | Yes | No |
| [Extensibility](@ref openvino_docs_Extensibility_UG_Intro) | Yes | Yes | No | No |

For more details on plugin-specific feature limitations, see the corresponding plugin pages.

## Querying Available Devices and Properties
The OpenVINO Runtime API features dedicated methods for querying devices and their capabilities. For information on how to query available devices and check their properties, see the [Query Device Properties](config_properties.md) page.

The [Hello Query Device C++](../../../samples/cpp/hello_query_device/README.md) and [Hello Query Device Python](../../../samples/python/hello_query_device/README.md) samples provide example code showing how to query devices. 


