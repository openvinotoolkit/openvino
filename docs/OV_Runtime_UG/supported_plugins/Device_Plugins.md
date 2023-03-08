# Inference Device Support {#openvino_docs_OV_UG_Working_with_devices}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_query_api
   openvino_docs_OV_UG_supported_plugins_CPU
   openvino_docs_OV_UG_supported_plugins_GPU
   openvino_docs_OV_UG_supported_plugins_GNA
   openvino_docs_OV_UG_supported_plugins_ARM_CPU

@endsphinxdirective

OpenVINO™ Runtime can infer deep learning models using the following device types:

* [CPU](CPU.md)    
* [GPU](GPU.md)    
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

## Enumerating Available Devices
The OpenVINO Runtime API features dedicated methods of enumerating devices and their capabilities. See the [Hello Query Device C++ Sample](../../../samples/cpp/hello_query_device/README.md). This is an example output from the sample (truncated to device names only):

```sh
  ./hello_query_device
  Available devices:
      Device: CPU
  ...
      Device: GPU.0
  ...
      Device: GPU.1
  ...
      Device: GNA
```

A simple programmatic way to enumerate the devices and use with the multi-device is as follows:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/MULTI2.cpp
       :language: cpp
       :fragment: [part2]

@endsphinxdirective

Beyond the typical "CPU", "GPU", and so on, when multiple instances of a device are available, the names are more qualified. For example, this is how two GPUs can be listed (iGPU is always GPU.0):

```
...
    Device: GPU.0
...
    Device: GPU.1
```

So, the explicit configuration to use both would be "MULTI:GPU.1,GPU.0". Accordingly, the code that loops over all available devices of the "GPU" type only is as follows:

@sphinxdirective

.. tab:: C++

    .. doxygensnippet:: docs/snippets/MULTI3.cpp
       :language: cpp
       :fragment: [part3]

@endsphinxdirective



