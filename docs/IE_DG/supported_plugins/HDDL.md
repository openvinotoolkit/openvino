# HDDL Plugin {#openvino_docs_IE_DG_supported_plugins_HDDL}

## Introducing HDDL Plugin

The Inference Engine HDDL plugin is developed for inference of neural networks on Intel&reg; Vision Accelerator Design with Intel&reg; Movidius&trade; VPUs which is designed for use cases those require large throughput of deep learning inference. It provides dozens amount of throughput as MYRIAD Plugin.

## Installation on Linux* OS

For installation instructions, refer to the [Installation Guide for Linux\*](VPU.md).

## Installation on Windows* OS

For installation instructions, refer to the [Installation Guide for Windows\*](Supported_Devices.md).

## Supported networks

For the "Supported Networks", please reference to [MYRIAD Plugin](MYRIAD.md)

## Supported Configuration Parameters

See VPU common configuration parameters for the [VPU Plugins](VPU.md).
When specifying key values as raw strings (that is, when using Python API), omit the `KEY_` prefix.

In addition to common parameters for Myriad plugin and HDDL plugin, HDDL plugin accepts the following options:

| Parameter Name                        | Parameter Values | Default      | Description                                                                     |
| :---                                  | :---             | :---         | :---                                                                            |
| KEY_PERF_COUNT                        | YES/NO           | NO           | Enable performance counter option.                                               |
| KEY_VPU_HDDL_GRAPH_TAG                | string           | empty string | Allows to execute network on specified count of devices.                        |
| KEY_VPU_HDDL_STREAM_ID                | string           | empty string | Allows to execute inference on a specified device.                              |
| KEY_VPU_HDDL_DEVICE_TAG               | string           | empty string | Allows to allocate/deallocate networks on specified devices.                    |
| KEY_VPU_HDDL_BIND_DEVICE              | YES/NO           | NO           | Whether the network should bind to a device. Refer to vpu_plugin_config.hpp.    |
| KEY_VPU_HDDL_RUNTIME_PRIORITY         | signed int       | 0            | Specify the runtime priority of a device among all devices that running a same network Refer to vpu_plugin_config.hpp. |

## See Also

* [Supported Devices](Supported_Devices.md)
* [VPU Plugins](VPU.md)
* [MYRIAD Plugin](MYRIAD.md)
