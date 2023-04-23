# AUTO Plugin Architecture

This guide encompasses existed architectural ideas and guidelines for AUTO Plugin development.

## Main Architectual Design Concepts

AUTO plugin is meta plugin in OpenVINO that doesn’t bind to a specific type of hardware device. When a user choose AUTO as target device, AUTO plugin automatically discovers the accelerators and hardware features of the platform, and selects one or more suitable devices for the task based on application configuration.

The logic behind the choice is as follows:
* Check what supported devices are available.
* Check performance hint of input setting (For detailed information of performance hint, please read more on the [ov::hint::PerformanceMode](https://docs.openvino.ai/latest/openvino_docs_OV_UG_Performance_Hints.html)).
* Check precisions of the input model (for detailed information on precisions read more on the [ov::device::capabilities](https://docs.openvino.ai/latest/namespaceov_1_1device_1_1capability.html)).
* Select the highest-priority device capable of supporting the given model for LATENCY hint and THROUGHPUT hint. Or Select all devices capable of supporting the given model for CUMULATIVE THROUGHPUT hint.
* If model’s precision is FP32 but there is no device capable of supporting it, offload the model to a device supporting FP16.

The AUTO plugin is also default plugin for OpenVINO when user has not selected any device in the application.

## Specific Features of AUTO plugin

### Accelerating First Inference Latency (FIL)

Compiling the model to accelerator-optimized kernels may take some time. Then AUTO starts inference with the CPU of the system by default, as it provides very low latency and can start inference with no additional delays. While the CPU is performing inference, AUTO continues to load the model to the device best suited for the purpose and transfers the task to it when ready. 

Note user can disable this acceleration feature by excluding CPU from the priority list or disable ov::intel_auto::enable_startup_fallback.

## See also
 * [AUTO Plugin README](../README.md)
 * [OpenVINO™ README](../../../../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)