# AUTO Plugin Architecture

This guide includes the current architectural ideas and guidelines for AUTO Plugin development.

## Main Architectual Design Concepts

AUTO is a meta plugin in OpenVINO that doesn’t bind to a specific type of hardware device. When the user chooses AUTO as the target device, the AUTO plugin automatically discovers all processing units and hardware features of the platform, and selects one or more devices suitable for the task based on the application's configuration.

The logic behind the choice is as follows:
* Check what supported devices are available.
* Check performance hint of input setting (For detailed information of performance hint, please read more on the [ov::hint::PerformanceMode](https://docs.openvino.ai/2025/openvino-workflow/running-inference/optimize-inference/high-level-performance-hints.html)).
* Check precisions of the input model.
* Select the highest-priority device capable of supporting the given model for LATENCY hint and THROUGHPUT hint. Or Select all devices capable of supporting the given model for CUMULATIVE THROUGHPUT hint.
* If model’s precision is FP32 but there is no device capable of supporting it, offload the model to a device supporting FP16.

The AUTO plugin is also the default plugin for OpenVINO, if the user does not select a device explicitly in their application.

## Specific Features of AUTO plugin

### Accelerating First Inference Latency (FIL)

Compiling the model to accelerator-optimized kernels may take some time. When AUTO selects one accelerator, it can start inference with the system's CPU by default, as it provides very low latency and can start inference with no additional delays. While the CPU is performing inference, AUTO continues to load the model to the device best suited for the purpose and transfers the task to it when ready.

![alt text](https://docs.openvino.ai/2025/_images/autoplugin_accelerate.svg "AUTO cuts first inference latency (FIL) by running inference on the CPU until the GPU is ready")

The user can disable this acceleration feature by excluding CPU from the priority list or disabling `ov::intel_auto::enable_startup_fallback`. Its default value is `true`.

### Inference on Multiple Devices

The `ov::hint::performance_mode` property enables you to specify a performance option for AUTO to be more efficient for particular use cases, while the `CUMULATIVE_THROUGHPUT` hint enables running inference on multiple devices at once for higher throughput.

With CUMULATIVE_THROUGHPUT, AUTO loads the model to all available devices in the candidate list. Then, it runs inference on-device based on the device's priority.

### Runtime fallback

When inference with one device fails, AUTO can automatically fall back to a different device, sending it the same infer request. For the LATENCY or THROUGHPUT hints, AUTO will select a new device from the candidate list that best fits the task. For CUMULATVIE_THROUGHPUT, it will remove the failing device from the execution device list.

The user can disable this feature by disabling `ov::intel_auto::enable_runtime_fallback`. Its default value is `true`.

## See also
 * [AUTO Plugin README](../README.md)
 * [OpenVINO™ README](../../../../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)