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

Compiling the model to accelerator-optimized kernels may take some time. When AUTO select one accelerator, AUTO can start inference with the CPU of the system by default, as it provides very low latency and can start inference with no additional delays. While the CPU is performing inference, AUTO continues to load the model to the device best suited for the purpose and transfers the task to it when ready. 

Below is the example of CPU acceleration during GPU compilation. 
![alt text](https://docs.openvino.ai/latest/_images/autoplugin_accelerate.svg "AUTO cuts first inference latency (FIL) by running inference on the CPU until the GPU is ready")

User can disable this acceleration feature by excluding CPU from the priority list or disable ov::intel_auto::enable_startup_fallback.The default value of ov::intel_auto::enable_startup_fallback is true.

### Inference on Multiple Devices

The ov::hint::performance_mode property enables you to specify a performance option for AUTO to be more efficient for particular use cases. And the CUMULATIVE_THROUGHPUT hint enables running inference on multiple devices for higher throughput. 

With CUMULATIVE_THROUGHPUT, AUTO loads the network model to all available devices in the candidate list, and then runs inference on them based on the devices priority.

### Runtime fallback

When the inference of the currently selected device fails, AUTO can automatically fall back this infer request to other device. For LATENCY hint or THROUGHPUT hint, AUTO will select new capable device from the next cadidate device. For CUMULATVIE_THROUGHPUT, AUTO will remove failed device from execution device list.

User can disable this feature by disable ov::intel_auto::enable_runtime_fallback.The default value of ov::intel_auto::enable_runtime_fallback is true.

## See also
 * [AUTO Plugin README](../README.md)
 * [OpenVINO™ README](../../../../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)