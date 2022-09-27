## Optimizing for the Latency {#openvino_docs_deployment_optimization_guide_latency}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_Model_caching_overview

@endsphinxdirective

A significant portion of deep learning use cases involve applications loading a single model and using a single input at a time, which is the of typical "consumer" scenario.
While an application can create more than one request if needed, for example to support [asynchronous inputs population](@ref async_api), its **inference performance depends on how many requests are being inferenced in parallel** on a device.

Similarly, when multiple models are served on the same device, it is important whether the models are executed simultaneously or in a chain, for example, in the inference pipeline.
As expected, the easiest way to achieve **low latency is by running only one inference at a time** on one device. Accordingly, any additional concurrency usually results in latency rising fast.

However, some conventional "root" devices (i.e., CPU or GPU) can be in fact internally composed of several "sub-devices". In many cases, letting OpenVINO leverage the "sub-devices" transparently helps to improve application's throughput (e.g., serve multiple clients simultaneously) without degrading latency. For example, multi-socket CPUs can deliver as many requests at the same minimal latency as there are NUMA nodes in the system. Similarly, a multi-tile GPU, which is essentially multiple GPUs in a single package, can deliver a multi-tile scalability with the number of inference requests, while preserving the single-tile latency.

Typically, human expertise is required to get more "throughput" out of the device, even in the inherently latency-oriented cases. OpenVINO can take this configuration burden via [high-level performance hints](../OV_Runtime_UG/performance_hints.md), the `ov::hint::PerformanceMode::LATENCY` specified for the `ov::hint::performance_mode` property for the `compile_model`.

> **NOTE**: [OpenVINO performance hints](../OV_Runtime_UG/performance_hints.md) is a recommended way for performance configuration, which is both device-agnostic and future-proof.

When multiple models are to be used simultaneously, consider running inference on separate devices for each of them. Finally, when multiple models are executed in parallel on a device, using additional `ov::hint::model_priority` may help to define relative priorities of the models. Refer to the documentation on the [matrix features support for OpenVINO devices](@ref features_support_matrix) to check if your device supports the feature.

**First-Inference Latency and Model Load/Compile Time**

In some cases, model loading and compilation contribute to the "end-to-end" latency more than usual. 
For example, when the model is used exactly once, or when it is unloaded and reloaded in a cycle, to free the memory for another inference due to on-device memory limitations.

Such a "first-inference latency" scenario may pose an additional limitation on the model load\compilation time, as inference accelerators (other than the CPU) usually require a certain level of model compilation upon loading.
The [model caching](../OV_Runtime_UG/Model_caching_overview.md) option is a way to lessen the impact over multiple application runs. If model caching is not possible, for example, it may require write permissions for the application, the CPU offers the fastest model load time almost every time. 

Another way of dealing with first-inference latency is using the [AUTO device selection inference mode](../OV_Runtime_UG/auto_device_selection.md). It starts inference on the CPU, while waiting for the actual accelerator to load the model. At that point, it shifts to the new device seamlessly.

Finally, note that any [throughput-oriented options](./dldt_deployment_optimization_tput.md) may significantly increase the model uptime.
