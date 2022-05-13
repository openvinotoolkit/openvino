## Optimizing for the Latency {#openvino_docs_deployment_optimization_guide_latency}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_UG_Model_caching_overview

@endsphinxdirective

In a regular "consumer" use case a significant fraction of applications is focused on the situations where typically a single model is loaded (and single input is used) at a time.
While an application can create more than one request if needed (for example to support [asynchronous inputs population](@ref async_api)), the inference performance depends on **how many requests are being inferenced in parallel** on a device.

Similarly, when multiple models are served on the same device, it is important whether the models are executed simultaneously, or in chain (for example, in the inference pipeline).
As expected, the easiest way to achieve the lowest latency is **running only one concurrent inference at a moment** on the device. Accordingly, any additional concurrency usually results in the latency growing fast.

However, some conventional "root" devices (i.e., CPU or GPU) can be in fact internally composed of several "sub-devices". In many cases letting the OpenVINO to transparently leverage the "sub-devices" helps to improve the application throughput (e.g., serve multiple clients simultaneously) without degrading the latency. For example, multi-socket CPUs can deliver as high number of requests (at the same minimal latency) as there are NUMA nodes in the machine. Similarly, a multi-tile GPU (which is essentially multiple GPUs in a single package), can deliver a multi-tile scalability with the number of inference requests, while preserving the single-tile latency.

Furthermore, human expertise is required to get more _throughput_ out of the device, even in the inherently latency-oriented cases. OpenVINO can take this configuration burden via [high-level performance hints](../OV_Runtime_UG/performance_hints.md), `ov::hint::PerformanceMode::LATENCY` specified for the `ov::hint::performance_mode` property for the compile_model.

> **NOTE**: [OpenVINO performance hints](../OV_Runtime_UG/performance_hints.md) is a recommended way for performance configuration, which is both device-agnostic and future-proof.

In the case when there are multiple models to be used simultaneously, consider using different devices for inferencing the different models. Finally, when multiple models are executed in parallel on the device, using additional `ov::hint::model_priority` may help to define relative priorities of the models (refer to the documentation on the [matrix features support for OpenVINO devices](@ref features_support_matrix) to check for the support of the feature by the specific device).

**First-Inference Latency and Model Load/Compile Time**

There are cases when model loading/compilation are heavily contributing to the _end-to-end_ latencies.
For example, when the model is used exactly once, or when due to on-device memory limitations the model is unloaded (to free the memory for another inference) and reloaded at some cadence.

Such a "first-inference latency" scenario however may pose an additional limitation on the model load\compilation time, as inference accelerators (other than the CPU) usually require certain level of model compilation upon loading.
The [model caching](../OV_Runtime_UG/Model_caching_overview.md) is a way to amortize the loading/compilation time over multiple application runs. If the model caching is not possible (for example, it requires write permissions for the applications), the CPU device almost exclusively offers the fastest model load time. Also, consider using the [AUTO device](../OV_Runtime_UG/auto_device_selection.md). It allows to transparently use the CPU for inference, while the actual accelerator loads the model (upon that, the inference hot-swapping also happens automatically).

Finally, notice that any [throughput-oriented options](./dldt_deployment_optimization_tput.md) may increase the model up time significantly.
