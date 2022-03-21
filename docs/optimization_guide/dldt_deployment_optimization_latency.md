## Optimizing for the Latency {#openvino_docs_deployment_optimization_guide_latency}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
    
   openvino_docs_IE_DG_Model_caching_overview

@endsphinxdirective

## Latency Specifics
A significant fraction of applications focused on the situations where typically a single model is loaded (and single input is used) at a time.
This is a regular "consumer" use case and a default (also for the legacy reasons) performance setup for any OpenVINO device.
Notice that an application can  create more than one request if needed (for example to support [asynchronous inputs population](./dldt_deployment_optimization_common.md)), the performance depends on **how many requests are being executed in parallel**.

Similarly, when multiple models are served on the same device, it is important whether the models are executed simultaneously, or in chain (for example in the inference pipeline).
As expected, the easiest way to achieve the lowest latency is **running only one concurrent inference at a moment**. Accordingly, any additional concurrency usually results in the latency growing fast.

However, some conventional "root" device (e.g. CPU or GPU) can be in fact internally composed of several "sub-devices".  In many cases letting the OpenVINO to transparently leverage the "sub-devices" helps to improve the application throughput (e.g. serve multiple clients simultaneously) without degrading the latency.  For example, the multi-socket CPUs can deliver as high number of requests (at the same minimal latency) as there are NUMA nodes in the machine. Similarly, a multi-tile GPU (which is essentially multiple GPUs in a single package), can deliver a multi-tile scalability with the number of inference requests, while preserving the single-tile latency.

Thus, human expertise is required to get more _throughput_ out of the device even in the latency-oriented cases. OpenVINO can take the configuration burden via [OpenVINO high-level performance hints](../OV_Runtime_UG/performance_hints.md).

> **NOTE**: [OpenVINO performance hints](./dldt_deployment_optimization_hints.md) is a recommended way for performance configuration, which is both device-agnostic and future-proof. 

In the case when there are multiple models to be used simultaneously, consider using different devices for inferencing the different models. Finally, when multiple models are executed in parallel on the device, using additional `ov::hint::model_priority` may help to define relative priorities of the models (please refer to the documentation on the [matrix features support for OpenVINO devices](ref @openvino_docs_OV_UG_features_support_matrix) to check for the support of the feature by the specific device).

## First-Inference Latency and Model Load/Compile Time
There are cases when model loading/compilation are heavily contributing to the end-to-end latencies.
For example when the model is used exactly once, or when due to on-device memory limitations the model is unloaded (to free the memory for another inference) and reloaded at some cadence.

Such a "first-inference latency" scenario however may pose an additional limitation on the model load\compilation time, as inference accelerators (other than the CPU) usually require certain level of model compilation upon loading.
The [model caching](../OV_Runtime_UG/Model_caching_overview.md) is a way to amortize the loading/compilation time over multiple application runs. If the model caching is not possible (as e.g. it requires write permissions for the applications), the CPU device is almost exclusively offers the fastest model load time. Also, consider using the [AUTO device](../OV_Runtime_UG/auto_device_selection.md). It allows to transparently use the CPU for inference, while the actual accelerator loads the model (upon that, the inference hot-swapping also happens automatically).

Finally, notice that any [throughput-oriented options](./dldt_deployment_optimization_tput.md) may increase the model up time significantly.
