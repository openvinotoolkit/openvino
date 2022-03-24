# High-level Performance Hints (Presets) {#openvino_docs_deployment_optimization_guide_hints}

Each of the OpenVINO's [supported devices](../OV_Runtime_UG/supported_plugins/Supported_Devices.md) offers a bunch of low-level performance settings. 
Tweaking this detailed configuration requires deep architecture understanding.

Also, while the resulting performance may be optimal for the specific combination of the device and the model that is inferred, it is actually neither device/model nor future-proof:
- Even within a family of the devices (like various CPUs), things like different number of CPU cores would eventually result in different execution configuration to be optimal.
- Similarly the optimal batch size is very much specific to the particular instance of the GPU.
- Compute vs memory-bandwidth requirements for the model being inferenced, as well as inference precision, possible model's quantization also contribute to the optimal parameters selection.
- Finally, the optimal execution parameters of one device do not transparently map to another device type, for example:
    - Both the CPU and GPU devices support the notion of the [streams](@ref throughput_streams), yet the optimal number of the streams is deduced very differently.
 
Here, to mitigate the performance configuration complexity the **Performance Hints** offer the high-level "presets" for the **latency** and **throughput**, as detailed in the [Performance Hints usage](../OV_Runtime_UG/performance_hints.md).

Beyond execution _parameters_ there is device-specific _scheduling_ that greatly affects the performance. 
Specifically, GPU-oriented optimizations like batching, which combines many (potentially tens) of inputs to achieve optimal throughput, do not always map well to the CPU, as e.g. detailed in the [internals](dldt_deployment_optimization_internals.md) sections.

The hints really hide the _execution_ specifics required to saturate the device. In the [internals](dldt_deployment_optimization_internals.md) sections you can find the implementation details (particularly how the OpenVINO implements the 'throughput' approach) for the specific devices. Keep in mind that the hints make this transparent to the application. For example, the hints obviates the need for explicit (application-side) batching or streams.

With the hints, it is enough to keep separate infer requests per camera or another source of input and process the requests in parallel using Async API as explained in the [application design considerations section](@ref throughput_app_design). The main requirement for the application to leverage the throughput is **running multiple inference requests in parallel**.


In summary, when the performance _portability_ is of concern, consider the Performance Hints as a solution. You may find further details and API examples [here](../OV_Runtime_UG/performance_hints.md). 