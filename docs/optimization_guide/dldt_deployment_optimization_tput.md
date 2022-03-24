# Optimizing for Throughput {#openvino_docs_deployment_optimization_guide_tput}

## General Throughput Considerations
As described in the section on the [latency-specific considerations](./dldt_deployment_optimization_latency.md) one possible use-case is delivering every single request at the minimal delay.
Throughput on the other hand, is about inference scenarios in which potentially large **number of inference requests are served simultaneously to improve the device utilization and total throughput**.

Also, the associated increase in latency is not linearly dependent on the number of requests executed in parallel.
Here, a trade-off between overall throughput and serial performance of individual requests can be achieved with the right OpenVINO performance configuration.

With the OpenVINO there are two low-level means of processing multiple inputs simultaneously: **batching** and **streams**, explained in this document. 
Both approaches can be used together or separately. However, all possible design decisions, are predicated on the requirement that sufficient parallelism exists, as detailed in the [next section](@ref throughput_app_design). 

Also, while low-level execution parameters tuning [(also detailed below)](ref throughput_advanced) may yield _maximum_ performance, the resulting configuration is not necessarily optimal for another device or model.
We encourage to use the new performance hints, [covered in this document](@ref throughput_hints), as the  _portable_ performance configuration approach for the OpenVINO.

@anchor throughput_app_design
## Throughput-Oriented Application Design
Most generally, throughput-oriented inference applications should:
* Expose substantial amounts of _inputs_ parallelism (e.g. process multiple video-sources)
* Decompose the data flow into a collection of concurrent inference requests that are aggressively scheduled to be executed in parallel
* Use the Async API with callbacks, to avoid any dependency on the requests' completion order, as explained in the [common-optimizations section](@ref openvino_docs_deployment_optimization_guide_common)


@anchor throughput_hints
## Basic OpenVINO Throughput Flow with Hints
Each of the OpenVINO's [supported devices](../OV_Runtime_UG/supported_plugins/Supported_Devices.md) offers a bunch of low-level performance settings. 
Tweaking this detailed configuration requires deep architecture understanding.
Additionally, the latency-throughput is not linearly dependent and very _device_ specific. It is also tightly integrated with _model_ characteristics.
To mitigate the performance configuration complexity the [Performance Hints](../OV_Runtime_UG/performance_hints.md) offer the high-level "presets" for the **latency** and **throughput**.

Beyond execution _parameters_ there is device-specific _scheduling_ that greatly affects the performance. 
Specifically, GPU-oriented tricks like batching, which combines many (potentially tens) of inputs to achieve optimal throughput, do not always map well to the CPU, as e.g. detailed in the [internals](dldt_deployment_optimization_internals.md) sections.

The hints really hide the _execution_ specifics required to saturate the device. For example, the hints obviates the need for explicit (application-side) batching or streams.
Instead, it is enough to keep a separate infer request per camera or another source of input and process the requests in parallel using Async API as explained in the [application design considerations section](@ref throughput_app_design).
The main requirement for the application to leverage the throughput is **running multiple inference requests in parallel**.

In the [internals](dldt_deployment_optimization_internals.md) sections you can find the implementation details (particularly how the OpenVINO implements the 'throughput' approach) for the specific devices. 
Keep in mind that the hints make this transparent to the application.

> **NOTE**: [OpenVINO performance hints](../OV_Runtime_UG/performance_hints.md) is a recommended way for performance configuration, which is both portable, device-agnostic and future-proof. 

## Advanced Flow: Explicit OpenVINO Streams
As detailed in the [common-optimizations section](@ref openvino_docs_deployment_optimization_guide_common) running multiple inference requests asynchronously is important for general application efficiency.
Internally, every device internally implements a queue. The queue acts as a buffer, storing the inference requests until retrieved by the device at its own pace. 
The devices may actually process multiple inference requests in parallel in order to improve the device utilization and overall throughput.
This configurable mean of this device-side parallelism is commonly referred as **streams**.

> **NOTE**: Notice that streams are **really executing the requests in parallel, but not in the lock step** (as e.g. the batching does), which makes the streams fully compatible with [dynamically-shaped inputs](../OV_Runtime_UG/ov_dynamic_shapes.md) when individual requests can have different shapes.

> **NOTE**: Most OpenVINO devices (including CPU, GPU and VPU) support the streams, yet the _optimal_ number of the streams is deduced very differently, please see the a dedicated section below.

Few general considerations:
* Using the streams does increase the latency of an individual request
   * When no number of streams is not specified, a device creates a bare minimum of streams (usually just one), as the latency-oriented case is default
   * Please find for the optimal number of the streams [below](ref throughput_advanced)
* Streams are memory-hungry, as every stream duplicates the intermediate buffers to do inference in parallel to the rest of streams
   * Always prefer streams over creating multiple `ov:Compiled_Model` instances for the same model, as weights memory is shared across streams, reducing the memory consumption
* Notice that the streams also inflate the model load (compilation) time.

### Further Details
For efficient asynchronous execution, the streams are actually handling the inference with a special pool of the threads (a thread per stream).
Each time you start inference requests (potentially from different application threads), they are actually muxed into a inference queue of the particular `ov:Compiled_Model`. 
If there is a vacant stream, it pops the request from the queue and actually expedites that to the on-device execution.
There are further device-specific details e.g. for the CPU, that you may find in the [internals](dldt_deployment_optimization_internals.md) section.

## Advanced Flow: Batching
Hardware accelerators like GPUs are optimized for massive compute parallelism, so the batching helps to saturate the device and leads to higher throughput.
While the streams (described earlier) already help to hide the communication overheads and certain bubbles in the scheduling, running multiple OpenCL kernels simultaneously is less GPU-efficient, compared to calling a kernel on the multiple inputs at once.   
As explained in the next section, the batching is a must to leverage maximum throughput on the GPUs.

There are two primary ways of using the batching to help application performance:
* Collecting the inputs explicitly on the application side and then _sending these batched requests to the OpenVINO_
   * Although this gives flexibility with the possible batching strategies, the approach requires redesigning the application logic
* _Sending individual requests_, while configuring the OpenVINO to collect and perform inference on the requests in batch [automatically](../OV_Runtime_UG/automatic_batching.md).
In both cases, optimal batch size is very device-specific. Also as explained below, the optimal batch size depends on the model, inference precision and other factors.
For OpenVINO devices that internally implement a dedicated heuristic, the `ov::optimal_batch_size` is a _device_ property (that accepts actual model as a parameter) to query the recommended batch size for the model.

@anchor throughput_advanced
## Advanced Flow:Choosing the Batch Size and Number of Streams
Predicting the inference performance is difficult and finding optimal execution parameters requires direct experiments with measurements.
Different devices behave differently with the batch sizes. The optimal batch size depends on the model, inference precision and other factors.
Similarly, different devices require different number of execution streams to saturate
Finally, in some cases  combination of streams and batching may be required to maximize the throughput.

One possible throughput optimization strategy is to **set an upper bound for latency and then increase the batch size or number of the streams until that tail latency is met (or the throughput is not growing anymore)**.
Also, consider [OpenVINO Deep Learning Workbench](@ref workbench_docs_Workbench_DG_Introduction) that builds handy latency vs throughput charts, iterating over possible values of the batch size and number of streams.


Few important configuration on the number of streams:
* Application can define the specific number of streams to use via the `ov::num_streams` e.g. as a configuration parameter of the `ov::Core::compile_model()`
   * Yet the _optimal_ number of the streams is both device- and model- specific


Below are general recommendations: 
* For the **CPU always prefer the streams** over the batching
   * Create as many streams as you application runs the requests simultaneously
   * Number of streams should be enough to meet the _average_ parallel slack rather than the peak load
   * _Maximum number of streams_ equals **total number of CPU cores**
      * As explained in the [CPU streams internals](dldt_deployment_optimization_internals.md), the CPU cores are evenly distributed between streams, so one core per stream is the finest-grained configuration
* For the **GPU**:
   * When the parallel slack is small (e.g. only 2-4 requests executed simultaneously), then using the streams for the GPU may suffice
      * Notice that the GPU runs 2 request per stream, so 4 requests can be served by 2 streams
      * Alternatively, consider single stream with small batch size (e.g. 2), that would total the same 4 inputs in flight  
   * _Maximum number of streams_ is usually 2, for more portability consider using the `ov::streams::AUTO` (`GPU_THROUGHPUT_AUTO` in the pre-OpenVINO 2.0 parlance)
   * Typically, for 4 and more requests the batching delivers better throughput for the GPUs
   * Batch size can be calculated as "number of inference requests executed in parallel" divided by the "number of requests that the streams consume"
      * E.g. if you process 16 cameras (by 16 requests inferenced _simultaneously_) with 2 GPU streams (each can process 2 requests), the batch size per request is 16/(2*2)=4 

> **NOTE**: When playing with [dynamically-shaped inputs](../OV_Runtime_UG/ov_dynamic_shapes.md) use only the streams (no batching), as they tolerate individual requests having different shapes. 

> **NOTE**: Using the [High-Level Performance Hints](../OV_Runtime_UG/performance_hints.md) explained in the next section, is the most portable and future-proof option, allowing the OpenVINO to find best combination of streams and batching for a given scenario and model. 

## Advanced Flow: Multi-Device Execution
OpenVINO offers _automatic_, [scalable multi-device inference](../OV_Runtime_UG/multi_device.md). This is simple _application-transparent_ way to improve the throughput. No need to re-architecture existing applications for any explicit multi-device support: no explicit network loading to each device, no separate per-device queues, no additional logic to balance the inference requests between devices, etc. From the application point of view, it is communicating to the single device that internally handles the actual machinery.
Just like with other throughput-oriented scenarios, there are two major pre-requisites for optimal multi-device performance:
*	Using the [Asynchronous API](@ref openvino_docs_deployment_optimization_guide_common) and [callbacks](../OV_Runtime_UG/ov_infer_request.md) in particular
*	Providing the multi-device (and hence the underlying devices) with enough data to crunch. As the inference requests are naturally independent data pieces, the multi-device performs load-balancing at the “requests” (outermost) level to minimize the scheduling overhead.

Notice that the resulting performance is usually a fraction of the “ideal” (plain sum) value, when the devices compete for a certain resources, like the memory-bandwidth which is shared between CPU and iGPU.
> **NOTE**: While the legacy approach of optimizing the parameters of each device separately works, the [OpenVINO performance hints](./dldt_deployment_optimization_hints.md) allow to configure all devices (that are part of the specific multi-device configuration) at once.
