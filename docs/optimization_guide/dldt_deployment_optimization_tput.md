# Optimizing for Throughput {#openvino_docs_deployment_optimization_guide_tput}

## General Throughput Considerations
As described in the section on the [latency-specific considerations](./dldt_deployment_optimization_latency.md) one possible use-case is delivering every single request at the minimal delay.
Throughput on the other hand, is about inference scenarios in which potentially large **number of inference requests are served simultaneously to improve the device utilization**.

Here, the overall application inference rate can be significantly improved with the right performance configuration.
Also, if the model is not already memory bandwidth-limited, the associated increase in latency is not linearly dependent on the number of requests executed in parallel.
With the OpenVINO there are two major means of processing multiple inputs simultaneously: **batching** and **streams**, explained in this document.

## OpenVINO Streams
As detailed in the [common-optimizations section](@ref openvino_docs_deployment_optimization_guide_common) running multiple inference requests asynchronously is important for general application efficiency.
The [Asynchronous API](./dldt_deployment_optimization_common.md) is in fact the "application side" of scheduling, as every device internally implements a queue. The queue acts as a buffer, storing the inference requests until retrieved by the device at its own pace. 

Further, the devices may actually process multiple inference requests in parallel in order to improve the device utilization and overall throughput. This parallelism is commonly referred as 'streams'. Some devices (like GPU) may run several requests per stream to amortize the host-side costs.
Notice that streams are **really executing the requests in parallel, but not in the lock step** (as e.g. the batching does), which makes the streams fully compatible with [dynamically-shaped inputs](../OV_Runtime_UG/ov_dynamic_shapes.md) when individual requests can have different shapes. 

For efficient asynchronous execution, the streams are actually handling inference with special pool of the threads.
So each time you start inference requests (potentially from different application threads), they are actually muxed into a inference queue of the particular `ov:Compiled_Model`. 
If there is a vacant stream, it pops the request from the queue and actually expedites that to the on-device execution.

The multi-streams approach is inherently throughput-oriented, as every stream requires a dedicated device memory to do inference in parallel to the rest of streams.
Although similar, the streams are always preferable compared to creating multiple `ov:Compiled_Model` instances for the same model, as weights memory is shared across streams, reducing the overall memory consumption.
Notice that the streams inflate the model load/compilation time.
Finally, using streams does increase the latency of an individual request, this is why for example the [latency hint](./dldt_deployment_optimization_hints.md) governs a device to create a bare minimum of streams (usually just one).
Please find the considerations for the optimal number of the streams in the later sections.

## Batching
Hardware accelerators like GPUs are optimized for massive compute parallelism, so the batching helps to saturate the device and leads to higher throughput.
While the streams (described) earlier already allow to hide the communication overheads and certain bubbles in the scheduling, running multiple OpenCL kernels simultaneously is less GPU-efficient, compared to calling a kernel on the multiple inputs at once.   
As explained in the next section, the batching is a must to leverage maximum throughput on the GPUs.

There are two primary ways of using the batching to help application performance:
* Collecting the inputs explicitly on the application side and then _sending these batched requests to the OpenVINO_
   * Although this gives flexibility with the possible batching strategies, the approach requires redesigning the application logic
* _Sending individual requests_, while configuring the OpenVINO to collect and perform inference on the requests in batch [automatically](../OV_Runtime_UG/automatic_batching.md).
In both cases, optimal batch size is very device-specific. Also as explained below, the optimal batch size depends on the model, inference precision and other factors.

## Choosing the Batch Size and Number of Streams
Predicting the inference performance is difficult and finding optimal execution parameters requires direct experiments with measurements.
One possible throughput optimization strategy is to **set an upper bound for latency and then increase the batch size or number of the streams until that tail latency is met (or the throughput is not growing anymore)**.
Also, consider [Deep Learning Workbench](@ref workbench_docs_Workbench_DG_Introduction) that builds handy latency vs throughput charts, iterating over possible values of the batch size and number of streams.

Different devices behave differently with the batch sizes. The optimal batch size depends on the model, inference precision and other factors. Similarly, different devices require different number of execution streams to maximize the throughput.
Below are general recommendations: 
* For the **CPU always prefer the streams** over the batching
   * Create as many streams as you application runs the requests simultaneously
   * Number of streams should be enough to meet the _average_ parallel slack rather than the peak load
   * _Maximum number of streams_ equals **total number of CPU cores**
      * As explained in the [CPU streams internals](dldt_deployment_optimization_internals.md), the CPU cores are evenly distributed between streams, so one core per stream is the finest-grained configuration
* For the **GPU**:
   * When the parallel slack is small (e.g. only 2-4 requests executed simultaneously), then using the streams for the GPU may suffice
      * Notice that the GPU runs 2 request per stream
   * _Maximum number of streams_ is usually 2, for more portability consider using the `ov::streams::AUTO` (`GPU_THROUGHPUT_AUTO` in the pre-OpenVINO 2.0 parlance)
   * Typically, for 4 and more requests the batching delivers better throughput for the GPUs
   * Batch size can be calculated as "number of inference requests executed _in parallel_" divided by the "number of requests that the streams consume"
      * E.g. if you process 16 cameras (by 16 requests inferenced _simultaneously_) with 2 GPU streams (each can process 2 requests), the batch size per request is 16/(2*2)=4 

> **NOTE**: When playing with [dynamically-shaped inputs](../OV_Runtime_UG/ov_dynamic_shapes.md) use only the streams (no batching), as they tolerate individual requests having different shapes. 

> **NOTE**: Using the [High-Level Performance Hints](../OV_Runtime_UG/performance_hints.md) explained in the next section, is the most portable and future-proof option, allowing the OpenVINO to find best combination of streams and batching for a given scenario and model. 

## OpenVINO Hints: Selecting Optimal Execution and Parameters **Automatically**
Overall, the latency-throughput is not linearly dependent and very _device_ specific. It is also tightly integrated with _model_ characteristics.
As for the possible inference devices the scenery had already become pretty diverse, the OpenVINO has introduced the dedicated notion of the high-level performance configuration "hints" to describe the target application scenarios.
The hints are described [here](./dldt_deployment_optimization_hints.md). 

The hints also obviates the need for explicit (application-side) batching. With the hints, the only requirement for the application is to run multiple individual requests using [Async API](./dldt_deployment_optimization_common.md) and let the OpenVINO decide whether to collect the requests and execute them in batch, streams, or both.

> **NOTE**: [OpenVINO performance hints](./dldt_deployment_optimization_hints.md) is a recommended way for performance configuration, which is both device-agnostic and future-proof. 

## Multi-Device Execution
OpenVINO offers _automatic_, [scalable multi-device inference](../OV_Runtime_UG/multi_device.md). This is simple _application-transparent_ way to improve the throughput. No need to re-architecture existing applications for any explicit multi-device support: no explicit network loading to each device, no separate per-device queues, no additional logic to balance the inference requests between devices, etc. From the application point of view, it is communicating to the single device that internally handles the actual machinery.
Just like with other throughput-oriented scenarios, there are two major pre-requisites for optimal multi-device performance:
*	Using the [Asynchronous API](@ref openvino_docs_deployment_optimization_guide_common) and [callbacks](../OV_Runtime_UG/ov_infer_request.md) in particular
*	Providing the multi-device (and hence the underlying devices) with enough data to crunch. As the inference requests are naturally independent data pieces, the multi-device performs load-balancing at the “requests” (outermost) level to minimize the scheduling overhead.

Notice that the resulting performance is usually a fraction of the “ideal” (plain sum) value, when the devices compete for a certain resources, like the memory-bandwidth which is shared between CPU and iGPU.
> **NOTE**: While the legacy approach of optimizing the parameters of each device separately works, the [OpenVINO performance hints](./dldt_deployment_optimization_hints.md) allow to configure all devices (that are part of the specific multi-device configuration) at once.
